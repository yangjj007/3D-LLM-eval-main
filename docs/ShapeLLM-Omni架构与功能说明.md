# ShapeLLM-Omni 架构与功能说明

本文档基于仓库代码与配置，系统梳理 **ShapeLLM-Omni** 的结构、数据流、能力边界以及（可从代码推断的）训练相关设计，便于理解「ShapeLLM 是什么、能做什么、大致是怎样被训练出来的」。

---

## 一、ShapeLLM-Omni 是什么

**ShapeLLM-Omni** 是一个**原生多模态大语言模型**，面向 **3D 生成与理解**（NeurIPS 2025 Spotlight）。核心特点：

- **多模态输入**：支持文本、图像、3D 网格（mesh/体素）作为输入。
- **3D 作为“语言”**：通过 **3D VQVAE** 把体素形状离散化成 **token 序列**（如 `<mesh0>` … `<mesh8191>`），与文本、图像一起送入 **Qwen2.5-VL**，使模型能「读 3D、写 3D、聊 3D」。
- **高质量几何输出**：生成或编辑得到的 3D token 序列，经 VQVAE 解码回体素后，再通过 **TRELLIS** 管线生成精细的 **mesh**、**Gaussian Splatting** 等表示，并可导出 GLB。

因此，从结构上可以概括为：

```
[ 文本 / 图像 / 3D ] → [ Qwen2.5-VL + 3D Token 词表 ] → [ 文本 + 3D Token 序列 ]
                                                              ↓
[ 3D Token 序列 ] → [ 3DVQVAE 解码 ] → [ 体素 ] → [ TRELLIS ] → [ Mesh / Gaussian / 渲染 ]
```

---

## 二、整体架构概览

### 2.1 三大核心部分

| 模块 | 作用 | 在仓库中的位置 |
|------|------|----------------|
| **ShapeLLM 主模型** | 多模态 LLM：理解/生成文本、图像、3D token | 基于 `Qwen2_5_VLForConditionalGeneration`，权重来自 `yejunliang23/ShapeLLM-7B-omni` |
| **3D VQVAE (3DVQVAE)** | 体素 ↔ 离散 token（8192 词表，1024 token/形状） | `trellis/models/sparse_structure_vqvae.py`，权重 `yejunliang23/3DVQVAE` |
| **TRELLIS 管线** | 文本/图像条件 + 体素结构 → 结构化潜在 → Mesh/Gaussian/辐射场 | `trellis/pipelines/`，预训练 `JeffreyXiang/TRELLIS-text-xlarge` 等 |

### 2.2 推理时的数据流（简化）

1. **输入侧**
   - **文本**：直接进入 Qwen2.5-VL。
   - **图像**：经 processor 与 vision 编码，与文本一起构成多模态输入。
   - **3D**：Mesh/OBJ/GLB → 归一化 → 体素化 64³ → 3DVQVAE 编码 → 1024 个 token → 转成 `<mesh-start><mesh i_1>…<mesh i_1024><mesh-end>` 拼进对话。

2. **模型侧**
   - 多模态输入（文本 + 可选图像 + 可选 3D token 序列）→ ShapeLLM（Qwen2.5-VL）→ 生成回复。
   - 若任务为「生成/编辑 3D」，回复中会包含一段 **3D token 序列**（同上格式）。

3. **输出侧**
   - 从回复中解析出 1024 个 `<mesh k>` 的索引 → 3DVQVAE 解码 → 64³ 体素 → 体素坐标送入 TRELLIS → `sample_slat` + `decode_slat` → 得到 mesh / gaussian 等 → 后处理为 GLB/视频。

---

## 三、核心组件详解

### 3.1 3D VQVAE（体素 ↔ 离散 Token）

**文件**：`trellis/models/sparse_structure_vqvae.py`  
**类**：`VQVAE3D`（内部含 `SparseStructure_vqEncoder`、`SparseStructure_vqDecoder`、`VectorQuantizer`）

- **输入**：`(B, 1, 64, 64, 64)` 二值体素（或浮点 occupancy）。
- **编码**：3D CNN 下采样到潜在空间，再经 VQ：潜在形状为 `(B, 8, 8, 16)`，展平后每格一个离散索引 → 共 **8×8×16 = 1024** 个 token。
- **词表大小**：默认 **8192**（`num_embeddings=8192`），即每个 token 取值 0..8191。
- **解码**：索引 → embedding →  reshape 成 `(B, 8, 16, 16, 16)` 再 3D CNN 上采样 → `(B, 1, 64, 64, 64)` 体素。

因此：**一个 3D 形状在 LLM 里被表示成 1024 个 token，每个 token 是 0~8191 的整数**，对应词元即 `<mesh0>` … `<mesh8191>`。

### 3.2 Mesh Token 的文本格式

- **序列格式**：`<mesh-start>` + 1024 个 `<mesh{k}>` + `<mesh-end>`。
- **用途**：
  - **输入**：把 3D 形状以「可读 token 序列」的形式交给 LLM，实现 3D 理解、描述、编辑指令等。
  - **输出**：LLM 生成 3D 时，在回复中生成同样格式的序列；解析后得到 1024 个索引，再交给 3DVQVAE 解码。

- **进入 LLM 的 3D token 数量规定**：**固定为 1024 个**。  
  - 来源：3DVQVAE 编码器把 64³ 体素下采样到 8×8×16 的潜在格点，每格 1 个离散索引，共 8×8×16 = **1024**（见 `sparse_structure_vqvae.py` 中 `encoding_indices.view(bs, h*w*d)` 即 `[bs, 1024]`）。  
  - 输入时：一个 3D 形状必定编码成 1024 个 `<mesh k>` 再送入 LLM（`app.py` 中 `token_to_words` 为 `for j in range(1024)`）。  
  - 输出时：从 LLM 回复解析出的 mesh 索引若不足 1024，会按代码**补齐到 1024**（如重复最后一个索引）；若超过则取前 1024 个，再交给 3DVQVAE 解码。  
  因此模型在训练与推理时，每个 3D 形状对应的 mesh token 长度均为 **1024**，无变长。

解析逻辑见 `app.py` / `main.py` 中的 `token_to_mesh` / 从 `response_text` 里按 `"><mesh"` 分割并提取数字。

### 3.3 ShapeLLM 主模型（Qwen2.5-VL + 3D 词表）

- **基座**：`Qwen2_5_VLForConditionalGeneration`（支持图像、文本多模态）。
- **扩展**：在词表中加入 3D 相关特殊 token，使模型能：
  - **读**：理解 `<mesh-start>…<mesh-end>` 表示的 3D 形状；
  - **写**：在生成时输出 3D token 序列。
- **结束符**：除常规 `eos_token_id` 外，使用 `159858` 作为 3D 相关结束符（见 `app.py` 中 `eos_token_id = [tokenizer.eos_token_id, 159858]`）。
- **权重**：`yejunliang23/ShapeLLM-7B-omni`（7B 规模）。

### 3.4 TRELLIS 管线（从体素到高质量 3D）

TRELLIS 负责：在「稀疏结构」（体素占用）和「文本/图像条件」下，生成**结构化潜在（SLat）**，再解码为多种 3D 表示。

- **文本→3D**：`TrellisTextTo3DPipeline`（`trellis/pipelines/trellis_text_to_3d.py`）
  - 文本 → CLIP 编码 → 条件；
  - **sparse structure**：可由「纯文本」从头采样（`sample_sparse_structure`），或由「已有体素坐标」给出（3D 编辑/变体）；
  - 在给定结构上 **sample_slat**（Flow Matching）→ **decode_slat** → mesh / gaussian / radiance_field。
- **图像→3D**：`TrellisImageTo3DPipeline`（`trellis/pipelines/trellis_image_to_3d.py`）
  - 图像 → DINOv2 等编码 → 条件；
  - 结构同样可从头采样或由体素坐标给出；
  - 同样经过 sample_slat → decode_slat → mesh / gaussian 等。

**关键方法**（以文本管线为例）：

- `get_cond(prompt)`：文本 → 条件 embedding。
- `sample_sparse_structure(cond)`：仅「文本→3D」时使用，从噪声采样 64³ 占用。
- `voxelize(mesh)`：Mesh → 64³ 体素坐标（与 3DVQVAE 的体素化一致）。
- `sample_slat(cond, coords)`：在给定体素坐标和条件下，采样 SLat（稀疏张量）。
- `decode_slat(slat, ['mesh','gaussian'])`：SLat → mesh + Gaussian，用于渲染和导出 GLB。

因此：**LLM 只负责生成「形状 token」；精细几何与外观由 TRELLIS 在体素结构上条件生成。**

### 3.5 表示与后处理

- **表示**（`trellis/representations/`）：Strivec（辐射场）、Octree、Gaussian、MeshExtractResult。
- **后处理**（`trellis/utils/postprocessing_utils.py`）：例如 `to_glb(gaussian, mesh, simplify=..., texture_size=...)` 生成可导出的 GLB。
- **渲染**（`trellis/utils/render_utils.py`）：如 `render_video` 用于预览与 Demo 视频。

---

## 四、能实现的功能（从代码与 README 推断）

1. **文本 → 3D**  
   用户输入文本描述 → ShapeLLM 生成带 3D token 的回复 → 解析 token → 3DVQVAE 解码 → TRELLIS 文本条件 refine → 输出 mesh/GLB/视频。

2. **图像 → 3D**  
   用户上传图像 → 与文本一起送入 ShapeLLM；若为「根据图像生成 3D」，则同样会生成 3D token 序列，再经 3DVQVAE + TRELLIS 图像条件管线得到 3D。

3. **3D 理解与描述**  
   用户上传 3D（mesh/GLB）→ 体素化 → 3DVQVAE 编码 → 转成 `<mesh-start>…<mesh-end>` 拼进 prompt → LLM 回复自然语言描述或分析。

4. **3D 编辑 / 变体**  
   用户提供 3D + 文本指令（如「生成一个变体」「改成某风格」）→ 同一套 3D token 输入 + 指令 → LLM 生成新的 3D token 序列 → 解码后用 TRELLIS 的 `run_variant(mesh, prompt)` 式流程（体素坐标 + 文本条件）生成编辑/变体结果。

5. **多轮对话**  
   Demo（`app.py`）支持多轮；`task_history` / `messages` 中可混合文本、图像、3D，实现多轮 3D 对话与连续编辑。

具体任务模板可参考 README 中提到的 `templates.txt`（若仓库提供）。

---

## 五、训练相关（从代码与配置能看出的部分）

README 注明：**训练代码尚未完全开源**；以下仅从现有代码和配置推断可能的训练设计。

### 5.1 3D VQVAE 的训练

- **配置**：`configs/vae/ss_vae_conv3d_16l8_fp16.json`
- **数据**：`SparseStructure` 数据集，64³ 体素，可选 `min_aesthetic_score` 过滤。
- **训练器**：`SparseStructureVaeTrainer`（`trellis/trainers/vae/` 下应有对应实现），loss 含 dice、KL 等。
- **流程**：体素 → Encoder → 潜在 → (VQ) → Decoder → 重建体素；VQ 的 commitment / codebook loss 一起训练。

### 5.2 TRELLIS 的训练（文本/图像条件 3D）

- **配置**：如 `configs/generation/slat_flow_txt_dit_L_64l8p2_fp16.json`
- **模型**：`ElasticSLatFlowModel`（Flow Matching），在 64³ 结构化潜在上、以文本/图像为条件。
- **数据**：`TextConditionedSLat`，带 `normalization`（mean/std）、`latent_model`、`slat_dec` 等；即先有「形状+外观」的潜在，再学条件生成。
- **训练器**：`TextConditionedSparseFlowMatchingCFGTrainer`，Classifier-Free Guidance（如 `p_uncond: 0.1`），优化器 AdamW、EMA、梯度裁剪等。

### 5.3 数据集与数据管线（dataset_toolkits）

- **build_metadata.py**：按数据集（如 3D-FUTURE、ObjaverseXL、ABO、Toys4k、HSSD）构建 `metadata.csv`，管理下载、体素化、特征等。
- **voxelize**：Mesh → 64³ 体素（与推理端一致）。
- **encode_latent / encode_ss_latent**：将体素/稀疏结构编码为潜在或 SS latent，供 TRELLIS 训练使用。
- **extract_feature**、**render**、**render_cond**：用于图像/多视角特征与渲染条件，可能用于图像→3D 或审美过滤（如 `min_aesthetic_score`）。

README 提到 **3D-Alpaca** 数据集（50k 高质量 3D 编辑对）和即将开放的更大 3D 编辑集，可推断：**ShapeLLM 的 3D 理解与生成能力**，依赖「文本+3D token」或「图像+3D token」的配对数据（如 3D-Alpaca）在 Qwen2.5-VL 上进行指令/对话微调；**3DVQVAE 与 TRELLIS 则先在大量 3D 数据上预训练**，再与 LLM 组合做推理。

### 5.4 数据管线大致流程（dataset_toolkits）

1. **build_metadata**：按数据集脚本（如 `datasets/ObjaverseXL`）生成 `metadata.csv`，包含 sha256、路径、是否已体素化等。
2. **download**：按元数据下载原始模型（若需要）。
3. **voxelize**：Mesh → 64³ 体素，写入如 `voxels/{sha256}.ply` 或等价表示。
4. **encode_latent / encode_ss_latent**：体素 → (SS) latent，供 TRELLIS 的 Flow 或 VAE 训练。
5. **render / render_cond / extract_feature**：多视角渲染、条件图、特征提取，用于图像条件或质量过滤。

---

## 六、推理流程小结（代码级）

1. **加载模型**  
   - 3DVQVAE：`VQVAE3D(num_embeddings=8192)`，加载 `yejunliang23/3DVQVAE` 权重。  
   - ShapeLLM：`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor`，加载 `yejunliang23/ShapeLLM-7B-omni`。  
   - TRELLIS：`TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")`，可选 `TrellisImageTo3DPipeline`。

2. **输入 3D 时**  
   - Mesh/GLB → 归一化到 [-0.5, 0.5] → 64³ 体素化（与 `trellis_text_to_3d.py` 的 `voxelize` 一致；注意 main.py 中有一致旋转约定）→ `vqvae.Encode(ss)` → 1024 个索引 → `token_to_words(token)` → `<mesh-start>…<mesh-end>` 拼入用户 prompt。

3. **调用 LLM**  
   - `processor.apply_chat_template` + `process_vision_info`（若有图）→ `model.generate(..., eos_token_id=[..., 159858])`。

4. **解析 3D 输出**  
   - 从生成文本中解析 `<mesh k>` → 得到 1024 个索引，不足则 padding（如重复最后 token）→ `encoding_indices`。

5. **从 token 到 3D 资产**  
   - `vqvae.Decode(encoding_indices)` → 体素 logits → 二值化 → 体素坐标；  
   - 体素坐标 → `pipeline_text.sample_slat(cond, coords)`（或 image 管线）→ `decode_slat(slat, ['mesh','gaussian'])`；  
   - `postprocessing_utils.to_glb(...)` → 导出 GLB；可选 `render_utils.render_video` 做预览。

---

## 七、仓库目录结构（与架构对应）

```
ShapeLLM-Omni/
├── app.py                    # Gradio Demo：多模态对话、3D 生成/理解/编辑
├── main.py                   # 脚本式推理示例：GLB → token → LLM → 解析 → TRELLIS → GLB
├── requirements.txt          # 依赖（PyTorch、transformers、trellis、qwen_vl_utils 等）
├── configs/
│   ├── vae/                  # VQVAE/VAE 配置（如 ss_vae_conv3d_16l8_fp16.json）
│   └── generation/           # TRELLIS 生成模型配置（slat_flow_txt/img_dit_*）
├── dataset_toolkits/         # 数据构建：元数据、体素化、编码、渲染等
│   ├── build_metadata.py
│   ├── voxelize.py
│   ├── encode_latent.py
│   ├── encode_ss_latent.py
│   ├── render.py
│   └── datasets/             # 各 3D 数据集接口（3D-FUTURE, ObjaverseXL, ABO, …）
├── trellis/                  # TRELLIS 核心（与微软 TRELLIS 同源）
│   ├── models/               # 3D VQVAE、Sparse Structure VAE/Flow、SLat VAE/Flow
│   ├── pipelines/            # TrellisTextTo3DPipeline、TrellisImageTo3DPipeline
│   ├── trainers/             # VAE、Flow Matching 训练器
│   ├── representations/      # Mesh、Gaussian、Strivec、Octree
│   ├── renderers/
│   └── utils/                # 渲染、后处理、数据等
└── docs/
    └── ShapeLLM-Omni架构与功能说明.md  # 本文档
```

---

## 八、总结

- **ShapeLLM-Omni** = **Qwen2.5-VL** + **3D Token 词表**（由 3DVQVAE 的 8192 个 code 定义）+ **TRELLIS**（体素→高质量 mesh/Gaussian）。
- **3D 在模型中的形态**：64³ 体素经 3DVQVAE 变为 **1024 个离散 token**（0~8191），以 `<mesh-start><mesh k_1>…<mesh k_1024><mesh-end>` 的形式与文本/图像一起输入输出。
- **能做的事**：文本/图像→3D、3D 理解与描述、3D 编辑/变体、多轮多模态对话；**训练**上，VQVAE 与 TRELLIS 部分有配置与数据管线可循，LLM 侧依赖 3D-Alpaca 等指令数据微调（具体训练脚本以官方后续发布为准）。

通过本文档可以较完整地把握 ShapeLLM-Omni 的「长什么样」以及「从数据到推理」的整条链路，便于在此基础上做二次开发或复现实验。

---

## 九、如何运行

- **环境**：按 README 或 [TRELLIS](https://github.com/microsoft/TRELLIS)、[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) 配置环境，或 `pip install -r requirements.txt`（需 CUDA、spconv、flash-attention 等）。
- **Demo**：`python app.py` 启动 Gradio 界面，支持上传图像/3D、输入文本、多轮对话与 3D 生成/编辑。
- **脚本推理**：`main.py` 中指定 `INPUT_GLB` 与 `PROMPT`，运行后将在 `./output_shapellm` 得到可视化与结果 GLB。
- **模型路径**：主模型默认从 `MODEL_DIR`（如 `/wangcm/ShapeLLM-7B-omni`）或 HuggingFace `yejunliang23/ShapeLLM-7B-omni` 加载；3DVQVAE 从 `yejunliang23/3DVQVAE` 自动下载。
