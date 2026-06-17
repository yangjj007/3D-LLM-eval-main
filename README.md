# 3D-LLM-eval-main

统一评测框架：支持 **ShapeLLM-Omni** 与 **Sparse-SDF-VQVAE + Qwen3-VL** 两套后端（`eval/adapters/`），多 GPU 数据并行、断点续测、样本级 `per_sample.jsonl` 与可选 mesh 导出。

## 1. 依赖安装

```bash
conda create -n eval3d python=3.10 -y
conda activate eval3d
cd /path/to/3D-LLM-eval-main
pip install -r requirements.txt

# 文本指标扩展（可选）
pip install pycocoevalcap bert-score sentence-transformers

# Sparse 后端：本仓库已自带 vendored ``trellis/models/autoencoders`` 与 ``trellis/utils/mesh_utils``。
# 若需从上游同步，见 scripts/vendor_sparse_trellis_from_med.ps1（或 .sh）。仍需 torchsparse / spconv 等（与训练侧一致）。
```

`requirements.txt` 已含：PyTorch、transformers、trimesh、open3d、scipy、spconv、**scikit-image**（ShapeLLM 体素转 mesh）、**sentence-transformers**（Sentence-BERT / SimCSE 指标）等。

**Hugging Face 缓存（建议固定在仓库内）**：在任务 YAML 中设置 `model.hf_cache_dir: ./eval_data/hf_cache`（[`eval/configs/default.yaml`](eval/configs/default.yaml) 已默认）。`eval/model_loader.py` 会把**相对路径**解析为**仓库根**下的绝对路径，避免工作目录变化导致下到别处；实际目录为 **`eval_data/hf_cache/hub`**（`hf_hub_download` / Trellis pipeline）、**`eval_data/hf_cache/transformers`**（Transformers / Trellis 内 CLIP）。从 Hub 拉 Trellis 时若遇断网、`Connection reset`、`RuntimeError: client has been closed` 等，会自动退避重试并尝试重置 hub 的 HTTP 会话。

## 2. 数据准备（mesh↔caption）

### 2.1 Sparse 后端：从 GLB 算 SDF（默认）

**Sparse（`sparse_sdf_qwen3`）不再依赖预计算 SDF 目录。** 推理时在 adapter 内用 `mesh_path`（GLB）调用 `mesh2sparse_sdf` 得到稀疏 SDF。任务 YAML 中：

- `model.sdf_from_mesh_only` 默认为 **true**：忽略数据里的 `sdf_path`，不从 `{uid}_r512.npz` 读取。
- `model.sdf_cache_dir`（可选）：把「从 GLB 算出的」稀疏 SDF 缓存在该目录以加速重复评测，与训练用离线 SDF 数据集不是同一概念。

任务数据使用与普通评测相同的 **`eval_data/understanding.json`**、**`generation.json`**、**`vqvae_recon.json`**（不再需要 `*_sparse.json`）。

若仍要为 JSON 写入预计算 `sdf_path` 并生成 `*_sparse.json`（legacy），见 `python -m eval.data.build_sparse_eval_datasets --help`。

### 2.2 从 metadata + GLB 驱动评测（推荐，可跳过手写 JSON）

在任务 YAML 的 `data:` 下设置 **`metadata_csv`** 与 **`glb_dir`**（与 `{file_identifier}.glb` 同目录），并**去掉或注释** `data_path`，即可在加载数据集时按 `metadata.csv` 在内存中构造样本，无需先运行 `build_eval_from_metadata`。

**understanding / generation** 仍要求 CSV 中含可解析的 **`captions`** 列（与 `build_eval_from_metadata` 一致）；**vqvae_recon** 仅需 `file_identifier` 与对应 GLB。

可选字段：

- `gen_caption_indices`：仅 **generation** 任务，逗号分隔 caption 索引，默认 `0`。
- `default_prompt`：仅 **understanding** 任务。
- **vqvae_recon**：仅需 `metadata_csv` + `glb_dir`；按存在 `{file_identifier}.glb` 的行列出样本，**不要求** CSV 中含 `captions`。

仍可通过 `python -m eval.data.build_eval_from_metadata ...` **显式写出** `understanding.json` / `generation.json`（例如给外部工具或离线检查）。

### 2.3 Objaverse / TRELLIS 数据流水线（集成 CLI）

在仓库根目录执行：

```bash
# 1) 拉取 TRELLIS-500K 元数据表（需能访问 HuggingFace 数据集 URL）
python -m eval.data.objaverse_eval_setup fetch-metadata --output_dir ./eval_data_objaverse

# 2) 按 metadata.csv 下载 GLB（需已存在 metadata.csv；依赖 objaverse 等，见 dataset_toolkits）
python -m eval.data.objaverse_eval_setup download-glb --output_dir ./eval_data_objaverse

# 3) 从本地已下载树中随机抽子集到扁平目录（脚本在 dataset_toolkits/sample_objaverse_glb_subset.py）
python -m eval.data.objaverse_eval_setup sample-subset \
  --input_dir /path/to/raw/hf-objaverse-v1 \
  --output_dir ./eval_data_objaverse/flat_5k \
  --num_samples 5000 \
  --seed 42

# 4) 可选：写出评测 JSON
python -m eval.data.objaverse_eval_setup build-eval-json \
  --metadata_csv ./eval_data_objaverse/flat_5k/metadata.csv \
  --glb_dir ./eval_data_objaverse/flat_5k \
  --output_dir ./eval_data_objaverse/flat_5k
```

完成后在 YAML 里配置 `metadata_csv` + `glb_dir` 指向子集目录，或直接 `data_path` 指向上一步生成的 JSON。

若仍要用 Objaverse + PointLLM 旧流程，见 `eval/EVAL_GUIDE.md` 与 `python -m eval.data.build_eval_datasets`。

编辑 `eval/configs/tasks/sparse_*.yaml` 中的 `model.vae_config`、`model.vae_ckpt`、`model.llm_path`（理解/生成任务）为**你的实际路径**。

## 3. 评测命令

在项目根目录执行。`--gpu_ids` 指定物理 GPU 编号；`--batch_size` 对 **sparse_sdf_qwen3** 的 understanding / vqvae 生效（ShapeLLM 侧仍强制 batch=1）。

### ShapeLLM-Omni

```bash
python -m eval.runner --config eval/configs/tasks/shapellm/understanding.yaml --adapter shapellm --gpu_ids 0 --no_resume
python -m eval.runner --config eval/configs/tasks/sparse_vqvae/understanding.yaml --adapter sparse_sdf_qwen3 --gpu_ids 0 --batch_size 1 --no_resume
python -m eval.runner --config eval/configs/tasks/shapellm/generation.yaml --gpu_ids 1 --no_resume
```

### Sparse-SDF-VQVAE + Qwen3-VL（全参 HF）

`understanding` / `generation` 使用 `transformers` 直接加载 LLM：
`AutoTokenizer.from_pretrained` → `AutoModelForCausalLM.from_pretrained` → 本地 Qwen 文本模板 → `model.generate`。
请在 sparse 任务 YAML 的 `model:` 中配置 HuggingFace checkpoint 路径，例如：

```yaml
llm_path: LLaMA-Factory/saves/qwen3-4b/sft_bpe_3d_stage2/checkpoint-55000
trust_remote_code: true
llm_dtype: bfloat16
```

`vqvae_recon` 只验证 Sparse SDF VQVAE，不会调用 Qwen/LLM。

```bash
python -m eval.runner --config eval/configs/tasks/sparse_vqvae/understanding.yaml --adapter sparse_sdf_qwen3 --gpu_ids 1 --batch_size 1
python -m eval.runner --config eval/configs/tasks/sparse_vqvae/generation.yaml --gpu_ids 0 --no_resume --batch_size 1
```

- **断点续跑**：重复同一命令；加 `--no_resume` 清空逻辑上已存在记录（需手动删输出目录或 rank 分片文件以彻底重算）。
- **输出目录**：`eval_results/<adapter>/<task>/` 下含 `per_sample.jsonl`、`aggregate.json`、`meshes/*.obj`（若启用）、以及 json/csv/tex 报告。

### Official external baselines

External baselines live behind the same `ModelAdapter` interface. Clone official
repositories first; cloned code and generated work dirs are ignored by git:

```bash
python -m eval.baselines.clone_official_repos
```

Enabled direct text-to-3D adapters:

```bash
python -m eval.runner --config eval/configs/tasks/baselines/trellis_generation.yaml --gpu_ids 0 --no_resume
python -m eval.runner --config eval/configs/tasks/baselines/sar3d_generation.yaml --gpu_ids 0 --no_resume
python -m eval.runner --config eval/configs/tasks/baselines/gaussiancube_generation.yaml --gpu_ids 0 --no_resume
python -m eval.runner --config eval/configs/tasks/baselines/shape_e_generation.yaml --gpu_ids 0 --no_resume
```

Enabled direct 3D understanding adapters:

```bash
python -m eval.runner --config eval/configs/tasks/baselines/pointllm_13b_understanding.yaml --gpu_ids 0 --no_resume
python -m eval.runner --config eval/configs/tasks/baselines/three_d_llm_understanding.yaml --gpu_ids 0 --no_resume
```

Bridge-mode baselines are also runnable through the same runner:

```bash
# Image-to-3D baselines need proxy images via model.input_image_dir/default_input_image/sample_image_map.
python -m eval.runner --config eval/configs/tasks/baselines/instantmesh_generation.yaml --gpu_ids 0 --no_resume
python -m eval.runner --config eval/configs/tasks/baselines/3dtopia_xl_generation.yaml --gpu_ids 0 --no_resume

# LGM's official text path is in app.py; the adapter calls its process() function without launching Gradio.
python -m eval.runner --config eval/configs/tasks/baselines/lgm_generation.yaml --gpu_ids 0 --no_resume

# 2D VLM baselines render mesh views first, then call official image inference.
python -m eval.runner --config eval/configs/tasks/baselines/instructblip_13b_understanding.yaml --gpu_ids 0 --no_resume
python -m eval.runner --config eval/configs/tasks/baselines/llava_13b_understanding.yaml --gpu_ids 0 --no_resume
```

To launch every registered baseline and automatically use all visible GPUs:

```bash
python -m eval.scripts.run_all_baselines --max_samples 10 --no_resume
python -m eval.scripts.run_all_baselines --dry_run
```

For local smoke tests without large weights/CUDA, use the explicit mock configs:

```bash
python -m eval.runner --config eval/configs/tasks/baselines/mock_trellis_generation.yaml --no_resume
python -m eval.runner --config eval/configs/tasks/baselines/mock_pointllm_understanding.yaml --no_resume
```

Bridge assumptions:

- `InstantMesh` and `3DTopia-XL` are official image-conditioned pipelines here; configure proxy images for each text prompt.
- `InstructBLIP-13B` and `LLaVA-13B` consume rendered 2D views of the input mesh.
- `LGM` uses its official Gradio `process(input_image=None, prompt=...)` path through `eval.baselines.run_lgm_text`.

### 按指标挑选样本（ours vs baseline）

```bash
python -m eval.analysis.sample_selector \
  --ours eval_results/sparse_sdf_qwen3/understanding/per_sample.jsonl \
  --baseline eval_results/shapellm/understanding/per_sample.jsonl \
  --metric bleu_1 --higher_is_better --min_gap 0.05 --top_k 50 \
  --out eval_results/picked/understanding_top50.jsonl
```

更多说明见 [eval/EVAL_GUIDE.md](eval/EVAL_GUIDE.md)。

## 4. 论文风格指标（Inception FD / KD）与 CLIP + Trellis 上色

### 4.1 指标说明

- **`fd_inception`** / **`kd_inception`**：对生成 mesh 与参考 mesh 做相同多视角渲染（PyVista，相机分布对齐 Trellis Hammersley），在 **Inception-V3**（2048 维池化特征）上计算 Fréchet Inception Distance；**`kd_inception`** 为 **高斯 RBF 核**下的无偏 **MMD²**，输出 **`100 × MMD²`**（与常见论文 ×10² 表尺度一致）。Inception 骨干通过 `torchmetrics` 构造；RBF 带宽 γ 默认 **`1 / feature_dim`**，可在 `metrics_config.render.kid_rbf_gamma` 覆盖。
- **`clip_score`**：对生成视图与 **首行 caption**（`prompt` 中 `caption\\n...` 的第一行）计算 CLIP 余弦相似度再平均；需 `transformers` 内置 CLIP。

聚合阶段在 `eval/runner.py::_merge_and_aggregate` 中写入 `aggregate.json`。

### 4.2 Trellis 白膜上色（Sparse 与 ShapeLLM）

- **Sparse（`sparse_sdf_qwen3`）**：在任务 YAML 中设置 `model.colorization.enabled: true`，并配置 `model.trellis_text_path`（默认 `JeffreyXiang/TRELLIS-text-xlarge`）与 `model.hf_cache_dir`。流程：稀疏 VQVAE → marching cubes 白膜 → `TrellisTextTo3DPipeline.run_variant` → `postprocessing_utils.to_glb` 导出带纹理 GLB。
- **ShapeLLM**：设置 `model.load_trellis_text: true`、`inference.run_trellis: true`，且 `model.colorization.enabled: true` 时，在 `GenerationEngine` 的 Trellis 输出上同样调用 `to_glb` 并保存。

Trellis 源码在仓库根目录 [`trellis/`](trellis/)（与评测共用一份）。启动时 `eval/utils/path_bootstrap.py` 会将 [`third_party/`](third_party/) 置于 `sys.path` 前部，便于可选依赖（如 [`third_party/vox2seq`](third_party/vox2seq)）；说明见 [`third_party/README.md`](third_party/README.md)。

### 4.3 Mock LLM（无 Qwen 权重先跑通链路）

在 `inference.mock_llm` 下设置：

```yaml
inference:
  mock_llm:
    enabled: true
    mesh_token_string: "<mesh_start><morton_0><mesh_0><mesh_end>"
```

此时 **`generation` / `sparse_mesh`** 任务会跳过加载 HF LLM；`model.llm_path` 可留空。用于服务器上先验证 VAE + Trellis + 指标流水线。

### 4.4 服务器自检脚本

```bash
cd /path/to/3D-LLM-eval-main
python eval/scripts/smoke_mock_sparse_pipeline.py
```

通过后可在 GPU 机器上运行：

```bash
python -m eval.runner --config eval/configs/tasks/sparse_vqvae/mock_smoke.yaml --adapter sparse_sdf_qwen3 --gpu_ids 1  --no_resume
```

（请先将 YAML 内 `vae_ckpt`、数据路径等改为你的环境。）

