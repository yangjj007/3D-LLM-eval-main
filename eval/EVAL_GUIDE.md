# ShapeLLM-Omni 评测框架使用指南

## 目录

1. [环境准备](#1-环境准备)
2. [数据集构建](#2-数据集构建)
3. [运行评测](#3-运行评测)
4. [评测任务详解](#4-评测任务详解)
5. [配置文件说明](#5-配置文件说明)
6. [结果输出与解读](#6-结果输出与解读)
7. [常见问题与排错](#7-常见问题与排错)
8. [进阶用法](#8-进阶用法)

---

## 1. 环境准备

### 1.1 前置条件

- Python 3.10 + PyTorch 2.4+
- conda 环境: `shapellm`
- 模型权重已缓存到本地 (`~/.cache/huggingface/hub/`)

### 1.2 激活环境

```bash
conda activate shapellm
cd /path/to/3D-LLM-eval-main
```

### 1.3 必须设置的环境变量

由于服务器无法访问 HuggingFace，需设置离线模式；同时为避免多 GPU 张量设备冲突，需指定单卡：

```bash
export HF_HUB_OFFLINE=1          # 跳过 HuggingFace 远程 HEAD 请求，直接用本地缓存
export CUDA_VISIBLE_DEVICES=7    # 指定单张 GPU（可改为 0-7 中任一空闲卡）
```

> **为什么需要 `CUDA_VISIBLE_DEVICES`?**
> ShapeLLM-7B 使用 `device_map="auto"`，在多卡环境下 transformers 会自动拆分模型到多张卡上，
> 导致 `RuntimeError: Expected all tensors to be on the same device` 错误。
> 指定单卡可避免此问题。7B 模型约需 14GB 显存，单张 RTX 5880 (48GB) 足够。

### 1.4 可选依赖安装

```bash
# CIDEr 指标（目前显示 -1.0 表示未安装）
pip install pycocoevalcap

# BERTScore 指标（目前显示 -1.0 表示未安装）
pip install bert-score
```

---

## 2. 数据集构建

### 2.1 一键构建（推荐）

数据构建脚本位于 `eval/data/build_eval_datasets.py`，支持自动下载 PointLLM GT、扫描本地 mesh、生成三个任务的评测数据集。

```bash
python -m eval.data.build_eval_datasets \
  --output_dir eval_data \
  --mesh_cache_dir ~/.objaverse/hf-objaverse-v1/glbs \
  --extra_mesh_dirs /path/to/your/mesh_library \
  --skip_download \
  --skip_alpaca \
  --update_configs \
  --max_vqvae 2000
```

### 2.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `eval_data` | 输出 JSON 数据集的目录 |
| `--mesh_cache_dir` | `None` | Objaverse mesh 缓存目录（扫描已下载的 .glb 文件） |
| `--extra_mesh_dirs` | `[]` | 额外的本地 mesh 目录（可指定多个，用空格分隔） |
| `--pointllm_scale` | `3000` | PointLLM GT 规模: `200`（小）或 `3000`（大） |
| `--max_alpaca_caption` | `1000` | 从 3D-Alpaca 取的 captioning 样本上限 |
| `--max_alpaca_gen` | `1200` | 从 3D-Alpaca 取的 text-to-3D 样本上限 |
| `--max_vqvae` | `None` (全部) | VQVAE 重建样本上限 |
| `--skip_download` | `False` | 跳过从 HuggingFace 下载 mesh（仅用本地缓存） |
| `--skip_alpaca` | `False` | 跳过 3D-Alpaca 数据加载 |
| `--update_configs` | `False` | 自动更新 YAML 配置文件中的数据路径 |
| `--download_processes` | `8` | 并行下载进程数 |

### 2.3 构建后的文件结构

```
eval_data/
├── understanding.json              # 理解任务数据集 (245 samples)
├── generation.json                 # 生成任务数据集 (379 samples)
├── vqvae_recon.json                # VQVAE重建数据集 (2000 samples)
├── PointLLM_brief_description_val_200_GT.json    # PointLLM GT (200)
├── PointLLM_brief_description_val_3000_GT.json   # PointLLM GT (3000)
└── dataset_meta.json               # 数据集元信息
```

### 2.4 数据集验证

构建完成后，可用以下命令验证数据完整性：

```bash
# 查看各数据集样本数和字段
python -c "
import json, os
for f in ['understanding.json', 'generation.json', 'vqvae_recon.json']:
    with open(f'eval_data/{f}') as fp:
        data = json.load(fp)
    valid = sum(1 for s in data if 'mesh_path' not in s or os.path.exists(s['mesh_path']))
    print(f'{f}: {len(data)} samples, {valid} valid mesh paths, keys={list(data[0].keys())}')
"
```

### 2.5 扩充数据集

当网络恢复或获取到更多 mesh 后，重新运行 build 脚本即可自动扩充：

```bash
# 方式 1: 如果有新的 mesh 目录
python -m eval.data.build_eval_datasets \
  --output_dir eval_data \
  --mesh_cache_dir ~/.objaverse/hf-objaverse-v1/glbs \
  --extra_mesh_dirs /path/to/new/meshes /path/to/your/mesh_library \
  --skip_download --update_configs

# 方式 2: 如果 HuggingFace 网络恢复，尝试下载全部 3000 mesh
python -m eval.data.build_eval_datasets \
  --output_dir eval_data \
  --update_configs \
  --download_processes 4
```

---

## 3. 运行评测

### 3.1 快速测试（推荐先跑通）

用少量样本验证 pipeline 可用性，避免长时间等待：

```bash
# VQVAE 重建 — 最快，不需要 LLM（约 2 分钟/2样本）
python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml --max_samples 2

# Understanding — 需加载 LLM（模型加载约 1-2 分钟，推理约 3 秒/样本）
python -m eval.runner --config eval/configs/tasks/understanding.yaml --max_samples 5

# Generation — 需加载 LLM（推理较慢，约 30 秒/样本）
python -m eval.runner --config eval/configs/tasks/generation_text2mesh.yaml --max_samples 5
```

### 3.2 完整评测

```bash
# 1. VQVAE 重建评测 (2000 samples, 预计 ~10 小时)
python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml

# 2. 3D 理解评测 (245 samples, 预计 ~15 分钟)
python -m eval.runner --config eval/configs/tasks/understanding.yaml

# 3. 3D 生成评测 (379 samples, 预计 ~3 小时)
python -m eval.runner --config eval/configs/tasks/generation_text2mesh.yaml
```

### 3.3 eval.runner 命令行参数

```
python -m eval.runner --config <yaml_path> [options]

必选:
  --config PATH           YAML 配置文件路径

可选:
  --task TASK             覆盖任务类型 (understanding / generation / vqvae_recon)
  --output_dir DIR        覆盖结果输出目录
  --max_samples N         限制最大评测样本数（用于快速测试）
```

### 3.4 后台运行

完整评测耗时较长，建议使用 `nohup` 或 `tmux`：

```bash
# 方式 1: nohup
nohup python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml \
  > eval_results/vqvae_recon/run.log 2>&1 &

# 方式 2: tmux
tmux new -s eval
python -m eval.runner --config eval/configs/tasks/understanding.yaml
# Ctrl+B D 退出 tmux, tmux attach -t eval 恢复
```

### 3.5 一键运行全部评测

```bash
#!/bin/bash
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=7

cd /path/to/3D-LLM-eval-main

echo "===== VQVAE Reconstruction ====="
python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml

echo "===== Understanding ====="
python -m eval.runner --config eval/configs/tasks/understanding.yaml

echo "===== Generation ====="
python -m eval.runner --config eval/configs/tasks/generation_text2mesh.yaml

echo "===== All Done ====="
```

---

## 4. 评测任务详解

### 4.1 VQVAE Reconstruction（VQVAE 重建）

**流程**: GLB mesh → 体素化 (64³) → VQVAE encode → 1024 tokens → VQVAE decode → 重建体素 → 与原始体素对比

**加载的模型**: 仅 VQVAE（不需要 LLM，显存占用小）

**评测指标**:
| 指标 | 含义 | 理想值 |
|------|------|--------|
| `voxel_iou` | 体素交并比 (Intersection over Union) | → 1.0 |
| `voxel_f1` | 体素 F1 分数 | → 1.0 |
| `voxel_precision` | 重建体素中正确的比例 | → 1.0 |
| `voxel_recall` | 原始体素被正确重建的比例 | → 1.0 |

**参考结果** (2 samples): IoU 0.869, F1 0.929

### 4.2 Understanding（3D 理解/描述）

**流程**: GLB mesh → 体素化 → VQVAE encode → mesh tokens → 拼入 prompt → LLM 生成文字描述 → 与 GT 对比

**加载的模型**: VQVAE + ShapeLLM-7B (Qwen2.5-VL)

**评测指标**:
| 指标 | 含义 | 说明 |
|------|------|------|
| `bleu_1~4` | BLEU n-gram 精度 | 越高越好, 通常 BLEU-1 > 0.3 算合理 |
| `rouge_l` | ROUGE-L (最长公共子序列) | 越高越好 |
| `meteor` | METEOR (考虑同义词) | 越高越好 |
| `cider` | CIDEr (TF-IDF 加权) | 需安装 pycocoevalcap |
| `bert_score` | BERTScore (语义相似度) | 需安装 bert-score |

**参考结果** (2 samples): BLEU-1 0.250, ROUGE-L 0.177, METEOR 0.146

### 4.3 Generation（3D 生成）

**流程**: 文本 prompt → LLM 生成 mesh tokens → 解析 `<mesh-start>...<mesh-end>` → VQVAE decode → 体素 → 可选: 与参考 mesh 对比

**加载的模型**: VQVAE + ShapeLLM-7B

**评测指标**:
| 指标 | 含义 | 说明 |
|------|------|------|
| `generation_success_rate` | 成功生成有效 mesh token 的比例 | → 1.0 |
| `avg_occupied_voxels` | 平均占用体素数 | 64³ 网格中非零体素数 |
| `chamfer_distance` | 倒角距离 (如有参考 mesh) | → 0.0, 越小越好 |
| `f_score` | F-Score (如有参考 mesh) | → 1.0 |

**参考结果** (2 samples): 成功率 100%, 平均占用体素 4748.5

---

## 5. 配置文件说明

### 5.1 配置文件层级

```
eval/configs/
├── default.yaml              # 全局默认配置（所有任务共享）
└── tasks/
    ├── understanding.yaml    # 理解任务配置（覆盖 default 的对应字段）
    ├── generation_text2mesh.yaml  # 生成任务配置
    └── vqvae_recon.yaml      # VQVAE 重建任务配置
```

加载逻辑: `default.yaml` 作为基础，任务配置深度合并覆盖。

### 5.2 核心配置字段

```yaml
# === 模型配置 ===
model:
  llm_path: "yejunliang23/ShapeLLM-7B-omni"   # LLM 路径 (HF repo 或本地路径)
  vqvae_repo: "yejunliang23/3DVQVAE"           # VQVAE 权重 repo
  vqvae_num_embeddings: 8192                    # codebook 大小
  load_llm: true              # 是否加载 LLM (VQVAE 任务设为 false)
  load_trellis_text: false    # 是否加载 Trellis text-to-3D (通常 false)
  load_trellis_image: false   # 是否加载 Trellis image-to-3D (通常 false)

# === 数据配置 ===
data:
  data_path: eval_data/understanding.json  # 数据集 JSON 路径
  max_samples: null                        # null=全部, 设数字限制样本数

# === 推理配置 ===
inference:
  max_new_tokens: 512        # 最大生成 token 数
  temperature: 0.0           # 0.0=贪心解码 (understanding), 0.7=采样 (generation)
  top_k: 1                   # Top-K 采样
  top_p: 1.0                 # Top-P (nucleus) 采样
  batch_size: 1              # 批大小（目前仅支持 1）

# === 评测指标 ===
metrics:
  - bleu
  - rouge_l
  - meteor

# === 报告输出 ===
reporting:
  formats: [json, csv, latex]              # 输出格式
  output_dir: eval_results/understanding   # 结果输出目录
```

### 5.3 常用配置修改示例

```yaml
# 使用本地 LLM 权重（不从 HuggingFace 下载）
model:
  llm_path: "/path/to/local/ShapeLLM-7B-omni"

# 只跑 100 个样本
data:
  max_samples: 100

# 生成任务使用更高温度
inference:
  temperature: 0.9
  top_k: 50
```

---

## 6. 结果输出与解读

### 6.1 输出文件

每个任务完成后，在 `output_dir` 下生成以下文件:

```
eval_results/understanding/
├── eval_results.json    # 完整结果（含每个样本的详细信息和指标）
├── eval_results.csv     # 每样本指标表格
├── eval_summary.csv     # 聚合指标摘要
└── eval_table.tex       # LaTeX 论文表格（可直接粘贴到论文中）
```

### 6.2 查看结果

```bash
# 查看聚合指标
cat eval_results/understanding/eval_summary.csv

# 查看 LaTeX 表格
cat eval_results/understanding/eval_table.tex

# 查看单样本详细结果
python -c "
import json
with open('eval_results/understanding/eval_results.json') as f:
    data = json.load(f)
# 查看第一个样本
sample = data['per_sample_results'][0]
print(f'Prompt: {sample[\"prompt\"][:100]}...')
print(f'Prediction: {sample[\"prediction\"][:200]}...')
print(f'Ground Truth: {sample[\"ground_truth\"][:200]}...')
print(f'Metrics: {sample.get(\"metrics\", {})}')
"
```

### 6.3 LaTeX 表格示例

生成的 LaTeX 表格可直接用于论文:

```latex
\begin{table}[htbp]
  \centering
  \caption{Evaluation Results — vqvae\_recon}
  \begin{tabular}{lcccc}
    \toprule
    Model & Voxel F1 & Voxel IoU & Precision & Recall \\
    \midrule
    ShapeLLM-7B-omni & 92.9\% & 86.9\% & 0.934 & 0.924 \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## 7. 常见问题与排错

### Q1: `RuntimeError: Expected all tensors to be on the same device`

**原因**: 多 GPU 环境下 `device_map="auto"` 导致张量分散到不同卡上。

**解决**: 设置 `export CUDA_VISIBLE_DEVICES=7`（或其他单张空闲 GPU）。

### Q2: HuggingFace 连接超时，大量 `ConnectionResetError` 日志

**原因**: 服务器无法访问 huggingface.co。

**解决**: 设置 `export HF_HUB_OFFLINE=1`。前提是模型权重已缓存在 `~/.cache/huggingface/hub/`。

### Q3: `CIDEr score unavailable` / `bert-score not installed`

**原因**: 可选依赖未安装。

**解决**:
```bash
pip install pycocoevalcap   # CIDEr
pip install bert-score      # BERTScore
```

这些指标未安装时会显示 `-1.0`，不影响其他指标的计算。

### Q4: NLTK `omw-1.4` 下载失败导致 METEOR 卡住

**原因**: NLTK 数据下载需要访问 GitHub，网络可能受限。

**解决**: 已修复为不阻塞模式。如仍有问题，手动下载:
```bash
curl -ksSL -o ~/nltk_data/corpora/omw-1.4.zip \
  "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/omw-1.4.zip"
cd ~/nltk_data/corpora && unzip -o omw-1.4.zip
```

### Q5: 数据集样本数不够 (Understanding 只有 245)

**原因**: PointLLM 3000 GT 中有 2755 个 Objaverse UID 的 mesh 文件未下载。

**解决**:
1. 等 HuggingFace 网络恢复后重新运行 build 脚本（去掉 `--skip_download`）
2. 或手动将 Objaverse GLB 文件放入 `~/.objaverse/hf-objaverse-v1/glbs/` 后重新构建

### Q6: `ModuleNotFoundError: No module named 'eval'`

**原因**: 没有在项目根目录下运行。

**解决**: 先 `cd` 到本仓库根目录（含 `eval/` 与 `eval_data/` 的 **3D-LLM-eval-main**）后再执行。

### Q7: 显存不足 (OOM)

**解决**:
- 选择显存充足的 GPU: `nvidia-smi` 查看空闲显存
- VQVAE 任务只需约 2GB，Understanding/Generation 约需 16GB
- 不要在同一张卡上运行其他大模型

### Q8: 如何更换评测的模型权重？

修改 YAML 配置或 `default.yaml`:
```yaml
model:
  llm_path: "/path/to/your/model"    # 替换为你的模型路径
  vqvae_repo: "/path/to/vqvae"      # 或 HF repo ID
```

---

## 8. 进阶用法

### 8.1 对比多个模型

```bash
# 模型 A
python -m eval.runner \
  --config eval/configs/tasks/understanding.yaml \
  --output_dir eval_results/model_a

# 修改 config 指向模型 B 后
python -m eval.runner \
  --config eval/configs/tasks/understanding.yaml \
  --output_dir eval_results/model_b

# 对比结果
paste eval_results/model_a/eval_summary.csv eval_results/model_b/eval_summary.csv
```

### 8.2 自定义数据集

手动创建 JSON 文件即可，需遵循以下 schema:

**Understanding**:
```json
[
  {
    "sample_id": "custom_001",
    "mesh_path": "/absolute/path/to/mesh.glb",
    "prompt": "Describe this 3D object.",
    "ground_truth": "A wooden chair with four legs.",
    "ground_truths": ["A wooden chair with four legs."]
  }
]
```

**Generation**:
```json
[
  {
    "sample_id": "gen_001",
    "prompt": "A red sports car",
    "source": "custom"
  }
]
```

**VQVAE Reconstruction**:
```json
[
  {
    "sample_id": "vqvae_001",
    "mesh_path": "/absolute/path/to/mesh.glb"
  }
]
```

然后修改对应 YAML 中的 `data.data_path` 指向你的 JSON 文件。

### 8.3 添加新的评测指标

在 `eval/metrics/` 下添加计算函数，然后注册到对应的 Metrics 类中:

```python
# eval/metrics/text_metrics.py
def compute_my_metric(predictions, references):
    # 你的计算逻辑
    return {"my_metric": score}

class TextMetrics:
    METRIC_FNS = {
        ...,
        "my_metric": compute_my_metric,
    }
```

然后在 YAML 配置的 `metrics` 列表中添加 `my_metric`。

### 8.4 完整项目结构参考

```
eval/
├── runner.py                    # 评测入口 (python -m eval.runner)
├── model_loader.py              # 模型加载 (VQVAE/LLM/Trellis)
├── EVAL_GUIDE.md                # 本文档
├── configs/
│   ├── default.yaml             # 全局默认配置
│   └── tasks/
│       ├── understanding.yaml
│       ├── generation_text2mesh.yaml
│       └── vqvae_recon.yaml
├── data/
│   ├── build_eval_datasets.py   # 一键数据构建脚本
│   ├── base_dataset.py          # 数据集基类
│   ├── understanding_dataset.py
│   ├── generation_dataset.py
│   └── vqvae_dataset.py
├── inference/
│   ├── base_engine.py           # 推理引擎基类
│   ├── understanding_engine.py  # mesh→tokens→LLM→text
│   ├── generation_engine.py     # text→LLM→tokens→VQVAE→voxels
│   └── vqvae_engine.py          # mesh→voxels→VQVAE encode/decode→compare
├── metrics/
│   ├── text_metrics.py          # BLEU, ROUGE-L, METEOR, CIDEr, BERTScore
│   ├── voxel_metrics.py         # IoU, F1, Precision, Recall
│   ├── mesh_metrics.py          # Chamfer Distance, EMD, F-Score
│   └── render_metrics.py        # PSNR, SSIM, LPIPS, FID
├── reporting/
│   ├── json_reporter.py
│   ├── csv_reporter.py
│   └── latex_reporter.py
└── utils/
    ├── mesh_processing.py       # mesh 加载、体素化
    └── token_utils.py           # mesh token 解析
```

---

## 9. 多模型 Adapter、断点续测与多 GPU（更新）

- **Adapter**：`adapter: shapellm`（默认）或 `adapter: sparse_sdf_qwen3`。实现位于 `eval/adapters/`。
- **Sparse 后端**：评测仓库内已包含 vendored `trellis`（SparseSDFVQVAE、mesh→sparse SDF）。权重与 Qwen 仍通过 YAML 的 `model.vae_ckpt` / `model.llm_path` 指向本地路径；与上游 Med 代码同步可运行 `scripts/vendor_sparse_trellis_from_med.ps1` 或 `.sh`。需已安装 `torchsparse` / `spconv` 等（见根目录 `README.md`）。
  - **SDF 来源**：默认 `model.sdf_from_mesh_only: true`，评测时从每条样本的 `mesh_path`（GLB）在线计算稀疏 SDF，不要求预计算 `{sha256}_r512.npz` 或 `*_sparse.json`。可选 `model.sdf_cache_dir` 仅缓存「本次从 GLB 算出」的结果以加速重复跑。
  - **数据**：可在 YAML 的 `data` 中配置 `metadata_csv` + `glb_dir`（并省略 `data_path`）直接加载；Objaverse 下载与子集见根目录 `README.md` 中的 `python -m eval.data.objaverse_eval_setup ...`。
- **断点续测**：每个任务在 `eval_results/<adapter>/<task>/per_sample.jsonl`（多卡时为 `per_sample.rank*.jsonl` 合并）中按行追加样本；再次运行相同命令会跳过已有 `sample_id`（`resume: false` 或 `--no_resume` 强制重跑）。
- **Mesh 导出**：`save_meshes: true` 时在 `meshes/*.obj` 保存**仅几何**的预测网格。
- **多 GPU**：`parallel.gpu_ids: [0,1,2,3]` 或 `--gpu_ids 0,1,2,3`；Linux 下 `torch.multiprocessing.spawn` 数据并行。**Windows** 当前退化为单进程单卡。
- **样本筛选**：`python -m eval.analysis.sample_selector --ours ... --baseline ... --metric bleu_1 --out picked.jsonl`
- **已知限制**：`sparse_sdf_qwen3` 的纯文本 **text→3D** 无法从仅 `<mesh_i>` 序列恢复坐标，故 `generation` 任务只做统计型指标，不做与 GT mesh 的 CD/HD 对齐（除非后续扩展坐标输出）。
