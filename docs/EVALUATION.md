# ShapeLLM-Omni Evaluation Framework

## 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [环境准备](#环境准备)
4. [快速开始](#快速开始)
5. [评测任务详解](#评测任务详解)
   - [3D Understanding 评测](#task-1-3d-understanding-评测)
   - [Text-to-3D Generation 评测](#task-2-text-to-3d-generation-评测)
   - [VQVAE Reconstruction 评测](#task-3-vqvae-reconstruction-评测)
6. [数据准备](#数据准备)
   - [PointLLM Objaverse Captioning](#方式-1pointllm-objaverse-captioning-benchmark)
   - [Toys4K Generation](#方式-2toys4k-generation-benchmark)
   - [3D-Alpaca](#方式-33d-alpaca-huggingface)
   - [本地 Mesh 目录](#方式-4本地-mesh-目录)
   - [下载 Objaverse Meshes](#下载-objaverse-meshes)
7. [数据格式规范](#数据格式规范)
8. [评测指标详解](#评测指标详解)
9. [配置系统](#配置系统)
10. [结果输出格式](#结果输出格式)
11. [扩展指南](#扩展指南)
12. [常见问题](#常见问题)

---

## 概述

本评测框架为 ShapeLLM-Omni（基于 Qwen2.5-VL-7B 的多模态 3D 大模型）提供了一套**模块化、可扩展**的学术级评测系统。覆盖模型的三大核心能力：

| 能力维度 | 任务类型 | 核心指标 |
|---------|---------|---------|
| **3D 理解** | Captioning / QA | BLEU, ROUGE-L, METEOR, CIDEr, BERTScore |
| **3D 生成** | Text-to-3D | Chamfer Distance, F-Score, IoU |
| **VQVAE 重建** | Encode-Decode 质量 | Voxel IoU, F1, Precision, Recall |

---

## 架构设计

```
eval/
├── configs/                     # YAML 配置
│   ├── default.yaml             # 全局默认
│   └── tasks/
│       ├── understanding.yaml
│       ├── generation_text2mesh.yaml
│       └── vqvae_recon.yaml
├── runner.py                    # 主入口
├── model_loader.py              # 模型加载
├── data/                        # 数据集
│   ├── base_dataset.py
│   ├── understanding_dataset.py
│   ├── generation_dataset.py
│   └── vqvae_dataset.py
├── inference/                   # 推理引擎
│   ├── base_engine.py
│   ├── understanding_engine.py
│   ├── generation_engine.py
│   └── vqvae_engine.py
├── metrics/                     # 指标计算
│   ├── text_metrics.py
│   ├── voxel_metrics.py
│   ├── mesh_metrics.py
│   ├── render_metrics.py
│   └── classification_metrics.py
├── reporting/                   # 结果输出
│   ├── json_reporter.py
│   ├── csv_reporter.py
│   ├── latex_reporter.py
│   └── wandb_reporter.py
└── utils/                       # 工具函数
    ├── mesh_processing.py
    └── token_utils.py
```

**数据流**：

```
YAML Config → Runner → ModelLoader → Dataset → InferenceEngine → Metrics → Reporters
```

---

## 环境准备

### 1. 基础依赖

确保已安装项目主体依赖（参见根目录 `requirements.txt`）。评测框架额外需要：

```bash
# 必需
pip install pyyaml nltk rouge-score

# 推荐（完整指标支持）
pip install bert-score sacrebleu pycocoevalcap

# 可选
pip install wandb                    # WandB 日志
pip install torchmetrics[image]      # FID 指标
pip install openai                   # GPT-Score (LLM-as-judge)
```

### 2. NLTK 数据

首次运行 METEOR 指标前，需要下载 NLTK 数据：

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

框架会在首次调用时自动下载，也可以手动提前执行。

### 3. 模型权重

评测需要以下模型权重（首次运行时自动从 HuggingFace 下载）：

| 模型 | HuggingFace Repo | 说明 |
|------|-------------------|------|
| 3D-VQVAE | `yejunliang23/3DVQVAE` | 所有任务都需要 |
| ShapeLLM-7B-omni | `yejunliang23/ShapeLLM-7B-omni` | Understanding / Generation 需要 |
| Trellis Text-XL | `JeffreyXiang/TRELLIS-text-xlarge` | Generation（可选，用于 mesh 精细化） |

如果使用本地权重，在 YAML 中修改对应路径即可。

---

## 快速开始

### 最快验证：VQVAE 重建评测

VQVAE 评测仅需 VQVAE 模型（无需 LLM），是最快的验证路径。

**Step 1**：准备一个简单的测试数据文件 `test_vqvae_data.json`：

```json
[
  {"sample_id": "test_001", "mesh_path": "./test.glb"}
]
```

**Step 2**：修改配置文件中的 `data_path`：

```bash
# 编辑 eval/configs/tasks/vqvae_recon.yaml
# 将 data_path 改为: "./test_vqvae_data.json"
```

**Step 3**：运行评测：

```bash
cd ShapeLLM-Omni
python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml
```

**Step 4**：查看结果：

```bash
cat eval_results/vqvae_recon/eval_results.json
cat eval_results/vqvae_recon/eval_summary.csv
```

### 完整评测示例

```bash
# 3D Understanding
python -m eval.runner --config eval/configs/tasks/understanding.yaml

# Text-to-3D Generation
python -m eval.runner --config eval/configs/tasks/generation_text2mesh.yaml

# 只跑前 10 个 sample（快速调试）
python -m eval.runner --config eval/configs/tasks/understanding.yaml --max_samples 10

# 指定输出目录
python -m eval.runner --config eval/configs/tasks/vqvae_recon.yaml --output_dir ./my_results
```

---

## 评测任务详解

### Task 1: 3D Understanding 评测

**目标**：评估模型对 3D 形状的理解能力（描述、问答）。

**流程**：
```
Mesh 文件 → 体素化 → VQVAE 编码 → Token 序列
  + 文本 Prompt → LLM 推理 → 文本回答
  vs. Ground Truth → 文本指标计算
```

**数据准备**：

```json
[
  {
    "sample_id": "cap3d_001",
    "mesh_path": "/data/objaverse/abc123.glb",
    "prompt": "Describe this 3D object in detail.",
    "ground_truth": "A wooden chair with four legs and a curved backrest.",
    "ground_truths": [
      "A wooden chair with four legs and a curved backrest.",
      "This is a chair made of wood, featuring a curved back."
    ]
  }
]
```

> **注意**：`ground_truths`（复数）支持多个参考答案，这对 BLEU/CIDEr 等指标非常重要。如果只有一个参考答案，使用 `ground_truth`（单数）即可。

**推理参数**：

评测时推荐使用 **greedy decoding**（`temperature: 0.0`）以确保结果可复现：

```yaml
inference:
  temperature: 0.0
  top_k: 1
  top_p: 1.0
  max_new_tokens: 512
```

**支持的指标**：

| 指标 | 描述 | 取值范围 | 方向 |
|------|------|---------|------|
| BLEU-1/2/3/4 | N-gram 重叠 | [0, 1] | ↑ |
| ROUGE-L | 最长公共子序列 F1 | [0, 1] | ↑ |
| METEOR | 同义词感知匹配 | [0, 1] | ↑ |
| CIDEr | TF-IDF 加权共识 | [0, 10] | ↑ |
| BERTScore | 上下文嵌入相似度 | [0, 1] | ↑ |
| GPT-Score | LLM-as-Judge (1-5 分) | [1, 5] | ↑ |

**配置示例**：

```yaml
task: understanding
data:
  data_path: "/data/cap3d_test.json"
metrics:
  - bleu
  - rouge_l
  - meteor
  - cider
  - bert_score
```

---

### Task 2: Text-to-3D Generation 评测

**目标**：评估模型从文本生成 3D 形状的质量。

**流程**：
```
文本 Prompt → LLM 生成 Mesh Token → VQVAE 解码 → 体素网格
  ↓ (可选)
  → Trellis 精细化 → Mesh / Gaussian
  vs. Reference Mesh → 3D 几何指标
```

**数据准备**：

```json
[
  {
    "sample_id": "gen_001",
    "prompt": "A drone with four propellers and a central body.",
    "reference_mesh_path": "/data/reference/drone.glb"
  }
]
```

> `reference_mesh_path` 是可选的。如果没有参考模型，框架只输出生成统计（成功率、体素数等）。

**两种评测模式**：

1. **仅 Voxel 级别**（快速，默认）：LLM → VQVAE decode → 体素与参考比较
2. **完整 Mesh 级别**（慢，需要 Trellis）：设置 `run_trellis: true` + `load_trellis_text: true`

**支持的指标**：

| 指标 | 描述 | 方向 |
|------|------|------|
| Chamfer Distance (CD) | 双向最近邻 L2 距离 | ↓ |
| Earth Mover's Distance (EMD) | 最优传输距离 | ↓ |
| F-Score@τ | τ 阈值内的点比例 | ↑ |
| Generation Success Rate | 成功生成 mesh token 的比例 | ↑ |
| PSNR / SSIM / LPIPS | 多视角渲染比较（需 Trellis）| ↑/↑/↓ |
| FID | 生成图像分布质量（需 Trellis）| ↓ |

---

### Task 3: VQVAE Reconstruction 评测

**目标**：评估 3D-VQVAE 的编码-解码保真度。

**流程**：
```
Mesh → 体素化 (64³) → VQVAE Encode → 1024 tokens → VQVAE Decode → 重建体素
  原始体素 vs. 重建体素 → 体素级指标
```

**数据准备**：

```json
[
  {"sample_id": "vq_001", "mesh_path": "/data/meshes/chair.glb"},
  {"sample_id": "vq_002", "mesh_path": "/data/meshes/table.obj"}
]
```

或者用纯文本文件（每行一个路径）：

```text
/data/meshes/chair.glb
/data/meshes/table.obj
/data/meshes/car.glb
```

**支持的指标**：

| 指标 | 公式 | 描述 |
|------|------|------|
| Voxel IoU | TP / (TP + FP + FN) | 整体重建质量 |
| Precision | TP / (TP + FP) | 多余体素的比例 |
| Recall | TP / (TP + FN) | 缺失体素的比例 |
| F1-Score | 2PR / (P+R) | 精确率和召回率的调和平均 |

> 此任务 **不需要 LLM**，设置 `load_llm: false` 可大幅加快加载速度。

---

## 数据准备

评测数据可以通过以下方式获取。框架提供了一个统一的数据准备 CLI 工具 `eval/data/prepare_data.py`，支持自动下载官方 benchmark 数据集或从本地 mesh 文件生成评测数据。

### 前置依赖

```bash
# 下载 Objaverse meshes（PointLLM / 3D-Alpaca 需要）
pip install objaverse

# 加载 3D-Alpaca HuggingFace 数据集
pip install datasets
```

### 方式 1：PointLLM Objaverse Captioning Benchmark

这是 ShapeLLM-Omni 论文中 **3D-to-Caption** 任务使用的官方 benchmark，来自 [PointLLM](https://github.com/InternRobotics/PointLLM)，包含 Objaverse 上 200 个精选样本。

**论文使用的 prompt**：`"<mesh>. Caption this 3D model in detail."`

**Step 1**：下载 Objaverse mesh 文件

```bash
# 方式 A：使用 objaverse Python 包批量下载
python -c "
import objaverse
import json

# 先下载 GT 获取所需 UID
import urllib.request
url = 'https://raw.githubusercontent.com/InternRobotics/PointLLM/main/data/PointLLM_brief_description_val_200_GT.json'
urllib.request.urlretrieve(url, 'pointllm_gt.json')

with open('pointllm_gt.json') as f:
    gt = json.load(f)
uids = [item['object_id'] for item in gt]

# 下载 mesh
objects = objaverse.load_objects(uids=uids, download_processes=4)
print(f'Downloaded {len(objects)} meshes')
"

# 方式 B：如果已有 Objaverse 本地缓存，直接指定目录即可
```

**Step 2**：生成评测数据 JSON

```bash
python -m eval.data.prepare_data pointllm \
    --output eval_data/pointllm_captioning.json \
    --mesh_dir /path/to/objaverse/meshes \
    --mesh_format glb
```

如果 PointLLM GT JSON 文件不在本地，脚本会自动从 GitHub 下载。也可以手动指定：

```bash
python -m eval.data.prepare_data pointllm \
    --output eval_data/pointllm_captioning.json \
    --mesh_dir /path/to/objaverse/meshes \
    --gt_json /path/to/PointLLM_brief_description_val_200_GT.json
```

**Step 3**：运行评测

```bash
# 修改 understanding.yaml 中的 data_path，或通过命令行传入
python -m eval.runner --config eval/configs/tasks/understanding.yaml \
    --data_path eval_data/pointllm_captioning.json
```

**生成的数据格式**：

```json
[
  {
    "sample_id": "pointllm_0000",
    "mesh_path": "/path/to/objaverse/abc123.glb",
    "prompt": "Caption this 3D model in detail.",
    "ground_truth": "A wooden chair with four legs...",
    "ground_truths": ["A wooden chair with four legs..."],
    "source": "pointllm_objaverse_captioning",
    "objaverse_uid": "abc123..."
  }
]
```

### 方式 2：Toys4K Generation Benchmark

ShapeLLM-Omni 论文中 **Text-to-3D** 和 **Image-to-3D** 任务使用 Toys4K 测试集，评测指标包括 FD（Frechet Distance）、KD（Kernel Distance）和 CLIP Score。

**论文使用的生成参数**：`top_k=8192, top_p=0.7, temperature=0.7`

**Step 1**：获取 Toys4K 数据集

Toys4K 数据集需要从其官方渠道下载。组织为如下目录结构：

```
toys4k/
├── test/
│   ├── metadata.json       # 或 captions.json，包含 prompt 文本
│   ├── meshes/
│   │   ├── 0001.glb
│   │   └── ...
│   └── images/             # Image-to-3D 任务需要
│       ├── 0001.png
│       └── ...
```

**Step 2**：生成评测数据

```bash
# Text-to-3D
python -m eval.data.prepare_data toys4k \
    --output eval_data/toys4k_text2mesh.json \
    --toys4k_dir /path/to/toys4k \
    --task text_to_3d

# Image-to-3D
python -m eval.data.prepare_data toys4k \
    --output eval_data/toys4k_img2mesh.json \
    --toys4k_dir /path/to/toys4k \
    --task image_to_3d
```

**Step 3**：运行评测

```bash
python -m eval.runner --config eval/configs/tasks/generation_text2mesh.yaml \
    --data_path eval_data/toys4k_text2mesh.json
```

### 方式 3：3D-Alpaca (HuggingFace)

[3D-Alpaca](https://huggingface.co/datasets/yejunliang23/3D-Alpaca) 是 ShapeLLM-Omni 的训练数据集（2.56M 样本，3.46B tokens），涵盖 4 种任务类型：text-to-3D、image-to-3D、3D-to-caption、3D-editing。可以从中采样子集用于评测。

```bash
# 从 3D-Alpaca 提取 3D-to-Caption 样本（前 500 条）
python -m eval.data.prepare_data alpaca \
    --output eval_data/alpaca_caption_500.json \
    --task_filter 3d_to_caption \
    --max_samples 500 \
    --mesh_dir /path/to/objaverse/meshes

# 提取 Text-to-3D 样本
python -m eval.data.prepare_data alpaca \
    --output eval_data/alpaca_text2mesh.json \
    --task_filter text_to_3d \
    --max_samples 200

# 导出所有任务类型
python -m eval.data.prepare_data alpaca \
    --output eval_data/alpaca_all.json \
    --max_samples 1000
```

> **注意**：3D-to-caption 类型的样本需要 Objaverse mesh 文件（通过 `--mesh_dir` 指定），脚本会自动匹配 UID 到本地文件路径。

### 方式 4：本地 Mesh 目录

如果只需要快速验证或评测自定义 mesh，可以直接从本地目录生成评测数据：

```bash
# VQVAE 重建评测（最快，不需要 LLM）
python -m eval.data.prepare_data local \
    --output eval_data/my_vqvae_test.json \
    --mesh_dir /path/to/my/meshes \
    --task vqvae_recon

# 3D Understanding 评测（需要提供 caption 参考答案）
python -m eval.data.prepare_data local \
    --output eval_data/my_caption_test.json \
    --mesh_dir /path/to/my/meshes \
    --task understanding \
    --captions_json /path/to/captions.json \
    --prompt "Describe this 3D object in detail."

# 3D Generation 评测（文件名作为 prompt）
python -m eval.data.prepare_data local \
    --output eval_data/my_gen_test.json \
    --mesh_dir /path/to/my/meshes \
    --task generation
```

其中 `captions.json` 格式为：

```json
{
  "chair": "A wooden chair with four legs and a curved backrest.",
  "table": "A rectangular dining table with a glass top."
}
```

### 下载 Objaverse Meshes

多个 benchmark 依赖 Objaverse mesh 文件。可以使用 `objaverse` Python 包按 UID 批量下载：

```python
from eval.data.prepare_data import download_objaverse_meshes

uids = ["abc123...", "def456..."]  # Objaverse UIDs
uid_to_path = download_objaverse_meshes(
    uids=uids,
    output_dir="/data/objaverse/meshes",
    processes=4,
)
```

或直接使用 objaverse 包：

```python
import objaverse
objects = objaverse.load_objects(uids=["abc123..."], download_processes=4)
```

> **提示**：`objaverse` 默认缓存在 `~/.objaverse/`，首次下载较慢，后续会使用缓存。

### 论文评测设置参考

| 任务 | Benchmark | 样本数 | Prompt 模板 | 生成参数 |
|------|-----------|--------|------------|---------|
| 3D-to-Caption | PointLLM Objaverse | 200 | `"<mesh>. Caption this 3D model in detail."` | temperature=0.0 (greedy) |
| Text-to-3D | Toys4K test | - | `"Please generate a 3D asset based on the prompt I provided: {prompt}"` | top_k=8192, top_p=0.7, temp=0.7 |
| Image-to-3D | Toys4K test | - | `"Create a 3D asset using the following image: <image>"` | top_k=8192, top_p=0.7, temp=0.7 |

---

## 数据格式规范

### 通用规则

- 支持 JSON (`.json`)、JSONL (`.jsonl`)、纯文本 (`.txt`) 格式
- JSON 文件应包含一个数组，每个元素是一个样本 dict
- JSONL 文件每行一个 JSON 对象
- 每个样本必须包含 `mesh_path` 或 `prompt`（取决于任务）
- `sample_id` 是可选的，框架会自动生成

### 支持的 Mesh 格式

通过 trimesh 库加载，支持：
- `.glb` / `.gltf` (推荐)
- `.obj`
- `.ply`
- `.stl`
- `.off`

### Understanding 数据集

```json
{
  "sample_id": "string (可选)",
  "mesh_path": "string (必需，mesh 文件路径)",
  "prompt": "string (可选，默认使用 config 中的 default_prompt)",
  "ground_truth": "string (必需，参考答案)",
  "ground_truths": ["string", "..."]  // 可选，多个参考答案
}
```

### Generation 数据集

```json
{
  "sample_id": "string (可选)",
  "prompt": "string (必需，生成提示词)",
  "reference_mesh_path": "string (可选，参考 mesh 路径)"
}
```

### VQVAE 数据集

```json
{
  "sample_id": "string (可选)",
  "mesh_path": "string (必需，mesh 文件路径)"
}
```

---

## 评测指标详解

### 文本指标 (`text_metrics.py`)

#### BLEU (Bilingual Evaluation Understudy)

衡量生成文本与参考文本之间的 n-gram 重叠程度。报告 BLEU-1 到 BLEU-4，使用 Method-1 平滑处理。

```python
from eval.metrics.text_metrics import compute_bleu
scores = compute_bleu(predictions, references, max_order=4)
# {"bleu_1": 0.72, "bleu_2": 0.55, "bleu_3": 0.41, "bleu_4": 0.31}
```

#### ROUGE-L

基于最长公共子序列（LCS）的 F1 指标，适合衡量句子结构相似性。

#### METEOR

综合考虑精确匹配、词干匹配和同义词匹配的指标，与人类判断相关性较高。

#### CIDEr

专为图像/场景描述设计的指标。使用 TF-IDF 权重，对信息量大的词赋予更高权重。需要安装 `pycocoevalcap`。

#### BERTScore

使用预训练语言模型（默认 DeBERTa-XL-MNLI）计算 token 级别的语义相似度，能捕捉同义改写。需要安装 `bert-score`。

#### GPT-Score

使用 GPT-4o-mini 作为评判者，从准确性、完整性、流畅性三个维度对生成文本评分（1-5 分）。需要设置 `OPENAI_API_KEY` 环境变量。

### 体素指标 (`voxel_metrics.py`)

所有体素指标在 64³ 二值网格上计算：

- **IoU**: 交并比，最通用的重建质量指标
- **Precision**: 预测为 occupied 的体素中正确的比例（避免"胖了"）
- **Recall**: 真实 occupied 体素被正确恢复的比例（避免"瘦了"）
- **F1**: Precision 和 Recall 的调和平均

```python
from eval.metrics.voxel_metrics import VoxelMetrics
metrics = VoxelMetrics.compute(results, ["voxel_iou", "voxel_f1"])
```

### Mesh 指标 (`mesh_metrics.py`)

基于点云采样的几何比较：

- **Chamfer Distance**: 双向的点到最近邻距离的均方和。值越小越好。
- **EMD**: 将一个点云最优地"搬运"到另一个的最小代价。
- **F-Score@τ**: 在距离阈值 τ 内匹配的点的比例。报告多个阈值。

```python
from eval.metrics.mesh_metrics import MeshMetrics
metrics = MeshMetrics.compute(results, ["chamfer_distance", "f_score"])
```

### 渲染指标 (`render_metrics.py`)

从多视角渲染图像进行像素级比较。复用 `trellis/utils/loss_utils.py` 中的实现：

- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性
- **LPIPS**: 感知相似性（基于 VGG 特征）
- **FID**: Fréchet Inception Distance（分布级别质量）

---

## 配置系统

### 配置层级

```
default.yaml (全局默认) ← task.yaml (任务覆盖) ← 命令行参数 (最高优先级)
```

### 命令行参数

```
python -m eval.runner --config CONFIG_PATH [OPTIONS]

必需参数:
  --config PATH     YAML 配置文件路径

可选参数:
  --task TYPE        覆盖任务类型 (understanding / generation / vqvae_recon)
  --output_dir DIR   覆盖输出目录
  --max_samples N    只评测前 N 个样本（快速调试用）
```

### 完整配置项参考

```yaml
task: understanding | generation | vqvae_recon

model:
  llm_path: str              # LLM 模型路径 (HF repo 或本地路径)
  vqvae_repo: str            # VQVAE HF repo
  vqvae_filename: str        # VQVAE 权重文件名
  vqvae_num_embeddings: int  # Codebook 大小 (默认 8192)
  dtype: str                 # torch dtype (默认 "auto")
  device_map: str            # 设备映射 (默认 "auto")
  load_llm: bool             # 是否加载 LLM
  load_trellis_text: bool    # 是否加载 Trellis Text Pipeline
  load_trellis_image: bool   # 是否加载 Trellis Image Pipeline

data:
  data_path: str             # 数据文件路径
  max_samples: int | null    # 最大样本数 (null = 全部)
  default_prompt: str        # 默认 prompt (仅 understanding)

inference:
  max_new_tokens: int        # LLM 最大生成 token 数
  temperature: float         # 采样温度 (0 = greedy)
  top_k: int                 # Top-K 采样
  top_p: float               # Nucleus 采样
  seed: int                  # 随机种子
  batch_size: int            # 批大小
  run_trellis: bool          # 是否运行 Trellis 精细化 (仅 generation)

metrics:
  - bleu                     # 指标列表
  - rouge_l
  - ...

reporting:
  formats: [json, csv, latex, wandb]
  output_dir: str
  wandb:
    enabled: bool
    project: str
    entity: str
    run_name: str
```

---

## 结果输出格式

### JSON (`eval_results.json`)

```json
{
  "task": "understanding",
  "model": "ShapeLLM-7B-omni",
  "timestamp": "2026-04-17T15:30:00",
  "config": { ... },
  "num_samples": 100,
  "aggregate_metrics": {
    "bleu_1": 0.723,
    "bleu_4": 0.312,
    "rouge_l": 0.456,
    "meteor": 0.389,
    "cider": 1.234,
    "bert_score_f1": 0.867
  },
  "per_sample_results": [
    {
      "sample_id": "cap3d_001",
      "prediction": "A wooden chair with ...",
      "ground_truth": "A chair made of wood ...",
      "metrics": { "bleu_1": 0.81, "rouge_l": 0.52, ... }
    }
  ]
}
```

### CSV (`eval_results.csv`)

每行一个样本，包含所有标量字段。便于用 pandas 做进一步分析：

```python
import pandas as pd
df = pd.read_csv("eval_results/understanding/eval_results.csv")
print(df.describe())
```

### CSV Summary (`eval_summary.csv`)

```
metric,value
bleu_1,0.723000
bleu_4,0.312000
...
```

### LaTeX (`eval_table.tex`)

直接复制到论文中使用的表格：

```latex
\begin{table}[htbp]
  \centering
  \caption{Evaluation Results — understanding}
  \begin{tabular}{lcccccc}
    \toprule
    Model & BLEU-1 & BLEU-4 & ROUGE-L & METEOR & CIDEr & BERTScore \\
    \midrule
    ShapeLLM-7B-omni & 0.723 & 0.312 & 0.456 & 0.389 & 1.234 & 0.867 \\
    \bottomrule
  \end{tabular}
\end{table}
```

### WandB

启用后自动上传到 Weights & Biases，包含：
- 聚合指标（Summary）
- 每样本结果（Table）
- 完整配置（Config）

---

## 扩展指南

### 添加新的评测任务

1. 在 `eval/data/` 中创建新的 Dataset 类，继承 `EvalDataset`
2. 在 `eval/inference/` 中创建新的 Engine 类，继承 `InferenceEngine`
3. 在 `eval/runner.py` 中添加新的 `run_xxx()` 函数和 `TASK_RUNNERS` 注册
4. 创建对应的 YAML 配置文件

### 添加新的评测指标

1. 在 `eval/metrics/` 中添加计算函数
2. 在对应的 `XxxMetrics` 类中注册
3. 在 `eval/metrics/__init__.py` 中注册到 `METRIC_REGISTRY`

### 添加新的报告格式

1. 在 `eval/reporting/` 中创建新的 Reporter 类
2. 实现 `save()` 静态方法
3. 在 `eval/reporting/__init__.py` 中注册到 `REPORTER_REGISTRY`

### 使用自定义模型权重

在 YAML 中指定本地路径：

```yaml
model:
  llm_path: "/your/local/path/to/ShapeLLM-7B-omni"
  vqvae_repo: "yejunliang23/3DVQVAE"  # 或本地路径
```

### 对接新的 Benchmark

创建 `eval/configs/benchmarks/your_benchmark.yaml`，指定数据路径和任务特定参数。例如：

```yaml
task: understanding
data:
  data_path: "/data/benchmarks/objaverse_cap3d/test.json"
  default_prompt: "Describe this 3D object."
metrics:
  - bleu
  - rouge_l
  - cider
reporting:
  output_dir: "./eval_results/cap3d"
```

---

## 常见问题

### Q: 评测很慢，如何加速？

1. 使用 `--max_samples 10` 快速调试
2. VQVAE 评测不需要 LLM，设置 `load_llm: false`
3. Generation 评测设置 `run_trellis: false` 跳过 Trellis 精细化
4. 确保 GPU 可用（`torch.cuda.is_available() == True`）

### Q: 如何复现论文中的结果？

- 使用 `temperature: 0.0`（greedy decoding）确保确定性输出
- 设置固定 `seed: 42`
- 确保使用相同版本的模型权重

### Q: 如何评测自己训练的模型？

修改 YAML 中的 `model.llm_path` 指向你的 checkpoint 目录：

```yaml
model:
  llm_path: "/path/to/your/finetuned-model"
```

### Q: BERTScore / CIDEr 报 -1.0？

这表示对应的依赖库未安装：

```bash
pip install bert-score         # for BERTScore
pip install pycocoevalcap      # for CIDEr
```

### Q: 如何只运行部分指标？

在 YAML 的 `metrics` 列表中只保留你需要的指标：

```yaml
metrics:
  - bleu
  - rouge_l
```

### Q: Mesh 文件加载失败？

- 确保 `trimesh` 和 `open3d` 已正确安装
- 确认 mesh 文件路径正确且文件完整
- 支持的格式：.glb, .gltf, .obj, .ply, .stl, .off

### Q: 如何在多 GPU 上运行？

当前框架使用 `device_map: "auto"`，transformers 会自动进行模型并行。如需数据并行，可修改 `inference/` 中的 Engine 实现。

---

## API 参考

### 独立使用指标模块

```python
# 文本指标
from eval.metrics.text_metrics import TextMetrics
scores = TextMetrics.compute(
    predictions=["A blue car"],
    references=[["A blue car with wheels", "A car colored blue"]],
    metric_names=["bleu", "rouge_l", "meteor"]
)

# 体素指标
from eval.metrics.voxel_metrics import voxel_iou, voxel_f1
import torch
pred = torch.randint(0, 2, (64, 64, 64))
gt = torch.randint(0, 2, (64, 64, 64))
print(f"IoU: {voxel_iou(pred, gt):.4f}")
print(f"F1:  {voxel_f1(pred, gt):.4f}")

# Mesh 指标
from eval.metrics.mesh_metrics import chamfer_distance
import numpy as np
a = np.random.randn(1000, 3).astype(np.float32)
b = np.random.randn(1000, 3).astype(np.float32)
print(f"CD: {chamfer_distance(a, b):.6f}")
```

### 独立使用工具函数

```python
# Mesh → Token 字符串
from eval.utils.mesh_processing import load_vertices, positions_to_voxel_tensor
from eval.utils.token_utils import token_to_words

positions = load_vertices("model.glb")         # (N, 3) numpy array
voxel = positions_to_voxel_tensor(positions)    # (1, 1, 64, 64, 64) tensor

# 解析 LLM 输出中的 mesh tokens
from eval.utils.token_utils import parse_mesh_tokens, pad_tokens
tokens = parse_mesh_tokens("<mesh-start><mesh42><mesh771><mesh-end>")
padded = pad_tokens(tokens, 1024)  # pad to 1024
```
