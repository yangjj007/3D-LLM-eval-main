from .text_metrics import TextMetrics, clear_text_metric_caches
from .voxel_metrics import VoxelMetrics
from .mesh_metrics import MeshMetrics
from .render_metrics import RenderMetrics
from .classification_metrics import ClassificationMetrics

METRIC_REGISTRY = {
    "bleu": TextMetrics,
    "rouge_l": TextMetrics,
    "meteor": TextMetrics,
    "cider": TextMetrics,
    "bert_score": TextMetrics,
    "gpt_score": TextMetrics,
    "voxel_iou": VoxelMetrics,
    "voxel_f1": VoxelMetrics,
    "chamfer_distance": MeshMetrics,
    "emd": MeshMetrics,
    "f_score": MeshMetrics,
    "psnr": RenderMetrics,
    "ssim": RenderMetrics,
    "lpips": RenderMetrics,
    "fid": RenderMetrics,
    "fd_inception": RenderMetrics,
    "kd_inception": RenderMetrics,
    "clip_score": RenderMetrics,
    "accuracy": ClassificationMetrics,
    "top_k_accuracy": ClassificationMetrics,
}
