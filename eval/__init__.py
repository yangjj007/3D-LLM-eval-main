"""
ShapeLLM-Omni Evaluation Framework

A modular, extensible evaluation system for 3D multimodal large language models.
Covers three capability dimensions:
  - 3D Understanding (captioning, QA)
  - 3D Generation (text-to-3D)
  - VQVAE Reconstruction quality
"""

__version__ = "0.1.0"

# Optional ``third_party`` on path (e.g. vox2seq); Trellis is repo-root ``trellis/``.
try:
    from eval.utils.path_bootstrap import ensure_third_party_on_path

    ensure_third_party_on_path()
except Exception:
    pass
