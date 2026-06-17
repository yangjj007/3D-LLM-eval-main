# third_party

- **`trellis/`** — Removed; use the **repository root** [`../trellis`](../trellis) package (single source of truth for SLAT / `to_glb` / pipelines).
- **`vox2seq/`** — Optional CUDA extension from ShapeLLM-Omni; install with `pip install -e third_party/vox2seq` if Trellis sparse attention requires it.
- **`LLaMA-Factory-src/`** — Existing vendored LLaMA-Factory snapshot.

HF checkpoints (Trellis, CLIP, VQVAE, etc.) are **not** stored here; configure paths in task YAML under `model.*`.
