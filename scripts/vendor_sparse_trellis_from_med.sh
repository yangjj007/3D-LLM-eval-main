#!/usr/bin/env bash
# Sync Sparse VQVAE-related trellis code from a Med-3D-LLM checkout into this repo.
# Usage:
#   export SOURCE=/path/to/Med-3D-LLM-main
#   bash scripts/vendor_sparse_trellis_from_med.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${SOURCE:-${1:-../Med-3D-LLM-main}}"
if [[ ! -d "$SOURCE/trellis" ]]; then
  echo "ERROR: SOURCE=$SOURCE is not a Med-3D-LLM tree (missing trellis/)." >&2
  echo "Clone it first, e.g.: git clone <your-med-repo-url> Med-3D-LLM-main" >&2
  exit 1
fi
mkdir -p "$ROOT/trellis/models/autoencoders" "$ROOT/trellis/utils" "$ROOT/eval/configs/vae"
cp -v "$SOURCE/trellis/models/autoencoders/"*.py "$ROOT/trellis/models/autoencoders/"
cp -v "$SOURCE/trellis/utils/mesh_utils.py" "$ROOT/trellis/utils/mesh_utils.py"
if [[ -f "$SOURCE/configs/vae/sdf_vqvae_stage2.json" ]]; then
  cp -v "$SOURCE/configs/vae/sdf_vqvae_stage2.json" "$ROOT/eval/configs/vae/sdf_vqvae_stage2.json"
fi
echo "Done. Vendored from: $SOURCE"
