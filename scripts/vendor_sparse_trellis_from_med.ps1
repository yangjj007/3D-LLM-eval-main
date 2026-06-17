# Sync Sparse VQVAE-related trellis code from a Med-3D-LLM checkout into this repo.
# Usage:
#   $env:SOURCE = "E:\path\to\Med-3D-LLM-main"
#   .\scripts\vendor_sparse_trellis_from_med.ps1
# Or: .\scripts\vendor_sparse_trellis_from_med.ps1 E:\path\to\Med-3D-LLM-main
param(
    [string] $Source = ""
)
$Root = Split-Path $PSScriptRoot -Parent
if (-not $Source) {
    $Source = if ($env:SOURCE) { $env:SOURCE } else { Join-Path $Root "..\Med-3D-LLM-main" }
}
if (-not (Test-Path -LiteralPath $Source)) {
    Write-Error "SOURCE path not found: $Source. Clone Med-3D-LLM first or set `$env:SOURCE."
    exit 1
}
$Source = (Resolve-Path -LiteralPath $Source).Path
if (-not (Test-Path (Join-Path $Source "trellis"))) {
    Write-Error "SOURCE must be Med-3D-LLM root (contains trellis\): $Source"
    exit 1
}
$aeSrc = Join-Path $Source "trellis\models\autoencoders"
$aeDst = Join-Path $Root "trellis\models\autoencoders"
$utilDst = Join-Path $Root "trellis\utils"
$vaeDst = Join-Path $Root "eval\configs\vae"
New-Item -ItemType Directory -Force -Path $aeDst, $utilDst, $vaeDst | Out-Null
Copy-Item (Join-Path $aeSrc "*.py") -Destination $aeDst -Force
Copy-Item (Join-Path $Source "trellis\utils\mesh_utils.py") -Destination (Join-Path $utilDst "mesh_utils.py") -Force
$stage2 = Join-Path $Source "configs\vae\sdf_vqvae_stage2.json"
if (Test-Path $stage2) {
    Copy-Item $stage2 (Join-Path $vaeDst "sdf_vqvae_stage2.json") -Force
}
Write-Host "Done. Vendored from: $Source"
