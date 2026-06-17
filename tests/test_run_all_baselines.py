import subprocess
import sys
from pathlib import Path


def test_run_all_baselines_dry_run_lists_bridge_and_direct_baselines():
    proc = subprocess.run(
        [sys.executable, "-m", "eval.scripts.run_all_baselines", "--dry_run", "--gpu_ids", "0", "--max_samples", "1"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "trellis:enabled" in proc.stdout
    assert "instantmesh:bridged" in proc.stdout
    assert "llava_13b:bridged" in proc.stdout
    assert "eval.runner" in proc.stdout
