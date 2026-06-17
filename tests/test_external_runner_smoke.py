import json
import subprocess
import sys
from pathlib import Path


def test_runner_external_generation_mock(tmp_path):
    repo = tmp_path / "TRELLIS"
    repo.mkdir()
    (repo / "example_text.py").write_text("", encoding="utf-8")
    data_path = tmp_path / "generation.json"
    data_path.write_text(json.dumps([{"sample_id": "g1", "prompt": "a chair"}]), encoding="utf-8")
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config.write_text(
        f"""
adapter: trellis
task: generation
save_meshes: false
data:
  data_path: {data_path.as_posix()}
model:
  baseline_repo_dir: {repo.as_posix()}
inference:
  mock_external:
    enabled: true
metrics: []
reporting:
  formats: [json]
  output_dir: {output_dir.as_posix()}
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "eval.runner", "--config", str(config), "--no_resume"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    per_sample = output_dir / "per_sample.jsonl"
    assert per_sample.exists()
    row = json.loads(per_sample.read_text(encoding="utf-8").splitlines()[-1])
    assert row["sample_id"] == "g1"
    assert row["extra"]["mock_external"] is True


def test_runner_external_understanding_mock(tmp_path):
    repo = tmp_path / "PointLLM"
    (repo / "pointllm" / "eval").mkdir(parents=True)
    (repo / "pointllm" / "eval" / "eval_objaverse.py").write_text("", encoding="utf-8")
    data_path = tmp_path / "understanding.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "u1",
                    "mesh_path": "missing.glb",
                    "prompt": "caption",
                    "ground_truth": "a chair",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    config.write_text(
        f"""
adapter: pointllm_13b
task: understanding
data:
  data_path: {data_path.as_posix()}
model:
  baseline_repo_dir: {repo.as_posix()}
inference:
  mock_external:
    enabled: true
metrics: []
reporting:
  formats: [json]
  output_dir: {output_dir.as_posix()}
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "eval.runner", "--config", str(config), "--no_resume"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    per_sample = output_dir / "per_sample.jsonl"
    assert per_sample.exists()
    row = json.loads(per_sample.read_text(encoding="utf-8").splitlines()[-1])
    assert row["sample_id"] == "u1"
    assert row["extra"]["debug"]["mock_external"] is True
