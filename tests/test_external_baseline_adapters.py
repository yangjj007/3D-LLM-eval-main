from pathlib import Path

from eval.adapters.base import MeshInput
from PIL import Image

from eval.adapters.text_to_3d_baselines import InstantMeshAdapter, SAR3DAdapter, TrellisAdapter
from eval.adapters.understanding_baselines import InstructBLIP13BAdapter, PointLLM13BAdapter


def test_mock_generation_returns_mesh(tmp_path):
    repo = tmp_path / "TRELLIS"
    repo.mkdir()
    (repo / "example_text.py").write_text("", encoding="utf-8")
    cfg = {
        "model": {"baseline_repo_dir": str(repo)},
        "inference": {"mock_external": {"enabled": True}},
    }
    adapter = TrellisAdapter()
    adapter.load(cfg, "cpu")
    result = adapter.generate_from_text(["a chair"], ["sample_1"], cfg)[0]
    assert result.pred_mesh is not None
    assert result.extra["mock_external"] is True
    assert result.extra["caption"] == "a chair"


def test_sar3d_command_uses_official_text_json(tmp_path, monkeypatch):
    repo = tmp_path / "SAR3D"
    repo.mkdir()
    (repo / "test.py").write_text("", encoding="utf-8")
    cfg = {
        "model": {
            "baseline_repo_dir": str(repo),
            "vqvae_pretrained_path": "vqvae.pt",
            "ar_ckpt_path": "text.pth",
        },
        "inference": {"work_dir": str(tmp_path / "work")},
    }
    adapter = SAR3DAdapter()
    adapter.load(cfg, "cpu")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]

    monkeypatch.setattr(adapter, "_run_subprocess", fake_run)
    monkeypatch.setattr(adapter, "_find_first_output", lambda *args, **kwargs: None)
    result = adapter.generate_from_text(["a blue cup"], ["cup 1"], cfg)[0]

    command = captured["command"]
    assert command[0] == "torchrun"
    assert any(str(part).startswith("--text_json_path=") for part in command)
    assert any(str(part).startswith("--save_path=") for part in command)
    assert captured["cwd"] == repo.resolve()
    assert result.extra["caption"] == "a blue cup"


def test_instantmesh_command_uses_proxy_image(tmp_path, monkeypatch):
    repo = tmp_path / "InstantMesh"
    repo.mkdir()
    (repo / "run.py").write_text("", encoding="utf-8")
    image = tmp_path / "proxy.png"
    Image.new("RGB", (8, 8), "red").save(image)
    cfg = {
        "model": {"baseline_repo_dir": str(repo), "default_input_image": str(image)},
        "inference": {"work_dir": str(tmp_path / "work"), "diffusion_steps": 2},
    }
    adapter = InstantMeshAdapter()
    adapter.load(cfg, "cpu")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

    monkeypatch.setattr(adapter, "_run_subprocess", fake_run)
    monkeypatch.setattr(adapter, "_find_first_output", lambda *args, **kwargs: None)
    adapter.generate_from_text(["a chair"], ["sample_1"], cfg)
    command = captured["command"]
    assert command[:2] == [str(Path(command[0])), "run.py"] or command[1] == "run.py"
    assert "--diffusion_steps" in command
    assert any(str(part).endswith(".png") for part in command)


def test_pointllm_mock_caption_rows(tmp_path):
    repo = tmp_path / "PointLLM"
    (repo / "pointllm" / "eval").mkdir(parents=True)
    (repo / "pointllm" / "eval" / "eval_objaverse.py").write_text("", encoding="utf-8")
    cfg = {
        "model": {"baseline_repo_dir": str(repo)},
        "inference": {"mock_external": {"enabled": True}},
    }
    adapter = PointLLM13BAdapter()
    adapter.load(cfg, "cpu")
    rows = adapter.caption_from_shape(
        [
            MeshInput(
                sample_id="shape_1",
                mesh_path="missing.glb",
                prompt="describe",
                ground_truth="a shape",
            )
        ],
        cfg,
    )
    assert rows[0]["sample_id"] == "shape_1"
    assert rows[0]["prediction"].startswith("mock external caption")
    assert rows[0]["ground_truths"] == ["a shape"]


def test_instructblip_mock_caption_rows(tmp_path):
    repo = tmp_path / "LAVIS"
    (repo / "projects" / "instructblip").mkdir(parents=True)
    (repo / "projects" / "instructblip" / "README.md").write_text("", encoding="utf-8")
    cfg = {
        "model": {"baseline_repo_dir": str(repo)},
        "inference": {"mock_external": {"enabled": True}},
    }
    adapter = InstructBLIP13BAdapter()
    adapter.load(cfg, "cpu")
    rows = adapter.caption_from_shape([MeshInput(sample_id="s", mesh_path="missing.glb")], cfg)
    assert rows[0]["debug"]["mock_external"] is True
