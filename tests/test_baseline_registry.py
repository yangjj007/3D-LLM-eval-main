from pathlib import Path

import yaml

from eval.adapters import ADAPTER_REGISTRY
from eval.baselines.registry import BASELINE_SPECS, enabled_specs, skipped_specs
from eval.utils.path_bootstrap import repo_root


def test_enabled_baselines_have_adapters_and_configs():
    enabled = list(enabled_specs())
    assert {spec.name for spec in enabled} >= {
        "sar3d",
        "trellis",
        "gaussiancube",
        "shape_e",
        "three_d_llm",
        "pointllm_13b",
        "3dtopia_xl",
        "lgm",
        "instructblip_13b",
        "llava_13b",
    }
    for spec in enabled:
        assert spec.repo_url.startswith("https://github.com/")
        assert spec.entrypoint
        assert spec.config_path
        assert spec.adapter in ADAPTER_REGISTRY


def test_bridge_baselines_are_marked_and_registered():
    bridged = [spec for spec in BASELINE_SPECS.values() if spec.status == "bridged"]
    assert {spec.name for spec in bridged} == {
        "instructblip_13b",
        "llava_13b",
        "lgm",
    }
    for spec in bridged:
        assert spec.notes
        assert spec.adapter in ADAPTER_REGISTRY


def test_skipped_baselines_are_documented_and_not_registered():
    skipped = list(skipped_specs())
    assert {spec.name for spec in skipped} >= {"instantmesh"}
    for spec in skipped:
        assert spec.skip_reason
        assert spec.adapter is None or spec.adapter not in ADAPTER_REGISTRY


def test_no_unknown_status_values():
    assert {spec.status for spec in BASELINE_SPECS.values()} <= {"enabled", "bridged", "skipped"}


def test_real_baseline_configs_use_metadata_glb_defaults():
    for spec in enabled_specs():
        if not spec.config_path or "mock_" in spec.config_path:
            continue
        cfg_path = repo_root() / spec.config_path
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        data = cfg.get("data", {})
        assert data.get("metadata_csv") == "eval_data/metadata.csv", spec.name
        assert data.get("glb_dir") == "eval_data", spec.name
        assert "data_path" not in data, spec.name
