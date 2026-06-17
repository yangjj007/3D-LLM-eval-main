from eval.adapters import ADAPTER_REGISTRY
from eval.baselines.registry import BASELINE_SPECS, enabled_specs, skipped_specs


def test_enabled_baselines_have_adapters_and_configs():
    enabled = list(enabled_specs())
    assert {spec.name for spec in enabled} >= {
        "sar3d",
        "trellis",
        "gaussiancube",
        "shape_e",
        "three_d_llm",
        "pointllm_13b",
    }
    for spec in enabled:
        assert spec.repo_url.startswith("https://github.com/")
        assert spec.entrypoint
        assert spec.config_path
        assert spec.adapter in ADAPTER_REGISTRY


def test_skipped_baselines_are_documented_and_not_registered():
    skipped = list(skipped_specs())
    assert {spec.name for spec in skipped} >= {
        "instantmesh",
        "instructblip_13b",
        "llava_13b",
        "3dtopia_xl",
        "lgm",
    }
    for spec in skipped:
        assert spec.skip_reason
        assert spec.adapter is None or spec.adapter not in ADAPTER_REGISTRY


def test_no_unknown_status_values():
    assert {spec.status for spec in BASELINE_SPECS.values()} <= {"enabled", "skipped"}
