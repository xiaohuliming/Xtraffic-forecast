from scripts.remote.linux.protocol_v3_event_gpu_probe import (
    SUPPORTED_PROBE_MODELS,
    parse_args,
)


def test_baseline_probe_accepts_convergence_recipe_and_budget(tmp_path) -> None:
    args = parse_args(
        [
            "--region",
            "Orange",
            "--models",
            "dcrnn",
            "--data-dir",
            str(tmp_path / "cache"),
            "--graph-dir",
            str(tmp_path / "graphs"),
            "--output",
            str(tmp_path / "probe.json"),
            "--epochs",
            "100",
            "--recipe-profile",
            "convergence100",
        ]
    )

    assert args.epochs == 100
    assert args.recipe_profile == "convergence100"
    assert "graphwavenet" in SUPPORTED_PROBE_MODELS
