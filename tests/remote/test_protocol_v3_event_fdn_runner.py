from __future__ import annotations

from pathlib import Path

from scripts.remote.windows.protocol_v3_runner import ProtocolV3Inputs


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "config/protocol_v3/table4_event_fdn_v1.json"
)


def _inputs(tmp_path: Path) -> ProtocolV3Inputs:
    return ProtocolV3Inputs(
        results_root=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        graph_dir=tmp_path / "graphs",
        cache_bundle_id="bundle-v3-test",
        cache_bundle_digest="a" * 64,
        code_sha="b" * 40,
        python_executable="python",
    )


def test_event_fdn_manifest_is_exact_three_region_seed42_matrix():
    from scripts.remote.windows.protocol_v3_event_fdn_runner import (
        load_fdn_manifest,
    )

    manifest = load_fdn_manifest(MANIFEST_PATH)

    assert manifest.training == {
        "epochs": 30,
        "patience": 6,
        "device": "cuda:0",
        "batch_size": 32,
        "stride": 1,
        "train_frac": 0.7,
        "val_frac": 0.15,
    }
    assert [task.region for task in manifest.tasks] == [
        "Alameda",
        "ContraCosta",
        "Orange",
    ]
    assert {task.kind for task in manifest.tasks} == {"fourier_dual_net"}
    assert {task.model for task in manifest.tasks} == {"fourier_dual_net"}
    assert {task.variant for task in manifest.tasks} == {"learnable_k3"}
    assert {task.seed for task in manifest.tasks} == {42}
    assert {task.sampling_protocol for task in manifest.tasks} == {"event"}


def test_event_fdn_invocation_uses_the_no_leak_event_contract(tmp_path):
    from scripts.remote.windows.protocol_v3_event_fdn_runner import (
        build_fdn_invocation,
        load_fdn_manifest,
    )

    manifest = load_fdn_manifest(MANIFEST_PATH)
    task = manifest.tasks[0]
    invocation = build_fdn_invocation(
        task,
        manifest,
        _inputs(tmp_path),
        tmp_path / "attempt",
    )
    command = list(invocation.command)

    assert command[command.index("--protocol") + 1] == "event"
    assert command[command.index("--decomp_mode") + 1] == "learnable"
    assert command[command.index("--K") + 1] == "3"
    assert command[command.index("--epochs") + 1] == "30"
    assert command[command.index("--patience") + 1] == "6"
    assert command[command.index("--batch_size") + 1] == "32"
    assert invocation.result_relpath == Path("work/learnable_K3/Alameda")
    assert invocation.required_metrics == ("all", "affected", "unaffected")
    assert set(invocation.required_artifacts) == {
        "summary.json",
        "ckpt_best.pt",
        "test_predictions.npz",
        "train.log",
        "decomp_mask.json",
    }
    assert invocation.config["optimizer"] == {
        "name": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 5.0,
    }
    assert invocation.config["scheduler"] == {
        "name": "CosineAnnealingLR",
        "t_max": "epochs_times_training_batches",
        "eta_min": 0.00001,
        "step_unit": "optimizer_step",
    }
