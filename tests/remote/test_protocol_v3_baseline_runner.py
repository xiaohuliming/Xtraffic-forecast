from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np

from scripts.remote.windows.protocol_v3_baseline_runner import (
    BASELINE_METHODS,
    build_baseline_invocation,
    load_baseline_manifest,
)
from scripts.remote.windows.protocol_v3_runner import ProtocolV3Inputs
from scripts.train_baseline_zoo import save_outputs
from dist_net.training_runtime import TrainingRunContext


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "config"
    / "protocol_v3"
    / "table4_fullwindow_baselines_phase1_v1.json"
)
EVENT_MANIFEST = (
    REPO_ROOT
    / "config"
    / "protocol_v3"
    / "table4_event_baselines_phase1_v1.json"
)
EVENT_LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "remote"
    / "windows"
    / "protocol_v3_event_baseline_phase1_5080.cmd"
)


def _inputs(tmp_path: Path) -> ProtocolV3Inputs:
    return ProtocolV3Inputs(
        results_root=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        graph_dir=tmp_path / "graphs",
        cache_bundle_id="bundle-test",
        cache_bundle_digest="a" * 64,
        code_sha="b" * 40,
        python_executable="python",
    )


def test_phase1_manifest_is_the_exact_supported_table4_subset() -> None:
    manifest = load_baseline_manifest(MANIFEST)

    assert len(manifest.tasks) == 33
    assert manifest.training == {
        "epochs": 30,
        "patience": 6,
        "device": "cuda:0",
        "batch_size": 32,
        "stride": 1,
        "train_frac": 0.7,
        "val_frac": 0.15,
    }
    assert {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    } == {
        (region, method, 42, "full_window")
        for region in ("Alameda", "ContraCosta", "Orange")
        for method in BASELINE_METHODS
    }


def test_event_manifest_is_the_exact_supported_table4_subset() -> None:
    manifest = load_baseline_manifest(EVENT_MANIFEST)

    assert len(manifest.tasks) == 33
    assert {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    } == {
        (region, method, 42, "event")
        for region in ("Alameda", "ContraCosta", "Orange")
        for method in BASELINE_METHODS
    }


def test_baseline_invocation_uses_v3_target_disjoint_70_15_15_split(
    tmp_path: Path,
) -> None:
    manifest = load_baseline_manifest(MANIFEST)
    task = next(task for task in manifest.tasks if task.model == "lstm")

    invocation = build_baseline_invocation(
        task,
        manifest,
        _inputs(tmp_path),
        tmp_path / "attempt",
    )

    command = list(invocation.command)
    assert command[3] == "scripts/train_baseline_zoo.py"
    assert command[command.index("--model") + 1] == "lstm"
    assert command[command.index("--protocol") + 1] == "full_window"
    assert command[command.index("--train_frac") + 1] == "0.7"
    assert command[command.index("--val_frac") + 1] == "0.15"
    assert invocation.result_relpath == Path("work") / "Alameda"
    assert invocation.required_artifacts == (
        "summary.json",
        "ckpt_best.pt",
        "test_predictions.npz",
        "train.log",
    )
    assert invocation.required_metrics == ("all",)


def test_history_last_has_no_fake_checkpoint_contract(tmp_path: Path) -> None:
    manifest = load_baseline_manifest(MANIFEST)
    task = next(task for task in manifest.tasks if task.model == "hl")

    invocation = build_baseline_invocation(
        task,
        manifest,
        _inputs(tmp_path),
        tmp_path / "attempt",
    )

    assert invocation.required_artifacts == (
        "summary.json",
        "test_predictions.npz",
        "train.log",
    )


def test_graphwavenet_keeps_its_fullwindow_output_suffix(tmp_path: Path) -> None:
    manifest = load_baseline_manifest(MANIFEST)
    task = next(
        task for task in manifest.tasks if task.model == "graphwavenet"
    )

    invocation = build_baseline_invocation(
        task,
        manifest,
        _inputs(tmp_path),
        tmp_path / "attempt",
    )

    assert invocation.result_relpath == Path("work_fullwindow") / "Alameda"


def test_event_graphwavenet_uses_unsuffixed_output_root(tmp_path: Path) -> None:
    manifest = load_baseline_manifest(EVENT_MANIFEST)
    task = next(
        task for task in manifest.tasks if task.model == "graphwavenet"
    )

    invocation = build_baseline_invocation(
        task,
        manifest,
        _inputs(tmp_path),
        tmp_path / "attempt",
    )

    command = list(invocation.command)
    assert command[command.index("--protocol") + 1] == "event"
    assert invocation.result_relpath == Path("work") / "Alameda"


def test_event_launcher_isolated_from_fullwindow_queue() -> None:
    text = EVENT_LAUNCHER.read_text(encoding="utf-8")

    assert "table4_event_baselines_phase1_v1.json" in text
    assert "v3_event_repo_v1" in text
    assert "v3_event_phase1_5080_v1" in text
    assert "v3_baselines_phase1_5080_v3" not in text


def test_fullwindow_summary_represents_empty_affected_subset_as_null(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "result"
    context = TrainingRunContext.prepare(
        output_dir,
        data_dir=tmp_path / "cache",
        region="Alameda",
        model="hl",
        variant="base",
        seed=42,
        sampling_protocol="full_window",
        environment={},
    )
    region = SimpleNamespace(
        region_idx=np.asarray([0, 1], dtype=np.int64),
        result_metadata=lambda: {
            "protocol_version": "protocol_v3_noleak_dedup",
            "cache_schema_version": 3,
            "sampling_protocol": "full_window",
        },
    )
    prediction = np.zeros((1, 12, 2), dtype=np.float32)
    actual = np.ones((1, 12, 2), dtype=np.float32)
    valid = np.ones((1, 12, 2), dtype=bool)
    affected = np.zeros((1, 2), dtype=bool)

    save_outputs(
        output_dir,
        np.asarray([0], dtype=np.int64),
        np.asarray([12], dtype=np.int64),
        region,
        prediction,
        actual,
        valid,
        affected,
        best_val=None,
        n_params=0,
        run_context=context,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["all"] == 1.0
    assert summary["affected"] is None
    assert summary["unaffected"] == 1.0
