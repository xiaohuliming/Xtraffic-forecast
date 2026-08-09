from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.remote.linux.protocol_v3_event_incremental_backup import (
    BackupError,
    discover_completed_tasks,
)
from scripts.remote.windows.protocol_v3_baseline_runner import (
    build_baseline_invocation,
    load_baseline_manifest,
)
from scripts.remote.windows.protocol_v3_runner import (
    ProtocolV3Inputs,
    derive_config_hash,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "config/protocol_v3/table4_event_baselines_phase1_v1.json"


def _write_case(tmp_path: Path) -> tuple[Path, Path, str]:
    manifest = load_baseline_manifest(CANONICAL)
    task = manifest.tasks[0]
    results_root = tmp_path / "gpu00"
    cache = tmp_path / "cache"
    graphs = tmp_path / "graphs"
    cache.mkdir()
    graphs.mkdir()
    run_inputs = ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=cache,
        graph_dir=graphs,
        cache_bundle_id="bundle-test",
        cache_bundle_digest="a" * 64,
        code_sha="b" * 40,
        python_executable=sys.executable,
    )
    invocation = build_baseline_invocation(
        task,
        manifest,
        run_inputs,
        results_root / ".staging" / task.task_id / "__TEST__",
    )
    config_hash = derive_config_hash(
        task, manifest, run_inputs, invocation_config=invocation.config
    )
    result = results_root / task.output_dir
    result.mkdir(parents=True)
    actual = np.full((1, 12, 2), 10.0, dtype=np.float32)
    np.savez_compressed(
        result / "test_predictions.npz",
        region_code=np.asarray([0], dtype=np.int64),
        sample_start=np.asarray([100], dtype=np.int64),
        region_node_idx=np.asarray([0, 1], dtype=np.int64),
        pred_raw_flow=actual + 1.0,
        actual_future_flow=actual,
        y_mask_flow=np.ones_like(actual, dtype=np.bool_),
        affected_mask=np.asarray([[True, False]], dtype=np.bool_),
    )
    summary = {
        "task_id": task.task_id,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "protocol_version": manifest.protocol_version,
        "sampling_protocol": task.sampling_protocol,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": "bundle-test",
        "cache_bundle_digest": "a" * 64,
        "code_sha": "b" * 40,
        "config_hash": config_hash,
        "all": 1.0,
        "h3": 1.0,
        "h6": 1.0,
        "h12": 1.0,
        "mae_h": [1.0] * 12,
        "rmse": 1.0,
        "rmse_h": [1.0] * 12,
        "mape": 10.0,
        "mape_h": [10.0] * 12,
        "valid_count": 24,
    }
    (result / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (result / "test_predictions.npz").touch(exist_ok=True)
    (result / "train.log").write_text("complete\n", encoding="utf-8")
    state_path = results_root / "state/tasks" / f"{task.task_id}.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "state": "completed",
                "task_id": task.task_id,
                "attempt": 1,
                "returncode": 0,
                "config_hash": config_hash,
                "final_dir": str(result),
                "log_path": str(
                    results_root / "logs" / task.task_id / "attempt_0001.log"
                ),
            }
        ),
        encoding="utf-8",
    )
    log = results_root / "logs" / task.task_id / "attempt_0001.log"
    log.parent.mkdir(parents=True)
    log.write_text("runner complete\n", encoding="utf-8")
    plan = {
        "plan_version": 1,
        "matrix_id": "test_matrix",
        "host_id": "test_host",
        "canonical_manifest": str(CANONICAL),
        "cache_bundle_id": "bundle-test",
        "cache_bundle_digest": "a" * 64,
        "cache_dir": str(cache),
        "graph_dir": str(graphs),
        "python_executable": sys.executable,
        "expected_task_count": 1,
        "sources": [
            {
                "source_id": "gpu00",
                "results_root": str(results_root),
                "code_sha": "b" * 40,
                "task_ids": [task.task_id],
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, state_path, task.task_id


def test_discovers_only_validated_completed_task_files(tmp_path: Path) -> None:
    plan_path, _, task_id = _write_case(tmp_path)

    snapshots = discover_completed_tasks(plan_path)

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.task_id == task_id
    paths = {item.remote_relative for item in snapshot.files}
    assert f"sources/gpu00/results/tasks/{task_id}/summary.json" in paths
    assert f"sources/gpu00/results/state/tasks/{task_id}.json" in paths
    assert f"sources/gpu00/results/logs/{task_id}/attempt_0001.log" in paths


def test_running_task_is_not_snapshotted(tmp_path: Path) -> None:
    plan_path, state_path, _ = _write_case(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    assert discover_completed_tasks(plan_path) == []


def test_known_unchanged_completed_state_is_not_revalidated(tmp_path: Path) -> None:
    plan_path, _, _ = _write_case(tmp_path)
    first = discover_completed_tasks(plan_path)

    assert discover_completed_tasks(
        plan_path,
        known_state_sha256={first[0].task_id: first[0].state_sha256},
    ) == []


def test_completed_task_with_wrong_final_dir_fails_closed(tmp_path: Path) -> None:
    plan_path, state_path, task_id = _write_case(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["final_dir"] = str(tmp_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(BackupError, match=f"final_dir mismatch: {task_id}"):
        discover_completed_tasks(plan_path)


def test_completed_task_symlink_fails_closed(tmp_path: Path) -> None:
    plan_path, _, task_id = _write_case(tmp_path)
    result = tmp_path / "gpu00" / "tasks" / task_id
    (result / "unsafe").symlink_to(result / "summary.json")

    with pytest.raises(BackupError, match="symlink"):
        discover_completed_tasks(plan_path)
