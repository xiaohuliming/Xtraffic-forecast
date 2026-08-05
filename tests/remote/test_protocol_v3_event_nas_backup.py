from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.remote.windows.protocol_v3_baseline_runner import (
    build_baseline_invocation,
    load_baseline_manifest,
)
from scripts.remote.windows.protocol_v3_runner import (
    ProtocolV3Inputs,
    derive_config_hash,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = (
    ROOT / "config/protocol_v3/table4_event_baselines_phase1_v1.json"
)
POSTFLIGHT = (
    ROOT
    / "scripts/remote/windows/protocol_v3_event_host_postflight.py"
)
FINALIZER = (
    ROOT
    / "scripts/remote/windows/protocol_v3_event_nas_finalize.ps1"
)
BACKUP_PLAN = (
    ROOT / "config/protocol_v3/event_baseline_5080_nas_backup_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result(
    *,
    task,
    manifest,
    results_root: Path,
    cache_dir: Path,
    graph_dir: Path,
    code_sha: str,
) -> None:
    run_inputs = ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id="bundle-test",
        cache_bundle_digest="a" * 64,
        code_sha=code_sha,
        python_executable=sys.executable,
    )
    invocation = build_baseline_invocation(
        task,
        manifest,
        run_inputs,
        results_root / ".staging" / task.task_id / "__TEST__",
    )
    config_hash = derive_config_hash(
        task,
        manifest,
        run_inputs,
        invocation_config=invocation.config,
    )
    result_dir = results_root / task.output_dir
    result_dir.mkdir(parents=True)
    summary = {
        "task_id": task.task_id,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "protocol_version": manifest.protocol_version,
        "sampling_protocol": task.sampling_protocol,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": code_sha,
        "config_hash": config_hash,
        "all": 1.0,
        "valid_count": 24,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    actual = np.full((1, 12, 2), 10.0, dtype=np.float32)
    np.savez_compressed(
        result_dir / "test_predictions.npz",
        region_code=np.asarray([0], dtype=np.int64),
        sample_start=np.asarray([100], dtype=np.int64),
        region_node_idx=np.asarray([0, 1], dtype=np.int64),
        pred_raw_flow=actual + 1.0,
        actual_future_flow=actual,
        y_mask_flow=np.ones_like(actual, dtype=np.bool_),
        affected_mask=np.asarray([[True, False]], dtype=np.bool_),
    )
    for artifact in invocation.required_artifacts:
        path = result_dir / artifact
        if not path.exists():
            path.write_bytes(f"artifact:{artifact}\n".encode())
    state = results_root / "state/tasks" / f"{task.task_id}.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "state": "completed",
                "task_id": task.task_id,
                "config_hash": config_hash,
            }
        ),
        encoding="utf-8",
    )
    log = results_root / "logs" / task.task_id / "attempt_0001.log"
    log.parent.mkdir(parents=True)
    log.write_text("complete\n", encoding="utf-8")
    (results_root / "queue_status.json").write_text(
        json.dumps({"state": "failed", "completed_tasks": 1}),
        encoding="utf-8",
    )


def _case(tmp_path: Path):
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    manifest = load_baseline_manifest(CANONICAL)
    tasks = manifest.tasks[:2]
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    cache_dir.mkdir()
    graph_dir.mkdir()
    sources = []
    for index, task in enumerate(tasks, start=1):
        root = tmp_path / f"source{index}"
        code_sha = str(index) * 40
        _write_result(
            task=task,
            manifest=manifest,
            results_root=root,
            cache_dir=cache_dir,
            graph_dir=graph_dir,
            code_sha=code_sha,
        )
        sources.append(
            {
                "source_id": f"source{index}",
                "results_root": str(root),
                "code_sha": code_sha,
                "task_ids": [task.task_id],
            }
        )
    gate_root = Path(sources[-1]["results_root"])
    (gate_root / "queue_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "total_tasks": 1,
                "completed_tasks": 1,
                "current_task": None,
            }
        ),
        encoding="utf-8",
    )
    exit_code = tmp_path / "runner.exitcode"
    exit_code.write_text("0\n", encoding="utf-8")
    plan = {
        "plan_version": 1,
        "matrix_id": "test_event_matrix",
        "host_id": "test_host",
        "canonical_manifest": str(CANONICAL),
        "canonical_manifest_sha256": _sha256(CANONICAL),
        "protocol_version": "protocol_v3_noleak_dedup",
        "cache_schema_version": 3,
        "cache_bundle_id": "bundle-test",
        "cache_bundle_digest": "a" * 64,
        "cache_dir": str(cache_dir),
        "graph_dir": str(graph_dir),
        "python_executable": sys.executable,
        "expected_task_count": 2,
        "sources": sources,
        "completion_gate": {
            "results_root": str(gate_root),
            "runner_exit_code": str(exit_code),
            "expected_tasks": 1,
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return pf, plan_path, sources, tasks


def test_multiroot_postflight_revalidates_results_and_namespaces_export(
    tmp_path: Path,
) -> None:
    pf, plan_path, sources, tasks = _case(tmp_path)

    report = pf.build_postflight_report(plan_path)

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 2
    assert report["task_state_counts"] == {"completed": 2}
    assert [item["task_id"] for item in report["tasks"]] == [
        task.task_id for task in tasks
    ]
    paths = {item["path"] for item in report["files"]}
    assert (
        f"sources/source1/results/tasks/{tasks[0].task_id}/summary.json"
        in paths
    )
    assert (
        f"sources/source2/results/tasks/{tasks[1].task_id}/summary.json"
        in paths
    )
    assert report["completion_gate"]["runner_exit_code"] == 0
    assert report["sources"][0]["code_sha"] == sources[0]["code_sha"]


def test_multiroot_postflight_fails_closed_on_tampered_summary(
    tmp_path: Path,
) -> None:
    pf, plan_path, sources, tasks = _case(tmp_path)
    summary_path = (
        Path(sources[0]["results_root"])
        / "tasks"
        / tasks[0].task_id
        / "summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["all"] = 99.0
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(pf.PostflightError, match="prediction metric"):
        pf.build_postflight_report(plan_path)


def test_nas_finalizer_is_resumable_atomic_and_never_shuts_down() -> None:
    text = FINALIZER.read_text(encoding="utf-8")

    assert "protocol_v3_event_host_postflight.py" in text
    assert "runner.exitcode" in text
    assert "Get-RemoteReusableBytes" in text
    assert "sha256sum -c ../metadata/transfer.sha256" in text
    assert ".incoming" in text
    assert "mv '$RemoteIncoming' '$RemoteFinal'" in text
    assert "FINALIZE_SUCCESS" in text
    assert "machine_stays_on" in text
    assert "train_baseline_zoo" in text
    assert "train_graphwavenet" in text
    assert "shutdown" not in text.lower()


def test_5080_backup_plan_is_exactly_alameda_and_contra_costa() -> None:
    plan = json.loads(BACKUP_PLAN.read_text(encoding="utf-8"))
    manifest = load_baseline_manifest(CANONICAL)
    task_by_id = {task.task_id: task for task in manifest.tasks}
    assigned = [
        task_id
        for source in plan["sources"]
        for task_id in source["task_ids"]
    ]

    assert plan["host_id"] == "event_rtx5080_alameda_contra"
    assert plan["expected_task_count"] == 22
    assert len(assigned) == len(set(assigned)) == 22
    assert [task.task_id for task in manifest.tasks[:22]] == assigned
    assert {task_by_id[task_id].region for task_id in assigned} == {
        "Alameda",
        "ContraCosta",
    }
    assert plan["canonical_manifest_sha256"] == _sha256(CANONICAL)
    assert plan["completion_gate"]["expected_tasks"] == 13
