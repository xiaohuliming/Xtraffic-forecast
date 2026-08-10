from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.remote.linux.protocol_v3_event_multi_gpu import (
    assign_tasks_lpt,
    build_nas_plan,
    load_calibration,
    select_region_tasks,
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
CALIBRATION = (
    ROOT / "config/protocol_v3/event_baseline_gpu_calibration_v1.json"
)
CACHE_BUNDLE_ID = "bundle-test"
CACHE_BUNDLE_DIGEST = "a" * 64
CODE_SHA = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_with_training(manifest, training):
    return replace(manifest, training=training)


def _write_result(*, task, manifest, results_root, cache_dir, graph_dir) -> None:
    run_inputs = ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
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
        "cache_bundle_id": CACHE_BUNDLE_ID,
        "cache_bundle_digest": CACHE_BUNDLE_DIGEST,
        "code_sha": CODE_SHA,
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
    if task.model == "graphwavenet":
        # The production GraphWaveNet trainer intentionally publishes only its
        # required scalar MAE. Paper-table horizon metrics must be derived from
        # the bound prediction artifact instead of assumed to exist here.
        for name in ("h3", "h6", "h12", "mae_h", "rmse", "rmse_h", "mape", "mape_h"):
            summary.pop(name)
    (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    actual = np.full((1, 12, 2), 10.0, dtype=np.float32)
    np.savez_compressed(
        result_dir / "test_predictions.npz",
        region_code=np.asarray(
            [{"Alameda": 0, "ContraCosta": 1, "Orange": 2}[task.region]],
            dtype=np.int64,
        ),
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
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {"state": "completed", "task_id": task.task_id, "config_hash": config_hash}
        ),
        encoding="utf-8",
    )


def _build_host(
    tmp_path: Path,
    *,
    microbatch: dict[str, int] | None = None,
    regions: tuple[str, ...] = ("ContraCosta", "Orange"),
):
    """Materialize a realistic interleaved four-GPU host and its NAS plan."""
    manifest = load_baseline_manifest(CANONICAL)
    calibration = load_calibration(CALIBRATION)
    tasks = select_region_tasks(manifest, regions)
    microbatch = microbatch or {}
    shards = assign_tasks_lpt(
        tasks, 4, calibration, microbatch_sizes=microbatch
    )
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    results_root = tmp_path / "results"
    control_root = tmp_path / "control"
    for path in (cache_dir, graph_dir, results_root, control_root):
        path.mkdir(parents=True, exist_ok=True)

    total = 0
    for gpu, shard in shards.items():
        shard_root = results_root / f"gpu{gpu:02d}"
        for task in shard:
            training = dict(manifest.training)
            if task.task_id in microbatch:
                training["microbatch_size"] = microbatch[task.task_id]
            _write_result(
                task=task,
                manifest=_manifest_with_training(manifest, training),
                results_root=shard_root,
                cache_dir=cache_dir,
                graph_dir=graph_dir,
            )
            total += 1
        (shard_root / "queue_status.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "total_tasks": len(shard),
                    "completed_tasks": len(shard),
                    "current_task": None,
                }
            ),
            encoding="utf-8",
        )
    (results_root / "queue_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "total_tasks": total,
                "completed_tasks": total,
                "skipped_valid_tasks": 0,
                "failed_count": 0,
                "current_task": None,
            }
        ),
        encoding="utf-8",
    )
    (control_root / "runner.exitcode").write_text("0\n", encoding="utf-8")

    plan = build_nas_plan(
        matrix_id="protocol_v3_event_cc_orange_test",
        host_id="event_test_multi_gpu",
        manifest=manifest,
        canonical_manifest=CANONICAL,
        canonical_manifest_sha256=_sha256(CANONICAL),
        shards=shards,
        results_root=results_root,
        control_root=control_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
        microbatch_sizes=microbatch,
    )
    plan_path = tmp_path / "nas_backup_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path, plan, shards


def _publish_verified_host(tmp_path: Path) -> tuple[Path, Path]:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    build_root = tmp_path / "source"
    target = "v3e033_event_d2stgnn_orange_seed42"
    plan_path, _, _ = _build_host(
        build_root,
        microbatch={target: 16},
        regions=("Alameda", "ContraCosta", "Orange"),
    )
    control = build_root / "control_outputs"
    report_path = control / "postflight.json"
    file_list = control / "transfer.files"
    transfer_manifest = control / "transfer.sha256"
    marker = control / "_TRANSFER_COMPLETE.json"
    report = pf.build_postflight_report(plan_path)
    pf.write_outputs(
        report=report,
        report_path=report_path,
        file_list_path=file_list,
        transfer_manifest_path=transfer_manifest,
        marker_path=marker,
    )

    published = tmp_path / "published" / report["host_id"]
    results = published / "results"
    metadata = published / "metadata"
    metadata.mkdir(parents=True)
    for record in report["files"]:
        destination = results / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["source_path"], destination)
    for source, name in (
        (marker, "_TRANSFER_COMPLETE.json"),
        (plan_path, plan_path.name),
        (report_path, "postflight.json"),
        (CANONICAL, CANONICAL.name),
        (file_list, "transfer.files"),
        (transfer_manifest, "transfer.sha256"),
    ):
        shutil.copyfile(source, metadata / name)
    return published, plan_path


def test_interleaved_four_gpu_host_passes_the_real_postflight(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, _, _ = _build_host(tmp_path)

    report = pf.build_postflight_report(plan_path)

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 22
    assert report["task_state_counts"] == {"completed": 22}
    assert report["completion_gate"]["runner_exit_code"] == 0
    assert report["source_layout"] == "interleaved"


def test_microbatch_task_survives_the_real_postflight(tmp_path: Path) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    target = "v3e032_event_dgcrn_orange_seed42"
    plan_path, plan, _ = _build_host(tmp_path, microbatch={target: 8})

    report = pf.build_postflight_report(plan_path)

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 22
    assert plan["task_overrides"] == {target: {"microbatch_size": 8}}
    assert report["task_overrides"] == {target: {"microbatch_size": 8}}
    entry = next(item for item in report["tasks"] if item["task_id"] == target)
    assert entry["invocation_config"]["microbatch_size"] == 8


def test_postflight_rejects_an_undeclared_microbatch_result(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    target = "v3e032_event_dgcrn_orange_seed42"
    plan_path, plan, _ = _build_host(tmp_path, microbatch={target: 8})
    plan.pop("task_overrides")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    with pytest.raises(pf.PostflightError):
        pf.build_postflight_report(plan_path)


def test_postflight_rejects_a_wrong_microbatch_declaration(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    target = "v3e032_event_dgcrn_orange_seed42"
    plan_path, plan, _ = _build_host(tmp_path, microbatch={target: 8})
    plan["task_overrides"] = {target: {"microbatch_size": 16}}
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    with pytest.raises(pf.PostflightError):
        pf.build_postflight_report(plan_path)


def test_postflight_rejects_an_override_for_an_unassigned_task(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, plan, _ = _build_host(tmp_path)
    plan["task_overrides"] = {"v3e001_event_hl_alameda_seed42": {"microbatch_size": 8}}
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    with pytest.raises(pf.PostflightError):
        pf.build_postflight_report(plan_path)


def test_interleaved_layout_still_requires_canonical_order_inside_a_source(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, plan, _ = _build_host(tmp_path)
    source = next(item for item in plan["sources"] if len(item["task_ids"]) > 1)
    source["task_ids"] = list(reversed(source["task_ids"]))
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    with pytest.raises(pf.PostflightError, match="canonical"):
        pf.build_postflight_report(plan_path)


def test_interleaved_layout_requires_deterministic_source_order(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, plan, _ = _build_host(tmp_path)
    plan["sources"] = list(reversed(plan["sources"]))
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    with pytest.raises(pf.PostflightError, match="source"):
        pf.build_postflight_report(plan_path)


def test_default_layout_keeps_the_strict_contiguous_contract(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, plan, _ = _build_host(tmp_path)
    plan.pop("source_layout")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    with pytest.raises(pf.PostflightError, match="canonical manifest order"):
        pf.build_postflight_report(plan_path)


def test_shipped_5080_plan_still_validates_under_the_default_layout() -> None:
    plan = json.loads(
        (
            ROOT / "config/protocol_v3/event_baseline_5080_nas_backup_v1.json"
        ).read_text(encoding="utf-8")
    )
    manifest = load_baseline_manifest(CANONICAL)
    assigned = [
        task_id for source in plan["sources"] for task_id in source["task_ids"]
    ]
    canonical = [task.task_id for task in manifest.tasks]

    assert "source_layout" not in plan
    assert "task_overrides" not in plan
    assert assigned == [task_id for task_id in canonical if task_id in set(assigned)]
