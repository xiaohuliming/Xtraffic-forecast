from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from scripts.remote.linux.protocol_v3_event_extension_multi_gpu import (
    assign_tasks_lpt,
    build_extension_nas_plan,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    CANONICAL_MANIFEST,
    SUPPORT_MANIFEST,
    ExtensionInputs,
    build_extension_invocation,
    load_extension_manifest,
    verify_extension_support,
)
from scripts.remote.windows.protocol_v3_runner import (
    ProtocolV3Inputs,
    derive_config_hash,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE_BUNDLE_ID = "bundle-test"
CACHE_BUNDLE_DIGEST = "a" * 64
CODE_SHA = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result(
    *,
    task,
    manifest,
    results_root: Path,
    cache_dir: Path,
    graph_dir: Path,
    extension: ExtensionInputs,
) -> None:
    protocol = ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
    )
    invocation = build_extension_invocation(
        task,
        manifest,
        extension,
        results_root / ".staging" / task.task_id / "__TEST__",
    )
    config_hash = derive_config_hash(
        task, manifest, protocol, invocation_config=invocation.config
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


def _build_extension_host(tmp_path: Path):
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=ROOT / "archive/node_order.npy",
    )
    extension = ExtensionInputs(
        support_root=support.support_root,
        node_order=support.node_order,
        support_manifest_sha256=support.manifest_sha256,
        igstgnn_data_parallel=False,
        max_microbatch_size=0,
    )
    seconds = {task.task_id: float(index + 1) for index, task in enumerate(manifest.tasks)}
    shards = assign_tasks_lpt(manifest.tasks, gpu_count=4, task_seconds=seconds)
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    results_root = tmp_path / "results"
    control_root = tmp_path / "control"
    for path in (cache_dir, graph_dir, results_root, control_root):
        path.mkdir(parents=True, exist_ok=True)
    for gpu, tasks in shards.items():
        shard_root = results_root / f"gpu{gpu:02d}"
        for task in tasks:
            _write_result(
                task=task,
                manifest=manifest,
                results_root=shard_root,
                cache_dir=cache_dir,
                graph_dir=graph_dir,
                extension=extension,
            )
        (shard_root / "queue_status.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "total_tasks": len(tasks),
                    "completed_tasks": len(tasks),
                    "current_task": None,
                }
            ),
            encoding="utf-8",
        )
    (results_root / "queue_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "total_tasks": 9,
                "completed_tasks": 9,
                "skipped_valid_tasks": 0,
                "failed_count": 0,
                "current_task": None,
            }
        ),
        encoding="utf-8",
    )
    (control_root / "runner.exitcode").write_text("0\n", encoding="ascii")
    loading_audit = control_root / "loading_equivalence.json"
    loading_audit.write_text('{"status":"EAGER_LAZY_EQUIVALENCE_OK"}\n', encoding="utf-8")
    plan = build_extension_nas_plan(
        matrix_id="protocol_v3_event_extensions_test",
        host_id="event_extensions_test",
        manifest=manifest,
        canonical_manifest=CANONICAL_MANIFEST,
        canonical_manifest_sha256=_sha256(CANONICAL_MANIFEST),
        support_manifest=SUPPORT_MANIFEST,
        support_manifest_sha256=support.manifest_sha256,
        loading_audit=loading_audit,
        loading_audit_sha256=_sha256(loading_audit),
        support_root=support.support_root,
        node_order=support.node_order,
        shards=shards,
        results_root=results_root,
        control_root=control_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
    )
    plan_path = tmp_path / "nas_backup_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path, plan


def test_extension_host_passes_the_shared_read_only_postflight(tmp_path: Path) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, _ = _build_extension_host(tmp_path)
    report = pf.build_postflight_report(plan_path)

    assert report["status"] == "verified"
    assert report["runner_kind"] == "event_extension"
    assert report["validated_tasks"] == 9
    assert report["support_manifest_sha256"] == _sha256(SUPPORT_MANIFEST)
    assert report["loading_audit_sha256"] == _sha256(
        Path(report["loading_audit"])
    )
    assert report["task_state_counts"] == {"completed": 9}


def test_extension_postflight_accepts_runner_revalidated_results(tmp_path: Path) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, plan = _build_extension_host(tmp_path)
    results_root = Path(plan["completion_gate"]["results_root"])
    for state_path in results_root.glob("gpu*/state/tasks/*.json"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["state"] = "skipped_valid"
        state_path.write_text(json.dumps(state), encoding="utf-8")

    report = pf.build_postflight_report(plan_path)

    assert report["validated_tasks"] == 9
    assert report["task_state_counts"] == {"skipped_valid": 9}
    assert {item["state"] for item in report["tasks"]} == {"skipped_valid"}


def test_extension_postflight_rejects_a_changed_support_binding(tmp_path: Path) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    plan_path, plan = _build_extension_host(tmp_path)
    plan["support_manifest_sha256"] = "c" * 64
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    try:
        pf.build_postflight_report(plan_path)
    except pf.PostflightError as exc:
        assert "support manifest" in str(exc)
    else:
        raise AssertionError("changed support manifest binding was accepted")


def _publish_extension_host(tmp_path: Path) -> Path:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    source = tmp_path / "source"
    plan_path, _ = _build_extension_host(source)
    control = source / "postflight_outputs"
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
    for source_path, name in (
        (marker, "_TRANSFER_COMPLETE.json"),
        (plan_path, plan_path.name),
        (report_path, "postflight.json"),
        (CANONICAL_MANIFEST, CANONICAL_MANIFEST.name),
        (SUPPORT_MANIFEST, SUPPORT_MANIFEST.name),
        (file_list, "transfer.files"),
        (transfer_manifest, "transfer.sha256"),
    ):
        shutil.copyfile(source_path, metadata / name)
    return published


def test_extension_composite_revalidates_the_published_tree(tmp_path: Path) -> None:
    from scripts.remote.linux.protocol_v3_event_full33_composite import (
        build_extension9_composite_index,
    )

    published = _publish_extension_host(tmp_path)
    output = tmp_path / "composite" / "index.json"
    index = build_extension9_composite_index(
        published_root=published,
        output_path=output,
        expected_matrix_id="protocol_v3_event_extensions_test",
        expected_host_id="event_extensions_test",
        expected_sources=4,
    )

    assert index["status"] == "verified_event_extensions_single_host_9"
    assert index["runner_kind"] == "event_extension"
    assert index["task_count"] == 9
    assert output.is_file()


def test_extension_launch_and_publication_scripts_are_fail_closed() -> None:
    launch = (
        ROOT / "scripts/remote/linux/protocol_v3_event_extension_launch_host.sh"
    ).read_text(encoding="utf-8")
    finalizer = (
        ROOT / "scripts/remote/linux/protocol_v3_event_nas_finalize.sh"
    ).read_text(encoding="utf-8")
    post_publish = (
        ROOT / "scripts/remote/linux/protocol_v3_event_extension_post_publish.sh"
    ).read_text(encoding="utf-8")

    assert "EXPECTED_PLAN_DIGEST" in launch
    assert "EXPECTED_TASKS=9" in launch
    assert "SUPPORT_MANIFEST=" in launch
    assert "SHUTDOWN_ON_SUCCESS" in launch
    assert "protocol_v3_event_extension_multi_gpu" in finalizer
    assert "train_protocol_v3_event_extension" in finalizer
    assert "MemoryMax=3221225472" in post_publish
    assert "protocol_v3_event_extension_composite" in post_publish
    assert "protocol_v3_event_merge_verified.sh" in post_publish
