from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from scripts.remote.linux.protocol_v3_event_convergence_multi_gpu import (
    assign_tasks_grouped_lpt,
    build_convergence_nas_plan,
    default_microbatch_sizes,
    parse_args,
)
from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    CANONICAL_MANIFEST,
    ConvergenceInputs,
    build_convergence_invocation,
    load_convergence_manifest,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    SUPPORT_MANIFEST,
    ExtensionInputs,
    verify_extension_support,
)
from scripts.remote.windows.protocol_v3_runner import (
    ProtocolV3Inputs,
    derive_config_hash,
)


ROOT = CANONICAL_MANIFEST.parents[2]
CACHE_BUNDLE_ID = "bundle-test"
CACHE_BUNDLE_DIGEST = "a" * 64
CODE_SHA = "b" * 40


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_convergence_multi_gpu_keeps_memory_fallback_groups_isolated() -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    microbatches = default_microbatch_sizes(manifest.tasks)
    task_seconds = {
        task.task_id: float(index + 1)
        for index, task in enumerate(manifest.tasks)
    }

    shards = assign_tasks_grouped_lpt(
        manifest.tasks,
        gpu_count=8,
        task_seconds=task_seconds,
        microbatch_sizes=microbatches,
    )

    assert microbatches == {
        "v3c033_event_sttn_orange_seed42": 16,
        "v3c036_event_d2stgnn_orange_seed42": 16,
        "v3c038_event_igstgnn_orange_seed42": 8,
        "v3c039_event_ndr_net_orange_seed42": 8,
    }
    assigned = [task.task_id for tasks in shards.values() for task in tasks]
    assert sorted(assigned) == sorted(task.task_id for task in manifest.tasks)
    assert len(assigned) == len(set(assigned)) == 39
    for tasks in shards.values():
        assert len({microbatches.get(task.task_id) for task in tasks}) == 1


def test_convergence_nas_plan_binds_mixed_runner_support_and_diagnostics(
    tmp_path,
) -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    microbatches = default_microbatch_sizes(manifest.tasks)
    task_seconds = {task.task_id: 1.0 for task in manifest.tasks}
    shards = assign_tasks_grouped_lpt(
        manifest.tasks,
        gpu_count=4,
        task_seconds=task_seconds,
        microbatch_sizes=microbatches,
    )

    plan = build_convergence_nas_plan(
        matrix_id=manifest.matrix_id,
        host_id="convergence-host",
        manifest=manifest,
        canonical_manifest=CANONICAL_MANIFEST,
        canonical_manifest_sha256="1" * 64,
        support_manifest=tmp_path / "support.json",
        support_manifest_sha256="2" * 64,
        loading_audit=tmp_path / "loading.json",
        loading_audit_sha256="3" * 64,
        support_root=tmp_path / "support",
        node_order=tmp_path / "node_order.npy",
        shards=shards,
        results_root=tmp_path / "results",
        control_root=tmp_path / "control",
        cache_dir=tmp_path / "cache",
        graph_dir=tmp_path / "graphs",
        cache_bundle_id="bundle-test",
        cache_bundle_digest="4" * 64,
        code_sha="5" * 40,
        python_executable="python",
        microbatch_sizes=microbatches,
    )

    assert plan["runner_kind"] == "event_convergence"
    assert plan["expected_task_count"] == 39
    assert plan["task_overrides"] == {
        task_id: {"microbatch_size": size}
        for task_id, size in sorted(microbatches.items())
    }
    assert plan["required_task_artifacts"] == ["training_diagnostics.json"]


def test_convergence_multi_gpu_supports_plan_only_gate(tmp_path) -> None:
    args = parse_args(
        [
            "--calibration",
            str(tmp_path / "calibration.json"),
            "--loading-audit",
            str(tmp_path / "loading.json"),
            "--control-root",
            str(tmp_path / "control"),
            "--results-root",
            str(tmp_path / "results"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--graph-dir",
            str(tmp_path / "graphs"),
            "--support-root",
            str(tmp_path / "support"),
            "--node-order",
            str(tmp_path / "node_order.npy"),
            "--cache-bundle-id",
            "bundle-test",
            "--cache-bundle-digest",
            "a" * 64,
            "--code-sha",
            "b" * 40,
            "--host-id",
            "host-test",
            "--gpu-count",
            "8",
            "--plan-only",
        ]
    )

    assert args.matrix_id == "protocol_v3_event_convergence_seed42_39_v1"
    assert args.gpu_count == 8
    assert args.plan_only is True


def test_convergence_host_passes_shared_read_only_postflight(tmp_path) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf

    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    task = next(task for task in manifest.tasks if task.model == "dcrnn")
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    results_root = tmp_path / "results"
    control_root = tmp_path / "control"
    for path in (cache_dir, graph_dir, results_root, control_root):
        path.mkdir(parents=True)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=ROOT / "archive/node_order.npy",
    )
    protocol = ProtocolV3Inputs(
        results_root=results_root / "gpu00",
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
    )
    inputs = ConvergenceInputs(
        protocol=protocol,
        extension=ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=False,
            max_microbatch_size=0,
            cache_dir=cache_dir,
            graph_dir=graph_dir,
            python_executable=sys.executable,
        ),
    )
    invocation = build_convergence_invocation(
        task,
        manifest,
        inputs,
        protocol.results_root / ".staging" / task.task_id / "__TEST__",
    )
    config_hash = derive_config_hash(
        task, manifest, protocol, invocation_config=invocation.config
    )
    result_dir = protocol.results_root / task.output_dir
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
        "valid_count": 24,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
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
            path.write_text("{}\n", encoding="utf-8")
    state_path = protocol.results_root / "state/tasks" / f"{task.task_id}.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {"state": "completed", "task_id": task.task_id, "config_hash": config_hash}
        ),
        encoding="utf-8",
    )
    (protocol.results_root / "queue_status.json").write_text(
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
    (results_root / "queue_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "total_tasks": 1,
                "completed_tasks": 1,
                "skipped_valid_tasks": 0,
                "failed_count": 0,
                "current_task": None,
            }
        ),
        encoding="utf-8",
    )
    (control_root / "runner.exitcode").write_text("0\n", encoding="ascii")
    loading_audit = control_root / "loading.json"
    loading_audit.write_text("{}\n", encoding="utf-8")
    plan = build_convergence_nas_plan(
        matrix_id=manifest.matrix_id,
        host_id="convergence-test",
        manifest=manifest,
        canonical_manifest=CANONICAL_MANIFEST,
        canonical_manifest_sha256=_sha256(CANONICAL_MANIFEST),
        support_manifest=SUPPORT_MANIFEST,
        support_manifest_sha256=support.manifest_sha256,
        loading_audit=loading_audit,
        loading_audit_sha256=_sha256(loading_audit),
        support_root=support.support_root,
        node_order=support.node_order,
        shards={0: (task,)},
        results_root=results_root,
        control_root=control_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
        microbatch_sizes={},
    )
    plan_path = tmp_path / "nas_backup_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = pf.build_postflight_report(plan_path)

    assert report["runner_kind"] == "event_convergence"
    assert report["validated_tasks"] == 1
