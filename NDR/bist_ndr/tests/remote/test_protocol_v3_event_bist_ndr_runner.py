from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_BUNDLE_ID = "bundle-test"
CACHE_BUNDLE_DIGEST = "a" * 64
CODE_SHA = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bist_ndr_runner_bootstraps_from_its_script_path() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/remote/windows/protocol_v3_event_bist_ndr_runner.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert "--cache-bundle-id" in completed.stdout


def test_bist_ndr_manifest_is_the_exact_three_region_seed42_matrix() -> None:
    from scripts.remote.windows.protocol_v3_event_bist_ndr_runner import (
        CANONICAL_MANIFEST,
        load_bist_ndr_manifest,
    )

    manifest = load_bist_ndr_manifest(CANONICAL_MANIFEST)

    assert len(manifest.tasks) == 3
    assert {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    } == {
        (region, "bist_ndr", 42, "event")
        for region in ("Alameda", "ContraCosta", "Orange")
    }


def test_bist_ndr_invocation_binds_profile_and_orange_microbatch(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows.protocol_v3_event_bist_ndr_runner import (
        CANONICAL_MANIFEST,
        build_bist_ndr_invocation,
        load_bist_ndr_manifest,
    )
    from scripts.remote.windows.protocol_v3_event_extension_runner import (
        ExtensionInputs,
    )

    manifest = load_bist_ndr_manifest(CANONICAL_MANIFEST)
    task = next(task for task in manifest.tasks if task.region == "Orange")
    invocation = build_bist_ndr_invocation(
        task,
        manifest,
        ExtensionInputs(
            support_root=tmp_path / "support",
            node_order=tmp_path / "node_order.npy",
            support_manifest_sha256="a" * 64,
            igstgnn_data_parallel=False,
            max_microbatch_size=0,
        ),
        tmp_path / "attempt",
    )

    command = list(invocation.command)
    assert command[command.index("--method") + 1] == "bist_ndr"
    assert command[command.index("--variant") + 1] == "v3_event_bist_ndr"
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert command[command.index("--run-contract") + 1] == "formal30"
    assert invocation.config["early_stopping_patience"] is None
    assert invocation.config["method_profile_sha256"] == (
        "b32340f8d4f94be2318dd1e0b8ddbcefce105005bc6bf7c5cc0548bcb9bc5ba8"
    )
    assert invocation.config["max_microbatch_size"] == 8
    assert invocation.required_artifacts == (
        "summary.json",
        "ckpt_best.pt",
        "test_predictions.npz",
        "train.log",
    )


def test_bist_ndr_results_pass_the_shared_read_only_postflight(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf
    from scripts.remote.windows.protocol_v3_event_bist_ndr_runner import (
        CANONICAL_MANIFEST,
        build_bist_ndr_invocation,
        load_bist_ndr_manifest,
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

    manifest = load_bist_ndr_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            PROJECT_ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=PROJECT_ROOT / "archive/node_order.npy",
    )
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    results_root = tmp_path / "results"
    control_root = tmp_path / "control"
    for path in (cache_dir, graph_dir, results_root, control_root):
        path.mkdir(parents=True)
    extension = ExtensionInputs(
        support_root=support.support_root,
        node_order=support.node_order,
        support_manifest_sha256=support.manifest_sha256,
        igstgnn_data_parallel=False,
        max_microbatch_size=0,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        python_executable=sys.executable,
    )
    protocol = ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
    )
    for task in manifest.tasks:
        invocation = build_bist_ndr_invocation(
            task,
            manifest,
            extension,
            results_root / ".staging" / task.task_id / "__TEST__",
        )
        config_hash = derive_config_hash(
            task,
            manifest,
            protocol,
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
        (result_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
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
                {
                    "state": "completed",
                    "task_id": task.task_id,
                    "config_hash": config_hash,
                }
            ),
            encoding="utf-8",
        )
    queue = {
        "state": "completed",
        "total_tasks": 3,
        "completed_tasks": 3,
        "skipped_valid_tasks": 0,
        "failed_count": 0,
        "current_task": None,
    }
    (results_root / "queue_status.json").write_text(
        json.dumps(queue), encoding="utf-8"
    )
    exit_path = control_root / "runner.exitcode"
    exit_path.write_text("0\n", encoding="ascii")
    loading_audit = control_root / "loading_equivalence.json"
    loading_audit.write_text(
        '{"status":"EAGER_LAZY_EQUIVALENCE_OK"}\n', encoding="utf-8"
    )
    plan = {
        "plan_version": 1,
        "runner_kind": "event_bist_ndr",
        "matrix_id": "protocol_v3_event_bist_ndr_test",
        "host_id": "event_rtx5080_bist_ndr_test",
        "canonical_manifest": str(CANONICAL_MANIFEST),
        "canonical_manifest_sha256": _sha256(CANONICAL_MANIFEST),
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "expected_task_count": 3,
        "cache_bundle_id": CACHE_BUNDLE_ID,
        "cache_bundle_digest": CACHE_BUNDLE_DIGEST,
        "cache_dir": str(cache_dir),
        "graph_dir": str(graph_dir),
        "python_executable": sys.executable,
        "support_manifest": str(SUPPORT_MANIFEST),
        "support_manifest_sha256": support.manifest_sha256,
        "loading_audit": str(loading_audit),
        "loading_audit_sha256": _sha256(loading_audit),
        "support_root": str(support.support_root),
        "node_order": str(support.node_order),
        "extension_options": {
            "igstgnn_data_parallel": False,
            "max_microbatch_size": 0,
            "task_microbatch_sizes": {
                "v3bn003_event_bist_ndr_orange_seed42": 8
            },
        },
        "source_layout": "canonical_contiguous",
        "sources": [
            {
                "source_id": "gpu00",
                "results_root": str(results_root),
                "code_sha": CODE_SHA,
                "task_ids": [task.task_id for task in manifest.tasks],
            }
        ],
        "completion_gate": {
            "results_root": str(results_root),
            "expected_tasks": 3,
            "runner_exit_code": str(exit_path),
        },
    }
    plan_path = tmp_path / "postflight_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    report = pf.build_postflight_report(plan_path)

    assert report["status"] == "verified"
    assert report["runner_kind"] == "event_bist_ndr"
    assert report["validated_tasks"] == 3
    assert report["task_state_counts"] == {"completed": 3}
