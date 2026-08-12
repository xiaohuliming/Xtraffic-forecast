from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from scripts.remote.windows.protocol_v3_runner import ProtocolV3Inputs


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "config/protocol_v3/table4_event_fdn_v1.json"
)
CACHE_BUNDLE_ID = "bundle-v3-test"
CACHE_BUNDLE_DIGEST = "a" * 64
CODE_SHA = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_event_fdn_results_pass_the_shared_read_only_postflight(
    tmp_path: Path,
) -> None:
    from scripts.remote.windows import protocol_v3_event_host_postflight as pf
    from scripts.remote.windows.protocol_v3_event_fdn_runner import (
        build_fdn_invocation,
        load_fdn_manifest,
    )
    from scripts.remote.windows.protocol_v3_runner import derive_config_hash

    manifest = load_fdn_manifest(MANIFEST_PATH)
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    results_root = tmp_path / "results"
    control_root = tmp_path / "control"
    for path in (cache_dir, graph_dir, results_root, control_root):
        path.mkdir(parents=True)
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
        invocation = build_fdn_invocation(
            task,
            manifest,
            protocol,
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
            "affected": 1.0,
            "unaffected": 1.0,
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
    (results_root / "queue_status.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "total_tasks": 3,
                "completed_tasks": 3,
                "skipped_valid_tasks": 0,
                "failed_count": 0,
                "current_task": None,
            }
        ),
        encoding="utf-8",
    )
    exit_path = control_root / "runner.exitcode"
    exit_path.write_text("0\n", encoding="ascii")
    plan = {
        "plan_version": 1,
        "runner_kind": "event_fdn",
        "matrix_id": "protocol_v3_event_fdn_test",
        "host_id": "event_rtx5080_fdn_test",
        "canonical_manifest": str(MANIFEST_PATH),
        "canonical_manifest_sha256": _sha256(MANIFEST_PATH),
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "expected_task_count": 3,
        "cache_bundle_id": CACHE_BUNDLE_ID,
        "cache_bundle_digest": CACHE_BUNDLE_DIGEST,
        "cache_dir": str(cache_dir),
        "graph_dir": str(graph_dir),
        "python_executable": sys.executable,
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
    assert report["runner_kind"] == "event_fdn"
    assert report["validated_tasks"] == 3
    assert report["task_state_counts"] == {"completed": 3}
