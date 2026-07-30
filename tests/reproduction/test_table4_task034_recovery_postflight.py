from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
)
from scripts.reproduction.igstgnn_table4.task034_recovery_postflight import (
    build_task034_recovery_postflight,
    load_task034_recovery_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.public_material_v2.json"
)
PLAN = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "task034_4gpu_recovery_v1.json"
)
CONTROLLER = (
    PROJECT_ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "igstgnn_table4_recover_task034_4090d4.sh"
)
MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357e"
    "b5320393417a83d344adb05151023a78"
)


def _write_completed_recovery(tmp_path: Path) -> tuple[Path, Path]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    task = next(
        task
        for task in manifest["tasks"]
        if task["task_id"] == "034_orange_sttn"
    )
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    results_base = tmp_path / "results"
    task_dir = results_base / plan["results_root"] / task["output_dir"]
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    train_log = tmp_path / "train.log"
    train_log.write_text("complete\n", encoding="utf-8")
    labels = np.arange(12, dtype=np.float32).reshape(1, 12, 1)
    predictions = labels + np.float32(0.5)
    write_completed_task_artifacts(
        target_dir=task_dir,
        task=task,
        predictions=predictions,
        labels=labels,
        checkpoint_source=checkpoint,
        train_log_source=train_log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "training_code_sha256": "3" * 64,
            "host": {
                "hostname": plan["recovery_hostname"],
                "system": "Linux",
                "release": "test",
                "machine": "x86_64",
            },
            "device": {
                "requested": "cuda:0",
                "type": "cuda",
                "resolved_index": 0,
                "name": plan["gpu_sku"],
            },
            "runtime": {
                "python": "3.11.15",
                "torch": "2.3.1+cu121",
                "numpy": "1.24.4",
            },
            "execution_adapter": {
                "name": "replica_device_full_batch_v1",
                "device_ids": [0, 1, 2, 3],
                "full_batch_preserved": True,
                "optimizer_step_contract_preserved": True,
                "controller_sha256": "4" * 64,
            },
            "recovery_execution": {
                "adapter": "multi-gpu-full-batch",
                "controller_sha256": "4" * 64,
                "base_training_code_sha256": "3" * 64,
            },
        },
    )
    inventory = {
        "schema_version": 1,
        "status": "verified",
        "host_id": plan["recovery_host_id"],
        "hostname": plan["recovery_hostname"],
        "gpu_sku": plan["gpu_sku"],
        "gpu_count": plan["gpu_count"],
        "gpu_uuids": plan["gpu_uuids"],
        "captured_at": "2026-07-30T12:00:00Z",
    }
    inventory_path = tmp_path / "hardware_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results_base, inventory_path


def test_task034_recovery_plan_locks_one_cross_host_task() -> None:
    plan = load_task034_recovery_plan(
        PLAN,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    assert plan["task_id"] == "034_orange_sttn"
    assert plan["adapter"] == "multi-gpu-full-batch"
    assert plan["gpu_count"] == 4
    assert len(plan["gpu_uuids"]) == 4
    assert plan["source_host_id"] == "table4_4090d5"
    assert plan["recovery_host_id"] == "table4_4090d4"
    assert plan["execution_layout"] == "cross_host_single_task_recovery"


def test_task034_postflight_revalidates_exact_five_file_export(tmp_path) -> None:
    results_base, inventory_path = _write_completed_recovery(tmp_path)
    report_path = tmp_path / "postflight.json"

    report = build_task034_recovery_postflight(
        plan_path=PLAN,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
        results_base=results_base,
        hardware_inventory_path=inventory_path,
        report_path=report_path,
    )

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 1
    assert report["file_count"] == 5
    assert report["task_id"] == "034_orange_sttn"
    assert report["gpu_uuids"] == json.loads(
        PLAN.read_text(encoding="utf-8")
    )["gpu_uuids"]
    assert report_path.is_file()


def test_task034_postflight_rejects_changed_gpu_identity(tmp_path) -> None:
    results_base, inventory_path = _write_completed_recovery(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["gpu_uuids"][0] = "GPU-unexpected"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="hardware inventory"):
        build_task034_recovery_postflight(
            plan_path=PLAN,
            manifest_path=MANIFEST,
            expected_manifest_sha256=MANIFEST_SHA256,
            results_base=results_base,
            hardware_inventory_path=inventory_path,
            report_path=tmp_path / "postflight.json",
        )


def test_task034_controller_is_fail_closed_and_cannot_expand_the_queue() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert "TASK_ID=034_orange_sttn" in source
    assert "TASK_ID=${" not in source
    assert (
        "PLAN_SHA256=022d3d9a0c635f94ce5f7b7760c022ba"
        "38162bbccfe68675b33cd6bbe78834c5"
    ) in source
    assert source.index("--smoke") < source.index("--results-root")
    assert 'EXPECTED_PLAN_ID="$PLAN_ID"' in source
    assert (
        'EXPECTED_EXECUTION_LAYOUT="cross_host_single_task_recovery"'
        in source
    )
    assert 'SHUTDOWN_BIN="/usr/bin/shutdown"' in source
    assert "bash \"$FINALIZER\"" in source
    assert '"$SHUTDOWN_BIN" -h now' not in source
