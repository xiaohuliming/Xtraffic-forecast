from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4.distributed_host_postflight import (
    build_host_postflight,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
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
    / "distributed_4090d_9gpu_v1.json"
)
MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357e"
    "b5320393417a83d344adb05151023a78"
)


def _completed_host_results(
    tmp_path: Path,
    host_id: str,
    *,
    mixed_hostname: bool = False,
) -> Path:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    results_base = tmp_path / "results"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    train_log = tmp_path / "train.log"
    train_log.write_text("complete\n", encoding="utf-8")
    labels = np.arange(12, dtype=np.float32).reshape(1, 12, 1)
    predictions = labels + np.float32(0.5)
    for shard in plan["shards"]:
        if shard["host_id"] != host_id:
            continue
        shard_root = results_base / shard["results_root"]
        for task_index, task_id in enumerate(shard["task_ids"]):
            task = task_by_id[task_id]
            training_code_sha256 = (
                "4" * 64
                if task["method"] == "bist"
                else "5" * 64
                if task["method"] == "igstgnn"
                else "3" * 64
            )
            write_completed_task_artifacts(
                target_dir=shard_root / task["output_dir"],
                task=task,
                predictions=predictions,
                labels=labels,
                checkpoint_source=checkpoint,
                train_log_source=train_log,
                provenance={
                    "protocol_id": "igstgnn_table4_official_v1",
                    "training_code_sha256": training_code_sha256,
                    "device": {
                        "requested": "cuda:0",
                        "type": "cuda",
                        "resolved_index": 0,
                        "name": "NVIDIA GeForce RTX 4090 D",
                    },
                    "host": {
                        "hostname": (
                            "unexpected-host"
                            if mixed_hostname
                            and shard["gpu_index"] == 0
                            and task_index == 0
                            else host_id
                        ),
                        "system": "Linux",
                        "release": "test",
                        "machine": "x86_64",
                    },
                    "runtime": {
                        "python": "3.11.15",
                        "torch": "2.3.1+cu121",
                        "numpy": "1.24.4",
                    },
                },
            )
        now = "2026-07-28T12:00:00.000000Z"
        completed = [
            {
                "task_id": task_id,
                "command": ["python", "-m", "task_runner"],
                "started_at": now,
                "finished_at": now,
            }
            for task_id in shard["task_ids"]
        ]
        status = {
            "schema_version": 1,
            "state": "completed",
            "protocol_id": "igstgnn_table4_official_v1",
            "matrix_id": "igstgnn_table4_public_repro_39_v2",
            "manifest_path": str(MANIFEST),
            "manifest_sha256": MANIFEST_SHA256,
            "splits_root": "/data/splits",
            "results_root": str(shard_root),
            "device": "cuda:0",
            "selected_task_ids": shard["task_ids"],
            "current": None,
            "completed": completed,
            "skipped": [],
            "failed": [],
            "command": ["python", "-m", "task_runner"],
            "started_at": now,
            "updated_at": now,
            "finished_at": now,
            "selected_task_count": len(shard["task_ids"]),
            "completed_count": len(shard["task_ids"]),
            "skipped_count": 0,
            "failed_count": 0,
            "verified_task_count": len(shard["task_ids"]),
        }
        (shard_root / "queue_status.json").write_text(
            json.dumps(status),
            encoding="utf-8",
        )
    return results_base


def test_host_postflight_validates_every_queue_task_and_export_file(tmp_path):
    results_base = _completed_host_results(tmp_path, "table4_4090d5")
    report_path = tmp_path / "postflight.json"

    report = build_host_postflight(
        plan_path=PLAN,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
        host_id="table4_4090d5",
        results_base=results_base,
        report_path=report_path,
    )

    assert report["status"] == "verified"
    assert report["host_id"] == "table4_4090d5"
    assert report["validated_shards"] == 5
    assert report["validated_tasks"] == 13
    assert report["file_count"] == 70
    assert report["export_paths"] == sorted(report["export_paths"])
    assert report_path.is_file()


def test_host_postflight_rejects_mixed_runtime_host_identity(tmp_path):
    results_base = _completed_host_results(
        tmp_path,
        "table4_4090d5",
        mixed_hostname=True,
    )

    with pytest.raises(ValueError, match="runtime host identity"):
        build_host_postflight(
            plan_path=PLAN,
            manifest_path=MANIFEST,
            expected_manifest_sha256=MANIFEST_SHA256,
            host_id="table4_4090d5",
            results_base=results_base,
            report_path=tmp_path / "postflight.json",
        )
