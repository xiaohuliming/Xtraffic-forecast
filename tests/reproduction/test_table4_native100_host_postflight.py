from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    load_native100_compare_manifest,
)
from scripts.reproduction.igstgnn_table4.native100_host_postflight import (
    build_native100_host_postflight,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "reproduction" / "igstgnn_table4"
MANIFEST = CONFIG / "matrix.native100_compare_v1.json"
BASE = CONFIG / "matrix.public_material_v2.json"
PLAN = CONFIG / "distributed_native100_5080_5090_v1.json"
MATRIX_SHA = "e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246"
BASE_SHA = "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
PLAN_SHA = "d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e"


def _write_task(root: Path, task: dict, *, gpu: str) -> None:
    checkpoint = root.parent / "checkpoint.pt"
    log = root.parent / "train.log"
    checkpoint.write_bytes(b"checkpoint\n")
    log.write_text("complete\n", encoding="utf-8")
    labels = np.full((1, 12, 2), 10.0, dtype=np.float32)
    write_completed_task_artifacts(
        target_dir=root / task["output_dir"],
        task=task,
        predictions=labels + np.float32(1.0),
        labels=labels,
        checkpoint_source=checkpoint,
        train_log_source=log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "matrix_id": "igstgnn_table4_bist_ndr_native100_18_v1",
            "claim_scope": "same_table_native_recipe_100_epoch_supplement",
            "training_code_sha256": "3" * 64,
            "host": {
                "hostname": "test-host",
                "system": "Windows",
                "release": "11",
                "machine": "AMD64",
            },
            "runtime": {"python": "3.13", "torch": "2.10", "numpy": "2.2"},
            "device": {"type": "cuda", "name": gpu},
        },
        claim_scope="same_table_native_recipe_100_epoch_supplement",
    )


def test_native100_5080_postflight_exports_only_declared_tasks(tmp_path: Path) -> None:
    manifest = load_native100_compare_manifest(
        MANIFEST,
        expected_sha256=MATRIX_SHA,
        base_manifest_path=BASE,
        expected_base_manifest_sha256=BASE_SHA,
    )
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    host = plan["hosts"][0]
    results = tmp_path / "results"
    (results / "tasks").mkdir(parents=True)
    for task_id in [*host["task_ids"], *host["excluded_existing_task_ids"]]:
        _write_task(results, task_by_id[task_id], gpu=host["gpu_sku"])
    queue = {
        "schema_version": 1,
        "state": "completed",
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "manifest_path": str(MANIFEST),
        "manifest_sha256": MATRIX_SHA,
        "base_manifest_path": str(BASE),
        "base_manifest_sha256": BASE_SHA,
        "splits_root": "splits",
        "results_root": str(results),
        "device": "cuda:0",
        "selected_task_ids": host["task_ids"],
        "current": None,
        "completed": host["task_ids"],
        "skipped": [],
        "failed": [],
        "command": None,
        "started_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-01T01:00:00Z",
        "finished_at": "2026-08-01T01:00:00Z",
        "selected_task_count": 12,
        "completed_count": 12,
        "skipped_count": 0,
        "failed_count": 0,
        "verified_task_count": 12,
    }
    (results / "queue_status.json").write_text(
        json.dumps(queue), encoding="utf-8"
    )

    report = build_native100_host_postflight(
        plan_path=PLAN,
        expected_plan_sha256=PLAN_SHA,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MATRIX_SHA,
        base_manifest_path=BASE,
        expected_base_manifest_sha256=BASE_SHA,
        host_id=host["host_id"],
        results_root=results,
        report_path=tmp_path / "postflight.json",
        file_list_path=tmp_path / "transfer.files",
        transfer_manifest_path=tmp_path / "transfer.sha256",
        marker_path=tmp_path / "marker.json",
    )

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 12
    assert report["excluded_tasks_validated"] == 3
    assert report["file_count"] == 61
    exported = (tmp_path / "transfer.files").read_text().splitlines()
    assert len(exported) == 61
    assert not any("n100007" in path for path in exported)
    marker = json.loads((tmp_path / "marker.json").read_text())
    assert marker["files_digest"] == hashlib.sha256(
        (tmp_path / "transfer.sha256").read_bytes()
    ).hexdigest()
