from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    load_native100_compare_manifest,
)
from scripts.reproduction.igstgnn_table4.native100_composite_postflight import (
    build_native100_composite_index,
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


def _write_task(root: Path, task: dict, *, gpu: str, hostname: str) -> None:
    checkpoint = root.parent / f"{hostname}.pt"
    log = root.parent / f"{hostname}.log"
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
            "task_id": task["task_id"],
            "method": task["method"],
            "region": task["region"],
            "seed": task["seed"],
            "config_sha256": task["config_sha256"],
            "split_bundle_manifest_sha256": task[
                "split_bundle_manifest_sha256"
            ],
            "training_code_sha256": "3" * 64,
            "host": {
                "hostname": hostname,
                "system": "Windows" if "5080" in hostname else "Linux",
                "release": "11",
                "machine": "AMD64",
            },
            "runtime": {"python": "3.13", "torch": "2.10", "numpy": "2.2"},
            "device": {"type": "cuda", "name": gpu},
        },
        claim_scope="same_table_native_recipe_100_epoch_supplement",
    )


def _write_queue(results: Path, manifest: dict, task_ids: list[str]) -> None:
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
        "selected_task_ids": task_ids,
        "current": None,
        "completed": task_ids,
        "skipped": [],
        "failed": [],
        "command": None,
        "started_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-01T01:00:00Z",
        "finished_at": "2026-08-01T01:00:00Z",
        "selected_task_count": len(task_ids),
        "completed_count": len(task_ids),
        "skipped_count": 0,
        "failed_count": 0,
        "verified_task_count": len(task_ids),
    }
    (results / "queue_status.json").write_text(json.dumps(queue), encoding="utf-8")


def test_native100_composite_validates_two_exact_published_sources(tmp_path: Path) -> None:
    manifest = load_native100_compare_manifest(
        MANIFEST,
        expected_sha256=MATRIX_SHA,
        base_manifest_path=BASE,
        expected_base_manifest_sha256=BASE_SHA,
    )
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    hosts_base = tmp_path / "hosts"
    for host in plan["hosts"]:
        working = tmp_path / f"working-{host['host_id']}"
        (working / "tasks").mkdir(parents=True)
        all_ids = [*host["task_ids"], *host["excluded_existing_task_ids"]]
        for task_id in all_ids:
            _write_task(
                working,
                task_by_id[task_id],
                gpu=host["gpu_sku"],
                hostname=host["host_id"],
            )
        _write_queue(working, manifest, host["task_ids"])
        control = tmp_path / f"control-{host['host_id']}"
        report = build_native100_host_postflight(
            plan_path=PLAN,
            expected_plan_sha256=PLAN_SHA,
            manifest_path=MANIFEST,
            expected_manifest_sha256=MATRIX_SHA,
            base_manifest_path=BASE,
            expected_base_manifest_sha256=BASE_SHA,
            host_id=host["host_id"],
            results_root=working,
            report_path=control / "postflight.json",
            file_list_path=control / "transfer.files",
            transfer_manifest_path=control / "transfer.sha256",
            marker_path=control / "_TRANSFER_COMPLETE.json",
        )
        published = hosts_base / host["host_id"]
        for record in report["files"]:
            relative = PurePosixPath(record["path"])
            target = published / "results" / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(working.joinpath(*relative.parts), target)
        metadata = published / "metadata"
        metadata.mkdir(parents=True)
        for source, name in (
            (control / "postflight.json", "postflight.json"),
            (control / "transfer.files", "transfer.files"),
            (control / "transfer.sha256", "transfer.sha256"),
            (control / "_TRANSFER_COMPLETE.json", "_TRANSFER_COMPLETE.json"),
            (PLAN, "distributed_native100_5080_5090_v1.json"),
            (MANIFEST, "matrix.native100_compare_v1.json"),
            (BASE, "matrix.public_material_v2.json"),
        ):
            shutil.copy2(source, metadata / name)

    index = build_native100_composite_index(
        hosts_base=hosts_base,
        plan_path=PLAN,
        expected_plan_sha256=PLAN_SHA,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MATRIX_SHA,
        base_manifest_path=BASE,
        expected_base_manifest_sha256=BASE_SHA,
        output_path=tmp_path / "composite_index.json",
    )

    assert index["status"] == "verified_native100_dual_host_18"
    assert index["source_count"] == 2
    assert index["task_count"] == 18
    assert index["file_count"] == 92
