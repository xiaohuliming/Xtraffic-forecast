from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.reproduction.igstgnn_table4.ndr_extension_report import (
    build_ndr_extension_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
)
BASE_MANIFEST = CONFIG_ROOT / "matrix.public_material_v2.json"
EXTENSION_MANIFEST = CONFIG_ROOT / "matrix.ndr_extension_v1.json"
PROFILE = CONFIG_ROOT / "ndr_extension_profile_v1.json"
PLAN = CONFIG_ROOT / "distributed_ndr_extension_4gpu_rebalance_v2.json"
ARTIFACT_NAMES = (
    "checkpoint.pt",
    "predictions.npz",
    "provenance.json",
    "summary.json",
    "train.log",
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_ndr_snapshot(tmp_path: Path) -> tuple[Path, str]:
    snapshot = tmp_path / "snapshot"
    metadata = snapshot / "metadata"
    results = snapshot / "results"
    metadata.mkdir(parents=True)
    results.mkdir()
    for source in (BASE_MANIFEST, EXTENSION_MANIFEST, PROFILE, PLAN):
        (metadata / source.name).write_bytes(source.read_bytes())
    manifest = json.loads(EXTENSION_MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    shard_by_task = {
        task_id: shard["results_root"]
        for shard in plan["shards"]
        for task_id in shard["task_ids"]
    }
    records: list[dict[str, Any]] = []
    for shard in plan["shards"]:
        relative = f"{shard['results_root']}/queue_status.json"
        payload = f"{shard['shard_id']} complete\n".encode()
        records.append(
            {
                "path": relative,
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    for task_number, task in enumerate(manifest["tasks"], start=1):
        task_root_relative = (
            f"{shard_by_task[task['task_id']]}/tasks/{task['task_id']}"
        )
        task_root = results / task_root_relative
        task_root.mkdir(parents=True)
        metrics = {
            horizon: {
                metric: float(task_number) + offset
                for metric, offset in (
                    ("mae", 0.01),
                    ("rmse", 0.02),
                    ("mape_percent", 0.03),
                )
            }
            for horizon in ("h3", "h6", "h12", "average")
        }
        summary_payload = _json_bytes(
            {
                "schema_version": 1,
                "status": "completed",
                "protocol_id": "igstgnn_table4_official_v1",
                "claim_scope": "same_protocol_project_method_extension",
                "validity_label": "official_protocol_with_known_overlap",
                "task": task,
                "artifact_allowlist": list(ARTIFACT_NAMES),
                "metrics": metrics,
                "valid_count": 12,
                "prediction_shape": [1, 12, 1],
                "prediction_dtype": "float32",
                "label_dtype": "float32",
            }
        )
        provenance_payload = _json_bytes(
            {
                "schema_version": 1,
                "protocol_id": "igstgnn_table4_official_v1",
                "matrix_id": "igstgnn_table4_ndr_extension_9_v1",
                "matrix_manifest_sha256": hashlib.sha256(
                    EXTENSION_MANIFEST.read_bytes()
                ).hexdigest(),
                "base_manifest_sha256": hashlib.sha256(
                    BASE_MANIFEST.read_bytes()
                ).hexdigest(),
                "claim_scope": "same_protocol_project_method_extension",
                "validity_label": "official_protocol_with_known_overlap",
                "task_id": task["task_id"],
                "method": task["method"],
                "region": task["region"],
                "seed": task["seed"],
                "config_sha256": task["config_sha256"],
                "split_bundle_manifest_sha256": task[
                    "split_bundle_manifest_sha256"
                ],
                "training_code_sha256": "1" * 64,
            }
        )
        payload_by_name = {
            "checkpoint.pt": b"checkpoint\n",
            "predictions.npz": b"predictions\n",
            "provenance.json": provenance_payload,
            "summary.json": summary_payload,
            "train.log": b"complete\n",
        }
        for name in ARTIFACT_NAMES:
            payload = payload_by_name[name]
            relative = f"{task_root_relative}/{name}"
            records.append(
                {
                    "path": relative,
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                }
            )
            if name in {"summary.json", "provenance.json"}:
                (task_root / name).write_bytes(payload)
    records.sort(key=lambda record: record["path"])
    export_paths = [record["path"] for record in records]
    files_digest = _sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    postflight = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_ndr_extension_9_v1",
        "plan_id": "igstgnn_table4_ndr_extension_4gpu_rebalance_v2",
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            EXTENSION_MANIFEST.read_bytes()
        ).hexdigest(),
        "base_manifest_sha256": hashlib.sha256(
            BASE_MANIFEST.read_bytes()
        ).hexdigest(),
        "host_id": "table4_4090d4",
        "gpu_sku": "NVIDIA GeForce RTX 4090 D",
        "execution_layout": "single_host_gpu_sku_homogeneous_restart",
        "validity_label": "official_protocol_with_known_overlap",
        "claim_scope": "same_protocol_project_method_extension",
        "validated_shards": 4,
        "validated_tasks": 9,
        "validated_task_ids": [
            task_id
            for shard in plan["shards"]
            for task_id in shard["task_ids"]
        ],
        "file_count": 49,
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files_digest": files_digest,
        "export_paths": export_paths,
        "files": records,
    }
    postflight_payload = _json_bytes(postflight)
    (metadata / "postflight.json").write_bytes(postflight_payload)
    file_list_payload = "".join(
        f"{relative}\n" for relative in export_paths
    ).encode()
    transfer_manifest_payload = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    ).encode()
    (metadata / "transfer.files").write_bytes(file_list_payload)
    (metadata / "transfer.sha256").write_bytes(transfer_manifest_payload)
    marker = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": postflight["protocol_id"],
        "matrix_id": postflight["matrix_id"],
        "plan_id": postflight["plan_id"],
        "plan_sha256": postflight["plan_sha256"],
        "manifest_sha256": postflight["manifest_sha256"],
        "base_manifest_sha256": postflight["base_manifest_sha256"],
        "host_id": postflight["host_id"],
        "claim_scope": postflight["claim_scope"],
        "validated_tasks": 9,
        "file_count": 49,
        "total_bytes": postflight["total_bytes"],
        "files_digest": files_digest,
        "file_list_sha256": _sha256(file_list_payload),
        "transfer_manifest_sha256": _sha256(transfer_manifest_payload),
        "postflight_sha256": _sha256(postflight_payload),
    }
    marker_payload = _json_bytes(marker)
    (metadata / "transfer_complete.json").write_bytes(marker_payload)
    sha_log_payload = "".join(
        f"{relative}: OK\n" for relative in export_paths
    ).encode()
    (snapshot / "ndr_sha256_check.log").write_bytes(sha_log_payload)
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "formal_root": "/fixture",
        "marker_sha256": _sha256(marker_payload),
        "postflight_sha256": _sha256(postflight_payload),
        "file_list_sha256": _sha256(file_list_payload),
        "transfer_manifest_sha256": _sha256(transfer_manifest_payload),
        "sha256_check_log_sha256": _sha256(sha_log_payload),
        "actual_file_list_sha256": _sha256(file_list_payload),
        "file_count": 49,
        "total_bytes": postflight["total_bytes"],
        "files_digest": files_digest,
        "validated_tasks": 9,
        "task_ids": postflight["validated_task_ids"],
        "verified_at": "2026-07-30T00:00:00Z",
    }
    receipt_payload = _json_bytes(receipt)
    (snapshot / "ndr_extension_remote_verify.json").write_bytes(
        receipt_payload
    )
    return snapshot, _sha256(receipt_payload)


def test_ndr_report_verifies_nine_tasks_and_builds_three_seed_rows(
    tmp_path: Path,
) -> None:
    snapshot, receipt_sha256 = _write_ndr_snapshot(tmp_path)
    output_dir = tmp_path / "ndr_report"

    report = build_ndr_extension_report(
        snapshot_root=snapshot,
        expected_receipt_sha256=receipt_sha256,
        output_dir=output_dir,
    )

    assert report["status"] == "verified_ndr_extension_report"
    assert report["task_count"] == 9
    assert report["region_count"] == 3
    assert report["seed_count"] == 3
    assert report["region_summaries"][0]["region"] == "Alameda"
    assert report["region_summaries"][0]["metrics"]["average"]["mae"][
        "mean"
    ] == 2.01
    assert (output_dir / "ndr_extension_index.json").is_file()
    assert (output_dir / "ndr_seed_results.csv").is_file()
    assert (output_dir / "ndr_region_summary.csv").is_file()
    assert (output_dir / "ndr_extension.md").is_file()
