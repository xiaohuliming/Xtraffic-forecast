from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.reproduction.igstgnn_table4.composite_postflight import (
    CompositeValidationError,
    build_table4_composite_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.public_material_v2.json"
)
MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357e"
    "b5320393417a83d344adb05151023a78"
)
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
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_source(
    relay_base: Path,
    *,
    source_id: str,
    tasks: list[dict[str, Any]],
    host_id: str,
    plan_id: str,
    execution_layout: str,
    task_prefix: str,
    task034_shape: bool = False,
) -> dict[str, Any]:
    root_relative = f"published/{source_id}"
    root = relay_base / root_relative
    results_root = root / "results"
    metadata_root = root / "metadata"
    results_root.mkdir(parents=True)
    metadata_root.mkdir()
    records: list[dict[str, Any]] = []
    for task in tasks:
        task_root = results_root / task_prefix / task["task_id"]
        task_root.mkdir(parents=True)
        payloads = {
            "checkpoint.pt": b"checkpoint\n",
            "predictions.npz": b"predictions\n",
            "provenance.json": _json_bytes(
                {
                    "protocol_id": "igstgnn_table4_official_v1",
                    "task_id": task["task_id"],
                    "method": task["method"],
                    "region": task["region"],
                    "seed": task["seed"],
                    "config_sha256": task["config_sha256"],
                    "split_bundle_manifest_sha256": task[
                        "split_bundle_manifest_sha256"
                    ],
                    "matrix_manifest_sha256": MANIFEST_SHA256,
                    "training_code_sha256": "1" * 64,
                }
            ),
            "summary.json": _json_bytes(
                {
                    "schema_version": 1,
                    "status": "completed",
                    "protocol_id": "igstgnn_table4_official_v1",
                    "claim_scope": "public_material_maximum_fidelity_reproduction",
                    "validity_label": "official_protocol_with_known_overlap",
                    "task": task,
                    "artifact_allowlist": list(ARTIFACT_NAMES),
                    "metrics": {
                        "h3": {"mae": 1.0, "rmse": 2.0, "mape_percent": 3.0},
                        "h6": {"mae": 1.1, "rmse": 2.1, "mape_percent": 3.1},
                        "h12": {"mae": 1.2, "rmse": 2.2, "mape_percent": 3.2},
                        "average": {
                            "mae": 1.1,
                            "rmse": 2.1,
                            "mape_percent": 3.1,
                        },
                    },
                    "valid_count": 12,
                    "prediction_shape": [1, 12, 1],
                    "prediction_dtype": "float32",
                    "label_dtype": "float32",
                }
            ),
            "train.log": b"complete\n",
        }
        for name in ARTIFACT_NAMES:
            payload = payloads[name]
            path = task_root / name
            path.write_bytes(payload)
            relative = path.relative_to(results_root).as_posix()
            records.append(
                {
                    "path": relative,
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                }
            )
    records.sort(key=lambda record: record["path"])
    export_paths = [record["path"] for record in records]
    files_digest = _sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    plan_sha256 = _sha256(f"{plan_id}\n".encode())
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_public_repro_39_v2",
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "manifest_sha256": MANIFEST_SHA256,
        "host_id": host_id,
        "gpu_sku": "NVIDIA GeForce RTX 4090 D",
        "execution_layout": execution_layout,
        "validated_shards": 1,
        "validated_tasks": len(tasks),
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files_digest": files_digest,
        "files": records,
        "export_paths": export_paths,
    }
    if task034_shape:
        report["task_id"] = tasks[0]["task_id"]
    else:
        report["tasks"] = [
            {
                "task_id": task["task_id"],
                "status": "completed_task_artifacts_verified",
            }
            for task in tasks
        ]
    postflight_payload = _json_bytes(report)
    (metadata_root / "postflight.json").write_bytes(postflight_payload)
    transfer_files_payload = "".join(
        f"{relative}\n" for relative in export_paths
    ).encode()
    transfer_manifest_payload = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    ).encode()
    (metadata_root / "transfer.files").write_bytes(transfer_files_payload)
    (metadata_root / "transfer.sha256").write_bytes(transfer_manifest_payload)
    marker = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": report["protocol_id"],
        "matrix_id": report["matrix_id"],
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "manifest_sha256": MANIFEST_SHA256,
        "host_id": host_id,
        "gpu_sku": report["gpu_sku"],
        "execution_layout": execution_layout,
        "validated_shards": 1,
        "validated_tasks": len(tasks),
        "file_count": len(records),
        "source_bytes": report["total_bytes"],
        "files_digest": files_digest,
        "file_list_sha256": _sha256(transfer_files_payload),
        "transfer_manifest_sha256": _sha256(transfer_manifest_payload),
        "postflight_sha256": _sha256(postflight_payload),
    }
    marker_payload = _json_bytes(marker)
    (root / "_TRANSFER_COMPLETE.json").write_bytes(marker_payload)
    return {
        "source_id": source_id,
        "root_relative": root_relative,
        "host_id": host_id,
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "execution_layout": execution_layout,
        "marker_sha256": _sha256(marker_payload),
        "postflight_sha256": marker["postflight_sha256"],
        "validated_shards": 1,
        "validated_tasks": len(tasks),
        "file_count": len(records),
        "source_bytes": report["total_bytes"],
        "files_digest": files_digest,
        "task_ids": [task["task_id"] for task in tasks],
        "auxiliary_paths": [],
    }


def _write_three_source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    relay_base = tmp_path / "relay"
    relay_base.mkdir()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    source_specs = [
        _write_source(
            relay_base,
            source_id="host4_original_26",
            tasks=tasks[:26],
            host_id="table4_4090d4",
            plan_id="igstgnn_table4_4090d_9gpu_v1",
            execution_layout="dual_host_gpu_sku_homogeneous",
            task_prefix="shards/host4/tasks",
        ),
        _write_source(
            relay_base,
            source_id="host5_partial_12",
            tasks=tasks[26:33] + tasks[34:],
            host_id="table4_4090d5",
            plan_id="igstgnn_table4_host5_partial_export_12_v1",
            execution_layout="single_host_partial_original_export",
            task_prefix="shards/host5/tasks",
        ),
        _write_source(
            relay_base,
            source_id="task034_recovery",
            tasks=[tasks[33]],
            host_id="table4_4090d4",
            plan_id="igstgnn_table4_task034_4gpu_recovery_v1",
            execution_layout="cross_host_single_task_recovery",
            task_prefix="tasks",
            task034_shape=True,
        ),
    ]
    spec = {
        "schema_version": 1,
        "composite_id": "igstgnn_table4_public_repro_39_composite_v1",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_public_repro_39_v2",
        "manifest_sha256": MANIFEST_SHA256,
        "claim_scope": "public_material_maximum_fidelity_reproduction",
        "validity_label": "official_protocol_with_known_overlap",
        "sources": source_specs,
    }
    spec_path = tmp_path / "source_spec.json"
    spec_path.write_bytes(_json_bytes(spec))
    return relay_base, spec_path


def test_composite_verifies_three_sources_and_exact_39_tasks(
    tmp_path: Path,
) -> None:
    relay_base, spec_path = _write_three_source_fixture(tmp_path)
    output_path = tmp_path / "composite" / "composite_index.json"

    index = build_table4_composite_index(
        manifest_path=MANIFEST,
        source_spec_path=spec_path,
        relay_base=relay_base,
        output_path=output_path,
        control_git_sha="2" * 40,
    )

    assert index["status"] == "verified"
    assert index["control_git_sha"] == "2" * 40
    assert len(index["validator_sha256"]) == 64
    assert index["matrix_validity_status"] == (
        "verified_official_protocol_with_known_overlap_39"
    )
    assert index["source_count"] == 3
    assert index["task_count"] == 39
    assert [task["task_id"] for task in index["tasks"]] == [
        f"{task_number:03d}_{region}_{method}"
        for task_number, region, method in (
            (
                number,
                task["region"].lower(),
                task["method"],
            )
            for number, task in enumerate(
                json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"],
                start=1,
            )
        )
    ]
    assert output_path.is_file()


def test_composite_rejects_artifact_changed_after_source_attestation(
    tmp_path: Path,
) -> None:
    relay_base, spec_path = _write_three_source_fixture(tmp_path)
    prediction = next(relay_base.rglob("predictions.npz"))
    prediction.write_bytes(prediction.read_bytes() + b"changed\n")

    with pytest.raises(CompositeValidationError, match="artifact changed"):
        build_table4_composite_index(
            manifest_path=MANIFEST,
            source_spec_path=spec_path,
            relay_base=relay_base,
            output_path=tmp_path / "composite_index.json",
            control_git_sha="2" * 40,
        )
