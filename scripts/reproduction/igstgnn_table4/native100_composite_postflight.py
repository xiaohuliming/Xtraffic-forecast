from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    NATIVE100_CLAIM_SCOPE,
    load_native100_compare_manifest,
)
from scripts.reproduction.igstgnn_table4.secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TASK_ARTIFACT_FILES = frozenset(
    {
        "checkpoint.pt",
        "predictions.npz",
        "provenance.json",
        "summary.json",
        "train.log",
    }
)
_ARTIFACT_ENTRIES = {
    "checkpoint": "checkpoint.pt",
    "predictions": "predictions.npz",
    "provenance": "provenance.json",
    "train_log": "train.log",
}
_METRIC_NAMES = ("mae", "rmse", "mape_percent")
_METRIC_ULP_LIMIT = 16
_METADATA_FILES = frozenset(
    {
        "_TRANSFER_COMPLETE.json",
        "distributed_native100_5080_5090_v1.json",
        "matrix.native100_compare_v1.json",
        "matrix.public_material_v2.json",
        "postflight.json",
        "transfer.files",
        "transfer.sha256",
    }
)
_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "protocol_id",
        "matrix_id",
        "manifest_sha256",
        "base_manifest_sha256",
        "plan_id",
        "plan_sha256",
        "host_id",
        "claim_scope",
        "validated_tasks",
        "file_count",
        "total_bytes",
        "files_digest",
        "postflight_sha256",
        "transfer_manifest_sha256",
    }
)


class Native100CompositeError(ValueError):
    """Raised when the two published native100 sources are not composable."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Native100CompositeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise Native100CompositeError(f"non-finite JSON constant: {value}")


def _read_json(path: Path, *, purpose: str, max_bytes: int) -> dict[str, Any]:
    try:
        payload = read_verified_regular_file(
            path,
            purpose=purpose,
            max_bytes=max_bytes,
            require_nonempty=True,
        )
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except Native100CompositeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise Native100CompositeError(
            f"cannot read strict {purpose}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise Native100CompositeError(f"{purpose} must be an object")
    return value


def _hash_file(path: Path, *, purpose: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open_verified_regular_file(
        path,
        purpose=purpose,
        require_nonempty=True,
    ) as opened:
        for block in iter(lambda: opened.handle.read(1024 * 1024), b""):
            digest.update(block)
        return opened.stat_result.st_size, digest.hexdigest()


def _sha256(path: Path, *, purpose: str) -> str:
    return _hash_file(path, purpose=purpose)[1]


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Native100CompositeError(f"{label} is invalid")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        raise Native100CompositeError(f"{label} is unsafe: {value!r}")
    return value


def _finite_metric(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Native100CompositeError(f"{label} is not numeric")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise Native100CompositeError(f"{label} is not finite and nonnegative")
    return number


def _metric_close(actual: float, expected: float) -> bool:
    spacing = abs(float(np.spacing(np.float32(expected))))
    tolerance = _METRIC_ULP_LIMIT * max(spacing, float(np.finfo(np.float32).tiny))
    return abs(actual - expected) <= tolerance


def _numpy_official_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    null_value = np.float32(labels.min()) if labels.min() < 1 else np.float32(0)
    per_horizon: list[dict[str, float]] = []
    for horizon in range(12):
        prediction = predictions[:, horizon, ...]
        label = labels[:, horizon, ...]
        mask = np.asarray(label != null_value, dtype=np.float32)
        mask = mask / mask.mean(dtype=np.float32)
        mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            absolute = np.abs(prediction - label)
            squared = (prediction - label) ** 2
            percentage = absolute / label
            absolute = np.where(
                np.isnan(absolute * mask), np.zeros_like(absolute), absolute * mask
            )
            squared = np.where(
                np.isnan(squared * mask), np.zeros_like(squared), squared * mask
            )
            percentage = np.where(
                np.isnan(percentage * mask),
                np.zeros_like(percentage),
                percentage * mask,
            )
        per_horizon.append(
            {
                "mae": float(absolute.mean(dtype=np.float32)),
                "rmse": float(np.sqrt(squared.mean(dtype=np.float32))),
                "mape_percent": float(percentage.mean(dtype=np.float32)) * 100.0,
            }
        )
    average = {
        name: float(np.mean([row[name] for row in per_horizon]))
        for name in _METRIC_NAMES
    }
    return {
        "h3": per_horizon[2],
        "h6": per_horizon[5],
        "h12": per_horizon[11],
        "average": average,
        "per_horizon": per_horizon,
        "null_value": float(null_value),
    }


def _validate_metric_contract(summary_metrics: Any, recomputed: dict[str, Any]) -> None:
    if not isinstance(summary_metrics, dict) or set(summary_metrics) != {
        "h3",
        "h6",
        "h12",
        "average",
        "per_horizon",
        "null_value",
    }:
        raise Native100CompositeError("task metric schema mismatch")
    per_horizon = summary_metrics["per_horizon"]
    if not isinstance(per_horizon, list) or len(per_horizon) != 12:
        raise Native100CompositeError("task per-horizon metric schema mismatch")
    for index, row in enumerate(per_horizon):
        if not isinstance(row, dict) or set(row) != set(_METRIC_NAMES):
            raise Native100CompositeError("task horizon metric schema mismatch")
        for name in _METRIC_NAMES:
            expected = _finite_metric(row[name], label=f"summary horizon {index} {name}")
            actual = recomputed["per_horizon"][index][name]
            if not _metric_close(actual, expected):
                raise Native100CompositeError(
                    f"recomputed task metric exceeds {_METRIC_ULP_LIMIT} float32 ULP: "
                    f"horizon={index + 1} metric={name}"
                )
    for label, index in (("h3", 2), ("h6", 5), ("h12", 11)):
        if summary_metrics[label] != per_horizon[index]:
            raise Native100CompositeError(f"task {label} metric is internally inconsistent")
    average = summary_metrics["average"]
    if not isinstance(average, dict) or set(average) != set(_METRIC_NAMES):
        raise Native100CompositeError("task average metric schema mismatch")
    for name in _METRIC_NAMES:
        expected = _finite_metric(average[name], label=f"summary average {name}")
        internal = float(np.mean([float(row[name]) for row in per_horizon]))
        if expected != internal:
            raise Native100CompositeError("task average metric is internally inconsistent")
        if not _metric_close(recomputed["average"][name], expected):
            raise Native100CompositeError(
                f"recomputed average metric exceeds {_METRIC_ULP_LIMIT} float32 ULP: {name}"
            )
    if summary_metrics["null_value"] != recomputed["null_value"]:
        raise Native100CompositeError("task null value is inconsistent")


def _validate_published_task(task_dir: Path, task: dict[str, Any]) -> None:
    if _actual_files(task_dir) != _TASK_ARTIFACT_FILES:
        raise Native100CompositeError(f"task artifact allowlist mismatch: {task['task_id']}")
    summary = _read_json(
        task_dir / "summary.json",
        purpose="native100 task summary",
        max_bytes=1024 * 1024,
    )
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "completed"
        or summary.get("protocol_id") != "igstgnn_table4_official_v1"
        or summary.get("validity_label") != "official_protocol_with_known_overlap"
        or summary.get("claim_scope") != NATIVE100_CLAIM_SCOPE
        or summary.get("task") != task
        or summary.get("artifact_allowlist") != sorted(_TASK_ARTIFACT_FILES)
        or summary.get("prediction_dtype") != "float32"
        or summary.get("label_dtype") != "float32"
    ):
        raise Native100CompositeError(f"task summary identity mismatch: {task['task_id']}")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(_ARTIFACT_ENTRIES):
        raise Native100CompositeError(f"task artifact schema mismatch: {task['task_id']}")
    for name, filename in _ARTIFACT_ENTRIES.items():
        entry = artifacts[name]
        if (
            not isinstance(entry, dict)
            or set(entry) != {"bytes", "sha256"}
            or isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] <= 0
            or not isinstance(entry["sha256"], str)
            or _SHA256.fullmatch(entry["sha256"]) is None
        ):
            raise Native100CompositeError(f"task artifact entry mismatch: {filename}")
        size, digest = _hash_file(task_dir / filename, purpose="native100 task artifact")
        if size != entry["bytes"] or digest != entry["sha256"]:
            raise Native100CompositeError(f"task artifact changed: {filename}")
    provenance = _read_json(
        task_dir / "provenance.json",
        purpose="native100 task provenance",
        max_bytes=1024 * 1024,
    )
    if (
        provenance.get("protocol_id") != "igstgnn_table4_official_v1"
        or provenance.get("matrix_id") != "igstgnn_table4_bist_ndr_native100_18_v1"
        or provenance.get("claim_scope") != NATIVE100_CLAIM_SCOPE
        or provenance.get("task_id") != task["task_id"]
        or provenance.get("method") != task["method"]
        or provenance.get("region") != task["region"]
        or provenance.get("seed") != task["seed"]
        or provenance.get("config_sha256") != task["config_sha256"]
        or provenance.get("split_bundle_manifest_sha256")
        != task["split_bundle_manifest_sha256"]
        or not isinstance(provenance.get("training_code_sha256"), str)
        or _SHA256.fullmatch(provenance["training_code_sha256"]) is None
    ):
        raise Native100CompositeError(f"task provenance mismatch: {task['task_id']}")
    with open_verified_regular_file(
        task_dir / "predictions.npz",
        purpose="native100 prediction artifact",
        require_nonempty=True,
    ) as opened:
        with np.load(opened.handle, allow_pickle=False) as archive:
            if set(archive.files) != {"predictions", "labels", "valid_mask"}:
                raise Native100CompositeError("prediction archive members mismatch")
            predictions = archive["predictions"]
            labels = archive["labels"]
            stored_valid_mask = archive["valid_mask"]
    if (
        predictions.dtype != np.dtype("float32")
        or labels.dtype != np.dtype("float32")
        or predictions.shape != labels.shape
        or predictions.ndim < 2
        or predictions.shape[1] != 12
        or predictions.size == 0
        or not np.isfinite(predictions).all()
        or not np.isfinite(labels).all()
    ):
        raise Native100CompositeError(f"prediction array contract mismatch: {task['task_id']}")
    null_value = np.float32(labels.min()) if labels.min() < 1 else np.float32(0)
    expected_valid_mask = np.asarray(labels != null_value, dtype=np.bool_)
    if (
        stored_valid_mask.dtype != np.dtype("bool")
        or stored_valid_mask.shape != labels.shape
        or not np.array_equal(stored_valid_mask, expected_valid_mask)
        or summary.get("prediction_shape") != list(predictions.shape)
        or summary.get("valid_count") != int(expected_valid_mask.sum())
    ):
        raise Native100CompositeError(f"prediction mask contract mismatch: {task['task_id']}")
    _validate_metric_contract(
        summary.get("metrics"),
        _numpy_official_metrics(predictions, labels),
    )


def _actual_files(root: Path) -> set[str]:
    files: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            entry = directory_path / name
            info = os.lstat(entry)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise Native100CompositeError(f"unsafe directory entry: {entry}")
        for name in filenames:
            entry = directory_path / name
            info = os.lstat(entry)
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise Native100CompositeError(f"unsafe file entry: {entry}")
            files.add(entry.relative_to(root).as_posix())
    return files


def _load_plan(
    path: Path,
    *,
    expected_sha256: str,
    manifest_sha256: str,
    base_manifest_sha256: str,
) -> dict[str, Any]:
    if _sha256(path, purpose="native100 composite plan") != expected_sha256:
        raise Native100CompositeError("plan SHA-256 mismatch")
    plan = _read_json(path, purpose="native100 composite plan", max_bytes=1024 * 1024)
    if (
        plan.get("schema_version") != 1
        or plan.get("plan_id") != "igstgnn_table4_native100_5080_5090_v1"
        or plan.get("protocol_id") != "igstgnn_table4_official_v1"
        or plan.get("matrix_id") != "igstgnn_table4_bist_ndr_native100_18_v1"
        or plan.get("manifest_sha256") != manifest_sha256
        or plan.get("base_manifest_sha256") != base_manifest_sha256
        or plan.get("execution_layout") != "dual_host_region_stratified"
        or not isinstance(plan.get("hosts"), list)
        or len(plan["hosts"]) != 2
    ):
        raise Native100CompositeError("plan identity mismatch")
    return plan


def _parse_transfer_manifest(payload: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise Native100CompositeError("transfer manifest is not UTF-8") from error
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise Native100CompositeError("transfer manifest line is invalid")
        relative = _safe_relative(match.group(2), label="transfer path")
        if relative in seen:
            raise Native100CompositeError("transfer manifest path is duplicated")
        seen.add(relative)
        records.append({"path": relative, "sha256": match.group(1)})
    if not records or [item["path"] for item in records] != sorted(seen):
        raise Native100CompositeError("transfer manifest is not canonical")
    return records


def _validate_source(
    *,
    hosts_base: Path,
    host_plan: dict[str, Any],
    plan: dict[str, Any],
    plan_sha256: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    base_manifest_sha256: str,
    task_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    host_id = host_plan["host_id"]
    source = hosts_base / host_id
    if source.is_symlink() or {entry.name for entry in source.iterdir()} != {
        "metadata",
        "results",
    }:
        raise Native100CompositeError(f"source tree mismatch: {host_id}")
    metadata = source / "metadata"
    results = source / "results"
    if (
        metadata.is_symlink()
        or results.is_symlink()
        or {entry.name for entry in metadata.iterdir()} != _METADATA_FILES
    ):
        raise Native100CompositeError(f"source metadata mismatch: {host_id}")
    if _actual_files(source) != {
        *(f"metadata/{name}" for name in _METADATA_FILES),
        *(
            f"results/{relative}"
            for relative in (
                path.relative_to(results).as_posix()
                for path in results.rglob("*")
                if path.is_file()
            )
        ),
    }:
        raise Native100CompositeError(f"unsafe source filesystem: {host_id}")

    marker_path = metadata / "_TRANSFER_COMPLETE.json"
    postflight_path = metadata / "postflight.json"
    transfer_path = metadata / "transfer.sha256"
    file_list_path = metadata / "transfer.files"
    marker = _read_json(marker_path, purpose="native100 transfer marker", max_bytes=1024 * 1024)
    postflight = _read_json(postflight_path, purpose="native100 postflight", max_bytes=16 * 1024 * 1024)
    if set(marker) != _MARKER_KEYS:
        raise Native100CompositeError(f"marker schema mismatch: {host_id}")
    expected_task_ids = host_plan["task_ids"]
    common = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "manifest_sha256": manifest_sha256,
        "base_manifest_sha256": base_manifest_sha256,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan_sha256,
        "host_id": host_id,
        "claim_scope": NATIVE100_CLAIM_SCOPE,
        "validated_tasks": len(expected_task_ids),
    }
    if any(marker.get(key) != value for key, value in common.items()):
        raise Native100CompositeError(f"marker identity mismatch: {host_id}")
    if any(postflight.get(key) != value for key, value in common.items()):
        raise Native100CompositeError(f"postflight identity mismatch: {host_id}")
    if postflight.get("task_ids") != expected_task_ids:
        raise Native100CompositeError(f"postflight task order mismatch: {host_id}")
    if marker["postflight_sha256"] != _sha256(
        postflight_path, purpose="native100 postflight hash"
    ):
        raise Native100CompositeError(f"postflight hash mismatch: {host_id}")
    transfer_payload = read_verified_regular_file(
        transfer_path,
        purpose="native100 transfer manifest",
        max_bytes=16 * 1024 * 1024,
        require_nonempty=True,
    )
    transfer_sha256 = hashlib.sha256(transfer_payload).hexdigest()
    if (
        marker["transfer_manifest_sha256"] != transfer_sha256
        or marker["files_digest"] != transfer_sha256
        or postflight.get("files_digest") != transfer_sha256
    ):
        raise Native100CompositeError(f"transfer digest mismatch: {host_id}")
    records = _parse_transfer_manifest(transfer_payload)
    file_list_payload = read_verified_regular_file(
        file_list_path,
        purpose="native100 transfer file list",
        max_bytes=16 * 1024 * 1024,
        require_nonempty=True,
    )
    expected_file_list = "".join(f"{item['path']}\n" for item in records).encode()
    if file_list_payload != expected_file_list:
        raise Native100CompositeError(f"transfer file list mismatch: {host_id}")
    actual_result_files = _actual_files(results)
    expected_result_files = {item["path"] for item in records}
    if actual_result_files != expected_result_files:
        raise Native100CompositeError(f"published result allowlist mismatch: {host_id}")
    postflight_records = postflight.get("files")
    if not isinstance(postflight_records, list) or len(postflight_records) != len(records):
        raise Native100CompositeError(f"postflight records mismatch: {host_id}")
    postflight_by_path = {item.get("path"): item for item in postflight_records if isinstance(item, dict)}
    total_bytes = 0
    for record in records:
        path_record = postflight_by_path.get(record["path"])
        if (
            not isinstance(path_record, dict)
            or path_record.get("sha256") != record["sha256"]
            or isinstance(path_record.get("size_bytes"), bool)
            or not isinstance(path_record.get("size_bytes"), int)
            or path_record["size_bytes"] <= 0
        ):
            raise Native100CompositeError(f"postflight file record mismatch: {host_id}")
        size, digest = _hash_file(
            results.joinpath(*PurePosixPath(record["path"]).parts),
            purpose="native100 published artifact",
        )
        if size != path_record["size_bytes"] or digest != record["sha256"]:
            raise Native100CompositeError(f"published artifact changed: {record['path']}")
        total_bytes += size
    if (
        marker["file_count"] != len(records)
        or postflight.get("file_count") != len(records)
        or marker["total_bytes"] != total_bytes
        or postflight.get("total_bytes") != total_bytes
    ):
        raise Native100CompositeError(f"published source totals mismatch: {host_id}")
    for control_name, expected in (
        ("distributed_native100_5080_5090_v1.json", plan_sha256),
        ("matrix.native100_compare_v1.json", manifest_sha256),
        ("matrix.public_material_v2.json", base_manifest_sha256),
    ):
        if _sha256(metadata / control_name, purpose="native100 bound control") != expected:
            raise Native100CompositeError(f"bound control mismatch: {control_name}")

    task_records: list[dict[str, Any]] = []
    for task_id in expected_task_ids:
        task = task_by_id[task_id]
        task_dir = results / task["output_dir"]
        try:
            _validate_published_task(task_dir, task)
        except (OSError, ValueError, RuntimeError) as error:
            raise Native100CompositeError(
                f"published task validation failed: {task_id}"
            ) from error
        task_records.append(
            {
                "task_id": task_id,
                "host_id": host_id,
                "source_root_relative": host_id,
                "task_relative": f"results/{task['output_dir']}",
                "summary_sha256": _sha256(
                    task_dir / "summary.json", purpose="native100 summary"
                ),
            }
        )
    source_record = {
        "host_id": host_id,
        "source_root_relative": host_id,
        "marker_sha256": _sha256(marker_path, purpose="native100 marker"),
        "postflight_sha256": marker["postflight_sha256"],
        "files_digest": marker["files_digest"],
        "validated_tasks": len(expected_task_ids),
        "task_ids": expected_task_ids,
        "file_count": len(records),
        "total_bytes": total_bytes,
    }
    return source_record, task_records


def build_native100_composite_index(
    *,
    hosts_base: Path | str,
    plan_path: Path | str,
    expected_plan_sha256: str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    base_manifest_path: Path | str,
    expected_base_manifest_sha256: str,
    output_path: Path | str,
) -> dict[str, Any]:
    base = Path(hosts_base).resolve(strict=True)
    plan_source = Path(plan_path)
    manifest_source = Path(manifest_path)
    manifest = load_native100_compare_manifest(
        manifest_source,
        expected_sha256=expected_manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=expected_base_manifest_sha256,
    )
    plan = _load_plan(
        plan_source,
        expected_sha256=expected_plan_sha256,
        manifest_sha256=expected_manifest_sha256,
        base_manifest_sha256=expected_base_manifest_sha256,
    )
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    assigned = [task_id for host in plan["hosts"] for task_id in host["task_ids"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(task_by_id):
        raise Native100CompositeError("plan is not an exact 18-task partition")
    source_records: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    for host_plan in plan["hosts"]:
        source_record, source_tasks = _validate_source(
            hosts_base=base,
            host_plan=host_plan,
            plan=plan,
            plan_sha256=expected_plan_sha256,
            manifest=manifest,
            manifest_sha256=expected_manifest_sha256,
            base_manifest_sha256=expected_base_manifest_sha256,
            task_by_id=task_by_id,
        )
        source_records.append(source_record)
        task_records.extend(source_tasks)
    by_task = {item["task_id"]: item for item in task_records}
    if len(by_task) != 18 or set(by_task) != set(task_by_id):
        raise Native100CompositeError("published sources do not cover 18 tasks")
    ordered_tasks = [by_task[task["task_id"]] for task in manifest["tasks"]]
    index = {
        "schema_version": 1,
        "status": "verified_native100_dual_host_18",
        "verified_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "manifest_sha256": expected_manifest_sha256,
        "base_manifest_sha256": expected_base_manifest_sha256,
        "profile_sha256": manifest["profile_sha256"],
        "plan_id": plan["plan_id"],
        "plan_sha256": expected_plan_sha256,
        "execution_layout": plan["execution_layout"],
        "claim_scope": NATIVE100_CLAIM_SCOPE,
        "validity_label": manifest["validity_label"],
        "source_count": len(source_records),
        "task_count": len(ordered_tasks),
        "file_count": sum(item["file_count"] for item in source_records),
        "total_bytes": sum(item["total_bytes"] for item in source_records),
        "sources": source_records,
        "tasks": ordered_tasks,
        "caveats": [
            "official_protocol_with_known_overlap",
            "not_protocol_v3_noleak_dedup",
            "dual_host_region_stratified",
            "native_recipes_differ_except_for_the_shared_100_epoch_cap",
        ],
    }
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise Native100CompositeError(f"composite output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    payload = (
        json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts-base", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    index = build_native100_composite_index(
        hosts_base=args.hosts_base,
        plan_path=args.plan,
        expected_plan_sha256=args.plan_sha256,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
        base_manifest_path=args.base_manifest,
        expected_base_manifest_sha256=args.base_manifest_sha256,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": index["status"],
                "source_count": index["source_count"],
                "task_count": index["task_count"],
                "file_count": index["file_count"],
                "total_bytes": index["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
