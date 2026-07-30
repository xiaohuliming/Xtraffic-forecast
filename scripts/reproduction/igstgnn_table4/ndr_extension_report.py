from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import uuid
from pathlib import Path
from typing import Any

from .ndr_extension_manifest import (
    EXTENSION_REGIONS,
    EXTENSION_SEEDS,
    load_ndr_extension_manifest,
)
from .secure_io import read_verified_regular_file


_ARTIFACT_ALLOWLIST = (
    "checkpoint.pt",
    "predictions.npz",
    "provenance.json",
    "summary.json",
    "train.log",
)
_CLAIM_SCOPE = "same_protocol_project_method_extension"
_MATRIX_ID = "igstgnn_table4_ndr_extension_9_v1"
_PLAN_ID = "igstgnn_table4_ndr_extension_4gpu_rebalance_v2"
_PROTOCOL_ID = "igstgnn_table4_official_v1"
_VALIDITY_LABEL = "official_protocol_with_known_overlap"
_METRIC_GROUPS = ("h3", "h6", "h12", "average")
_METRIC_NAMES = ("mae", "rmse", "mape_percent")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 32 * 1024 * 1024
_EXPECTED_RESULT_FILE_COUNT = 49
_EXPECTED_TASK_COUNT = 9


class NDRExtensionReportError(ValueError):
    """Raised when an NDR extension snapshot cannot support a report."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise NDRExtensionReportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise NDRExtensionReportError(f"non-finite JSON constant: {value}")


def _ensure_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NDRExtensionReportError(
            f"{label} contains a non-finite number"
        )
    if isinstance(value, list):
        for item in value:
            _ensure_finite(item, label=label)
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_finite(item, label=label)


def _read_bytes(
    path: Path,
    *,
    purpose: str,
    max_bytes: int = _MAX_JSON_BYTES,
) -> bytes:
    try:
        return bytes(
            read_verified_regular_file(
                path,
                purpose=purpose,
                max_bytes=max_bytes,
                require_nonempty=True,
            )
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise NDRExtensionReportError(
            f"cannot read {purpose}: {path}"
        ) from error


def _read_json(
    path: Path,
    *,
    purpose: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_bytes(path, purpose=purpose)
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, label=f"{purpose} SHA-256")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise NDRExtensionReportError(f"{purpose} SHA-256 mismatch")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except NDRExtensionReportError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise NDRExtensionReportError(
            f"{purpose} is not strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise NDRExtensionReportError(f"{purpose} must contain an object")
    _ensure_finite(value, label=purpose)
    return value, payload


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NDRExtensionReportError(f"{label} must be a SHA-256 digest")
    return value


def _require_identity(
    value: dict[str, Any],
    *,
    purpose: str,
    expected: dict[str, Any],
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise NDRExtensionReportError(
                f"unexpected {purpose} field: {key}"
            )


def _parse_file_list(payload: bytes) -> list[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise NDRExtensionReportError(
            "transfer file list is not UTF-8"
        ) from error
    if not text.endswith("\n"):
        raise NDRExtensionReportError(
            "transfer file list must end with a newline"
        )
    paths = text.splitlines()
    if (
        len(paths) != _EXPECTED_RESULT_FILE_COUNT
        or len(paths) != len(set(paths))
        or paths != sorted(paths)
        or any(
            not path
            or path.startswith("/")
            or ".." in Path(path).parts
            for path in paths
        )
    ):
        raise NDRExtensionReportError(
            "transfer file list is not the canonical 49-file allowlist"
        )
    return paths


def _parse_transfer_manifest(payload: bytes) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise NDRExtensionReportError(
            "transfer manifest is not UTF-8"
        ) from error
    if not text.endswith("\n"):
        raise NDRExtensionReportError(
            "transfer manifest must end with a newline"
        )
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise NDRExtensionReportError(
                "transfer manifest line has an invalid shape"
            )
        digest = _require_sha256(
            line[:64],
            label="transfer manifest file digest",
        )
        path = line[66:]
        if (
            not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise NDRExtensionReportError(
                "transfer manifest contains an unsafe path"
            )
        records.append({"path": path, "sha256": digest})
    paths = [record["path"] for record in records]
    if (
        len(records) != _EXPECTED_RESULT_FILE_COUNT
        or len(paths) != len(set(paths))
        or paths != sorted(paths)
    ):
        raise NDRExtensionReportError(
            "transfer manifest is not canonical"
        )
    return records


def _parse_checksum_log(payload: bytes) -> list[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise NDRExtensionReportError(
            "checksum log is not UTF-8"
        ) from error
    if not text.endswith("\n"):
        raise NDRExtensionReportError(
            "checksum log must end with a newline"
        )
    suffix = ": OK"
    paths = []
    for line in text.splitlines():
        if not line.endswith(suffix):
            raise NDRExtensionReportError(
                "checksum log contains a failed or malformed entry"
            )
        paths.append(line[: -len(suffix)])
    if len(paths) != len(set(paths)):
        raise NDRExtensionReportError(
            "checksum log contains duplicate entries"
        )
    return paths


def _load_plan(
    path: Path,
    *,
    expected_sha256: str,
    expected_manifest_sha256: str,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    plan, _ = _read_json(
        path,
        purpose="NDR extension rebalance plan",
        expected_sha256=expected_sha256,
    )
    _require_identity(
        plan,
        purpose="NDR extension plan",
        expected={
            "schema_version": 1,
            "plan_id": _PLAN_ID,
            "protocol_id": _PROTOCOL_ID,
            "matrix_id": _MATRIX_ID,
            "manifest_sha256": expected_manifest_sha256,
            "base_manifest_sha256": expected_base_manifest_sha256,
            "gpu_sku": "NVIDIA GeForce RTX 4090 D",
            "execution_layout": "single_host_gpu_sku_homogeneous_restart",
        },
    )
    hosts = plan.get("hosts")
    if (
        not isinstance(hosts, list)
        or len(hosts) != 1
        or hosts[0].get("host_id") != "table4_4090d4"
        or hosts[0].get("gpu_count") != 4
        or hosts[0].get("regions") != list(EXTENSION_REGIONS)
    ):
        raise NDRExtensionReportError(
            "NDR extension plan host binding is invalid"
        )
    shards = plan.get("shards")
    if not isinstance(shards, list) or len(shards) != 4:
        raise NDRExtensionReportError(
            "NDR extension plan must contain four shards"
        )
    task_paths: dict[str, str] = {}
    for gpu_index, shard in enumerate(shards):
        expected_shard_id = f"ndr_table4_4090d4_gpu{gpu_index}"
        if (
            not isinstance(shard, dict)
            or shard.get("shard_id") != expected_shard_id
            or shard.get("host_id") != "table4_4090d4"
            or shard.get("gpu_index") != gpu_index
            or shard.get("results_root") != f"shards/{expected_shard_id}"
            or not isinstance(shard.get("task_ids"), list)
        ):
            raise NDRExtensionReportError(
                "NDR extension shard binding is invalid"
            )
        for task_id in shard["task_ids"]:
            if not isinstance(task_id, str) or task_id in task_paths:
                raise NDRExtensionReportError(
                    "NDR extension plan task partition is invalid"
                )
            task_paths[task_id] = (
                f"{shard['results_root']}/tasks/{task_id}"
            )
    if len(task_paths) != _EXPECTED_TASK_COUNT:
        raise NDRExtensionReportError(
            "NDR extension plan must partition nine tasks"
        )
    return {**plan, "task_paths": task_paths}


def _verify_snapshot_metadata(
    snapshot_root: Path,
    *,
    expected_receipt_sha256: str,
) -> dict[str, Any]:
    metadata = snapshot_root / "metadata"
    receipt, receipt_payload = _read_json(
        snapshot_root / "ndr_extension_remote_verify.json",
        purpose="NDR remote verification receipt",
        expected_sha256=expected_receipt_sha256,
    )
    _require_identity(
        receipt,
        purpose="NDR verification receipt",
        expected={
            "schema_version": 1,
            "status": "verified",
            "file_count": _EXPECTED_RESULT_FILE_COUNT,
            "validated_tasks": _EXPECTED_TASK_COUNT,
        },
    )
    marker, marker_payload = _read_json(
        metadata / "transfer_complete.json",
        purpose="NDR transfer marker",
        expected_sha256=_require_sha256(
            receipt.get("marker_sha256"),
            label="receipt marker_sha256",
        ),
    )
    postflight, postflight_payload = _read_json(
        metadata / "postflight.json",
        purpose="NDR source-host postflight",
        expected_sha256=_require_sha256(
            receipt.get("postflight_sha256"),
            label="receipt postflight_sha256",
        ),
    )
    file_list_payload = _read_bytes(
        metadata / "transfer.files",
        purpose="NDR transfer file list",
    )
    transfer_manifest_payload = _read_bytes(
        metadata / "transfer.sha256",
        purpose="NDR transfer manifest",
    )
    checksum_log_payload = _read_bytes(
        snapshot_root / "ndr_sha256_check.log",
        purpose="NDR remote checksum log",
    )
    digest_bindings = (
        (
            "file_list_sha256",
            file_list_payload,
            "transfer file list",
        ),
        (
            "transfer_manifest_sha256",
            transfer_manifest_payload,
            "transfer manifest",
        ),
        (
            "sha256_check_log_sha256",
            checksum_log_payload,
            "remote checksum log",
        ),
    )
    for key, payload, label in digest_bindings:
        expected = _require_sha256(
            receipt.get(key),
            label=f"receipt {key}",
        )
        if hashlib.sha256(payload).hexdigest() != expected:
            raise NDRExtensionReportError(f"{label} SHA-256 mismatch")
    expected_common = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": _PROTOCOL_ID,
        "matrix_id": _MATRIX_ID,
        "plan_id": _PLAN_ID,
        "claim_scope": _CLAIM_SCOPE,
        "host_id": "table4_4090d4",
        "validated_tasks": _EXPECTED_TASK_COUNT,
        "file_count": _EXPECTED_RESULT_FILE_COUNT,
    }
    _require_identity(
        marker,
        purpose="NDR transfer marker",
        expected=expected_common,
    )
    _require_identity(
        postflight,
        purpose="NDR source-host postflight",
        expected={
            **expected_common,
            "validity_label": _VALIDITY_LABEL,
            "gpu_sku": "NVIDIA GeForce RTX 4090 D",
            "execution_layout": "single_host_gpu_sku_homogeneous_restart",
            "validated_shards": 4,
        },
    )
    marker_postflight_fields = (
        "plan_sha256",
        "manifest_sha256",
        "base_manifest_sha256",
    )
    for key in marker_postflight_fields:
        if marker.get(key) != postflight.get(key):
            raise NDRExtensionReportError(
                f"NDR metadata binding disagrees for {key}"
            )
    shared_fields = (
        "total_bytes",
        "files_digest",
    )
    for key in shared_fields:
        values = (receipt.get(key), marker.get(key), postflight.get(key))
        if values[0] != values[1] or values[1] != values[2]:
            raise NDRExtensionReportError(
                f"NDR metadata binding disagrees for {key}"
            )
    if (
        marker.get("postflight_sha256")
        != hashlib.sha256(postflight_payload).hexdigest()
        or marker.get("file_list_sha256")
        != hashlib.sha256(file_list_payload).hexdigest()
        or marker.get("transfer_manifest_sha256")
        != hashlib.sha256(transfer_manifest_payload).hexdigest()
        or receipt.get("actual_file_list_sha256")
        != hashlib.sha256(file_list_payload).hexdigest()
    ):
        raise NDRExtensionReportError(
            "NDR transfer marker does not bind the verified metadata"
        )
    file_paths = _parse_file_list(file_list_payload)
    transfer_records = _parse_transfer_manifest(
        transfer_manifest_payload
    )
    if [record["path"] for record in transfer_records] != file_paths:
        raise NDRExtensionReportError(
            "NDR transfer manifest paths disagree with the file list"
        )
    checksum_paths = _parse_checksum_log(checksum_log_payload)
    if checksum_paths != file_paths:
        raise NDRExtensionReportError(
            "NDR remote checksum log does not cover the exact file list"
        )
    postflight_files = postflight.get("files")
    if (
        not isinstance(postflight_files, list)
        or len(postflight_files) != _EXPECTED_RESULT_FILE_COUNT
        or postflight.get("export_paths") != file_paths
    ):
        raise NDRExtensionReportError(
            "NDR postflight file inventory is invalid"
        )
    canonical_files_payload = json.dumps(
        postflight_files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if (
        hashlib.sha256(canonical_files_payload).hexdigest()
        != postflight.get("files_digest")
    ):
        raise NDRExtensionReportError(
            "NDR postflight file digest is invalid"
        )
    postflight_by_path: dict[str, dict[str, Any]] = {}
    transfer_by_path = {
        record["path"]: record for record in transfer_records
    }
    total_bytes = 0
    for record in postflight_files:
        if not isinstance(record, dict):
            raise NDRExtensionReportError(
                "NDR postflight file record is invalid"
            )
        path = record.get("path")
        digest = _require_sha256(
            record.get("sha256"),
            label="postflight file SHA-256",
        )
        size = record.get("size_bytes")
        if (
            not isinstance(path, str)
            or path in postflight_by_path
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or transfer_by_path.get(path, {}).get("sha256") != digest
        ):
            raise NDRExtensionReportError(
                "NDR postflight file record does not match transfer metadata"
            )
        total_bytes += size
        postflight_by_path[path] = record
    if (
        list(postflight_by_path) != file_paths
        or total_bytes != postflight.get("total_bytes")
        or receipt.get("total_bytes") != total_bytes
        or receipt.get("files_digest") != postflight.get("files_digest")
    ):
        raise NDRExtensionReportError(
            "NDR file inventory totals are invalid"
        )
    return {
        "receipt": receipt,
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "marker": marker,
        "postflight": postflight,
        "files_by_path": postflight_by_path,
        "file_paths": file_paths,
    }


def _metric(
    metrics: Any,
    *,
    task_id: str,
    group: str,
    name: str,
) -> float:
    if not isinstance(metrics, dict) or not isinstance(
        metrics.get(group), dict
    ):
        raise NDRExtensionReportError(
            f"{task_id} metrics do not contain {group}.{name}"
        )
    value = metrics[group].get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise NDRExtensionReportError(
            f"{task_id} {group}.{name} must be finite and positive"
        )
    return float(value)


def _load_tasks(
    snapshot_root: Path,
    *,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    task_paths = plan["task_paths"]
    canonical_tasks = manifest.get("tasks")
    if not isinstance(canonical_tasks, list) or len(canonical_tasks) != 9:
        raise NDRExtensionReportError(
            "NDR extension manifest must contain nine tasks"
        )
    expected_ids = [task["task_id"] for task in canonical_tasks]
    if set(task_paths) != set(expected_ids):
        raise NDRExtensionReportError(
            "NDR plan does not partition the canonical task set"
        )
    if set(metadata["receipt"].get("task_ids", [])) != set(expected_ids):
        raise NDRExtensionReportError(
            "NDR receipt does not cover the canonical task set"
        )
    if set(metadata["postflight"].get("validated_task_ids", [])) != set(
        expected_ids
    ):
        raise NDRExtensionReportError(
            "NDR postflight does not cover the canonical task set"
        )
    local_result_paths = sorted(
        path.relative_to(snapshot_root / "results").as_posix()
        for path in (snapshot_root / "results").glob("**/*")
        if path.is_file()
    )
    expected_local_paths = sorted(
        f"{task_paths[task_id]}/{name}"
        for task_id in expected_ids
        for name in ("provenance.json", "summary.json")
    )
    if local_result_paths != expected_local_paths:
        raise NDRExtensionReportError(
            "local NDR snapshot must contain exactly 18 summary files"
        )
    output_tasks: list[dict[str, Any]] = []
    for expected_task in canonical_tasks:
        task_id = expected_task["task_id"]
        relative_root = task_paths[task_id]
        summary_relative = f"{relative_root}/summary.json"
        provenance_relative = f"{relative_root}/provenance.json"
        summary_expected = metadata["files_by_path"].get(summary_relative)
        provenance_expected = metadata["files_by_path"].get(
            provenance_relative
        )
        if summary_expected is None or provenance_expected is None:
            raise NDRExtensionReportError(
                f"NDR transfer inventory is missing {task_id} metadata"
            )
        summary, summary_payload = _read_json(
            snapshot_root / "results" / summary_relative,
            purpose=f"{task_id} summary",
            expected_sha256=summary_expected["sha256"],
        )
        provenance, provenance_payload = _read_json(
            snapshot_root / "results" / provenance_relative,
            purpose=f"{task_id} provenance",
            expected_sha256=provenance_expected["sha256"],
        )
        if (
            len(summary_payload) != summary_expected["size_bytes"]
            or len(provenance_payload) != provenance_expected["size_bytes"]
        ):
            raise NDRExtensionReportError(
                f"{task_id} metadata size does not match the inventory"
            )
        _require_identity(
            summary,
            purpose=f"{task_id} summary",
            expected={
                "schema_version": 1,
                "status": "completed",
                "protocol_id": _PROTOCOL_ID,
                "claim_scope": _CLAIM_SCOPE,
                "validity_label": _VALIDITY_LABEL,
                "task": expected_task,
                "artifact_allowlist": list(_ARTIFACT_ALLOWLIST),
            },
        )
        if (
            isinstance(summary.get("valid_count"), bool)
            or not isinstance(summary.get("valid_count"), int)
            or summary["valid_count"] <= 0
        ):
            raise NDRExtensionReportError(
                f"{task_id} summary valid_count is invalid"
            )
        _require_identity(
            provenance,
            purpose=f"{task_id} provenance",
            expected={
                "schema_version": 1,
                "protocol_id": _PROTOCOL_ID,
                "matrix_id": _MATRIX_ID,
                "matrix_manifest_sha256": metadata["marker"][
                    "manifest_sha256"
                ],
                "base_manifest_sha256": metadata["marker"][
                    "base_manifest_sha256"
                ],
                "claim_scope": _CLAIM_SCOPE,
                "validity_label": _VALIDITY_LABEL,
                "task_id": task_id,
                "method": expected_task["method"],
                "region": expected_task["region"],
                "seed": expected_task["seed"],
                "config_sha256": expected_task["config_sha256"],
                "split_bundle_manifest_sha256": expected_task[
                    "split_bundle_manifest_sha256"
                ],
            },
        )
        task_training_code_sha256 = _require_sha256(
            provenance.get("training_code_sha256"),
            label=f"{task_id} training_code_sha256",
        )
        metrics = {
            group: {
                name: _metric(
                    summary.get("metrics"),
                    task_id=task_id,
                    group=group,
                    name=name,
                )
                for name in _METRIC_NAMES
            }
            for group in _METRIC_GROUPS
        }
        output_tasks.append(
            {
                "task_id": task_id,
                "region": expected_task["region"],
                "seed": expected_task["seed"],
                "method": "ndr_net",
                "metrics": metrics,
                "valid_count": summary["valid_count"],
                "training_code_sha256": task_training_code_sha256,
                "summary_sha256": summary_expected["sha256"],
                "provenance_sha256": provenance_expected["sha256"],
            }
        )
    return output_tasks


def _summarize_regions(
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for region in EXTENSION_REGIONS:
        region_tasks = [task for task in tasks if task["region"] == region]
        if [task["seed"] for task in region_tasks] != list(EXTENSION_SEEDS):
            raise NDRExtensionReportError(
                f"{region} does not contain the canonical three seeds"
            )
        metrics: dict[str, dict[str, dict[str, float]]] = {}
        for group in _METRIC_GROUPS:
            metrics[group] = {}
            for name in _METRIC_NAMES:
                values = [
                    task["metrics"][group][name] for task in region_tasks
                ]
                metrics[group][name] = {
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values),
                }
        summaries.append(
            {
                "region": region,
                "method": "ndr_net",
                "seeds": list(EXTENSION_SEEDS),
                "metrics": metrics,
            }
        )
    return summaries


def _seed_csv_bytes(tasks: list[dict[str, Any]]) -> bytes:
    metric_columns = [
        f"{group}_{name}"
        for group in _METRIC_GROUPS
        for name in _METRIC_NAMES
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "task_id",
            "region",
            "method",
            "seed",
            "valid_count",
            *metric_columns,
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for task in tasks:
        writer.writerow(
            {
                "task_id": task["task_id"],
                "region": task["region"],
                "method": task["method"],
                "seed": task["seed"],
                "valid_count": task["valid_count"],
                **{
                    f"{group}_{name}": (
                        f"{task['metrics'][group][name]:.6f}"
                    )
                    for group in _METRIC_GROUPS
                    for name in _METRIC_NAMES
                },
            }
        )
    return buffer.getvalue().encode("utf-8")


def _region_csv_bytes(summaries: list[dict[str, Any]]) -> bytes:
    metric_columns = [
        f"{group}_{name}_{statistic}"
        for group in _METRIC_GROUPS
        for name in _METRIC_NAMES
        for statistic in ("mean", "sample_std")
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("region", "method", "seeds", *metric_columns),
        lineterminator="\n",
    )
    writer.writeheader()
    for summary in summaries:
        writer.writerow(
            {
                "region": summary["region"],
                "method": summary["method"],
                "seeds": ",".join(str(seed) for seed in summary["seeds"]),
                **{
                    f"{group}_{name}_{statistic}": (
                        f"{summary['metrics'][group][name][statistic]:.6f}"
                    )
                    for group in _METRIC_GROUPS
                    for name in _METRIC_NAMES
                    for statistic in ("mean", "sample_std")
                },
            }
        )
    return buffer.getvalue().encode("utf-8")


def _mean_std(value: dict[str, float]) -> str:
    return f"{value['mean']:.2f} ± {value['sample_std']:.2f}"


def _markdown_bytes(
    tasks: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    *,
    receipt_sha256: str,
) -> bytes:
    lines = [
        "# NDR-Net 的 IGSTGNN Table 4 同协议扩展",
        "",
        "该结果使用 IGSTGNN Table 4 的公开协议和已知重叠切分。",
        "NDR-Net 是项目方法扩展，不是 IGSTGNN 原论文的方法行。",
        "每个地区报告种子 42、1、2 的均值和样本标准差。",
        "",
        f"结果服务器复验回执 SHA-256：`{receipt_sha256}`。",
        "",
        "## 三种子汇总",
        "",
        "| Data | Method | H3 MAE | H3 RMSE | H3 MAPE | "
        "H6 MAE | H6 RMSE | H6 MAPE | H12 MAE | H12 RMSE | "
        "H12 MAPE | Avg MAE | Avg RMSE | Avg MAPE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|",
    ]
    for summary in summaries:
        cells = []
        for group in _METRIC_GROUPS:
            for name in _METRIC_NAMES:
                cells.append(_mean_std(summary["metrics"][group][name]))
        lines.append(
            f"| {summary['region'].replace('_', ' ')} | NDR-Net | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 单种子结果",
            "",
            "| Task | Data | Seed | H3 MAE | H6 MAE | H12 MAE | "
            "Avg MAE | Avg RMSE | Avg MAPE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in tasks:
        metrics = task["metrics"]
        lines.append(
            f"| {task['task_id']} | {task['region'].replace('_', ' ')} | "
            f"{task['seed']} | {metrics['h3']['mae']:.6f} | "
            f"{metrics['h6']['mae']:.6f} | "
            f"{metrics['h12']['mae']:.6f} | "
            f"{metrics['average']['mae']:.6f} | "
            f"{metrics['average']['rmse']:.6f} | "
            f"{metrics['average']['mape_percent']:.6f} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive_output(
    output_dir: Path,
    *,
    files: dict[str, bytes],
) -> None:
    if output_dir.exists():
        raise NDRExtensionReportError(
            f"output directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / (
        f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        for name, payload in files.items():
            target = staging / name
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            for path in staging.iterdir():
                if path.is_file():
                    path.unlink()
            staging.rmdir()
        raise


def build_ndr_extension_report(
    *,
    snapshot_root: Path | str,
    expected_receipt_sha256: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Verify a nine-task NDR snapshot and write its three-seed report."""

    snapshot = Path(snapshot_root)
    destination = Path(output_dir)
    _require_sha256(
        expected_receipt_sha256,
        label="expected receipt SHA-256",
    )
    metadata = _verify_snapshot_metadata(
        snapshot,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    marker = metadata["marker"]
    manifest_sha256 = _require_sha256(
        marker.get("manifest_sha256"),
        label="extension manifest SHA-256",
    )
    base_manifest_sha256 = _require_sha256(
        marker.get("base_manifest_sha256"),
        label="base manifest SHA-256",
    )
    try:
        manifest = load_ndr_extension_manifest(
            snapshot / "metadata" / "matrix.ndr_extension_v1.json",
            expected_sha256=manifest_sha256,
            base_manifest_path=(
                snapshot
                / "metadata"
                / "matrix.public_material_v2.json"
            ),
            expected_base_manifest_sha256=base_manifest_sha256,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise NDRExtensionReportError(
            "NDR extension manifest binding is invalid"
        ) from error
    plan_sha256 = _require_sha256(
        marker.get("plan_sha256"),
        label="NDR plan SHA-256",
    )
    plan = _load_plan(
        snapshot
        / "metadata"
        / "distributed_ndr_extension_4gpu_rebalance_v2.json",
        expected_sha256=plan_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    tasks = _load_tasks(
        snapshot,
        manifest=manifest,
        plan=plan,
        metadata=metadata,
    )
    region_summaries = _summarize_regions(tasks)
    report = {
        "schema_version": 1,
        "status": "verified_ndr_extension_report",
        "protocol_id": _PROTOCOL_ID,
        "matrix_id": _MATRIX_ID,
        "base_matrix_id": "igstgnn_table4_public_repro_39_v2",
        "claim_scope": _CLAIM_SCOPE,
        "validity_label": _VALIDITY_LABEL,
        "execution_layout": plan["execution_layout"],
        "host_id": "table4_4090d4",
        "gpu_sku": plan["gpu_sku"],
        "task_count": len(tasks),
        "region_count": len(region_summaries),
        "seed_count": len(EXTENSION_SEEDS),
        "seeds": list(EXTENSION_SEEDS),
        "receipt_sha256": metadata["receipt_sha256"],
        "marker_sha256": hashlib.sha256(
            _read_bytes(
                snapshot / "metadata" / "transfer_complete.json",
                purpose="NDR transfer marker",
            )
        ).hexdigest(),
        "postflight_sha256": metadata["receipt"]["postflight_sha256"],
        "manifest_sha256": manifest_sha256,
        "base_manifest_sha256": base_manifest_sha256,
        "profile_sha256": manifest["profile_sha256"],
        "plan_sha256": plan_sha256,
        "files_digest": marker["files_digest"],
        "source_file_count": marker["file_count"],
        "source_bytes": marker["total_bytes"],
        "training_code_sha256": sorted(
            {task["training_code_sha256"] for task in tasks}
        ),
        "tasks": tasks,
        "region_summaries": region_summaries,
        "reporting_note": (
            "NDR-Net is a three-seed project-method extension and is not "
            "an original IGSTGNN Table 4 method row."
        ),
    }
    seed_csv = _seed_csv_bytes(tasks)
    region_csv = _region_csv_bytes(region_summaries)
    markdown = _markdown_bytes(
        tasks,
        region_summaries,
        receipt_sha256=metadata["receipt_sha256"],
    )
    _write_exclusive_output(
        destination,
        files={
            "ndr_extension_index.json": _json_bytes(report),
            "ndr_seed_results.csv": seed_csv,
            "ndr_region_summary.csv": region_csv,
            "ndr_extension.md": markdown,
        },
    )
    return report
