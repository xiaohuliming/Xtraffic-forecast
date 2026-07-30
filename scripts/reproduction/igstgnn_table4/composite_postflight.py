from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .manifest import load_table4_manifest
from .secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
    verify_secure_directory,
)


_MAX_JSON_BYTES = 16 * 1024 * 1024
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_ARTIFACT_NAMES = frozenset(
    {
        "checkpoint.pt",
        "predictions.npz",
        "provenance.json",
        "summary.json",
        "train.log",
    }
)
_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "composite_id",
        "protocol_id",
        "matrix_id",
        "manifest_sha256",
        "claim_scope",
        "validity_label",
        "sources",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "root_relative",
        "host_id",
        "plan_id",
        "plan_sha256",
        "execution_layout",
        "marker_sha256",
        "postflight_sha256",
        "validated_shards",
        "validated_tasks",
        "file_count",
        "source_bytes",
        "files_digest",
        "task_ids",
        "auxiliary_paths",
    }
)
_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "protocol_id",
        "matrix_id",
        "plan_id",
        "plan_sha256",
        "manifest_sha256",
        "host_id",
        "gpu_sku",
        "execution_layout",
        "validated_shards",
        "validated_tasks",
        "file_count",
        "source_bytes",
        "files_digest",
        "file_list_sha256",
        "transfer_manifest_sha256",
        "postflight_sha256",
    }
)


class CompositeValidationError(ValueError):
    """Raised when the three published sources cannot form one verified matrix."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CompositeValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CompositeValidationError(f"non-finite JSON constant: {value}")


def _ensure_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CompositeValidationError(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _ensure_finite(item, label=label)
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_finite(item, label=label)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = read_verified_regular_file(
            path,
            purpose=label,
            max_bytes=_MAX_JSON_BYTES,
            require_nonempty=True,
        )
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except CompositeValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CompositeValidationError(
            f"cannot read strict {label}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CompositeValidationError(f"{label} must contain an object")
    _ensure_finite(value, label=label)
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CompositeValidationError("value is not strict JSON") from error


def _json_output(value: Any) -> bytes:
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


def _hash_file(path: Path, *, label: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    try:
        with open_verified_regular_file(
            path,
            purpose=label,
            require_nonempty=True,
        ) as opened:
            for block in iter(lambda: opened.handle.read(1024 * 1024), b""):
                digest.update(block)
            size = opened.stat_result.st_size
    except (OSError, ValueError, RuntimeError) as error:
        raise CompositeValidationError(
            f"cannot hash verified {label}: {path}"
        ) from error
    return size, digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CompositeValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_string(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise CompositeValidationError(f"{label} must be a nonempty safe string")
    return value


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CompositeValidationError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _safe_relative(value: Any, *, label: str) -> str:
    text = _require_string(value, label=label)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != text:
        raise CompositeValidationError(f"{label} must be a canonical relative path")
    return text


def _verify_directory(path: Path, *, label: str) -> Path:
    try:
        return verify_secure_directory(path, purpose=label)
    except (OSError, ValueError, RuntimeError) as error:
        raise CompositeValidationError(
            f"{label} is missing or unsafe: {path}"
        ) from error


def _list_names(path: Path) -> set[str]:
    try:
        return {entry.name for entry in os.scandir(path)}
    except OSError as error:
        raise CompositeValidationError(f"cannot list directory: {path}") from error


def _parse_transfer_files(path: Path) -> list[str]:
    try:
        payload = read_verified_regular_file(
            path,
            purpose="transfer file list",
            max_bytes=_MAX_JSON_BYTES,
            require_nonempty=True,
        )
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise CompositeValidationError("cannot read transfer file list") from error
    if not lines or lines != sorted(lines) or len(lines) != len(set(lines)):
        raise CompositeValidationError(
            "transfer file list must be nonempty, sorted, and unique"
        )
    for line in lines:
        _safe_relative(line, label="transfer path")
    return lines


def _parse_transfer_manifest(path: Path) -> dict[str, str]:
    try:
        payload = read_verified_regular_file(
            path,
            purpose="transfer SHA-256 manifest",
            max_bytes=_MAX_JSON_BYTES,
            require_nonempty=True,
        )
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise CompositeValidationError("cannot read transfer SHA-256 manifest") from error
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise CompositeValidationError("invalid transfer manifest row")
        digest = _require_sha256(line[:64], label="transfer digest")
        relative = _safe_relative(line[66:], label="transfer path")
        if relative in result:
            raise CompositeValidationError(
                f"duplicate transfer manifest path: {relative}"
            )
        result[relative] = digest
    if list(result) != sorted(result) or not result:
        raise CompositeValidationError(
            "transfer manifest must be nonempty and canonically sorted"
        )
    return result


def _source_task_ids(report: dict[str, Any]) -> list[str]:
    tasks = report.get("tasks")
    if isinstance(tasks, list):
        result = []
        for item in tasks:
            if not isinstance(item, dict):
                raise CompositeValidationError("postflight task must be an object")
            result.append(_require_string(item.get("task_id"), label="task_id"))
        return result
    task_id = report.get("task_id")
    if isinstance(task_id, str):
        return [task_id]
    raise CompositeValidationError("postflight report does not identify its tasks")


def _validate_common_identity(
    value: dict[str, Any],
    *,
    spec: dict[str, Any],
    source: dict[str, Any],
    label: str,
) -> None:
    expected = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": spec["protocol_id"],
        "matrix_id": spec["matrix_id"],
        "manifest_sha256": spec["manifest_sha256"],
        "host_id": source["host_id"],
        "plan_id": source["plan_id"],
        "plan_sha256": source["plan_sha256"],
        "execution_layout": source["execution_layout"],
        "validated_shards": source["validated_shards"],
        "validated_tasks": source["validated_tasks"],
        "file_count": source["file_count"],
        "files_digest": source["files_digest"],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise CompositeValidationError(
                f"{label} {key} does not match the locked source specification"
            )


def _validate_source(
    *,
    relay_base: Path,
    spec: dict[str, Any],
    source: dict[str, Any],
    manifest_tasks: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(source) != _SOURCE_KEYS:
        raise CompositeValidationError("source specification keys are invalid")
    source_id = _require_string(source["source_id"], label="source_id")
    if _SOURCE_ID_RE.fullmatch(source_id) is None:
        raise CompositeValidationError(f"unsafe source_id: {source_id}")
    relative_root = _safe_relative(
        source["root_relative"],
        label=f"{source_id} root_relative",
    )
    for key in ("plan_sha256", "marker_sha256", "postflight_sha256", "files_digest"):
        _require_sha256(source[key], label=f"{source_id} {key}")
    for key in (
        "validated_shards",
        "validated_tasks",
        "file_count",
        "source_bytes",
    ):
        _require_int(source[key], label=f"{source_id} {key}", minimum=1)
    task_ids = source.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
        or task_ids != sorted(task_ids)
    ):
        raise CompositeValidationError(
            f"{source_id} task_ids must be sorted and unique"
        )
    if source["validated_tasks"] != len(task_ids):
        raise CompositeValidationError(
            f"{source_id} validated_tasks does not match task_ids"
        )
    auxiliary_paths = source.get("auxiliary_paths")
    if not isinstance(auxiliary_paths, list):
        raise CompositeValidationError(
            f"{source_id} auxiliary_paths must be a list"
        )
    auxiliary = {
        _safe_relative(item, label=f"{source_id} auxiliary path")
        for item in auxiliary_paths
    }
    if len(auxiliary) != len(auxiliary_paths):
        raise CompositeValidationError(
            f"{source_id} auxiliary_paths contain duplicates"
        )

    root = _verify_directory(
        relay_base / Path(*PurePosixPath(relative_root).parts),
        label=f"{source_id} source root",
    )
    try:
        root.relative_to(relay_base)
    except ValueError as error:
        raise CompositeValidationError(
            f"{source_id} source root escapes relay base"
        ) from error
    if _list_names(root) != {"_TRANSFER_COMPLETE.json", "metadata", "results"}:
        raise CompositeValidationError(
            f"{source_id} source root does not have the exact published layout"
        )
    metadata_root = _verify_directory(
        root / "metadata",
        label=f"{source_id} metadata root",
    )
    results_root = _verify_directory(
        root / "results",
        label=f"{source_id} results root",
    )
    if _list_names(metadata_root) != {
        "postflight.json",
        "transfer.files",
        "transfer.sha256",
    }:
        raise CompositeValidationError(
            f"{source_id} metadata allowlist does not match"
        )

    marker_path = root / "_TRANSFER_COMPLETE.json"
    marker_size, marker_sha256 = _hash_file(
        marker_path,
        label=f"{source_id} transfer marker",
    )
    if marker_size <= 0 or marker_sha256 != source["marker_sha256"]:
        raise CompositeValidationError(f"{source_id} transfer marker changed")
    marker = _read_json(marker_path, label=f"{source_id} transfer marker")
    if set(marker) != _MARKER_KEYS:
        raise CompositeValidationError(f"{source_id} transfer marker keys changed")
    _validate_common_identity(
        marker,
        spec=spec,
        source=source,
        label=f"{source_id} transfer marker",
    )
    if marker.get("source_bytes") != source["source_bytes"]:
        raise CompositeValidationError(f"{source_id} source_bytes changed")

    postflight_path = metadata_root / "postflight.json"
    _, postflight_sha256 = _hash_file(
        postflight_path,
        label=f"{source_id} postflight",
    )
    if (
        postflight_sha256 != source["postflight_sha256"]
        or marker["postflight_sha256"] != postflight_sha256
    ):
        raise CompositeValidationError(f"{source_id} postflight changed")
    postflight = _read_json(postflight_path, label=f"{source_id} postflight")
    _validate_common_identity(
        postflight,
        spec=spec,
        source=source,
        label=f"{source_id} postflight",
    )
    if postflight.get("total_bytes") != source["source_bytes"]:
        raise CompositeValidationError(f"{source_id} postflight byte count changed")
    observed_task_ids = _source_task_ids(postflight)
    if sorted(observed_task_ids) != task_ids:
        raise CompositeValidationError(
            f"{source_id} postflight task set does not match"
        )

    file_list_path = metadata_root / "transfer.files"
    transfer_manifest_path = metadata_root / "transfer.sha256"
    _, file_list_sha256 = _hash_file(
        file_list_path,
        label=f"{source_id} transfer file list",
    )
    _, transfer_manifest_sha256 = _hash_file(
        transfer_manifest_path,
        label=f"{source_id} transfer manifest",
    )
    if (
        file_list_sha256 != marker["file_list_sha256"]
        or transfer_manifest_sha256 != marker["transfer_manifest_sha256"]
    ):
        raise CompositeValidationError(
            f"{source_id} transfer metadata changed"
        )
    file_list = _parse_transfer_files(file_list_path)
    transfer_manifest = _parse_transfer_manifest(transfer_manifest_path)
    if file_list != list(transfer_manifest):
        raise CompositeValidationError(
            f"{source_id} transfer file list and manifest disagree"
        )

    records = postflight.get("files")
    export_paths = postflight.get("export_paths")
    if not isinstance(records, list) or not isinstance(export_paths, list):
        raise CompositeValidationError(
            f"{source_id} postflight lacks export records"
        )
    normalized_records: list[dict[str, Any]] = []
    record_digests: dict[str, str] = {}
    total_bytes = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise CompositeValidationError(
                f"{source_id} postflight file record is invalid"
            )
        relative = _safe_relative(record["path"], label="postflight file path")
        digest = _require_sha256(
            record["sha256"],
            label="postflight file digest",
        )
        size = _require_int(
            record["size_bytes"],
            label="postflight file size",
            minimum=1,
        )
        if relative in record_digests:
            raise CompositeValidationError(
                f"{source_id} has duplicate postflight path: {relative}"
            )
        normalized_records.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
        record_digests[relative] = digest
        total_bytes += size
    if export_paths != sorted(record_digests):
        raise CompositeValidationError(
            f"{source_id} export paths are not canonical"
        )
    if (
        len(records) != source["file_count"]
        or total_bytes != source["source_bytes"]
        or file_list != export_paths
        or transfer_manifest != record_digests
    ):
        raise CompositeValidationError(
            f"{source_id} file counts, paths, bytes, or hashes disagree"
        )
    files_digest = hashlib.sha256(_canonical_json(normalized_records)).hexdigest()
    if files_digest != source["files_digest"]:
        raise CompositeValidationError(f"{source_id} files_digest changed")

    actual_paths: list[str] = []
    task_artifacts: dict[str, dict[str, tuple[str, int, str]]] = {
        task_id: {} for task_id in task_ids
    }
    for directory, directory_names, file_names in os.walk(
        results_root,
        followlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)
        if current.is_symlink():
            raise CompositeValidationError(
                f"{source_id} results contain a symlink directory"
            )
        for file_name in file_names:
            path = current / file_name
            relative = path.relative_to(results_root).as_posix()
            actual_paths.append(relative)
            if relative not in transfer_manifest:
                raise CompositeValidationError(
                    f"{source_id} results contain an unexpected file: {relative}"
                )
            size, digest = _hash_file(
                path,
                label=f"{source_id} result artifact",
            )
            record = next(
                item for item in normalized_records if item["path"] == relative
            )
            if size != record["size_bytes"] or digest != record["sha256"]:
                raise CompositeValidationError(
                    f"{source_id} result artifact changed: {relative}"
                )
            parts = PurePosixPath(relative).parts
            if "tasks" in parts:
                task_index = parts.index("tasks")
                if len(parts) == task_index + 3:
                    task_id = parts[task_index + 1]
                    artifact_name = parts[task_index + 2]
                    if (
                        task_id in task_artifacts
                        and artifact_name in _ARTIFACT_NAMES
                    ):
                        task_artifacts[task_id][artifact_name] = (
                            relative,
                            size,
                            digest,
                        )
                        continue
            if relative not in auxiliary:
                raise CompositeValidationError(
                    f"{source_id} file is not a locked task artifact "
                    f"or auxiliary path: {relative}"
                )
    if sorted(actual_paths) != file_list:
        raise CompositeValidationError(
            f"{source_id} actual result tree does not match transfer metadata"
        )

    task_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        artifacts = task_artifacts[task_id]
        if set(artifacts) != _ARTIFACT_NAMES:
            raise CompositeValidationError(
                f"{source_id} task {task_id} does not have the exact five artifacts"
            )
        task = manifest_tasks.get(task_id)
        if task is None:
            raise CompositeValidationError(
                f"{source_id} contains non-canonical task {task_id}"
            )
        summary_path = results_root / Path(
            *PurePosixPath(artifacts["summary.json"][0]).parts
        )
        provenance_path = results_root / Path(
            *PurePosixPath(artifacts["provenance.json"][0]).parts
        )
        summary = _read_json(summary_path, label=f"{task_id} summary")
        provenance = _read_json(provenance_path, label=f"{task_id} provenance")
        if (
            summary.get("schema_version") != 1
            or summary.get("status") != "completed"
            or summary.get("protocol_id") != spec["protocol_id"]
            or summary.get("claim_scope") != spec["claim_scope"]
            or summary.get("validity_label") != spec["validity_label"]
            or summary.get("task") != task
            or summary.get("artifact_allowlist") != sorted(_ARTIFACT_NAMES)
        ):
            raise CompositeValidationError(
                f"{source_id} task {task_id} summary identity is invalid"
            )
        expected_provenance = {
            "protocol_id": spec["protocol_id"],
            "matrix_manifest_sha256": spec["manifest_sha256"],
            "task_id": task_id,
            "method": task["method"],
            "region": task["region"],
            "seed": task["seed"],
            "config_sha256": task["config_sha256"],
            "split_bundle_manifest_sha256": task[
                "split_bundle_manifest_sha256"
            ],
        }
        for key, expected_value in expected_provenance.items():
            if provenance.get(key) != expected_value:
                raise CompositeValidationError(
                    f"{source_id} task {task_id} provenance {key} is invalid"
                )
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            raise CompositeValidationError(
                f"{source_id} task {task_id} lacks metrics"
            )
        required_metrics = {"h3", "h6", "h12", "average"}
        if not required_metrics.issubset(metrics):
            raise CompositeValidationError(
                f"{source_id} task {task_id} lacks Table 4 metrics"
            )
        task_rows.append(
            {
                "task_id": task_id,
                "method": task["method"],
                "region": task["region"],
                "seed": task["seed"],
                "source_id": source_id,
                "source_host_id": source["host_id"],
                "source_execution_layout": source["execution_layout"],
                "relative_task_root": str(
                    PurePosixPath(artifacts["summary.json"][0]).parent
                ),
                "training_code_sha256": _require_sha256(
                    provenance.get("training_code_sha256"),
                    label=f"{task_id} training_code_sha256",
                ),
                "summary_sha256": artifacts["summary.json"][2],
                "predictions_sha256": artifacts["predictions.npz"][2],
                "provenance_sha256": artifacts["provenance.json"][2],
                "metrics": metrics,
                "valid_count": summary.get("valid_count"),
                "prediction_shape": summary.get("prediction_shape"),
            }
        )
    source_row = {
        "source_id": source_id,
        "root_relative": relative_root,
        "host_id": source["host_id"],
        "plan_id": source["plan_id"],
        "plan_sha256": source["plan_sha256"],
        "execution_layout": source["execution_layout"],
        "marker_sha256": marker_sha256,
        "postflight_sha256": postflight_sha256,
        "validated_shards": source["validated_shards"],
        "validated_tasks": source["validated_tasks"],
        "file_count": source["file_count"],
        "source_bytes": source["source_bytes"],
        "files_digest": source["files_digest"],
        "task_ids": task_ids,
    }
    return source_row, task_rows


def _write_index(path: Path, payload: bytes, source_roots: list[Path]) -> None:
    if path.exists() or path.is_symlink():
        raise CompositeValidationError(f"composite index already exists: {path}")
    for source_root in source_roots:
        try:
            path.resolve(strict=False).relative_to(source_root)
        except ValueError:
            continue
        raise CompositeValidationError(
            "composite index output must be outside every source root"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = _verify_directory(path.parent, label="composite output directory")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as error:
        raise CompositeValidationError(
            f"composite index already exists: {path}"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def build_table4_composite_index(
    *,
    manifest_path: Path | str,
    source_spec_path: Path | str,
    relay_base: Path | str,
    output_path: Path | str | None = None,
    control_git_sha: str,
) -> dict[str, Any]:
    if (
        not isinstance(control_git_sha, str)
        or _GIT_SHA_RE.fullmatch(control_git_sha) is None
    ):
        raise CompositeValidationError(
            "control_git_sha must be a full lowercase Git SHA"
        )
    spec_path = Path(source_spec_path)
    spec = _read_json(spec_path, label="Table 4 composite source specification")
    if set(spec) != _SPEC_KEYS:
        raise CompositeValidationError(
            "composite source specification keys are invalid"
        )
    if spec.get("schema_version") != 1:
        raise CompositeValidationError(
            "unsupported composite source specification schema"
        )
    for key in (
        "composite_id",
        "protocol_id",
        "matrix_id",
        "claim_scope",
        "validity_label",
    ):
        _require_string(spec.get(key), label=key)
    manifest_sha256 = _require_sha256(
        spec.get("manifest_sha256"),
        label="manifest_sha256",
    )
    try:
        manifest = load_table4_manifest(
            manifest_path,
            expected_sha256=manifest_sha256,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CompositeValidationError(
            "canonical Table 4 manifest verification failed"
        ) from error
    if (
        spec["protocol_id"] != manifest["protocol_id"]
        or spec["matrix_id"] != manifest["matrix_id"]
        or spec["claim_scope"] != manifest["claim_scope"]
        or spec["validity_label"] != manifest["validity_label"]
    ):
        raise CompositeValidationError(
            "composite specification does not match the canonical manifest"
        )
    sources = spec.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise CompositeValidationError(
            "Table 4 composite must lock exactly three published sources"
        )
    source_ids = [source.get("source_id") for source in sources if isinstance(source, dict)]
    if len(source_ids) != 3 or len(set(source_ids)) != 3:
        raise CompositeValidationError("composite source IDs must be unique")

    relay_root = _verify_directory(Path(relay_base), label="relay base")
    manifest_tasks = {
        task["task_id"]: task for task in manifest["tasks"]
    }
    source_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    source_roots: list[Path] = []
    for source in sources:
        if not isinstance(source, dict):
            raise CompositeValidationError(
                "composite source specification must contain objects"
            )
        source_row, source_task_rows = _validate_source(
            relay_base=relay_root,
            spec=spec,
            source=source,
            manifest_tasks=manifest_tasks,
        )
        source_rows.append(source_row)
        task_rows.extend(source_task_rows)
        source_roots.append(relay_root / source_row["root_relative"])
    observed_task_ids = [task["task_id"] for task in task_rows]
    canonical_task_ids = list(manifest_tasks)
    duplicates = sorted(
        {
            task_id
            for task_id in observed_task_ids
            if observed_task_ids.count(task_id) > 1
        }
    )
    if duplicates:
        raise CompositeValidationError(
            f"duplicate canonical task IDs across sources: {duplicates}"
        )
    if set(observed_task_ids) != set(canonical_task_ids):
        missing = sorted(set(canonical_task_ids) - set(observed_task_ids))
        extra = sorted(set(observed_task_ids) - set(canonical_task_ids))
        raise CompositeValidationError(
            f"composite task coverage mismatch, missing={missing}, extra={extra}"
        )
    by_task = {task["task_id"]: task for task in task_rows}
    ordered_tasks = [by_task[task_id] for task_id in canonical_task_ids]
    _, spec_sha256 = _hash_file(
        spec_path,
        label="Table 4 composite source specification",
    )
    _, validator_sha256 = _hash_file(
        Path(__file__).resolve(),
        label="Table 4 composite validator",
    )
    payload = {
        "schema_version": 1,
        "status": "verified",
        "matrix_validity_status": (
            "verified_official_protocol_with_known_overlap_39"
        ),
        "composite_id": spec["composite_id"],
        "protocol_id": spec["protocol_id"],
        "matrix_id": spec["matrix_id"],
        "manifest_sha256": manifest_sha256,
        "source_spec_sha256": spec_sha256,
        "control_git_sha": control_git_sha,
        "validator_sha256": validator_sha256,
        "claim_scope": spec["claim_scope"],
        "validity_label": spec["validity_label"],
        "execution_layout": "three_source_published_composite",
        "source_count": len(source_rows),
        "task_count": len(ordered_tasks),
        "source_bytes": sum(source["source_bytes"] for source in source_rows),
        "file_count": sum(source["file_count"] for source in source_rows),
        "sources": source_rows,
        "tasks": ordered_tasks,
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "caveats": [
            "official_protocol_with_known_overlap",
            "public_material_maximum_fidelity_reproduction",
            "task034_cross_host_single_task_recovery",
            "author_baseline_adapters_and_exact_configs_not_public",
        ],
    }
    if output_path is not None:
        _write_index(Path(output_path), _json_output(payload), source_roots)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and index the three published IGSTGNN Table 4 sources",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-spec", required=True, type=Path)
    parser.add_argument("--relay-base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-git-sha", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = build_table4_composite_index(
            manifest_path=args.manifest,
            source_spec_path=args.source_spec,
            relay_base=args.relay_base,
            output_path=args.output,
            control_git_sha=args.control_git_sha,
        )
    except CompositeValidationError as error:
        print(f"COMPOSITE_BLOCKED {error}")
        return 1
    print(
        json.dumps(
            {
                "status": payload["status"],
                "matrix_validity_status": payload["matrix_validity_status"],
                "source_count": payload["source_count"],
                "task_count": payload["task_count"],
                "file_count": payload["file_count"],
                "source_bytes": payload["source_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
