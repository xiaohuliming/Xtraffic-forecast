from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import compute_official_table4_metrics
from .secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
    secure_io_provenance,
    verify_secure_directory,
)


_PROTOCOL_ID = "igstgnn_table4_official_v1"
_ALLOWLIST = frozenset(
    {
        "checkpoint.pt",
        "predictions.npz",
        "provenance.json",
        "summary.json",
        "train.log",
    }
)
_TASK_KEYS = frozenset(
    {
        "task_id",
        "method",
        "region",
        "seed",
        "source_id",
        "profile_id",
        "config_sha256",
        "split_bundle_manifest_sha256",
        "output_dir",
    }
)
_SHA256_KEYS = ("config_sha256", "split_bundle_manifest_sha256")
_AT_FDCWD_DARWIN = -2
_AT_FDCWD_LINUX = -100
_RENAME_EXCL_DARWIN = 0x00000004
_RENAME_NOREPLACE_LINUX = 0x00000001


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_task(task: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(task)
    if set(value) != _TASK_KEYS:
        raise ValueError("Task identity does not match the matrix schema")
    for key in (
        "task_id",
        "method",
        "region",
        "source_id",
        "profile_id",
        "output_dir",
    ):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"Task {key} must be a non-empty string")
    if value["seed"] is not None and (
        type(value["seed"]) is not int or value["seed"] < 0
    ):
        raise ValueError("Task seed must be null or a non-negative integer")
    for key in _SHA256_KEYS:
        if not _is_sha256(value[key]):
            raise ValueError(f"Task {key} must be a lowercase SHA-256")
    return value


def _hash_regular(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open_verified_regular_file(
        path,
        purpose="Result artifact",
        require_nonempty=True,
    ) as opened:
        while True:
            chunk = opened.handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        size = opened.stat_result.st_size
    return size, digest.hexdigest()


def _copy_regular(source: Path, destination: Path) -> None:
    with open_verified_regular_file(
        source,
        purpose="Result artifact source",
        require_nonempty=True,
    ) as opened:
        with destination.open("xb") as destination_handle:
            shutil.copyfileobj(
                opened.handle,
                destination_handle,
                length=1024 * 1024,
            )
            destination_handle.flush()
            os.fsync(destination_handle.fileno())


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory_exclusive(staging: Path, target: Path) -> None:
    result: int | None = None
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        source = os.fsencode(staging)
        destination = os.fsencode(target)
        if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
            result = libc.renameatx_np(
                _AT_FDCWD_DARWIN,
                ctypes.c_char_p(source),
                _AT_FDCWD_DARWIN,
                ctypes.c_char_p(destination),
                _RENAME_EXCL_DARWIN,
            )
        elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            result = libc.renameat2(
                _AT_FDCWD_LINUX,
                ctypes.c_char_p(source),
                _AT_FDCWD_LINUX,
                ctypes.c_char_p(destination),
                _RENAME_NOREPLACE_LINUX,
            )
    if result is not None:
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"Task directory already exists: {target}")
        raise OSError(error_number, os.strerror(error_number), target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Task directory already exists: {target}")
    try:
        os.rename(staging, target)
    except FileExistsError as error:
        raise FileExistsError(f"Task directory already exists: {target}") from error


def _prepare_arrays(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction_array = np.asarray(predictions)
    label_array = np.asarray(labels)
    if prediction_array.shape != label_array.shape:
        raise ValueError(
            "Predictions and labels must have the same shape, got "
            f"{prediction_array.shape} and {label_array.shape}"
        )
    if prediction_array.dtype != np.dtype("float32") or label_array.dtype != np.dtype(
        "float32"
    ):
        raise ValueError("Predictions and labels must use float32")
    if prediction_array.ndim < 2 or prediction_array.shape[1] != 12:
        raise ValueError("Predictions and labels must contain 12 horizons")
    if prediction_array.size == 0:
        raise ValueError("Predictions and labels must be non-empty")
    if not np.isfinite(prediction_array).all() or not np.isfinite(label_array).all():
        raise ValueError("Predictions and labels must contain only finite values")
    null_value = (
        np.float32(label_array.min()) if label_array.min() < 1 else np.float32(0)
    )
    valid_mask = np.asarray(label_array != null_value, dtype=np.bool_)
    return prediction_array, label_array, valid_mask


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(provenance)
    if value.get("protocol_id") != _PROTOCOL_ID:
        raise ValueError("Provenance protocol_id is invalid")
    if not _is_sha256(value.get("training_code_sha256")):
        raise ValueError("Provenance training_code_sha256 is invalid")
    value["secure_io_boundary"] = secure_io_provenance()
    _json_bytes(value)
    return value


def _artifact_entry(path: Path) -> dict[str, Any]:
    size, digest = _hash_regular(path)
    return {"bytes": size, "sha256": digest}


def write_completed_task_artifacts(
    *,
    target_dir: Path | str,
    task: Mapping[str, Any],
    predictions: np.ndarray,
    labels: np.ndarray,
    checkpoint_source: Path | str,
    train_log_source: Path | str,
    provenance: Mapping[str, Any],
) -> Path:
    target = Path(target_dir)
    task_value = _validated_task(task)
    provenance_value = _validate_provenance(provenance)
    prediction_array, label_array, valid_mask = _prepare_arrays(
        predictions,
        labels,
    )
    metrics = compute_official_table4_metrics(
        prediction_array,
        label_array,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Task directory already exists: {target}")
    staging = target.parent / (f".{target.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        _copy_regular(Path(checkpoint_source), staging / "checkpoint.pt")
        _copy_regular(Path(train_log_source), staging / "train.log")
        _write_exclusive(
            staging / "provenance.json",
            _json_bytes(provenance_value),
        )
        with (staging / "predictions.npz").open("xb") as handle:
            np.savez_compressed(
                handle,
                predictions=prediction_array,
                labels=label_array,
                valid_mask=valid_mask,
            )
            handle.flush()
            os.fsync(handle.fileno())

        artifacts = {
            name: _artifact_entry(staging / filename)
            for name, filename in (
                ("checkpoint", "checkpoint.pt"),
                ("predictions", "predictions.npz"),
                ("provenance", "provenance.json"),
                ("train_log", "train.log"),
            )
        }
        summary = {
            "schema_version": 1,
            "status": "completed",
            "protocol_id": _PROTOCOL_ID,
            "validity_label": "official_protocol_with_known_overlap",
            "claim_scope": "public_material_maximum_fidelity_reproduction",
            "task": task_value,
            "prediction_shape": list(prediction_array.shape),
            "prediction_dtype": "float32",
            "label_dtype": "float32",
            "valid_count": int(valid_mask.sum()),
            "metrics": metrics,
            "artifact_allowlist": sorted(_ALLOWLIST),
            "artifacts": artifacts,
        }
        _write_exclusive(staging / "summary.json", _json_bytes(summary))
        validate_completed_task_artifacts(
            staging,
            expected_task=task_value,
        )
        _publish_directory_exclusive(staging, target)
        staging = Path()
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)
    return target


def _read_bounded_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    payload = read_verified_regular_file(
        path,
        purpose="JSON result artifact",
        max_bytes=max_bytes,
        require_nonempty=True,
    )
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _verify_entry(
    directory: Path,
    *,
    filename: str,
    entry: Any,
) -> None:
    if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
        raise ValueError(f"Invalid artifact entry: {filename}")
    if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
        raise ValueError(f"Invalid artifact size: {filename}")
    if not _is_sha256(entry["sha256"]):
        raise ValueError(f"Invalid artifact SHA-256: {filename}")
    size, digest = _hash_regular(directory / filename)
    if size != entry["bytes"] or digest != entry["sha256"]:
        raise ValueError(f"Artifact integrity mismatch: {filename}")


def validate_completed_task_artifacts(
    task_dir: Path | str,
    *,
    expected_task: Mapping[str, Any],
) -> dict[str, Any]:
    directory = verify_secure_directory(
        task_dir,
        purpose="Task artifact path",
    )
    names = {path.name for path in directory.iterdir()}
    if names != _ALLOWLIST:
        raise ValueError("Task artifact allowlist mismatch")

    task_value = _validated_task(expected_task)
    summary = _read_bounded_json(
        directory / "summary.json",
        max_bytes=1024 * 1024,
    )
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "completed"
        or summary.get("protocol_id") != _PROTOCOL_ID
        or summary.get("validity_label") != "official_protocol_with_known_overlap"
        or summary.get("claim_scope") != "public_material_maximum_fidelity_reproduction"
        or summary.get("task") != task_value
        or summary.get("artifact_allowlist") != sorted(_ALLOWLIST)
        or summary.get("prediction_dtype") != "float32"
        or summary.get("label_dtype") != "float32"
    ):
        raise ValueError("Task summary identity or status is invalid")
    artifacts = summary.get("artifacts")
    expected_entries = {
        "checkpoint": "checkpoint.pt",
        "predictions": "predictions.npz",
        "provenance": "provenance.json",
        "train_log": "train.log",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_entries):
        raise ValueError("Task summary artifact entries are invalid")
    for key, filename in expected_entries.items():
        _verify_entry(
            directory,
            filename=filename,
            entry=artifacts[key],
        )

    provenance = _read_bounded_json(
        directory / "provenance.json",
        max_bytes=1024 * 1024,
    )
    _validate_provenance(provenance)
    with open_verified_regular_file(
        directory / "predictions.npz",
        purpose="Prediction result artifact",
        require_nonempty=True,
    ) as opened:
        with np.load(opened.handle, allow_pickle=False) as archive:
            if set(archive.files) != {"predictions", "labels", "valid_mask"}:
                raise ValueError("Prediction artifact members are invalid")
            predictions = np.array(archive["predictions"], copy=True)
            labels = np.array(archive["labels"], copy=True)
            stored_valid_mask = np.array(archive["valid_mask"], copy=True)
    predictions, labels, expected_valid_mask = _prepare_arrays(
        predictions,
        labels,
    )
    if (
        stored_valid_mask.dtype != np.dtype("bool")
        or stored_valid_mask.shape != labels.shape
        or not np.array_equal(stored_valid_mask, expected_valid_mask)
    ):
        raise ValueError("Prediction valid mask is inconsistent")
    if summary.get("prediction_shape") != list(predictions.shape):
        raise ValueError("Prediction shape does not match the summary")
    if summary.get("valid_count") != int(expected_valid_mask.sum()):
        raise ValueError("Prediction valid count does not match the summary")
    metrics = compute_official_table4_metrics(predictions, labels)
    if summary.get("metrics") != metrics:
        raise ValueError("Recomputed metrics do not match the summary")
    return {
        "status": "completed_task_artifacts_verified",
        "task_id": task_value["task_id"],
        "metrics": metrics,
        "artifacts": artifacts,
    }
