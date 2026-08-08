#!/usr/bin/env python3
"""Read-only composite verification for the single-host v3 event 33-task run."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import uuid
from typing import Mapping, Sequence

from scripts.remote.windows import protocol_v3_runner as runner
from scripts.remote.windows.protocol_v3_baseline_runner import (
    build_baseline_invocation,
    load_baseline_manifest,
)


EXPECTED_STATUS = "verified_event_full33_single_host_33"
EXPECTED_PROTOCOL = "protocol_v3_noleak_dedup"
EXPECTED_TASKS = 33
EXPECTED_SOURCES = 4
HEX_40 = set("0123456789abcdef")
HEX_64 = HEX_40


class Full33CompositeError(RuntimeError):
    """Raised when the published full-33 tree cannot be accepted."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Full33CompositeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise Full33CompositeError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Full33CompositeError(f"{label} is missing or not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except Full33CompositeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Full33CompositeError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Full33CompositeError(f"{label} must contain a JSON object")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Full33CompositeError("value is not strict finite JSON") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_record(path: Path) -> tuple[str, int, tuple[int, int, int, int]]:
    if path.is_symlink():
        raise Full33CompositeError(f"symbolic link is forbidden: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise Full33CompositeError(f"not a regular file: {path}")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = os.lstat(path)
    except OSError as exc:
        raise Full33CompositeError(f"cannot hash file: {path}") from exc
    fingerprint = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    if fingerprint != (
        int(after.st_dev), int(after.st_ino), int(after.st_size), int(after.st_mtime_ns)
    ) or fingerprint != (
        int(current.st_dev), int(current.st_ino), int(current.st_size), int(current.st_mtime_ns)
    ):
        raise Full33CompositeError(f"file changed while hashing: {path}")
    return digest.hexdigest(), int(before.st_size), fingerprint


def _sha256_file(path: Path) -> str:
    return _file_record(path)[0]


def _require_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise Full33CompositeError(f"{name} must be a nonempty string")
    return value


def _require_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Full33CompositeError(f"{name} must be an integer")
    return value


def _require_hex(payload: Mapping[str, object], name: str, length: int) -> str:
    value = _require_string(payload, name).lower()
    if len(value) != length or any(character not in HEX_64 for character in value):
        raise Full33CompositeError(f"{name} must be a {length}-character hex digest")
    return value


def _safe_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Full33CompositeError(f"{label} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise Full33CompositeError(f"unsafe {label}: {value}")
    normalized = path.as_posix()
    if normalized != value or normalized in (".", ""):
        raise Full33CompositeError(f"non-canonical {label}: {value}")
    return value


def _inventory(root: Path) -> dict[str, tuple[str, int, tuple[int, int, int, int]]]:
    if root.is_symlink() or not root.is_dir():
        raise Full33CompositeError(f"result root is missing or a symlink: {root}")
    records: dict[str, tuple[str, int, tuple[int, int, int, int]]] = {}
    casefolded: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise Full33CompositeError(f"symbolic link is forbidden: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        key = relative.casefold()
        if key in casefolded:
            raise Full33CompositeError(f"case-insensitive path collision: {relative}")
        casefolded.add(key)
        records[relative] = _file_record(path)
    return records


def _parse_transfer_manifest(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if len(line) < 67 or line[64:66] != "  ":
            raise Full33CompositeError(
                f"invalid transfer manifest line {line_number}"
            )
        digest = line[:64].lower()
        if any(character not in HEX_64 for character in digest):
            raise Full33CompositeError(
                f"invalid transfer digest on line {line_number}"
            )
        records.append((digest, _safe_relative(line[66:], label="transfer path")))
    if not records or len({path for _, path in records}) != len(records):
        raise Full33CompositeError("transfer manifest is empty or has duplicates")
    return records


def _manifest_for_task(manifest, override: Mapping[str, int] | None):
    if not override:
        return manifest
    training = dict(manifest.training)
    training.update(override)
    return dataclasses.replace(manifest, training=training)


def _finite_metric(summary: Mapping[str, object], name: str) -> float:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Full33CompositeError(f"summary metric {name} is missing")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise Full33CompositeError(f"summary metric {name} is invalid")
    return number


def _metric_array(summary: Mapping[str, object], name: str) -> list[float]:
    value = summary.get(name)
    if not isinstance(value, list) or len(value) != 12:
        raise Full33CompositeError(f"summary metric {name} must have 12 values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise Full33CompositeError(f"summary metric {name} is invalid")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise Full33CompositeError(f"summary metric {name} is invalid")
        result.append(number)
    return result


def _atomic_create(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise Full33CompositeError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise Full33CompositeError(f"output appeared during validation: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_full33_composite_index(
    *,
    published_root: Path | str,
    output_path: Path | str,
    expected_matrix_id: str,
    expected_host_id: str,
) -> dict[str, object]:
    root = Path(published_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise Full33CompositeError("published root is missing or a symlink")
    root = root.resolve(strict=True)
    results_root = root / "results"
    metadata_root = root / "metadata"
    if metadata_root.is_symlink() or not metadata_root.is_dir():
        raise Full33CompositeError("metadata root is missing or a symlink")
    report_path = metadata_root / "postflight.json"
    marker_path = metadata_root / "_TRANSFER_COMPLETE.json"
    transfer_path = metadata_root / "transfer.sha256"
    file_list_path = metadata_root / "transfer.files"
    report = _load_json(report_path, label="source-host postflight")
    marker = _load_json(marker_path, label="transfer marker")

    if report.get("status") != "verified":
        raise Full33CompositeError("source-host postflight is not verified")
    identity = {
        "matrix_id": expected_matrix_id,
        "host_id": expected_host_id,
        "protocol_version": EXPECTED_PROTOCOL,
        "validated_tasks": EXPECTED_TASKS,
        "source_layout": "interleaved",
    }
    for name, expected in identity.items():
        if report.get(name) != expected:
            raise Full33CompositeError(
                f"{name} mismatch: expected {expected!r}, got {report.get(name)!r}"
            )
    if _require_int(report, "cache_schema_version") != 3:
        raise Full33CompositeError("cache_schema_version must be 3")
    sources = report.get("sources")
    tasks = report.get("tasks")
    files = report.get("files")
    if not isinstance(sources, list) or len(sources) != EXPECTED_SOURCES:
        raise Full33CompositeError("source-host report must contain four GPU sources")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS:
        raise Full33CompositeError("source-host report must contain 33 tasks")
    if not isinstance(files, list) or not files:
        raise Full33CompositeError("source-host report has no files")
    if report.get("files_digest") != _sha256_json(files):
        raise Full33CompositeError("source-host files_digest mismatch")

    manifest_source = Path(_require_string(report, "canonical_manifest"))
    manifest_name = manifest_source.name
    plan_source = Path(_require_string(report, "plan_path"))
    plan_name = plan_source.name
    expected_metadata = {
        "_TRANSFER_COMPLETE.json",
        manifest_name,
        plan_name,
        "postflight.json",
        "transfer.files",
        "transfer.sha256",
    }
    actual_metadata = {path.name for path in metadata_root.iterdir()}
    if actual_metadata != expected_metadata:
        raise Full33CompositeError("metadata allowlist mismatch")
    manifest_path = metadata_root / manifest_name
    plan_path = metadata_root / plan_name
    if _sha256_file(manifest_path) != _require_hex(
        report, "canonical_manifest_sha256", 64
    ):
        raise Full33CompositeError("canonical manifest SHA-256 mismatch")
    if _sha256_file(plan_path) != _require_hex(report, "plan_sha256", 64):
        raise Full33CompositeError("NAS plan SHA-256 mismatch")
    plan = _load_json(plan_path, label="NAS backup plan")
    for name, expected in (
        ("matrix_id", expected_matrix_id),
        ("host_id", expected_host_id),
        ("protocol_version", EXPECTED_PROTOCOL),
        ("expected_task_count", EXPECTED_TASKS),
        ("source_layout", "interleaved"),
    ):
        if plan.get(name) != expected:
            raise Full33CompositeError(f"plan {name} mismatch")

    marker_expected = {
        "status": "verified",
        "matrix_id": expected_matrix_id,
        "host_id": expected_host_id,
        "validated_tasks": EXPECTED_TASKS,
        "file_count": report.get("file_count"),
        "total_bytes": report.get("total_bytes"),
        "files_digest": report.get("files_digest"),
        "plan_sha256": report.get("plan_sha256"),
        "postflight_sha256": _sha256_file(report_path),
        "file_list_sha256": _sha256_file(file_list_path),
        "transfer_manifest_sha256": _sha256_file(transfer_path),
    }
    for name, expected in marker_expected.items():
        if marker.get(name) != expected:
            raise Full33CompositeError(f"transfer marker {name} mismatch")

    transfer_records = _parse_transfer_manifest(transfer_path)
    transfer_paths = [path for _, path in transfer_records]
    listed_paths = file_list_path.read_text(encoding="utf-8").splitlines()
    report_paths = [_safe_relative(item.get("path"), label="report file path") for item in files if isinstance(item, dict)]
    if len(report_paths) != len(files):
        raise Full33CompositeError("invalid file record in source-host report")
    if transfer_paths != listed_paths or transfer_paths != report_paths:
        raise Full33CompositeError("transfer file lists do not match")

    inventory_before = _inventory(results_root)
    if set(inventory_before) != set(transfer_paths):
        raise Full33CompositeError("published result allowlist mismatch")
    report_by_path = {str(item["path"]): item for item in files if isinstance(item, dict)}
    for digest, relative in transfer_records:
        observed_digest, observed_size, _ = inventory_before[relative]
        record = report_by_path[relative]
        if (
            observed_digest != digest
            or digest != record.get("sha256")
            or observed_size != record.get("size_bytes")
        ):
            raise Full33CompositeError(f"result hash or size mismatch: {relative}")

    manifest = load_baseline_manifest(manifest_path)
    if len(manifest.tasks) != EXPECTED_TASKS:
        raise Full33CompositeError("canonical manifest is not the 33-task matrix")
    if manifest.protocol_version != EXPECTED_PROTOCOL:
        raise Full33CompositeError("canonical protocol mismatch")
    proof_by_task: dict[str, Mapping[str, object]] = {}
    source_by_task: dict[str, str] = {}
    for item in tasks:
        if not isinstance(item, dict):
            raise Full33CompositeError("invalid task proof")
        task_id = _require_string(item, "task_id")
        if task_id in proof_by_task:
            raise Full33CompositeError(f"duplicate task proof: {task_id}")
        proof_by_task[task_id] = item
        source_by_task[task_id] = _require_string(item, "source_id")
    canonical_ids = [task.task_id for task in manifest.tasks]
    if set(proof_by_task) != set(canonical_ids):
        raise Full33CompositeError("task proofs do not equal the canonical 33 tasks")

    source_reports: dict[str, Mapping[str, object]] = {}
    for item in sources:
        if not isinstance(item, dict):
            raise Full33CompositeError("invalid source record")
        source_id = _require_string(item, "source_id")
        if source_id in source_reports:
            raise Full33CompositeError(f"duplicate source: {source_id}")
        source_reports[source_id] = item
    if sorted(source_reports) != ["gpu00", "gpu01", "gpu02", "gpu03"]:
        raise Full33CompositeError("GPU source identities must be gpu00 through gpu03")

    cache_bundle_id = _require_string(report, "cache_bundle_id")
    cache_bundle_digest = _require_hex(report, "cache_bundle_digest", 64)
    cache_dir = Path(_require_string(plan, "cache_dir"))
    graph_dir = Path(_require_string(plan, "graph_dir"))
    python_executable = _require_string(plan, "python_executable")
    overrides = report.get("task_overrides")
    if not isinstance(overrides, dict):
        raise Full33CompositeError("task_overrides must be an object")

    task_records: list[dict[str, object]] = []
    code_shas: set[str] = set()
    for task in manifest.tasks:
        proof = proof_by_task[task.task_id]
        source_id = source_by_task[task.task_id]
        source_report = source_reports.get(source_id)
        if source_report is None:
            raise Full33CompositeError(f"unknown task source: {source_id}")
        source_task_ids = source_report.get("task_ids")
        if not isinstance(source_task_ids, list) or task.task_id not in source_task_ids:
            raise Full33CompositeError(f"task is absent from source record: {task.task_id}")
        code_sha = _require_hex(source_report, "code_sha", 40)
        code_shas.add(code_sha)
        local_source_root = results_root / "sources" / source_id / "results"
        run_inputs = runner.ProtocolV3Inputs(
            results_root=local_source_root,
            cache_dir=cache_dir,
            graph_dir=graph_dir,
            cache_bundle_id=cache_bundle_id,
            cache_bundle_digest=cache_bundle_digest,
            code_sha=code_sha,
            python_executable=python_executable,
        )
        raw_override = overrides.get(task.task_id)
        if raw_override is not None and not isinstance(raw_override, dict):
            raise Full33CompositeError(f"invalid task override: {task.task_id}")
        task_manifest = _manifest_for_task(manifest, raw_override)
        invocation = build_baseline_invocation(
            task,
            task_manifest,
            run_inputs,
            local_source_root / ".composite_probe" / task.task_id,
        )
        config_hash = runner.derive_config_hash(
            task,
            task_manifest,
            run_inputs,
            invocation_config=invocation.config,
        )
        if proof.get("config_hash") != config_hash:
            raise Full33CompositeError(f"config_hash mismatch: {task.task_id}")
        if proof.get("invocation_config") != dict(invocation.config):
            raise Full33CompositeError(f"invocation_config mismatch: {task.task_id}")
        result_dir = local_source_root / task.output_dir
        try:
            runner.validate_result(
                result_dir,
                task,
                task_manifest,
                run_inputs,
                expected_config_hash=config_hash,
                required_artifacts=invocation.required_artifacts,
                required_metrics=invocation.required_metrics,
            )
        except runner.ResultValidationError as exc:
            raise Full33CompositeError(f"{task.task_id}: {exc}") from exc
        summary_path = result_dir / "summary.json"
        summary = _load_json(summary_path, label=f"{task.task_id} summary")
        mae_h = _metric_array(summary, "mae_h")
        rmse_h = _metric_array(summary, "rmse_h")
        mape_h = _metric_array(summary, "mape_h")
        if [summary.get("h3"), summary.get("h6"), summary.get("h12")] != [
            mae_h[2], mae_h[5], mae_h[11]
        ]:
            raise Full33CompositeError(f"horizon MAE fields mismatch: {task.task_id}")
        relative_result_dir = result_dir.relative_to(root).as_posix()
        artifacts = {
            name: {
                "path": (result_dir / name).relative_to(root).as_posix(),
                "sha256": _sha256_file(result_dir / name),
            }
            for name in invocation.required_artifacts
        }
        task_records.append(
            {
                "task_id": task.task_id,
                "source_id": source_id,
                "region": task.region,
                "model": task.model,
                "variant": task.variant,
                "seed": task.seed,
                "output_dir": str(task.output_dir),
                "published_result_dir": relative_result_dir,
                "config_hash": config_hash,
                "summary_path": summary_path.relative_to(root).as_posix(),
                "summary_sha256": _sha256_file(summary_path),
                "artifacts": artifacts,
                "metrics": {
                    "h3": {"mae": mae_h[2], "rmse": rmse_h[2], "mape_percent": mape_h[2]},
                    "h6": {"mae": mae_h[5], "rmse": rmse_h[5], "mape_percent": mape_h[5]},
                    "h12": {"mae": mae_h[11], "rmse": rmse_h[11], "mape_percent": mape_h[11]},
                    "average": {
                        "mae": _finite_metric(summary, "all"),
                        "rmse": _finite_metric(summary, "rmse"),
                        "mape_percent": _finite_metric(summary, "mape"),
                    },
                },
            }
        )

    if len(code_shas) != 1:
        raise Full33CompositeError("all GPU sources must use one code SHA")
    inventory_after = _inventory(results_root)
    if {
        path: (value[0], value[1]) for path, value in inventory_after.items()
    } != {
        path: (value[0], value[1]) for path, value in inventory_before.items()
    }:
        raise Full33CompositeError("published result tree changed during validation")

    output = Path(output_path).expanduser().resolve(strict=False)
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise Full33CompositeError("composite index output must be outside published root")
    payload: dict[str, object] = {
        "index_schema_version": 1,
        "status": EXPECTED_STATUS,
        "execution_layout": "single_host_four_gpu_interleaved",
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matrix_id": expected_matrix_id,
        "host_id": expected_host_id,
        "protocol_version": EXPECTED_PROTOCOL,
        "cache_schema_version": 3,
        "cache_bundle_id": cache_bundle_id,
        "cache_bundle_digest": cache_bundle_digest,
        "code_sha": next(iter(code_shas)),
        "canonical_manifest_sha256": _sha256_file(manifest_path),
        "source_postflight_sha256": _sha256_file(report_path),
        "source_files_digest": report["files_digest"],
        "source_count": len(sources),
        "task_count": len(task_records),
        "published_root": str(root),
        "tasks": task_records,
    }
    _atomic_create(output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-matrix-id", required=True)
    parser.add_argument("--expected-host-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_full33_composite_index(
            published_root=args.published_root,
            output_path=args.output,
            expected_matrix_id=args.expected_matrix_id,
            expected_host_id=args.expected_host_id,
        )
    except (Full33CompositeError, OSError, runner.ManifestError) as exc:
        print(f"EVENT_FULL33_COMPOSITE_FAILED {exc}", file=sys.stderr)
        return 1
    print(
        "EVENT_FULL33_COMPOSITE_OK "
        f"tasks={payload['task_count']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
