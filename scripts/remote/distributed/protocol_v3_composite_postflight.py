#!/usr/bin/env python3
"""Verify two source-attested protocol-v3 shards and index them read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


def _discover_repo_root() -> Path:
    relative = Path("scripts/remote/windows/protocol_v3_runner.py")
    configured = os.environ.get("XTRAFFIC_REPO_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not (candidate / relative).is_file():
            raise RuntimeError(
                "XTRAFFIC_REPO_ROOT does not contain "
                f"{relative.as_posix()}: {candidate}"
            )
        return candidate
    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / relative).is_file():
            return candidate
    raise RuntimeError(
        "cannot locate the XTraffic repository; set XTRAFFIC_REPO_ROOT"
    )


REPO_ROOT = _discover_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.remote.windows import protocol_v3_runner as runner  # noqa: E402
from scripts.remote.distributed import (  # noqa: E402
    protocol_v3_distributed as distributed,
)


INDEX_SCHEMA_VERSION = 1
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RUN_INPUT_FIELDS = {
    "results_root",
    "cache_dir",
    "graph_dir",
    "cache_bundle_id",
    "cache_bundle_digest",
    "code_sha",
    "python_executable",
}
RUNTIME_FIELDS = {
    "hostname",
    "platform",
    "python",
    "python_executable",
    "requested_python_executable",
    "torch",
    "torch_cuda",
    "numpy",
    "h5py",
    "pandas",
    "gpu",
}


class CompositeValidationError(RuntimeError):
    """Raised when two shard attestations cannot form a verified matrix."""


@dataclass(frozen=True)
class FileRecord:
    sha256: str
    size_bytes: int
    fingerprint: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ExpectedFile:
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class TaskProof:
    config_hash: str
    invocation_config: Mapping[str, object]


@dataclass(frozen=True)
class ShardReport:
    source_path: Path
    source_sha256: str
    source_fingerprint: tuple[int, int, int, int, int]
    shard: str
    host: str
    local_results_root: Path
    source_run_inputs: Mapping[str, str]
    run_inputs: runner.ProtocolV3Inputs
    task_ids: tuple[str, ...]
    task_proofs: Mapping[str, TaskProof]
    runtime_provenance: Mapping[str, object]
    runtime_provenance_sha256: str
    expected_files: Mapping[str, ExpectedFile]
    files_digest: str
    source_manifest_raw_sha256: str
    source_manifest_canonical_json_sha256: str


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CompositeValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise CompositeValidationError(f"non-finite JSON constant {value}")


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except CompositeValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompositeValidationError(
            f"cannot read strict {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CompositeValidationError(
            f"{label} JSON must contain an object: {path}"
        )
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
        raise CompositeValidationError(
            f"value is not strict finite JSON: {exc}"
        ) from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositeValidationError(
            f"{label} must be a nonempty string"
        )
    if any(ord(character) < 32 for character in value):
        raise CompositeValidationError(
            f"{label} must not contain control characters"
        )
    return value


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CompositeValidationError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _hash_regular_file(path: Path) -> FileRecord:
    if path.is_symlink():
        raise CompositeValidationError(f"file is a symlink: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise CompositeValidationError(
                    f"path is not a regular file: {path}"
                )
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        current = os.lstat(path)
    except OSError as exc:
        raise CompositeValidationError(
            f"cannot hash file {path}: {exc}"
        ) from exc
    fingerprint = _stat_fingerprint(before)
    if (
        _stat_fingerprint(after) != fingerprint
        or _stat_fingerprint(current) != fingerprint
    ):
        raise CompositeValidationError(
            f"file changed while it was being hashed: {path}"
        )
    return FileRecord(
        sha256=digest.hexdigest(),
        size_bytes=int(before.st_size),
        fingerprint=fingerprint,
    )


def _safe_relative_path(value: object, *, label: str) -> str:
    text = _require_string(value, label=label)
    if "\\" in text:
        raise CompositeValidationError(
            f"{label} must use POSIX separators: {text!r}"
        )
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise CompositeValidationError(
            f"{label} is not a safe relative path: {text!r}"
        )
    return pure.as_posix()


def _parse_runtime(value: object) -> tuple[str, dict[str, object]]:
    if not isinstance(value, dict):
        raise CompositeValidationError(
            "source postflight runtime must be a JSON object"
        )
    missing = RUNTIME_FIELDS - set(value)
    if missing:
        raise CompositeValidationError(
            f"source postflight runtime is missing fields: {sorted(missing)}"
        )
    for field in RUNTIME_FIELDS - {"gpu"}:
        _require_string(value.get(field), label=f"runtime.{field}")
    gpu = value.get("gpu")
    if (
        not isinstance(gpu, list)
        or not gpu
        or any(not isinstance(item, str) or not item.strip() for item in gpu)
    ):
        raise CompositeValidationError(
            "runtime.gpu must be a nonempty array of nonempty strings"
        )
    runtime = dict(value)
    _canonical_json(runtime)
    host = _require_string(runtime["hostname"], label="runtime.hostname")
    if not NAME_RE.fullmatch(host):
        raise CompositeValidationError(
            f"runtime.hostname contains unsupported characters: {host!r}"
        )
    return host, runtime


def _parse_run_inputs(
    value: object,
    *,
    local_results_root: Path,
) -> tuple[dict[str, str], runner.ProtocolV3Inputs]:
    if not isinstance(value, dict) or set(value) != RUN_INPUT_FIELDS:
        actual = sorted(value) if isinstance(value, dict) else None
        raise CompositeValidationError(
            "source postflight run_inputs must contain exactly "
            f"{sorted(RUN_INPUT_FIELDS)}, got {actual!r}"
        )
    raw = {
        field: _require_string(
            value[field],
            label=f"run_inputs.{field}",
        )
        for field in RUN_INPUT_FIELDS
    }
    if not HEX_64_RE.fullmatch(raw["cache_bundle_digest"]):
        raise CompositeValidationError(
            "run_inputs.cache_bundle_digest must be a lowercase SHA-256"
        )
    if not HEX_40_RE.fullmatch(raw["code_sha"]):
        raise CompositeValidationError(
            "run_inputs.code_sha must be a lowercase 40-character Git SHA"
        )
    return raw, runner.ProtocolV3Inputs(
        results_root=local_results_root,
        cache_dir=Path(raw["cache_dir"]),
        graph_dir=Path(raw["graph_dir"]),
        cache_bundle_id=raw["cache_bundle_id"],
        cache_bundle_digest=raw["cache_bundle_digest"],
        code_sha=raw["code_sha"],
        python_executable=raw["python_executable"],
    )


def _parse_task_proofs(
    value: object,
) -> tuple[tuple[str, ...], dict[str, TaskProof]]:
    if not isinstance(value, list) or not value:
        raise CompositeValidationError(
            "source postflight tasks must be a nonempty array"
        )
    task_ids: list[str] = []
    proofs: dict[str, TaskProof] = {}
    casefold_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise CompositeValidationError(
                f"source postflight tasks[{index}] must be an object"
            )
        task_id = _require_string(
            item.get("task_id"),
            label=f"tasks[{index}].task_id",
        )
        key = task_id.casefold()
        if key in casefold_ids:
            raise CompositeValidationError(
                f"source postflight contains duplicate task {task_id!r}"
            )
        casefold_ids.add(key)
        config_hash = item.get("config_hash")
        if (
            not isinstance(config_hash, str)
            or not HEX_64_RE.fullmatch(config_hash)
        ):
            raise CompositeValidationError(
                f"tasks[{index}].config_hash must be a lowercase SHA-256"
            )
        invocation_config = item.get("invocation_config")
        if not isinstance(invocation_config, dict):
            raise CompositeValidationError(
                f"tasks[{index}].invocation_config must be an object"
            )
        _canonical_json(invocation_config)
        task_ids.append(task_id)
        proofs[task_id] = TaskProof(
            config_hash=config_hash,
            invocation_config=dict(invocation_config),
        )
    return tuple(task_ids), proofs


def _parse_expected_files(
    payload: Mapping[str, object],
) -> tuple[dict[str, ExpectedFile], str]:
    value = payload.get("files")
    if not isinstance(value, list) or not value:
        raise CompositeValidationError(
            "source postflight files must be a nonempty array"
        )
    records: dict[str, ExpectedFile] = {}
    casefold_paths: set[str] = set()
    ordered_paths: list[str] = []
    total_bytes = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise CompositeValidationError(
                f"source postflight files[{index}] must be an object"
            )
        path = _safe_relative_path(
            item.get("path"),
            label=f"files[{index}].path",
        )
        key = path.casefold()
        if key in casefold_paths:
            raise CompositeValidationError(
                f"source postflight has a duplicate file path: {path}"
            )
        casefold_paths.add(key)
        size = _require_int(
            item.get("size_bytes"),
            label=f"files[{index}].size_bytes",
        )
        sha256 = item.get("sha256")
        if (
            not isinstance(sha256, str)
            or not HEX_64_RE.fullmatch(sha256)
        ):
            raise CompositeValidationError(
                f"files[{index}].sha256 must be a lowercase SHA-256"
            )
        records[path] = ExpectedFile(sha256=sha256, size_bytes=size)
        ordered_paths.append(path)
        total_bytes += size
    if ordered_paths != sorted(ordered_paths):
        raise CompositeValidationError(
            "source postflight files must be sorted by path"
        )
    if payload.get("file_count") != len(records):
        raise CompositeValidationError(
            "source postflight file_count does not match files"
        )
    if payload.get("total_bytes") != total_bytes:
        raise CompositeValidationError(
            "source postflight total_bytes does not match files"
        )
    export_paths = payload.get("export_paths")
    if export_paths != ordered_paths:
        raise CompositeValidationError(
            "source postflight export_paths does not match files"
        )
    digest = _sha256_json(value)
    if payload.get("files_digest") != digest:
        raise CompositeValidationError(
            "source postflight files_digest does not match files"
        )
    return records, digest


def load_shard_report(
    path: Path | str,
    local_results_root: Path | str,
) -> ShardReport:
    source_input = Path(path)
    if source_input.is_symlink():
        raise CompositeValidationError(
            f"source postflight report must not be a symlink: {source_input}"
        )
    source_path = source_input.resolve(strict=True)
    source_before = _hash_regular_file(source_path)
    payload = _read_json_object(source_path, label="source postflight")
    source_after = _hash_regular_file(source_path)
    if source_after != source_before:
        raise CompositeValidationError(
            f"source postflight changed while it was read: {source_path}"
        )
    if payload.get("status") != "verified":
        raise CompositeValidationError(
            f"source postflight status is not verified: {source_path}"
        )
    shard = _require_string(payload.get("shard"), label="shard")
    if not NAME_RE.fullmatch(shard):
        raise CompositeValidationError(
            f"shard contains unsupported characters: {shard!r}"
        )

    root_input = Path(local_results_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise CompositeValidationError(
            "local results root is missing, not a directory, or a symlink: "
            f"{root_input}"
        )
    root = root_input.resolve(strict=True)
    raw_inputs, run_inputs = _parse_run_inputs(
        payload.get("run_inputs"),
        local_results_root=root,
    )
    identity = {
        "results_root": raw_inputs["results_root"],
        "cache_bundle_id": raw_inputs["cache_bundle_id"],
        "cache_bundle_digest": raw_inputs["cache_bundle_digest"],
        "code_sha": raw_inputs["code_sha"],
    }
    for field, expected in identity.items():
        if payload.get(field) != expected:
            raise CompositeValidationError(
                f"source postflight {field} mismatch: "
                f"expected {expected!r}, got {payload.get(field)!r}"
            )

    host, runtime = _parse_runtime(payload.get("runtime"))
    if (
        runtime["requested_python_executable"]
        != raw_inputs["python_executable"]
    ):
        raise CompositeValidationError(
            "runtime requested_python_executable does not match run_inputs"
        )
    task_ids, task_proofs = _parse_task_proofs(payload.get("tasks"))
    if payload.get("validated_tasks") != len(task_ids):
        raise CompositeValidationError(
            "source postflight validated_tasks does not match tasks"
        )
    expected_files, files_digest = _parse_expected_files(payload)
    source_manifest_raw_sha256 = payload.get("manifest_raw_sha256")
    source_manifest_canonical_json_sha256 = payload.get(
        "manifest_canonical_json_sha256"
    )
    if (
        not isinstance(source_manifest_raw_sha256, str)
        or not HEX_64_RE.fullmatch(source_manifest_raw_sha256)
    ):
        raise CompositeValidationError(
            "source postflight manifest_raw_sha256 must be a SHA-256"
        )
    if (
        not isinstance(source_manifest_canonical_json_sha256, str)
        or not HEX_64_RE.fullmatch(
            source_manifest_canonical_json_sha256
        )
    ):
        raise CompositeValidationError(
            "source postflight manifest_canonical_json_sha256 "
            "must be a SHA-256"
        )
    return ShardReport(
        source_path=source_path,
        source_sha256=source_before.sha256,
        source_fingerprint=source_before.fingerprint,
        shard=shard,
        host=host,
        local_results_root=root,
        source_run_inputs=raw_inputs,
        run_inputs=run_inputs,
        task_ids=task_ids,
        task_proofs=task_proofs,
        runtime_provenance=runtime,
        runtime_provenance_sha256=_sha256_json(runtime),
        expected_files=expected_files,
        files_digest=files_digest,
        source_manifest_raw_sha256=source_manifest_raw_sha256,
        source_manifest_canonical_json_sha256=(
            source_manifest_canonical_json_sha256
        ),
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _verify_shards(
    manifest: runner.Manifest,
    shards: Sequence[ShardReport],
) -> dict[str, ShardReport]:
    if len(shards) != 2:
        raise CompositeValidationError(
            f"exactly two shard reports are required, got {len(shards)}"
        )
    expected_shard_names = set(distributed.SHARD_SPECS)
    actual_shard_names = {item.shard for item in shards}
    if actual_shard_names != expected_shard_names:
        raise CompositeValidationError(
            "shard names must equal the fixed distributed contract "
            f"{sorted(expected_shard_names)}; got "
            f"{sorted(actual_shard_names)}"
        )
    if len({item.host.casefold() for item in shards}) != 2:
        raise CompositeValidationError("source hosts must be distinct")
    first, second = (item.local_results_root for item in shards)
    if (
        first == second
        or _is_within(first, second)
        or _is_within(second, first)
    ):
        raise CompositeValidationError(
            "local results roots must be distinct and non-nested"
        )

    expected = {task.task_id for task in manifest.tasks}
    assignment: dict[str, ShardReport] = {}
    overlaps: set[str] = set()
    unknown: set[str] = set()
    for shard in shards:
        if shard.source_run_inputs["cache_bundle_id"] != shards[0].source_run_inputs[
            "cache_bundle_id"
        ]:
            raise CompositeValidationError(
                "source postflight cache_bundle_id mismatch"
            )
        if (
            shard.source_run_inputs["cache_bundle_digest"]
            != shards[0].source_run_inputs["cache_bundle_digest"]
        ):
            raise CompositeValidationError(
                "source postflight cache_bundle_digest mismatch"
            )
        if shard.source_run_inputs["code_sha"] != shards[0].source_run_inputs[
            "code_sha"
        ]:
            raise CompositeValidationError(
                "source postflight code_sha mismatch"
            )
        for task_id in shard.task_ids:
            if task_id not in expected:
                unknown.add(task_id)
            if task_id in assignment:
                overlaps.add(task_id)
            else:
                assignment[task_id] = shard
    missing = expected - set(assignment)
    if overlaps or unknown or missing or len(assignment) != 45:
        raise CompositeValidationError(
            "source shard assignments must be a disjoint union of the "
            f"canonical 45 tasks: overlaps={sorted(overlaps)}, "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}, "
            f"assigned={len(assignment)}"
        )

    canonical_by_number = {
        distributed.task_number(task): task.task_id
        for task in manifest.tasks
    }
    for shard in shards:
        expected_ids = {
            canonical_by_number[number]
            for number in distributed.SHARD_SPECS[
                shard.shard
            ].task_numbers
        }
        actual_ids = set(shard.task_ids)
        if actual_ids != expected_ids:
            raise CompositeValidationError(
                f"shard {shard.shard} differs from its fixed task "
                "assignment: "
                f"missing={sorted(expected_ids - actual_ids)}, "
                f"extra={sorted(actual_ids - expected_ids)}"
            )
    return assignment


def _verify_report_identity(
    payload: Mapping[str, object],
    *,
    manifest: runner.Manifest,
    manifest_canonical_json_sha256: str,
    report_path: Path,
) -> None:
    expected = {
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "manifest_canonical_json_sha256": (
            manifest_canonical_json_sha256
        ),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise CompositeValidationError(
                f"source postflight {field} mismatch in {report_path}: "
                f"expected {value!r}, got {payload.get(field)!r}"
            )


def _verify_task_root(shard: ShardReport) -> None:
    task_root = shard.local_results_root / "tasks"
    if task_root.is_symlink() or not task_root.is_dir():
        raise CompositeValidationError(
            f"{shard.shard} task root is missing or a symlink: {task_root}"
        )
    actual: set[str] = set()
    for entry in task_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            raise CompositeValidationError(
                f"{shard.shard} has an invalid task entry: {entry}"
            )
        actual.add(entry.name)
    expected = set(shard.task_ids)
    if actual != expected:
        raise CompositeValidationError(
            f"{shard.shard} task directories mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _inventory_tree(root: Path) -> dict[str, FileRecord]:
    records: dict[str, FileRecord] = {}
    casefold_paths: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise CompositeValidationError(
                    f"result tree contains a symlink: {child}"
                )
            if child.is_dir():
                pending.append(child)
                continue
            if not child.is_file():
                raise CompositeValidationError(
                    f"result tree contains a non-regular entry: {child}"
                )
            relative = child.relative_to(root).as_posix()
            key = relative.casefold()
            if key in casefold_paths:
                raise CompositeValidationError(
                    "result tree has a case-insensitive path collision: "
                    f"{relative}"
                )
            casefold_paths.add(key)
            records[relative] = _hash_regular_file(child)
    return dict(sorted(records.items()))


def _verify_transferred_files(
    shard: ShardReport,
) -> dict[str, FileRecord]:
    actual = _inventory_tree(shard.local_results_root)
    expected_paths = set(shard.expected_files)
    actual_paths = set(actual)
    if actual_paths != expected_paths:
        raise CompositeValidationError(
            f"{shard.shard} transferred file set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    for path, expected in shard.expected_files.items():
        observed = actual[path]
        if (
            observed.size_bytes != expected.size_bytes
            or observed.sha256 != expected.sha256
        ):
            raise CompositeValidationError(
                f"{shard.shard} transferred file hash mismatch: {path}"
            )
    return actual


def _verify_inventory_unchanged(
    root: Path,
    records: Mapping[str, FileRecord],
) -> None:
    current_paths: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        for child in directory.iterdir():
            if child.is_symlink():
                raise CompositeValidationError(
                    f"result tree gained a symlink: {child}"
                )
            if child.is_dir():
                pending.append(child)
                continue
            if not child.is_file():
                raise CompositeValidationError(
                    f"result tree gained a non-regular entry: {child}"
                )
            relative = child.relative_to(root).as_posix()
            current_paths.add(relative)
            expected = records.get(relative)
            if expected is None:
                raise CompositeValidationError(
                    f"result tree gained a file: {child}"
                )
            if _stat_fingerprint(os.lstat(child)) != expected.fingerprint:
                raise CompositeValidationError(
                    f"result file changed during validation: {child}"
                )
    missing = set(records) - current_paths
    if missing:
        raise CompositeValidationError(
            f"result tree lost files during validation: {sorted(missing)}"
        )


def _expected_invocation_config(
    task: runner.Task,
    manifest: runner.Manifest,
    shard: ShardReport,
    invocation: runner.Invocation,
) -> dict[str, object]:
    expected = dict(invocation.config)
    expected["cache_dir"] = shard.source_run_inputs["cache_dir"]
    expected["graph_dir"] = shard.source_run_inputs["graph_dir"]
    return expected


def _validate_task(
    task: runner.Task,
    manifest: runner.Manifest,
    shard: ShardReport,
    inventory: Mapping[str, FileRecord],
) -> dict[str, object]:
    proof = shard.task_proofs[task.task_id]
    result_dir = shard.local_results_root / task.output_dir
    summary_path = result_dir / "summary.json"
    _read_json_object(summary_path, label=f"{task.task_id} summary")
    invocation = runner.build_production_invocation(
        task,
        manifest,
        shard.run_inputs,
        shard.local_results_root / ".composite_probe" / task.task_id,
    )
    expected_config = _expected_invocation_config(
        task,
        manifest,
        shard,
        invocation,
    )
    if _canonical_json(proof.invocation_config) != _canonical_json(
        expected_config
    ):
        raise CompositeValidationError(
            f"source invocation_config mismatch for {task.task_id}"
        )
    derived_hash = runner.derive_config_hash(
        task,
        manifest,
        shard.run_inputs,
        invocation_config=proof.invocation_config,
    )
    if derived_hash != proof.config_hash:
        raise CompositeValidationError(
            f"source config_hash derivation mismatch for {task.task_id}"
        )
    try:
        runner.validate_result(
            result_dir,
            task,
            manifest,
            shard.run_inputs,
            expected_config_hash=derived_hash,
            required_artifacts=invocation.required_artifacts,
            required_metrics=invocation.required_metrics,
        )
    except (runner.ManifestError, runner.ResultValidationError, OSError) as exc:
        raise CompositeValidationError(
            f"{shard.shard} result validation failed for "
            f"{task.task_id}: {exc}"
        ) from exc

    prefix = task.output_dir.as_posix() + "/"
    artifacts = {
        path[len(prefix) :]: record
        for path, record in inventory.items()
        if path.startswith(prefix)
    }
    summary = artifacts.get("summary.json")
    if summary is None:
        raise CompositeValidationError(
            f"summary.json is absent from source file proof: {task.task_id}"
        )
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "sampling_protocol": task.sampling_protocol,
        "shard": shard.shard,
        "host": shard.host,
        "runtime_provenance": dict(shard.runtime_provenance),
        "runtime_provenance_sha256": shard.runtime_provenance_sha256,
        "source_run_inputs": dict(shard.source_run_inputs),
        "source_postflight_sha256": shard.source_sha256,
        "config_hash": derived_hash,
        "summary": {
            "path": (task.output_dir / "summary.json").as_posix(),
            "sha256": summary.sha256,
            "size_bytes": summary.size_bytes,
        },
        "artifacts": {
            path: {
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
            }
            for path, record in sorted(artifacts.items())
        },
    }


def _atomic_create_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CompositeValidationError(
            f"output already exists and will not be overwritten: {path}"
        )
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise CompositeValidationError(
            f"output appeared during validation and was not overwritten: "
            f"{path}"
        ) from exc
    except OSError as exc:
        raise CompositeValidationError(
            f"cannot create composite index {path}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_composite_index(
    *,
    manifest_path: Path | str,
    shard_report_paths: Sequence[Path | str],
    results_roots: Sequence[Path | str],
    output_path: Path | str,
) -> dict[str, object]:
    if len(shard_report_paths) != len(results_roots):
        raise CompositeValidationError(
            "shard report and local results root counts must match"
        )
    manifest_input = Path(manifest_path)
    if manifest_input.is_symlink():
        raise CompositeValidationError(
            f"canonical manifest must not be a symlink: {manifest_input}"
        )
    manifest_source = manifest_input.resolve(strict=True)
    manifest_before = _hash_regular_file(manifest_source)
    manifest_payload = _read_json_object(
        manifest_source,
        label="canonical manifest",
    )
    manifest_canonical_json_sha256 = _sha256_json(manifest_payload)
    try:
        manifest = runner.load_manifest(
            manifest_source,
            require_production_matrix=True,
        )
    except (runner.ManifestError, OSError) as exc:
        raise CompositeValidationError(
            f"cannot load canonical manifest: {exc}"
        ) from exc
    if (
        _hash_regular_file(manifest_source) != manifest_before
        or manifest.digest != manifest_before.sha256
    ):
        raise CompositeValidationError(
            "canonical manifest changed while it was read"
        )

    shards = [
        load_shard_report(report, root)
        for report, root in zip(
            shard_report_paths,
            results_roots,
            strict=True,
        )
    ]
    for shard in shards:
        payload = _read_json_object(
            shard.source_path,
            label="source postflight",
        )
        _verify_report_identity(
            payload,
            manifest=manifest,
            manifest_canonical_json_sha256=(
                manifest_canonical_json_sha256
            ),
            report_path=shard.source_path,
        )
    assignment = _verify_shards(manifest, shards)
    for shard in shards:
        _verify_task_root(shard)

    output = Path(output_path).resolve(strict=False)
    for shard in shards:
        if _is_within(output, shard.local_results_root):
            raise CompositeValidationError(
                "composite index output must be outside both result roots"
            )
        if _is_within(shard.source_path, shard.local_results_root):
            raise CompositeValidationError(
                "source postflight report must be outside its result root"
            )

    inventories = {
        shard.shard: _verify_transferred_files(shard)
        for shard in shards
    }
    task_records = [
        _validate_task(
            task,
            manifest,
            assignment[task.task_id],
            inventories[assignment[task.task_id].shard],
        )
        for task in manifest.tasks
    ]

    for shard in shards:
        _verify_task_root(shard)
        _verify_inventory_unchanged(
            shard.local_results_root,
            inventories[shard.shard],
        )
        source = _hash_regular_file(shard.source_path)
        if (
            source.sha256 != shard.source_sha256
            or source.fingerprint != shard.source_fingerprint
        ):
            raise CompositeValidationError(
                f"source postflight changed during validation: "
                f"{shard.source_path}"
            )
    if _hash_regular_file(manifest_source) != manifest_before:
        raise CompositeValidationError(
            "canonical manifest changed during validation"
        )

    ordered_shards = sorted(shards, key=lambda item: item.shard.casefold())
    reference = ordered_shards[0]
    payload: dict[str, object] = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "status": "verified",
        "execution_layout": "dual_host_stratified",
        "verified_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": reference.source_run_inputs["cache_bundle_id"],
        "cache_bundle_digest": reference.source_run_inputs[
            "cache_bundle_digest"
        ],
        "code_sha": reference.source_run_inputs["code_sha"],
        "manifest_raw_sha256": manifest.digest,
        "manifest_canonical_json_sha256": (
            manifest_canonical_json_sha256
        ),
        "task_count": len(task_records),
        "shard_count": len(shards),
        "shards": [
            {
                "shard": shard.shard,
                "host": shard.host,
                "source_results_root": shard.source_run_inputs[
                    "results_root"
                ],
                "local_results_root": str(shard.local_results_root),
                "task_count": len(shard.task_ids),
                "task_ids": [
                    task.task_id
                    for task in manifest.tasks
                    if assignment[task.task_id] is shard
                ],
                "source_run_inputs": dict(shard.source_run_inputs),
                "runtime_provenance": dict(shard.runtime_provenance),
                "runtime_provenance_sha256": (
                    shard.runtime_provenance_sha256
                ),
                "source_postflight_path": str(shard.source_path),
                "source_postflight_sha256": shard.source_sha256,
                "source_files_digest": shard.files_digest,
                "source_manifest_raw_sha256": (
                    shard.source_manifest_raw_sha256
                ),
                "source_manifest_canonical_json_sha256": (
                    shard.source_manifest_canonical_json_sha256
                ),
            }
            for shard in ordered_shards
        ],
        "tasks": task_records,
    }
    _atomic_create_json(output, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "scripts/remote/windows/protocol_v3_manifest.json"
        ),
    )
    parser.add_argument(
        "--shard-report",
        type=Path,
        action="append",
        required=True,
        help="Source-host postflight report, repeat exactly twice.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        required=True,
        help="Transferred local result root paired by argument order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_composite_index(
            manifest_path=args.manifest,
            shard_report_paths=args.shard_report,
            results_roots=args.results_root,
            output_path=args.output,
        )
    except (
        CompositeValidationError,
        OSError,
        runner.ManifestError,
        runner.ResultValidationError,
    ) as exc:
        print(
            f"COMPOSITE_POSTFLIGHT_FAILED: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        "COMPOSITE_POSTFLIGHT_OK "
        f"tasks={payload['task_count']} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
