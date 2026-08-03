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

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    NATIVE100_CLAIM_SCOPE,
    load_native100_compare_manifest,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    validate_completed_task_artifacts,
)
from scripts.reproduction.igstgnn_table4.secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
            validate_completed_task_artifacts(
                task_dir,
                expected_task=task,
                expected_claim_scope=NATIVE100_CLAIM_SCOPE,
            )
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
