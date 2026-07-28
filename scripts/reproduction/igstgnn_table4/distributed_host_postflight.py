from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .distributed_plan import load_distributed_plan, sha256_file
from .manifest import load_table4_manifest
from .result_artifacts import validate_completed_task_artifacts
from .secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
    verify_secure_directory,
)


_ARTIFACT_FILENAMES = (
    "checkpoint.pt",
    "predictions.npz",
    "provenance.json",
    "summary.json",
    "train.log",
)
_QUEUE_KEYS = {
    "schema_version",
    "state",
    "protocol_id",
    "matrix_id",
    "manifest_path",
    "manifest_sha256",
    "splits_root",
    "results_root",
    "device",
    "selected_task_ids",
    "current",
    "completed",
    "skipped",
    "failed",
    "command",
    "started_at",
    "updated_at",
    "finished_at",
    "selected_task_count",
    "completed_count",
    "skipped_count",
    "failed_count",
    "verified_task_count",
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    payload = read_verified_regular_file(
        path,
        purpose="Table 4 distributed postflight JSON",
        max_bytes=max_bytes,
        require_nonempty=True,
    )
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON constant is forbidden: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _hash_regular(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open_verified_regular_file(
        path,
        purpose="Table 4 distributed export file",
        require_nonempty=True,
    ) as opened:
        for block in iter(lambda: opened.handle.read(1024 * 1024), b""):
            digest.update(block)
        return opened.stat_result.st_size, digest.hexdigest()


def _queue_task_ids(records: Any, *, field: str) -> list[str]:
    if not isinstance(records, list):
        raise ValueError(f"Queue {field} must be a list")
    task_ids: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Queue {field} records must be objects")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Queue {field} record has an invalid task_id")
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Queue {field} task IDs must be unique")
    return task_ids


def _validate_queue(
    queue: dict[str, Any],
    *,
    shard: dict[str, Any],
    manifest: dict[str, Any],
    expected_manifest_sha256: str,
) -> None:
    if set(queue) != _QUEUE_KEYS:
        raise ValueError(f"Queue schema mismatch for {shard['shard_id']}")
    expected_task_ids = shard["task_ids"]
    if (
        queue["schema_version"] != 1
        or queue["state"] != "completed"
        or queue["protocol_id"] != manifest["protocol_id"]
        or queue["matrix_id"] != manifest["matrix_id"]
        or queue["manifest_sha256"] != expected_manifest_sha256
        or queue["selected_task_ids"] != expected_task_ids
        or queue["current"] is not None
        or queue["failed"] != []
        or queue["selected_task_count"] != len(expected_task_ids)
        or queue["failed_count"] != 0
        or queue["verified_task_count"] != len(expected_task_ids)
        or queue["finished_at"] != queue["updated_at"]
    ):
        raise ValueError(f"Queue completion contract failed for {shard['shard_id']}")
    completed = _queue_task_ids(queue["completed"], field="completed")
    skipped = _queue_task_ids(queue["skipped"], field="skipped")
    if (
        set(completed) & set(skipped)
        or set(completed) | set(skipped) != set(expected_task_ids)
        or queue["completed_count"] != len(completed)
        or queue["skipped_count"] != len(skipped)
    ):
        raise ValueError(f"Queue task accounting failed for {shard['shard_id']}")


def _actual_regular_files(root: Path) -> set[str]:
    files: set[str] = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Unsafe export directory entry: {path}")
        for name in filenames:
            path = directory_path / name
            info = os.lstat(path)
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise ValueError(f"Unsafe export file entry: {path}")
            files.add(path.relative_to(root).as_posix())
    return files


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def build_host_postflight(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    host_id: str,
    results_base: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    plan_source = Path(plan_path)
    manifest_source = Path(manifest_path)
    plan = load_distributed_plan(
        plan_source,
        manifest_path=manifest_source,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    host_ids = {host["host_id"] for host in plan["hosts"]}
    if host_id not in host_ids:
        raise ValueError(f"Unknown distributed host_id: {host_id}")
    manifest = load_table4_manifest(
        manifest_source,
        expected_sha256=expected_manifest_sha256,
    )
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    base = verify_secure_directory(
        results_base,
        purpose="Table 4 distributed results base",
    )
    host_shards = [
        shard for shard in plan["shards"] if shard["host_id"] == host_id
    ]
    expected_paths: set[str] = set()
    task_reports: list[dict[str, Any]] = []
    queue_reports: list[dict[str, Any]] = []
    runtime_identities: set[tuple[str, ...]] = set()
    for shard in host_shards:
        shard_root = verify_secure_directory(
            base / shard["results_root"],
            purpose="Table 4 distributed shard results",
        )
        queue_path = shard_root / "queue_status.json"
        queue = _read_json(queue_path, max_bytes=4 * 1024 * 1024)
        _validate_queue(
            queue,
            shard=shard,
            manifest=manifest,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        queue_reports.append(
            {
                "shard_id": shard["shard_id"],
                "finished_at": queue["finished_at"],
                "completed_count": queue["completed_count"],
                "skipped_count": queue["skipped_count"],
            }
        )
        expected_paths.add(
            (Path(shard["results_root"]) / "queue_status.json").as_posix()
        )
        for task_id in shard["task_ids"]:
            task = task_by_id[task_id]
            task_dir = shard_root / task["output_dir"]
            validation = validate_completed_task_artifacts(
                task_dir,
                expected_task=task,
            )
            provenance = _read_json(
                task_dir / "provenance.json",
                max_bytes=8 * 1024 * 1024,
            )
            host = provenance.get("host")
            device = provenance.get("device")
            runtime = provenance.get("runtime")
            if (
                not isinstance(host, dict)
                or not isinstance(device, dict)
                or not isinstance(runtime, dict)
                or host.get("system") != "Linux"
                or host.get("machine") != "x86_64"
                or device.get("type") != "cuda"
                or device.get("name") != plan["gpu_sku"]
            ):
                raise ValueError(
                    f"Invalid runtime host identity for task {task_id}"
                )
            identity_fields = (
                host.get("hostname"),
                host.get("system"),
                host.get("release"),
                host.get("machine"),
                runtime.get("python"),
                runtime.get("torch"),
                runtime.get("numpy"),
                provenance.get("training_code_sha256"),
                device.get("name"),
            )
            if any(not isinstance(item, str) or not item for item in identity_fields):
                raise ValueError(
                    f"Invalid runtime host identity for task {task_id}"
                )
            runtime_identities.add(identity_fields)
            task_reports.append(
                {
                    "task_id": task_id,
                    "shard_id": shard["shard_id"],
                    "status": validation["status"],
                }
            )
            for filename in _ARTIFACT_FILENAMES:
                expected_paths.add(
                    (
                        Path(shard["results_root"])
                        / task["output_dir"]
                        / filename
                    ).as_posix()
                )

    if len(runtime_identities) != 1:
        raise ValueError("Distributed results contain mixed runtime host identity")
    actual_paths = _actual_regular_files(base)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(
            "Distributed export allowlist mismatch: "
            f"missing={missing}, extra={extra}"
        )
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for relative in sorted(expected_paths):
        size, digest = _hash_regular(base / relative)
        records.append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
            }
        )
        total_bytes += size
    files_digest = hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256_file(plan_source),
        "manifest_sha256": expected_manifest_sha256,
        "host_id": host_id,
        "gpu_sku": plan["gpu_sku"],
        "execution_layout": plan["execution_layout"],
        "verified_at": _utc_now(),
        "runtime_host_identity": dict(
            zip(
                (
                    "hostname",
                    "system",
                    "release",
                    "machine",
                    "python",
                    "torch",
                    "numpy",
                    "training_code_sha256",
                    "gpu_name",
                ),
                next(iter(runtime_identities)),
                strict=True,
            )
        ),
        "validated_shards": len(host_shards),
        "validated_tasks": len(task_reports),
        "queues": queue_reports,
        "tasks": task_reports,
        "files": records,
        "export_paths": [record["path"] for record in records],
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files_digest": files_digest,
    }
    _write_report(Path(report_path), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--results-base", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    report = build_host_postflight(
        plan_path=arguments.plan,
        manifest_path=arguments.manifest,
        expected_manifest_sha256=arguments.manifest_sha256,
        host_id=arguments.host_id,
        results_base=arguments.results_base,
        report_path=arguments.report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "host_id": report["host_id"],
                "validated_tasks": report["validated_tasks"],
                "file_count": report["file_count"],
                "files_digest": report["files_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
