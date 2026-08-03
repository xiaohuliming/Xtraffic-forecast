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


_ARTIFACTS = (
    "checkpoint.pt",
    "predictions.npz",
    "provenance.json",
    "summary.json",
    "train.log",
)
_QUEUE_KEYS = frozenset(
    {
        "schema_version",
        "state",
        "protocol_id",
        "matrix_id",
        "manifest_path",
        "manifest_sha256",
        "base_manifest_path",
        "base_manifest_sha256",
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
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Native100HostPostflightError(ValueError):
    """Raised when one native100 source host is not exportable."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Native100HostPostflightError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise Native100HostPostflightError(f"non-finite JSON constant: {value}")


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
    except Native100HostPostflightError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise Native100HostPostflightError(
            f"cannot read strict {purpose}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise Native100HostPostflightError(f"{purpose} must be an object")
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


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_task_ids(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise Native100HostPostflightError(f"{label} must be task IDs")
    if len(value) != len(set(value)):
        raise Native100HostPostflightError(f"{label} contains duplicates")
    return list(value)


def _actual_task_directories(tasks_root: Path) -> set[str]:
    try:
        entries = list(tasks_root.iterdir())
    except OSError as error:
        raise Native100HostPostflightError("cannot enumerate task root") from error
    names: set[str] = set()
    for entry in entries:
        info = os.lstat(entry)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise Native100HostPostflightError(
                f"unsafe task root entry: {entry}"
            )
        names.add(entry.name)
    return names


def _validate_task_directory(
    task_dir: Path,
    *,
    task: dict[str, Any],
    expected_gpu: str,
) -> dict[str, Any]:
    try:
        actual = {entry.name for entry in task_dir.iterdir()}
    except OSError as error:
        raise Native100HostPostflightError(
            f"cannot enumerate task directory: {task['task_id']}"
        ) from error
    if actual != set(_ARTIFACTS):
        raise Native100HostPostflightError(
            f"artifact allowlist mismatch: {task['task_id']}"
        )
    try:
        validation = validate_completed_task_artifacts(
            task_dir,
            expected_task=task,
            expected_claim_scope=NATIVE100_CLAIM_SCOPE,
        )
        provenance = _read_json(
            task_dir / "provenance.json",
            purpose="native100 provenance",
            max_bytes=8 * 1024 * 1024,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise Native100HostPostflightError(
            f"task artifact validation failed: {task['task_id']}"
        ) from error
    device = provenance.get("device")
    host = provenance.get("host")
    runtime = provenance.get("runtime")
    if (
        provenance.get("matrix_id") != "igstgnn_table4_bist_ndr_native100_18_v1"
        or provenance.get("claim_scope") != NATIVE100_CLAIM_SCOPE
        or not isinstance(device, dict)
        or device.get("type") != "cuda"
        or device.get("name") != expected_gpu
        or not isinstance(host, dict)
        or not isinstance(runtime, dict)
    ):
        raise Native100HostPostflightError(
            f"task runtime identity mismatch: {task['task_id']}"
        )
    training_code_sha256 = provenance.get("training_code_sha256")
    if not isinstance(training_code_sha256, str) or not _SHA256.fullmatch(
        training_code_sha256
    ):
        raise Native100HostPostflightError(
            f"task code identity mismatch: {task['task_id']}"
        )
    identity = {
        "hostname": host.get("hostname"),
        "system": host.get("system"),
        "release": host.get("release"),
        "machine": host.get("machine"),
        "python": runtime.get("python"),
        "torch": runtime.get("torch"),
        "numpy": runtime.get("numpy"),
        "gpu_name": device.get("name"),
    }
    if any(not isinstance(item, str) or not item for item in identity.values()):
        raise Native100HostPostflightError(
            f"task runtime fields are incomplete: {task['task_id']}"
        )
    return {
        "task_id": task["task_id"],
        "status": validation["status"],
        "runtime_identity": identity,
        "training_code_sha256": training_code_sha256,
    }


def build_native100_host_postflight(
    *,
    plan_path: Path | str,
    expected_plan_sha256: str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    base_manifest_path: Path | str,
    expected_base_manifest_sha256: str,
    host_id: str,
    results_root: Path | str,
    report_path: Path | str,
    file_list_path: Path | str,
    transfer_manifest_path: Path | str,
    marker_path: Path | str,
) -> dict[str, Any]:
    plan_source = Path(plan_path)
    manifest_source = Path(manifest_path)
    base_manifest_source = Path(base_manifest_path)
    results = Path(results_root)
    observed_plan_sha256 = _sha256(plan_source, purpose="native100 plan")
    if observed_plan_sha256 != expected_plan_sha256:
        raise Native100HostPostflightError("plan SHA-256 mismatch")
    plan = _read_json(
        plan_source,
        purpose="native100 distributed plan",
        max_bytes=1024 * 1024,
    )
    if (
        plan.get("schema_version") != 1
        or plan.get("plan_id") != "igstgnn_table4_native100_5080_5090_v1"
        or plan.get("protocol_id") != "igstgnn_table4_official_v1"
        or plan.get("matrix_id") != "igstgnn_table4_bist_ndr_native100_18_v1"
        or plan.get("manifest_sha256") != expected_manifest_sha256
        or plan.get("base_manifest_sha256") != expected_base_manifest_sha256
        or plan.get("execution_layout") != "dual_host_region_stratified"
    ):
        raise Native100HostPostflightError("distributed plan identity mismatch")
    hosts = plan.get("hosts")
    if not isinstance(hosts, list):
        raise Native100HostPostflightError("distributed plan hosts are invalid")
    matching = [item for item in hosts if item.get("host_id") == host_id]
    if len(matching) != 1:
        raise Native100HostPostflightError("host is not unique in the plan")
    host_plan = matching[0]
    selected_ids = _safe_task_ids(host_plan.get("task_ids"), label="plan tasks")
    excluded_ids = _safe_task_ids(
        host_plan.get("excluded_existing_task_ids", []),
        label="excluded plan tasks",
    )
    if set(selected_ids) & set(excluded_ids):
        raise Native100HostPostflightError("selected and excluded tasks overlap")

    manifest = load_native100_compare_manifest(
        manifest_source,
        expected_sha256=expected_manifest_sha256,
        base_manifest_path=base_manifest_source,
        expected_base_manifest_sha256=expected_base_manifest_sha256,
    )
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    if not set(selected_ids + excluded_ids) <= set(task_by_id):
        raise Native100HostPostflightError("plan contains unknown task IDs")

    root_entries = {entry.name for entry in results.iterdir()}
    if root_entries != {"queue_status.json", "tasks"}:
        raise Native100HostPostflightError("result root allowlist mismatch")
    queue = _read_json(
        results / "queue_status.json",
        purpose="native100 queue status",
        max_bytes=4 * 1024 * 1024,
    )
    if set(queue) != _QUEUE_KEYS:
        raise Native100HostPostflightError("queue schema mismatch")
    completed = _safe_task_ids(queue["completed"], label="completed tasks")
    skipped = _safe_task_ids(queue["skipped"], label="skipped tasks")
    if (
        queue["schema_version"] != 1
        or queue["state"] != "completed"
        or queue["protocol_id"] != manifest["protocol_id"]
        or queue["matrix_id"] != manifest["matrix_id"]
        or queue["manifest_sha256"] != expected_manifest_sha256
        or queue["base_manifest_sha256"] != expected_base_manifest_sha256
        or queue["selected_task_ids"] != selected_ids
        or queue["current"] is not None
        or queue["command"] is not None
        or queue["failed"] != []
        or queue["failed_count"] != 0
        or queue["selected_task_count"] != len(selected_ids)
        or queue["verified_task_count"] != len(selected_ids)
        or set(completed) & set(skipped)
        or set(completed) | set(skipped) != set(selected_ids)
        or queue["completed_count"] != len(completed)
        or queue["skipped_count"] != len(skipped)
        or queue["finished_at"] != queue["updated_at"]
    ):
        raise Native100HostPostflightError("queue completion contract failed")

    tasks_root = results / "tasks"
    actual_task_ids = _actual_task_directories(tasks_root)
    if actual_task_ids != set(selected_ids + excluded_ids):
        raise Native100HostPostflightError(
            "task directory allowlist mismatch: "
            f"missing={sorted(set(selected_ids + excluded_ids) - actual_task_ids)}, "
            f"extra={sorted(actual_task_ids - set(selected_ids + excluded_ids))}"
        )

    selected_reports = [
        _validate_task_directory(
            tasks_root / task_id,
            task=task_by_id[task_id],
            expected_gpu=host_plan["gpu_sku"],
        )
        for task_id in selected_ids
    ]
    excluded_reports = [
        _validate_task_directory(
            tasks_root / task_id,
            task=task_by_id[task_id],
            expected_gpu=host_plan["gpu_sku"],
        )
        for task_id in excluded_ids
    ]
    runtime_identities = {
        json.dumps(item["runtime_identity"], sort_keys=True)
        for item in selected_reports
    }
    if len(runtime_identities) != 1:
        raise Native100HostPostflightError("selected tasks mix runtime identities")

    relative_paths = ["queue_status.json"]
    for task_id in selected_ids:
        relative_paths.extend(
            f"tasks/{task_id}/{name}" for name in _ARTIFACTS
        )
    relative_paths = sorted(relative_paths)
    records: list[dict[str, Any]] = []
    for relative in relative_paths:
        pure = PurePosixPath(relative)
        source = results.joinpath(*pure.parts)
        size, digest = _hash_file(source, purpose="native100 export artifact")
        records.append(
            {"path": relative, "size_bytes": size, "sha256": digest}
        )
    transfer_payload = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in records
    ).encode("utf-8")
    file_list_payload = "".join(
        f"{item['path']}\n" for item in records
    ).encode("utf-8")
    report = {
        "schema_version": 1,
        "status": "verified",
        "verified_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "manifest_sha256": expected_manifest_sha256,
        "base_manifest_sha256": expected_base_manifest_sha256,
        "profile_sha256": manifest["profile_sha256"],
        "plan_id": plan["plan_id"],
        "plan_sha256": observed_plan_sha256,
        "execution_layout": plan["execution_layout"],
        "claim_scope": NATIVE100_CLAIM_SCOPE,
        "validity_label": manifest["validity_label"],
        "host_id": host_id,
        "gpu_sku": host_plan["gpu_sku"],
        "runtime_identity": selected_reports[0]["runtime_identity"],
        "validated_tasks": len(selected_ids),
        "task_ids": selected_ids,
        "excluded_task_ids": excluded_ids,
        "excluded_tasks_validated": len(excluded_reports),
        "tasks": selected_reports,
        "file_count": len(records),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "files_digest": hashlib.sha256(transfer_payload).hexdigest(),
        "files": records,
    }
    report_payload = _json_bytes(report, pretty=True)
    _atomic_write(Path(report_path), report_payload)
    _atomic_write(Path(file_list_path), file_list_payload)
    _atomic_write(Path(transfer_manifest_path), transfer_payload)
    marker = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "manifest_sha256": expected_manifest_sha256,
        "base_manifest_sha256": expected_base_manifest_sha256,
        "plan_id": plan["plan_id"],
        "plan_sha256": observed_plan_sha256,
        "host_id": host_id,
        "claim_scope": NATIVE100_CLAIM_SCOPE,
        "validated_tasks": len(selected_ids),
        "file_count": len(records),
        "total_bytes": report["total_bytes"],
        "files_digest": report["files_digest"],
        "postflight_sha256": hashlib.sha256(report_payload).hexdigest(),
        "transfer_manifest_sha256": hashlib.sha256(
            transfer_payload
        ).hexdigest(),
    }
    _atomic_write(Path(marker_path), _json_bytes(marker, pretty=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--transfer-manifest", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    args = parser.parse_args()
    report = build_native100_host_postflight(
        plan_path=args.plan,
        expected_plan_sha256=args.plan_sha256,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
        base_manifest_path=args.base_manifest,
        expected_base_manifest_sha256=args.base_manifest_sha256,
        host_id=args.host_id,
        results_root=args.results_root,
        report_path=args.report,
        file_list_path=args.file_list,
        transfer_manifest_path=args.transfer_manifest,
        marker_path=args.marker,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "host_id": report["host_id"],
                "validated_tasks": report["validated_tasks"],
                "file_count": report["file_count"],
                "total_bytes": report["total_bytes"],
                "files_digest": report["files_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
