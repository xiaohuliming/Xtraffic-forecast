from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .distributed_plan import sha256_file
from .manifest import load_table4_manifest
from .result_artifacts import validate_completed_task_artifacts
from .secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
    verify_secure_directory,
)


_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "protocol_id",
    "matrix_id",
    "manifest_sha256",
    "base_plan_id",
    "base_plan_sha256",
    "task_id",
    "source_host_id",
    "source_shard_id",
    "recovery_host_id",
    "recovery_hostname",
    "gpu_sku",
    "gpu_count",
    "gpu_uuids",
    "execution_layout",
    "adapter",
    "results_root",
    "relay_base",
}
_ARTIFACT_FILENAMES = (
    "checkpoint.pt",
    "predictions.npz",
    "provenance.json",
    "summary.json",
    "train.log",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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
        purpose="Table 4 task034 recovery JSON",
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
        raise ValueError("Table 4 task034 recovery JSON must contain an object")
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_task034_recovery_plan(
    plan_path: Path | str,
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    plan = _read_json(Path(plan_path), max_bytes=1024 * 1024)
    if set(plan) != _PLAN_KEYS or plan["schema_version"] != 1:
        raise ValueError("Task034 recovery plan schema mismatch")
    for field in _PLAN_KEYS - {"schema_version", "gpu_count", "gpu_uuids"}:
        _nonempty_string(plan[field], field=field)
    if (
        plan["plan_id"] != "igstgnn_table4_task034_4gpu_recovery_v1"
        or plan["protocol_id"] != "igstgnn_table4_official_v1"
        or plan["matrix_id"] != "igstgnn_table4_public_repro_39_v2"
        or plan["manifest_sha256"] != expected_manifest_sha256
        or plan["base_plan_id"] != "igstgnn_table4_4090d_9gpu_v1"
        or _SHA256_PATTERN.fullmatch(plan["base_plan_sha256"]) is None
        or plan["task_id"] != "034_orange_sttn"
        or plan["source_host_id"] != "table4_4090d5"
        or plan["source_shard_id"] != "table4_4090d5_gpu0"
        or plan["recovery_host_id"] != "table4_4090d4"
        or plan["gpu_sku"] != "NVIDIA GeForce RTX 4090 D"
        or plan["gpu_count"] != 4
        or plan["execution_layout"] != "cross_host_single_task_recovery"
        or plan["adapter"] != "multi-gpu-full-batch"
    ):
        raise ValueError("Task034 recovery plan identity mismatch")
    gpu_uuids = plan["gpu_uuids"]
    if (
        not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != plan["gpu_count"]
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(
            not isinstance(value, str) or not value.startswith("GPU-")
            for value in gpu_uuids
        )
    ):
        raise ValueError("Task034 recovery GPU UUID contract mismatch")
    pure_root = PurePosixPath(plan["results_root"])
    if (
        pure_root.is_absolute()
        or ".." in pure_root.parts
        or pure_root.parts[:1] != ("shards",)
        or str(pure_root) != plan["results_root"]
    ):
        raise ValueError("Task034 recovery results_root is unsafe")
    pure_relay = PurePosixPath(plan["relay_base"])
    if (
        not pure_relay.is_absolute()
        or ".." in pure_relay.parts
        or str(pure_relay) != plan["relay_base"]
    ):
        raise ValueError("Task034 recovery relay_base is unsafe")
    manifest = load_table4_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    if (
        manifest["protocol_id"] != plan["protocol_id"]
        or manifest["matrix_id"] != plan["matrix_id"]
    ):
        raise ValueError("Task034 recovery manifest identity mismatch")
    task = next(
        (item for item in manifest["tasks"] if item["task_id"] == plan["task_id"]),
        None,
    )
    if task is None or task["method"] != "sttn" or task["region"] != "Orange":
        raise ValueError("Task034 recovery task identity mismatch")
    return plan


def _hash_regular(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open_verified_regular_file(
        path,
        purpose="Table 4 task034 recovery export",
        require_nonempty=True,
    ) as opened:
        for block in iter(lambda: opened.handle.read(1024 * 1024), b""):
            digest.update(block)
        return opened.stat_result.st_size, digest.hexdigest()


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


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def build_task034_recovery_postflight(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    results_base: Path | str,
    hardware_inventory_path: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    plan_source = Path(plan_path)
    plan = load_task034_recovery_plan(
        plan_source,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest = load_table4_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    task = next(
        item for item in manifest["tasks"] if item["task_id"] == plan["task_id"]
    )
    inventory_source = Path(hardware_inventory_path)
    inventory = _read_json(inventory_source, max_bytes=1024 * 1024)
    expected_inventory = {
        "schema_version": 1,
        "status": "verified",
        "host_id": plan["recovery_host_id"],
        "hostname": plan["recovery_hostname"],
        "gpu_sku": plan["gpu_sku"],
        "gpu_count": plan["gpu_count"],
        "gpu_uuids": plan["gpu_uuids"],
    }
    if (
        set(inventory) != set(expected_inventory) | {"captured_at"}
        or any(inventory.get(key) != value for key, value in expected_inventory.items())
        or not isinstance(inventory.get("captured_at"), str)
        or not inventory["captured_at"]
    ):
        raise ValueError("Task034 recovery hardware inventory mismatch")

    base = verify_secure_directory(
        results_base,
        purpose="Table 4 task034 recovery results base",
    )
    result_root = verify_secure_directory(
        base / plan["results_root"],
        purpose="Table 4 task034 recovery result root",
    )
    task_dir = result_root / task["output_dir"]
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
    factory = provenance.get("factory")
    execution_adapter = (
        factory.get("execution_adapter")
        if isinstance(factory, dict)
        else None
    )
    recovery_execution = provenance.get("recovery_execution")
    controller_sha256 = (
        execution_adapter.get("controller_sha256")
        if isinstance(execution_adapter, dict)
        else None
    )
    if (
        not isinstance(host, dict)
        or host.get("hostname") != plan["recovery_hostname"]
        or host.get("system") != "Linux"
        or host.get("machine") != "x86_64"
        or not isinstance(device, dict)
        or device.get("type") != "cuda"
        or device.get("name") != plan["gpu_sku"]
        or not isinstance(runtime, dict)
        or not isinstance(execution_adapter, dict)
        or execution_adapter.get("name") != "replica_device_full_batch_v1"
        or execution_adapter.get("device_ids") != list(range(plan["gpu_count"]))
        or execution_adapter.get("full_batch_preserved") is not True
        or execution_adapter.get("optimizer_step_contract_preserved") is not True
        or not isinstance(controller_sha256, str)
        or _SHA256_PATTERN.fullmatch(controller_sha256) is None
        or not isinstance(recovery_execution, dict)
        or recovery_execution.get("adapter") != plan["adapter"]
        or recovery_execution.get("controller_sha256") != controller_sha256
        or recovery_execution.get("base_training_code_sha256")
        != provenance.get("training_code_sha256")
    ):
        raise ValueError("Task034 recovery provenance contract mismatch")

    actual_files = {
        path.relative_to(result_root).as_posix()
        for path in result_root.rglob("*")
        if path.is_file()
    }
    expected_files = {
        (Path(task["output_dir"]) / filename).as_posix()
        for filename in _ARTIFACT_FILENAMES
    }
    if actual_files != expected_files:
        raise ValueError("Task034 recovery export allowlist mismatch")
    records: list[dict[str, Any]] = []
    for relative in sorted(expected_files):
        size, digest = _hash_regular(result_root / relative)
        records.append(
            {"path": relative, "size_bytes": size, "sha256": digest}
        )
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
        "protocol_id": plan["protocol_id"],
        "matrix_id": plan["matrix_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256_file(plan_source),
        "manifest_sha256": expected_manifest_sha256,
        "task_id": plan["task_id"],
        "source_host_id": plan["source_host_id"],
        "host_id": plan["recovery_host_id"],
        "recovery_host_id": plan["recovery_host_id"],
        "recovery_hostname": plan["recovery_hostname"],
        "gpu_sku": plan["gpu_sku"],
        "gpu_count": plan["gpu_count"],
        "gpu_uuids": plan["gpu_uuids"],
        "execution_layout": plan["execution_layout"],
        "adapter": plan["adapter"],
        "verified_at": _utc_now(),
        "hardware_inventory_sha256": sha256_file(inventory_source),
        "training_code_sha256": provenance["training_code_sha256"],
        "controller_sha256": controller_sha256,
        "validated_shards": 1,
        "validated_tasks": 1,
        "validation_status": validation["status"],
        "metrics": validation["metrics"],
        "files": records,
        "export_paths": [record["path"] for record in records],
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files_digest": files_digest,
    }
    _write_report(Path(report_path), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--results-base", required=True, type=Path)
    parser.add_argument("--hardware-inventory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    report = build_task034_recovery_postflight(
        plan_path=arguments.plan,
        manifest_path=arguments.manifest,
        expected_manifest_sha256=arguments.manifest_sha256,
        results_base=arguments.results_base,
        hardware_inventory_path=arguments.hardware_inventory,
        report_path=arguments.report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "task_id": report["task_id"],
                "file_count": report["file_count"],
                "files_digest": report["files_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
