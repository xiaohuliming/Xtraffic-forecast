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


_EXPECTED_TASK_IDS = [
    "027_orange_hl",
    "028_orange_lstm",
    "029_orange_dcrnn",
    "030_orange_agcrn",
    "031_orange_stgcn",
    "032_orange_gwnet",
    "033_orange_astgcn",
    "035_orange_dstagnn",
    "036_orange_dgcrn",
    "037_orange_d2stgnn",
    "038_orange_bist",
    "039_orange_igstgnn",
]
_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "protocol_id",
    "matrix_id",
    "manifest_sha256",
    "base_plan_id",
    "base_plan_sha256",
    "host_id",
    "hostname",
    "gpu_sku",
    "gpu_count",
    "gpu_uuids",
    "execution_layout",
    "tasks",
    "relay_base",
}
_TASK_KEYS = {"task_id", "results_root"}
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
        purpose="Table 4 host5 partial export JSON",
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
        raise ValueError("Host5 partial export JSON must contain an object")
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_host5_partial_export_plan(
    plan_path: Path | str,
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    plan = _read_json(Path(plan_path), max_bytes=1024 * 1024)
    if set(plan) != _PLAN_KEYS or plan["schema_version"] != 1:
        raise ValueError("Host5 partial export plan schema mismatch")
    for field in _PLAN_KEYS - {
        "schema_version",
        "gpu_count",
        "gpu_uuids",
        "tasks",
    }:
        _nonempty_string(plan[field], field=field)
    if (
        plan["plan_id"] != "igstgnn_table4_host5_partial_export_12_v1"
        or plan["protocol_id"] != "igstgnn_table4_official_v1"
        or plan["matrix_id"] != "igstgnn_table4_public_repro_39_v2"
        or plan["manifest_sha256"] != expected_manifest_sha256
        or plan["base_plan_id"] != "igstgnn_table4_4090d_9gpu_v1"
        or _SHA256_PATTERN.fullmatch(plan["base_plan_sha256"]) is None
        or plan["host_id"] != "table4_4090d5"
        or plan["gpu_sku"] != "NVIDIA GeForce RTX 4090 D"
        or plan["gpu_count"] != 5
        or plan["execution_layout"] != "single_host_partial_original_export"
    ):
        raise ValueError("Host5 partial export plan identity mismatch")
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
        raise ValueError("Host5 partial export GPU UUID contract mismatch")
    assignments = plan["tasks"]
    if not isinstance(assignments, list) or len(assignments) != 12:
        raise ValueError("Host5 partial export must contain exactly 12 tasks")
    task_ids = []
    for assignment in assignments:
        if not isinstance(assignment, dict) or set(assignment) != _TASK_KEYS:
            raise ValueError("Host5 partial export task schema mismatch")
        task_id = _nonempty_string(assignment["task_id"], field="task_id")
        results_root = _nonempty_string(
            assignment["results_root"],
            field="results_root",
        )
        pure_root = PurePosixPath(results_root)
        if (
            pure_root.is_absolute()
            or ".." in pure_root.parts
            or pure_root.parts[:1] != ("shards",)
            or str(pure_root) != results_root
        ):
            raise ValueError("Host5 partial export results_root is unsafe")
        task_ids.append(task_id)
    if task_ids != _EXPECTED_TASK_IDS:
        raise ValueError("Host5 partial export task identity mismatch")
    relay_root = PurePosixPath(plan["relay_base"])
    if (
        not relay_root.is_absolute()
        or ".." in relay_root.parts
        or str(relay_root) != plan["relay_base"]
    ):
        raise ValueError("Host5 partial export relay_base is unsafe")
    manifest = load_table4_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    if (
        manifest["protocol_id"] != plan["protocol_id"]
        or manifest["matrix_id"] != plan["matrix_id"]
    ):
        raise ValueError("Host5 partial export manifest identity mismatch")
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    if any(
        task_id not in task_by_id or task_by_id[task_id]["region"] != "Orange"
        for task_id in task_ids
    ):
        raise ValueError("Host5 partial export contains a non-Orange task")
    return plan


def _hash_regular(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open_verified_regular_file(
        path,
        purpose="Table 4 host5 partial export file",
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


def build_host5_partial_export_postflight(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    results_base: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    plan_source = Path(plan_path)
    plan = load_host5_partial_export_plan(
        plan_source,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest = load_table4_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    base = verify_secure_directory(
        results_base,
        purpose="Table 4 host5 partial results base",
    )
    records: list[dict[str, Any]] = []
    task_reports: list[dict[str, Any]] = []
    runtime_identities: set[tuple[str, ...]] = set()
    shard_ids: set[str] = set()
    for assignment in plan["tasks"]:
        task = task_by_id[assignment["task_id"]]
        shard_ids.add(Path(assignment["results_root"]).name)
        task_dir = (
            base
            / assignment["results_root"]
            / task["output_dir"]
        )
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
            or host.get("hostname") != plan["hostname"]
            or host.get("system") != "Linux"
            or host.get("machine") != "x86_64"
            or not isinstance(device, dict)
            or device.get("type") != "cuda"
            or device.get("name") != plan["gpu_sku"]
            or not isinstance(runtime, dict)
        ):
            raise ValueError(
                f"Invalid host5 runtime identity for {task['task_id']}"
            )
        identity = (
            str(host.get("hostname")),
            str(host.get("system")),
            str(host.get("release")),
            str(host.get("machine")),
            str(runtime.get("python")),
            str(runtime.get("torch")),
            str(runtime.get("numpy")),
            str(device.get("name")),
        )
        if any(not value or value == "None" for value in identity):
            raise ValueError(
                f"Incomplete host5 runtime identity for {task['task_id']}"
            )
        runtime_identities.add(identity)
        task_reports.append(
            {
                "task_id": task["task_id"],
                "results_root": assignment["results_root"],
                "status": validation["status"],
                "metrics": validation["metrics"],
                "training_code_sha256": provenance["training_code_sha256"],
            }
        )
        for filename in _ARTIFACT_FILENAMES:
            relative = (
                Path(assignment["results_root"])
                / task["output_dir"]
                / filename
            ).as_posix()
            size, digest = _hash_regular(base / relative)
            records.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
    if len(runtime_identities) != 1:
        raise ValueError("Host5 partial export contains mixed runtime identity")
    records.sort(key=lambda record: record["path"])
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
        "host_id": plan["host_id"],
        "hostname": plan["hostname"],
        "gpu_sku": plan["gpu_sku"],
        "gpu_count": plan["gpu_count"],
        "gpu_uuids": plan["gpu_uuids"],
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
                    "gpu_name",
                ),
                next(iter(runtime_identities)),
                strict=True,
            )
        ),
        "validated_shards": len(shard_ids),
        "validated_tasks": len(task_reports),
        "tasks": task_reports,
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
    parser.add_argument("--report", required=True, type=Path)
    arguments = parser.parse_args()
    report = build_host5_partial_export_postflight(
        plan_path=arguments.plan,
        manifest_path=arguments.manifest,
        expected_manifest_sha256=arguments.manifest_sha256,
        results_base=arguments.results_base,
        report_path=arguments.report,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
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
