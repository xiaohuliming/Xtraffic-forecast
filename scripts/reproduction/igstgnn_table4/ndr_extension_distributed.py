from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Sequence

from .manifest import _reject_constant, _strict_object
from .ndr_extension_manifest import load_ndr_extension_manifest
from .result_artifacts import validate_completed_task_artifacts
from .secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
    verify_secure_directory,
)


_MAX_PLAN_BYTES = 256 * 1024
_HOST_IDS = ("table4_4090d4", "table4_4090d5")
_HOST_GPU_COUNTS = {
    "table4_4090d4": 4,
    "table4_4090d5": 5,
}
_HOST_REGIONS = {
    "table4_4090d4": ("Alameda", "Contra_Costa"),
    "table4_4090d5": ("Orange",),
}
_PLAN_SPECS = {
    "igstgnn_table4_ndr_extension_9gpu_v1": {
        "execution_layout": "dual_host_gpu_sku_homogeneous",
        "host_ids": ("table4_4090d4", "table4_4090d5"),
        "host_regions": _HOST_REGIONS,
        "shard_count": 7,
    },
    "igstgnn_table4_ndr_extension_4gpu_rebalance_v2": {
        "execution_layout": "single_host_gpu_sku_homogeneous_restart",
        "host_ids": ("table4_4090d4",),
        "host_regions": {
            "table4_4090d4": ("Alameda", "Contra_Costa", "Orange"),
        },
        "shard_count": 4,
    },
}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_RUNNER_MODULE = (
    "scripts.reproduction.igstgnn_table4.ndr_extension_matrix_runner"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
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


def load_ndr_extension_plan(
    path: Path | str,
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    base_manifest_path: Path | str,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    source = Path(path)
    payload = read_verified_regular_file(
        source,
        purpose="Table 4 NDR extension distributed plan",
        max_bytes=_MAX_PLAN_BYTES,
        require_nonempty=True,
    )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    plan = json.loads(
        bytes(payload).decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(plan, dict):
        raise ValueError("NDR extension distributed plan must be an object")
    manifest = load_ndr_extension_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=expected_base_manifest_sha256,
    )
    plan_id = plan.get("plan_id")
    plan_spec = _PLAN_SPECS.get(plan_id)
    if plan_spec is None:
        raise ValueError("Unexpected NDR extension plan field: plan_id")
    exact = {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_ndr_extension_9_v1",
        "manifest_sha256": expected_manifest_sha256,
        "base_manifest_sha256": expected_base_manifest_sha256,
        "gpu_sku": "NVIDIA GeForce RTX 4090 D",
        "execution_layout": plan_spec["execution_layout"],
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise ValueError(f"Unexpected NDR extension plan field: {key}")
    hosts = plan.get("hosts")
    expected_host_ids = plan_spec["host_ids"]
    if not isinstance(hosts, list) or len(hosts) != len(expected_host_ids):
        raise ValueError("NDR extension plan has an invalid host count")
    if [host.get("host_id") for host in hosts] != list(expected_host_ids):
        raise ValueError("NDR extension host order is invalid")
    for host in hosts:
        host_id = host["host_id"]
        if set(host) != {"host_id", "gpu_count", "regions"}:
            raise ValueError("NDR extension host schema is invalid")
        if (
            host["gpu_count"] != _HOST_GPU_COUNTS[host_id]
            or host["regions"]
            != list(plan_spec["host_regions"][host_id])
        ):
            raise ValueError("NDR extension host capacity or region binding is invalid")

    manifest_tasks = {
        task["task_id"]: task for task in manifest["tasks"]
    }
    shards = plan.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != plan_spec["shard_count"]
    ):
        raise ValueError("NDR extension plan has an invalid GPU shard count")
    observed_task_ids = []
    shard_ids = set()
    result_roots = set()
    host_gpu_pairs = set()
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {
            "shard_id",
            "host_id",
            "gpu_index",
            "results_root",
            "task_ids",
        }:
            raise ValueError("NDR extension shard schema is invalid")
        shard_id = shard["shard_id"]
        host_id = shard["host_id"]
        gpu_index = shard["gpu_index"]
        result_root = shard["results_root"]
        task_ids = shard["task_ids"]
        if (
            not isinstance(shard_id, str)
            or not shard_id
            or shard_id in shard_ids
            or host_id not in expected_host_ids
            or type(gpu_index) is not int
            or not 0 <= gpu_index < _HOST_GPU_COUNTS[host_id]
            or (host_id, gpu_index) in host_gpu_pairs
            or not isinstance(result_root, str)
            or result_root in result_roots
            or PurePosixPath(result_root).is_absolute()
            or ".." in PurePosixPath(result_root).parts
            or not isinstance(task_ids, list)
            or not task_ids
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError("NDR extension shard identity is invalid")
        for task_id in task_ids:
            task = manifest_tasks.get(task_id)
            if (
                task is None
                or task["region"] not in plan_spec["host_regions"][host_id]
            ):
                raise ValueError("NDR extension task is assigned to the wrong host")
        shard_ids.add(shard_id)
        result_roots.add(result_root)
        host_gpu_pairs.add((host_id, gpu_index))
        observed_task_ids.extend(task_ids)
    expected_task_ids = [task["task_id"] for task in manifest["tasks"]]
    if (
        len(observed_task_ids) != len(set(observed_task_ids))
        or set(observed_task_ids) != set(expected_task_ids)
    ):
        raise ValueError("NDR extension plan does not partition all nine tasks")
    return {**plan, "plan_sha256": observed_sha256}


def build_ndr_host_launch_specs(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    host_id: str,
    splits_root: Path | str,
    results_base: Path | str,
    python_executable: Path | str,
) -> list[dict[str, Any]]:
    plan = load_ndr_extension_plan(
        plan_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    declared_host_ids = {
        host["host_id"] for host in plan["hosts"]
    }
    if host_id not in declared_host_ids:
        raise ValueError(f"Unknown NDR extension host: {host_id}")
    manifest_source = Path(manifest_path).resolve()
    base_manifest_source = Path(base_manifest_path).resolve()
    split_source = Path(splits_root).resolve()
    result_base = Path(results_base).resolve()
    specs = []
    for shard in plan["shards"]:
        if shard["host_id"] != host_id:
            continue
        result_root = result_base / shard["results_root"]
        command = [
            str(python_executable),
            "-X",
            "utf8",
            "-m",
            _MATRIX_RUNNER_MODULE,
            "--manifest",
            str(manifest_source),
            "--manifest-sha256",
            manifest_sha256,
            "--base-manifest",
            str(base_manifest_source),
            "--base-manifest-sha256",
            base_manifest_sha256,
            "--splits-root",
            str(split_source),
            "--results-root",
            str(result_root),
            "--device",
            "cuda:0",
            "--python",
            str(python_executable),
        ]
        for task_id in shard["task_ids"]:
            command.extend(("--task-id", task_id))
        specs.append(
            {
                "shard_id": shard["shard_id"],
                "gpu_index": shard["gpu_index"],
                "cuda_visible_devices": str(shard["gpu_index"]),
                "results_root": str(result_root),
                "task_ids": list(shard["task_ids"]),
                "command": command,
            }
        )
    return specs


def _terminate_processes(processes: dict[str, subprocess.Popen[Any]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for process in processes.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_ndr_extension_host(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    host_id: str,
    splits_root: Path | str,
    results_base: Path | str,
    control_dir: Path | str,
    python_executable: Path | str,
    poll_seconds: float = 5,
) -> int:
    split_source = Path(splits_root).resolve()
    python_source = Path(python_executable).resolve()
    if not split_source.is_dir():
        raise ValueError("NDR extension split root is missing")
    if not python_source.is_file():
        raise ValueError("NDR extension Python executable is missing")
    control = Path(control_dir)
    control.mkdir(parents=True, exist_ok=True)
    logs = control / "logs"
    logs.mkdir(exist_ok=True)
    result_base = Path(results_base)
    result_base.mkdir(parents=True, exist_ok=True)
    specs = build_ndr_host_launch_specs(
        plan_path=plan_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
        host_id=host_id,
        splits_root=split_source,
        results_base=result_base,
        python_executable=python_source,
    )
    started_at = _utc_now()
    status_path = control / "host_status.json"
    status = {
        "schema_version": 1,
        "status": "running",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_ndr_extension_9_v1",
        "host_id": host_id,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "base_manifest_path": str(Path(base_manifest_path).resolve()),
        "base_manifest_sha256": base_manifest_sha256,
        "splits_root": str(split_source),
        "results_base": str(result_base.resolve()),
        "started_at": started_at,
        "updated_at": started_at,
        "finished_at": None,
        "shards": [
            {
                **spec,
                "pid": None,
                "returncode": None,
                "finished_at": None,
            }
            for spec in specs
        ],
    }
    _atomic_json(status_path, status)
    processes: dict[str, subprocess.Popen[Any]] = {}
    handles = []
    try:
        for record in status["shards"]:
            handle = (logs / f"{record['shard_id']}.runner.log").open(
                "ab",
                buffering=0,
            )
            handles.append(handle)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = record["cuda_visible_devices"]
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["PYTHONPATH"] = str(PROJECT_ROOT)
            process = subprocess.Popen(
                record["command"],
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
            processes[record["shard_id"]] = process
            record["pid"] = process.pid
            status["updated_at"] = _utc_now()
            _atomic_json(status_path, status)
        remaining = set(processes)
        while remaining:
            for record in status["shards"]:
                shard_id = record["shard_id"]
                if shard_id not in remaining:
                    continue
                returncode = processes[shard_id].poll()
                if returncode is None:
                    continue
                record["returncode"] = returncode
                record["finished_at"] = _utc_now()
                status["updated_at"] = record["finished_at"]
                remaining.remove(shard_id)
                _atomic_json(status_path, status)
                if returncode != 0:
                    _terminate_processes(
                        {
                            other: processes[other]
                            for other in remaining
                        }
                    )
                    for pending in status["shards"]:
                        if pending["shard_id"] in remaining:
                            pending["returncode"] = processes[
                                pending["shard_id"]
                            ].poll()
                            pending["finished_at"] = _utc_now()
                    remaining.clear()
                    break
            if remaining:
                time.sleep(poll_seconds)
    except BaseException:
        _terminate_processes(processes)
        status["status"] = "failed"
        status["updated_at"] = _utc_now()
        status["finished_at"] = status["updated_at"]
        _atomic_json(status_path, status)
        raise
    finally:
        for handle in handles:
            handle.close()
    status["status"] = (
        "completed"
        if all(record["returncode"] == 0 for record in status["shards"])
        else "failed"
    )
    status["updated_at"] = _utc_now()
    status["finished_at"] = status["updated_at"]
    _atomic_json(status_path, status)
    return 0 if status["status"] == "completed" else 1


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with open_verified_regular_file(
        path,
        purpose="NDR extension postflight export",
        require_nonempty=True,
    ) as opened:
        for chunk in iter(lambda: opened.handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        if size != opened.stat_result.st_size:
            raise ValueError("NDR extension export changed while hashing")
    return size, digest.hexdigest()


def validate_ndr_extension_host_results(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    host_id: str,
    results_base: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    plan = load_ndr_extension_plan(
        plan_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    manifest = load_ndr_extension_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    declared_host_ids = {
        host["host_id"] for host in plan["hosts"]
    }
    if host_id not in declared_host_ids:
        raise ValueError("Unknown NDR extension postflight host")
    tasks_by_id = {
        task["task_id"]: task for task in manifest["tasks"]
    }
    result_root = verify_secure_directory(
        results_base,
        purpose="NDR extension postflight results base",
    )
    host_shards = [
        shard for shard in plan["shards"] if shard["host_id"] == host_id
    ]
    files = []
    validated_task_ids = []
    for shard in host_shards:
        shard_root = verify_secure_directory(
            result_root / shard["results_root"],
            purpose="NDR extension postflight shard root",
        )
        if {item.name for item in shard_root.iterdir()} != {
            "queue_status.json",
            "tasks",
        }:
            raise ValueError("NDR extension shard contains extra or missing paths")
        tasks_root = verify_secure_directory(
            shard_root / "tasks",
            purpose="NDR extension postflight tasks root",
        )
        if {item.name for item in tasks_root.iterdir()} != set(shard["task_ids"]):
            raise ValueError("NDR extension shard task allowlist mismatch")
        status_payload = read_verified_regular_file(
            shard_root / "queue_status.json",
            purpose="NDR extension queue status",
            max_bytes=1024 * 1024,
            require_nonempty=True,
        )
        queue_status = json.loads(
            bytes(status_payload).decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
        completed_or_skipped = list(queue_status.get("completed", [])) + list(
            queue_status.get("skipped", [])
        )
        if (
            not isinstance(queue_status, dict)
            or queue_status.get("schema_version") != 1
            or queue_status.get("state") != "completed"
            or queue_status.get("protocol_id")
            != "igstgnn_table4_official_v1"
            or queue_status.get("matrix_id")
            != "igstgnn_table4_ndr_extension_9_v1"
            or queue_status.get("manifest_sha256") != manifest_sha256
            or queue_status.get("base_manifest_sha256")
            != base_manifest_sha256
            or queue_status.get("selected_task_ids") != shard["task_ids"]
            or set(completed_or_skipped) != set(shard["task_ids"])
            or len(completed_or_skipped) != len(shard["task_ids"])
            or queue_status.get("failed") != []
            or queue_status.get("current") is not None
        ):
            raise ValueError("NDR extension queue status is not complete and canonical")
        export_sources = [shard_root / "queue_status.json"]
        for task_id in shard["task_ids"]:
            task_dir = tasks_root / task_id
            validate_completed_task_artifacts(
                task_dir,
                expected_task=tasks_by_id[task_id],
                expected_claim_scope=(
                    "same_protocol_project_method_extension"
                ),
            )
            validated_task_ids.append(task_id)
            export_sources.extend(
                task_dir / filename
                for filename in (
                    "checkpoint.pt",
                    "predictions.npz",
                    "provenance.json",
                    "summary.json",
                    "train.log",
                )
            )
        for source in export_sources:
            size, digest = _hash_file(source)
            relative = source.relative_to(result_root).as_posix()
            files.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
    files.sort(key=lambda record: record["path"])
    expected_host_tasks = [
        task_id
        for shard in host_shards
        for task_id in shard["task_ids"]
    ]
    if validated_task_ids != expected_host_tasks:
        raise RuntimeError("NDR extension postflight task order changed")
    report = {
        "schema_version": 1,
        "status": "verified",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_ndr_extension_9_v1",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "manifest_sha256": manifest_sha256,
        "base_manifest_sha256": base_manifest_sha256,
        "host_id": host_id,
        "gpu_sku": plan["gpu_sku"],
        "execution_layout": plan["execution_layout"],
        "validity_label": manifest["validity_label"],
        "claim_scope": manifest["claim_scope"],
        "validated_shards": len(host_shards),
        "validated_tasks": len(validated_task_ids),
        "validated_task_ids": validated_task_ids,
        "file_count": len(files),
        "total_bytes": sum(record["size_bytes"] for record in files),
        "files_digest": hashlib.sha256(
            json.dumps(
                files,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "export_paths": [record["path"] for record in files],
        "files": files,
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, report)
    return report


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--host-id", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control the distributed NDR-Net Table 4 extension."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-host")
    _add_identity_arguments(run_parser)
    run_parser.add_argument("--splits-root", required=True, type=Path)
    run_parser.add_argument("--results-base", required=True, type=Path)
    run_parser.add_argument("--control-dir", required=True, type=Path)
    run_parser.add_argument("--python", required=True, type=Path)
    run_parser.add_argument("--poll-seconds", type=float, default=5)
    postflight_parser = subparsers.add_parser("postflight")
    _add_identity_arguments(postflight_parser)
    postflight_parser.add_argument("--results-base", required=True, type=Path)
    postflight_parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    identity = {
        "plan_path": args.plan,
        "manifest_path": args.manifest,
        "manifest_sha256": args.manifest_sha256,
        "base_manifest_path": args.base_manifest,
        "base_manifest_sha256": args.base_manifest_sha256,
        "host_id": args.host_id,
    }
    if args.command == "run-host":
        return run_ndr_extension_host(
            **identity,
            splits_root=args.splits_root,
            results_base=args.results_base,
            control_dir=args.control_dir,
            python_executable=args.python,
            poll_seconds=args.poll_seconds,
        )
    validate_ndr_extension_host_results(
        **identity,
        results_base=args.results_base,
        report_path=args.report,
    )
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
