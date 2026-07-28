from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .distributed_plan import load_distributed_plan, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_MODULE = "scripts.reproduction.igstgnn_table4.matrix_runner"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
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


def build_host_launch_specs(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    host_id: str,
    splits_root: Path | str,
    results_base: Path | str,
    python_executable: Path | str,
) -> list[dict[str, Any]]:
    plan = load_distributed_plan(
        plan_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if host_id not in {host["host_id"] for host in plan["hosts"]}:
        raise ValueError(f"Unknown distributed host_id: {host_id}")
    manifest_source = Path(manifest_path).resolve()
    split_source = Path(splits_root).resolve()
    result_base = Path(results_base).resolve()
    python_source = Path(python_executable)
    specs: list[dict[str, Any]] = []
    for shard in plan["shards"]:
        if shard["host_id"] != host_id:
            continue
        result_root = result_base / shard["results_root"]
        command = [
            str(python_source),
            "-X",
            "utf8",
            "-m",
            _MATRIX_MODULE,
            "--manifest",
            str(manifest_source),
            "--manifest-sha256",
            expected_manifest_sha256,
            "--splits-root",
            str(split_source),
            "--results-root",
            str(result_root),
            "--device",
            "cuda:0",
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
        remaining = max(0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_distributed_host(
    *,
    plan_path: Path | str,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    host_id: str,
    splits_root: Path | str,
    results_base: Path | str,
    control_dir: Path | str,
    python_executable: Path | str,
    poll_seconds: float = 5,
) -> int:
    split_source = Path(splits_root).resolve()
    if not split_source.is_dir():
        raise ValueError(f"Split root is missing: {split_source}")
    python_source = Path(python_executable).resolve()
    if not python_source.is_file():
        raise ValueError(f"Python executable is missing: {python_source}")
    control = Path(control_dir)
    control.mkdir(parents=True, exist_ok=True)
    logs = control / "logs"
    logs.mkdir(exist_ok=True)
    result_base = Path(results_base)
    result_base.mkdir(parents=True, exist_ok=True)
    specs = build_host_launch_specs(
        plan_path=plan_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        host_id=host_id,
        splits_root=split_source,
        results_base=result_base,
        python_executable=python_source,
    )
    started_at = _utc_now()
    status_path = control / "host_status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_public_repro_39_v2",
        "host_id": host_id,
        "plan_path": str(Path(plan_path).resolve()),
        "plan_sha256": sha256_file(plan_path),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": expected_manifest_sha256,
        "splits_root": str(split_source),
        "results_base": str(Path(results_base).resolve()),
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
    _atomic_write(status_path, status)
    processes: dict[str, subprocess.Popen[Any]] = {}
    log_handles: list[Any] = []
    try:
        for record in status["shards"]:
            log_path = logs / f"{record['shard_id']}.runner.log"
            log_handle = log_path.open("ab", buffering=0)
            log_handles.append(log_handle)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = record["cuda_visible_devices"]
            environment["PYTHONPATH"] = str(PROJECT_ROOT)
            process = subprocess.Popen(
                record["command"],
                cwd=str(PROJECT_ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
            processes[record["shard_id"]] = process
            record["pid"] = process.pid
            status["updated_at"] = _utc_now()
            _atomic_write(status_path, status)

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
                remaining.remove(shard_id)
                status["updated_at"] = record["finished_at"]
                _atomic_write(status_path, status)
            if remaining:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        _terminate_processes(processes)
        finished_at = _utc_now()
        status["status"] = "interrupted"
        status["updated_at"] = finished_at
        status["finished_at"] = finished_at
        for record in status["shards"]:
            process = processes.get(record["shard_id"])
            if process is not None and record["returncode"] is None:
                record["returncode"] = process.poll()
                record["finished_at"] = finished_at
        _atomic_write(status_path, status)
        return 130
    except BaseException:
        _terminate_processes(processes)
        finished_at = _utc_now()
        status["status"] = "failed"
        status["updated_at"] = finished_at
        status["finished_at"] = finished_at
        for record in status["shards"]:
            process = processes.get(record["shard_id"])
            if process is not None and record["returncode"] is None:
                record["returncode"] = process.poll()
                record["finished_at"] = finished_at
        _atomic_write(status_path, status)
        raise
    finally:
        for handle in log_handles:
            handle.close()

    finished_at = _utc_now()
    returncodes = [record["returncode"] for record in status["shards"]]
    status["status"] = (
        "completed" if all(returncode == 0 for returncode in returncodes) else "failed"
    )
    status["updated_at"] = finished_at
    status["finished_at"] = finished_at
    _atomic_write(status_path, status)
    return 0 if status["status"] == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--results-base", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=5)
    arguments = parser.parse_args(argv)
    return run_distributed_host(
        plan_path=arguments.plan,
        manifest_path=arguments.manifest,
        expected_manifest_sha256=arguments.manifest_sha256,
        host_id=arguments.host_id,
        splits_root=arguments.splits_root,
        results_base=arguments.results_base,
        control_dir=arguments.control_dir,
        python_executable=arguments.python,
        poll_seconds=arguments.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
