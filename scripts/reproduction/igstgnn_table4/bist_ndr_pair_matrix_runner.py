from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Sequence

from .bist_ndr_pair_manifest import (
    PAIR_CLAIM_SCOPE,
    load_bist_ndr_pair_manifest,
)
from .matrix_runner import _atomic_write_status, _select_tasks
from .result_artifacts import validate_completed_task_artifacts
from .secure_io import verify_secure_directory


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TASK_RUNNER_MODULE = (
    "scripts.reproduction.igstgnn_table4.bist_ndr_pair_runner"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def build_bist_ndr_pair_task_command(
    *,
    python_executable: Path | str,
    manifest_path: Path,
    manifest_sha256: str,
    base_manifest_path: Path,
    base_manifest_sha256: str,
    task_id: str,
    splits_root: Path,
    results_root: Path,
    device: str,
) -> list[str]:
    return [
        str(python_executable),
        "-X",
        "utf8",
        "-m",
        _TASK_RUNNER_MODULE,
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
        "--base-manifest",
        str(base_manifest_path),
        "--base-manifest-sha256",
        base_manifest_sha256,
        "--task-id",
        task_id,
        "--splits-root",
        str(splits_root),
        "--results-root",
        str(results_root),
        "--device",
        device,
    ]


def build_bist_ndr_pair_task_environment(
    base_environment: dict[str, str] | os._Environ[str],
) -> dict[str, str]:
    environment = dict(base_environment)
    existing_pythonpath = environment.get("PYTHONPATH")
    python_paths = [str(PROJECT_ROOT)]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def _task_target(results_root: Path, task: dict[str, Any]) -> Path:
    relative = Path(task["output_dir"])
    if relative.parts != ("tasks", task["task_id"]):
        raise ValueError("BiST-NDR paired task output path is not canonical")
    return results_root / relative


def run_bist_ndr_pair_matrix(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    splits_root: Path | str,
    results_root: Path | str,
    device: str,
    task_ids: Sequence[str] | None = None,
    python_executable: Path | str = sys.executable,
) -> int:
    manifest_source = Path(manifest_path).resolve()
    base_manifest_source = Path(base_manifest_path).resolve()
    split_source = Path(splits_root).resolve()
    manifest = load_bist_ndr_pair_manifest(
        manifest_source,
        expected_sha256=manifest_sha256,
        base_manifest_path=base_manifest_source,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    selected_tasks = _select_tasks(manifest["tasks"], task_ids)
    result_root = Path(results_root)
    result_root.mkdir(parents=True, exist_ok=True)
    result_root = verify_secure_directory(
        result_root,
        purpose="BiST-NDR paired shard result root",
    )
    (result_root / "tasks").mkdir(exist_ok=True)
    verify_secure_directory(
        result_root / "tasks",
        purpose="BiST-NDR paired shard tasks root",
    )
    status_path = result_root / "queue_status.json"
    started_at = _utc_now()
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "manifest_path": str(manifest_source),
        "manifest_sha256": manifest_sha256,
        "base_manifest_path": str(base_manifest_source),
        "base_manifest_sha256": base_manifest_sha256,
        "splits_root": str(split_source),
        "results_root": str(result_root),
        "device": device,
        "selected_task_ids": [task["task_id"] for task in selected_tasks],
        "current": None,
        "completed": [],
        "skipped": [],
        "failed": [],
        "command": None,
        "started_at": started_at,
        "updated_at": started_at,
        "finished_at": None,
    }
    _atomic_write_status(status_path, status)

    environment = build_bist_ndr_pair_task_environment(os.environ)
    for task in selected_tasks:
        task_id = task["task_id"]
        target = _task_target(result_root, task)
        if target.exists() or target.is_symlink():
            try:
                validate_completed_task_artifacts(
                    target,
                    expected_task=task,
                    expected_claim_scope=PAIR_CLAIM_SCOPE,
                )
            except Exception as error:
                status["failed"].append(
                    {
                        "task_id": task_id,
                        "phase": "validate_existing",
                        "error": str(error),
                        "returncode": None,
                    }
                )
                status["state"] = "failed"
                status["updated_at"] = _utc_now()
                status["finished_at"] = status["updated_at"]
                _atomic_write_status(status_path, status)
                return 2
            status["skipped"].append(task_id)
            status["updated_at"] = _utc_now()
            _atomic_write_status(status_path, status)
            continue
        command = build_bist_ndr_pair_task_command(
            python_executable=python_executable,
            manifest_path=manifest_source,
            manifest_sha256=manifest_sha256,
            base_manifest_path=base_manifest_source,
            base_manifest_sha256=base_manifest_sha256,
            task_id=task_id,
            splits_root=split_source,
            results_root=result_root,
            device=device,
        )
        task_started_at = _utc_now()
        status["current"] = {
            "task_id": task_id,
            "command": command,
            "started_at": task_started_at,
        }
        status["command"] = command
        status["updated_at"] = task_started_at
        _atomic_write_status(status_path, status)
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            status["failed"].append(
                {
                    "task_id": task_id,
                    "phase": "train",
                    "command": command,
                    "started_at": task_started_at,
                    "failed_at": _utc_now(),
                    "returncode": result.returncode,
                    "error": "BiST-NDR paired task runner failed",
                }
            )
            status["state"] = "failed"
            status["updated_at"] = status["failed"][-1]["failed_at"]
            status["finished_at"] = status["updated_at"]
            _atomic_write_status(status_path, status)
            return result.returncode
        try:
            validate_completed_task_artifacts(
                target,
                expected_task=task,
                expected_claim_scope=PAIR_CLAIM_SCOPE,
            )
        except Exception as error:
            status["failed"].append(
                {
                    "task_id": task_id,
                    "phase": "validate_new",
                    "command": command,
                    "started_at": task_started_at,
                    "failed_at": _utc_now(),
                    "returncode": 0,
                    "error": str(error),
                }
            )
            status["state"] = "failed"
            status["updated_at"] = status["failed"][-1]["failed_at"]
            status["finished_at"] = status["updated_at"]
            _atomic_write_status(status_path, status)
            return 3
        status["completed"].append(task_id)
        status["current"] = None
        status["command"] = None
        status["updated_at"] = _utc_now()
        _atomic_write_status(status_path, status)

    status["state"] = "completed"
    status["current"] = None
    status["command"] = None
    status["updated_at"] = _utc_now()
    status["finished_at"] = status["updated_at"]
    _atomic_write_status(status_path, status)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated BiST versus BiST-NDR paired shard."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_bist_ndr_pair_matrix(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        base_manifest_path=args.base_manifest,
        base_manifest_sha256=args.base_manifest_sha256,
        splits_root=args.splits_root,
        results_root=args.results_root,
        device=args.device,
        task_ids=args.task_ids,
        python_executable=args.python,
    )


if __name__ == "__main__":
    raise SystemExit(main())
