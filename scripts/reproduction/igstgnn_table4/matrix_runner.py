from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .manifest import load_table4_manifest
from .result_artifacts import validate_completed_task_artifacts
from .secure_io import verify_secure_directory


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TASK_RUNNER_MODULE = "scripts.reproduction.igstgnn_table4.task_runner"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_status(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        current = os.lstat(path)
        reparse_flag = getattr(
            stat,
            "FILE_ATTRIBUTE_REPARSE_POINT",
            0x400,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_nlink != 1
            or bool(getattr(current, "st_file_attributes", 0) & reparse_flag)
        ):
            raise ValueError(
                "Queue status path must not be a symlink, hardlink, "
                "reparse point, or non-regular file"
            )
    payload = {
        **value,
        "selected_task_count": len(value["selected_task_ids"]),
        "completed_count": len(value["completed"]),
        "skipped_count": len(value["skipped"]),
        "failed_count": len(value["failed"]),
        "verified_task_count": (len(value["completed"]) + len(value["skipped"])),
    }
    temporary = path.parent / f".q.{uuid.uuid4().hex[:16]}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
    )
    temporary_created = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        temporary_created = True
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        if temporary_created and (temporary.exists() or temporary.is_symlink()):
            temporary.unlink()
    if os.name != "nt":
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _select_tasks(
    tasks: Sequence[dict[str, Any]],
    task_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if task_ids is None:
        return list(tasks)
    requested = list(task_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("Task ID allowlist must not contain duplicates")
    known = {task["task_id"] for task in tasks}
    unknown = [task_id for task_id in requested if task_id not in known]
    if unknown:
        raise ValueError(
            "Task ID allowlist contains unknown tasks: " + ", ".join(unknown)
        )
    selected = set(requested)
    return [task for task in tasks if task["task_id"] in selected]


def _task_command(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    task_id: str,
    splits_root: Path,
    results_root: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        _TASK_RUNNER_MODULE,
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
        "--task-id",
        task_id,
        "--splits-root",
        str(splits_root),
        "--results-root",
        str(results_root),
        "--device",
        device,
    ]


def _task_target(
    *,
    results_root: Path,
    task: dict[str, Any],
) -> Path:
    relative = Path(task["output_dir"])
    if relative.is_absolute() or relative.parts != (
        "tasks",
        task["task_id"],
    ):
        raise ValueError("Task output path does not match its canonical identity")
    tasks_root = results_root / "tasks"
    if tasks_root.exists() or tasks_root.is_symlink():
        tasks_root = verify_secure_directory(
            tasks_root,
            purpose="Table 4 results tasks directory",
        )
    return tasks_root / task["task_id"]


def _finish_unsuccessful_task(
    *,
    status_path: Path,
    status: dict[str, Any],
    state: str,
    task_id: str,
    phase: str,
    command: list[str] | None,
    started_at: str,
    returncode: int | None,
    error: str,
    runner_returncode: int,
) -> int:
    failed_at = _utc_now()
    status["state"] = state
    status["failed"].append(
        {
            "task_id": task_id,
            "phase": phase,
            "command": command,
            "started_at": started_at,
            "failed_at": failed_at,
            "returncode": returncode,
            "error": error,
        }
    )
    status["updated_at"] = failed_at
    status["finished_at"] = failed_at
    _atomic_write_status(status_path, status)
    return runner_returncode


def run_table4_matrix(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    splits_root: Path | str,
    results_root: Path | str,
    device: str,
    task_ids: Sequence[str] | None = None,
) -> int:
    manifest_source = Path(manifest_path).resolve()
    split_source = Path(splits_root).resolve()
    result_root = Path(results_root)
    manifest = load_table4_manifest(
        manifest_source,
        expected_sha256=manifest_sha256,
    )
    selected_tasks = _select_tasks(manifest["tasks"], task_ids)
    result_root.mkdir(parents=True, exist_ok=True)
    result_root = verify_secure_directory(
        result_root,
        purpose="Table 4 matrix results root",
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

    for task in selected_tasks:
        task_id = task["task_id"]
        path_validation_started = _utc_now()
        try:
            target = _task_target(
                results_root=result_root,
                task=task,
            )
        except Exception as error:
            status["current"] = {
                "task_id": task_id,
                "command": None,
                "started_at": path_validation_started,
            }
            status["command"] = None
            status["updated_at"] = path_validation_started
            _atomic_write_status(status_path, status)
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="failed",
                task_id=task_id,
                phase="task_path_validation",
                command=None,
                started_at=path_validation_started,
                returncode=None,
                error=str(error),
                runner_returncode=4,
            )
        if target.exists() or target.is_symlink():
            validation_started = _utc_now()
            status["current"] = {
                "task_id": task_id,
                "command": None,
                "started_at": validation_started,
            }
            status["command"] = None
            status["updated_at"] = validation_started
            _atomic_write_status(status_path, status)
            try:
                validate_completed_task_artifacts(
                    target,
                    expected_task=task,
                )
            except KeyboardInterrupt:
                return _finish_unsuccessful_task(
                    status_path=status_path,
                    status=status,
                    state="interrupted",
                    task_id=task_id,
                    phase="existing_result_validation_interrupted",
                    command=None,
                    started_at=validation_started,
                    returncode=130,
                    error="matrix runner interrupted by user",
                    runner_returncode=130,
                )
            except SystemExit as error:
                exit_code = (
                    error.code if type(error.code) is int and error.code != 0 else 1
                )
                return _finish_unsuccessful_task(
                    status_path=status_path,
                    status=status,
                    state="interrupted",
                    task_id=task_id,
                    phase="existing_result_validation_system_exit",
                    command=None,
                    started_at=validation_started,
                    returncode=exit_code,
                    error=(
                        f"matrix runner received SystemExit with code {error.code!r}"
                    ),
                    runner_returncode=exit_code,
                )
            except Exception as error:
                return _finish_unsuccessful_task(
                    status_path=status_path,
                    status=status,
                    state="failed",
                    task_id=task_id,
                    phase="existing_result_validation",
                    command=None,
                    started_at=validation_started,
                    returncode=None,
                    error=str(error),
                    runner_returncode=4,
                )
            validated_at = _utc_now()
            status["skipped"].append(
                {
                    "task_id": task_id,
                    "command": None,
                    "started_at": validation_started,
                    "finished_at": validated_at,
                }
            )
            status["current"] = None
            status["updated_at"] = validated_at
            _atomic_write_status(status_path, status)
            continue

        command = _task_command(
            manifest_path=manifest_source,
            manifest_sha256=manifest_sha256,
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
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                check=False,
                shell=False,
            )
        except KeyboardInterrupt:
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="interrupted",
                task_id=task_id,
                phase="interrupted",
                command=command,
                started_at=task_started_at,
                returncode=130,
                error="matrix runner interrupted by user",
                runner_returncode=130,
            )
        except SystemExit as error:
            exit_code = error.code if type(error.code) is int and error.code != 0 else 1
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="interrupted",
                task_id=task_id,
                phase="system_exit",
                command=command,
                started_at=task_started_at,
                returncode=exit_code,
                error=(f"matrix runner received SystemExit with code {error.code!r}"),
                runner_returncode=exit_code,
            )
        except OSError as error:
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="failed",
                task_id=task_id,
                phase="task_launch",
                command=command,
                started_at=task_started_at,
                returncode=None,
                error=str(error),
                runner_returncode=3,
            )
        if completed.returncode != 0:
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="failed",
                task_id=task_id,
                phase="task_process",
                command=command,
                started_at=task_started_at,
                returncode=completed.returncode,
                error=(f"task runner exited with code {completed.returncode}"),
                runner_returncode=3,
            )
        try:
            target = _task_target(
                results_root=result_root,
                task=task,
            )
            validate_completed_task_artifacts(
                target,
                expected_task=task,
            )
        except KeyboardInterrupt:
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="interrupted",
                task_id=task_id,
                phase="completed_result_validation_interrupted",
                command=command,
                started_at=task_started_at,
                returncode=130,
                error="matrix runner interrupted by user",
                runner_returncode=130,
            )
        except SystemExit as error:
            exit_code = error.code if type(error.code) is int and error.code != 0 else 1
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="interrupted",
                task_id=task_id,
                phase="completed_result_validation_system_exit",
                command=command,
                started_at=task_started_at,
                returncode=exit_code,
                error=(f"matrix runner received SystemExit with code {error.code!r}"),
                runner_returncode=exit_code,
            )
        except Exception as error:
            return _finish_unsuccessful_task(
                status_path=status_path,
                status=status,
                state="failed",
                task_id=task_id,
                phase="completed_result_validation",
                command=command,
                started_at=task_started_at,
                returncode=0,
                error=str(error),
                runner_returncode=4,
            )
        task_finished_at = _utc_now()
        status["completed"].append(
            {
                "task_id": task_id,
                "command": command,
                "started_at": task_started_at,
                "finished_at": task_finished_at,
            }
        )
        status["current"] = None
        status["updated_at"] = task_finished_at
        _atomic_write_status(status_path, status)

    finished_at = _utc_now()
    status["state"] = "completed"
    status["current"] = None
    status["updated_at"] = finished_at
    status["finished_at"] = finished_at
    _atomic_write_status(status_path, status)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run a locked IGSTGNN Table 4 matrix or canonical allowlist.")
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help=(
            "Restrict execution to this canonical task ID. Repeat the "
            "option for more tasks; manifest order is always retained."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_table4_matrix(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        splits_root=args.splits_root,
        results_root=args.results_root,
        device=args.device,
        task_ids=args.task_ids,
    )


if __name__ == "__main__":
    raise SystemExit(main())
