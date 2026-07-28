from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TASK_ID = re.compile(r"^[0-9]{3}_[a-z0-9_]+_[a-z0-9_]+$")
_Runner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if temporary.exists():
            temporary.unlink()


def _default_runner(
    command: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        shell=False,
        text=True,
    )


def _matrix_command(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    splits_root: Path,
    results_root: Path,
    device: str,
    task_ids: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.reproduction.igstgnn_table4.matrix_runner",
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_sha256,
        "--splits-root",
        str(splits_root),
        "--results-root",
        str(results_root),
        "--device",
        device,
    ]
    for task_id in task_ids:
        command.extend(["--task-id", task_id])
    return command


def run_handoff(
    *,
    project_root: Path | str,
    manifest_path: Path | str,
    manifest_sha256: str,
    splits_root: Path | str,
    results_root: Path | str,
    device: str,
    pilot_task_ids: Sequence[str],
    status_path: Path | str,
    runner: _Runner = _default_runner,
) -> int:
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256")
    pilots = tuple(pilot_task_ids)
    if not pilots or len(set(pilots)) != len(pilots):
        raise ValueError("pilot_task_ids must be non-empty and unique")
    if any(_TASK_ID.fullmatch(task_id) is None for task_id in pilots):
        raise ValueError("pilot task ID is not canonical")
    if not device:
        raise ValueError("device must be non-empty")

    root = Path(project_root)
    manifest = Path(manifest_path)
    splits = Path(splits_root)
    results = Path(results_root)
    status_file = Path(status_path)
    started_at = _utc_now()
    status: dict[str, Any] = {
        "schema_version": 1,
        "state": "running",
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "splits_root": str(splits),
        "results_root": str(results),
        "device": device,
        "pilot_task_ids": list(pilots),
        "current_phase": None,
        "completed_phases": [],
        "failed_phase": None,
        "returncode": None,
        "started_at": started_at,
        "updated_at": started_at,
        "finished_at": None,
    }
    _atomic_write_json(status_file, status)

    phases = (
        ("pilot_gate", pilots),
        ("full_matrix", ()),
    )
    for phase, task_ids in phases:
        status["current_phase"] = phase
        status["updated_at"] = _utc_now()
        _atomic_write_json(status_file, status)
        command = _matrix_command(
            manifest_path=manifest,
            manifest_sha256=manifest_sha256,
            splits_root=splits,
            results_root=results,
            device=device,
            task_ids=task_ids,
        )
        try:
            completed = runner(command, cwd=root)
        except BaseException as error:
            finished_at = _utc_now()
            status.update(
                {
                    "state": "failed",
                    "current_phase": None,
                    "failed_phase": phase,
                    "returncode": 3,
                    "error": f"{type(error).__name__}: {error}",
                    "updated_at": finished_at,
                    "finished_at": finished_at,
                }
            )
            _atomic_write_json(status_file, status)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return 3
        returncode = int(completed.returncode)
        if returncode != 0:
            finished_at = _utc_now()
            status.update(
                {
                    "state": "failed",
                    "current_phase": None,
                    "failed_phase": phase,
                    "returncode": returncode,
                    "updated_at": finished_at,
                    "finished_at": finished_at,
                }
            )
            _atomic_write_json(status_file, status)
            return returncode
        status["completed_phases"].append(phase)
        status["updated_at"] = _utc_now()
        _atomic_write_json(status_file, status)

    finished_at = _utc_now()
    status.update(
        {
            "state": "completed",
            "current_phase": None,
            "returncode": 0,
            "updated_at": finished_at,
            "finished_at": finished_at,
        }
    )
    _atomic_write_json(status_file, status)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Table 4 pilot tasks as a fail-closed gate, then resume the "
            "complete immutable matrix."
        )
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--pilot-task-id",
        dest="pilot_task_ids",
        action="append",
        required=True,
    )
    parser.add_argument("--status", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run_handoff(
        project_root=args.project_root,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        splits_root=args.splits_root,
        results_root=args.results_root,
        device=args.device,
        pilot_task_ids=args.pilot_task_ids,
        status_path=args.status,
    )


if __name__ == "__main__":
    raise SystemExit(main())
