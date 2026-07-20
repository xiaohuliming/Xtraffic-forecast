#!/usr/bin/env python3
"""Stop a protocol-v3 runner process group after one task is promoted.

The watcher is intentionally external to the training checkout. It waits until
both the promoted summary and the runner-owned completed task state exist, then
terminates the old orchestrator process group. This keeps the current task
intact while preventing the next task from continuing under an obsolete shard.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_completed(results_root: Path, task_id: str) -> bool:
    summary_path = results_root / "tasks" / task_id / "summary.json"
    state_path = results_root / "state" / "tasks" / f"{task_id}.json"
    if not summary_path.is_file() or not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(state, dict)
        and state.get("state") == "completed"
        and state.get("task_id") == task_id
        and isinstance(summary, dict)
        and summary.get("task_id") == task_id
    )


def _process_cmdline(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"cannot read process {pid} command line: {exc}") from exc
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(
    *,
    pid: int,
    pgid: int,
    expected_command: str,
    grace_seconds: float,
) -> str:
    cmdline = _process_cmdline(pid)
    if cmdline is None:
        return "already_exited"
    if expected_command not in cmdline:
        raise RuntimeError(
            f"refusing to stop pid {pid}: expected command marker "
            f"{expected_command!r}, got {cmdline!r}"
        )
    actual_pgid = os.getpgid(pid)
    if actual_pgid != pgid:
        raise RuntimeError(
            f"refusing to stop pid {pid}: expected pgid {pgid}, "
            f"got {actual_pgid}"
        )
    if pgid == os.getpgrp():
        raise RuntimeError("refusing to terminate the watcher's own process group")

    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return "terminated"
        time.sleep(0.25)
    if _process_group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
        return "killed_after_grace"
    return "terminated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--pgid", type=int, required=True)
    parser.add_argument("--expected-command", required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--grace-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_base = {
        "task_id": args.task_id,
        "pid": args.pid,
        "pgid": args.pgid,
        "results_root": str(args.results_root.resolve()),
    }
    _atomic_write_json(
        args.status_path,
        {
            **status_base,
            "state": "waiting",
            "updated_at": _utc_now(),
        },
    )

    while True:
        if _task_completed(args.results_root, args.task_id):
            try:
                stop_result = _stop_process_group(
                    pid=args.pid,
                    pgid=args.pgid,
                    expected_command=args.expected_command,
                    grace_seconds=args.grace_seconds,
                )
            except (OSError, RuntimeError) as exc:
                _atomic_write_json(
                    args.status_path,
                    {
                        **status_base,
                        "state": "failed",
                        "error": str(exc),
                        "updated_at": _utc_now(),
                    },
                )
                print(f"BOUNDARY_STOP_FAILED: {exc}", file=sys.stderr, flush=True)
                return 2
            _atomic_write_json(
                args.status_path,
                {
                    **status_base,
                    "state": "completed",
                    "stop_result": stop_result,
                    "updated_at": _utc_now(),
                },
            )
            print(
                f"BOUNDARY_STOPPED task={args.task_id} result={stop_result}",
                flush=True,
            )
            return 0

        if _process_cmdline(args.pid) is None:
            error = "target orchestrator exited before the boundary task completed"
            _atomic_write_json(
                args.status_path,
                {
                    **status_base,
                    "state": "failed",
                    "error": error,
                    "updated_at": _utc_now(),
                },
            )
            print(f"BOUNDARY_STOP_FAILED: {error}", file=sys.stderr, flush=True)
            return 3
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
