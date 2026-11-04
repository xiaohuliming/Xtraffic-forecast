"""Run a frozen NDR development queue without a window or external transfer."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )
    os.replace(temp, path)


def validate_plan(plan, repo):
    claimed = plan.get("plan_digest")
    if claimed != digest({k: v for k, v in plan.items() if k != "plan_digest"}):
        raise ValueError("plan digest mismatch")
    if (
        plan["protocol_version"] != "protocol_v3_noleak_dedup"
        or plan["cache_schema_version"] != 3
    ):
        raise ValueError("unsupported data protocol")
    if (
        plan["test_evaluated"]
        or plan["automatic_transfer"]
        or plan["automatic_shutdown"]
    ):
        raise ValueError("development queue must retain no-test and keep-on policy")
    ids = [t["task_id"] for t in plan["tasks"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("queue task IDs must be unique")
    root = repo.resolve()
    for rel, expected in plan["source_files"].items():
        p = repo / rel
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError("unsafe or missing source path: " + rel)
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("source snapshot changed: " + rel)
    for task in plan["tasks"]:
        for key in ["train_command", "verify_command"]:
            command = task[key]
            if (
                not isinstance(command, list)
                or not command
                or command[0] != plan["python"]
            ):
                raise ValueError("worker command must use frozen Python executable")
        output = Path(task["output_dir"]).resolve()
        if not output.is_relative_to(Path(plan["run_root"]).resolve()):
            raise ValueError("task output escapes isolated run root")
    return plan


def gpu_ready(minimum_free_mib):
    value = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
        .strip()
        .splitlines()
    )
    if len(value) != 1:
        raise RuntimeError("research queue requires the single RTX 5080")
    name, free, usage = [s.strip() for s in value[0].split(",")]
    if "5080" not in name:
        raise RuntimeError("requested RTX 5080 is not present")
    return float(free) >= minimum_free_mib and float(usage) < 65, {
        "name": name,
        "free_mib": float(free),
        "utilization": float(usage),
    }


def execute(command, *, log, repo, state, status_path, deadline):
    flags = 0x08000000 if os.name == "nt" else 0
    with log.open("x", encoding="utf8") as stream:
        child = subprocess.Popen(
            command,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        state["worker_pid"] = child.pid
        state["worker_command"] = command
        atomic_json(status_path, state)
        while child.poll() is None:
            if time.time() >= deadline:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                state["worker_pid"] = None
                state["worker_exit_code"] = child.returncode
                atomic_json(status_path, state)
                raise TimeoutError(
                    "frozen queue wall-clock deadline reached; owned worker stopped"
                )
            time.sleep(5)
        state["worker_pid"] = None
        atomic_json(status_path, state)
        if child.returncode:
            raise RuntimeError(f"owned worker exited {child.returncode}; inspect {log}")


def run(plan_path):
    if os.name != "nt":
        raise RuntimeError("this queue is for the authorized Windows RTX 5080")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    repo = Path(plan["repo"])
    validate_plan(plan, repo)
    root = Path(plan["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    results.mkdir(exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = results / "queue_status.json"
    if status_path.exists():
        raise FileExistsError("queue state exists; audit before any explicit recovery")
    ctypes.windll.kernel32.GetConsoleWindow.restype = ctypes.c_void_p
    if ctypes.windll.kernel32.GetConsoleWindow():
        raise RuntimeError("queue must be launched with pythonw and no console window")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    deadline = time.time() + float(plan["wall_hours"]) * 3600
    state = {
        "status": "running",
        "plan_digest": plan["plan_digest"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "absolute_deadline_utc": datetime.fromtimestamp(
            deadline, timezone.utc
        ).isoformat(),
        "pid": os.getpid(),
        "worker_pid": None,
        "completed": [],
        "current_task": None,
        "test_evaluated": False,
        "automatic_shutdown": False,
        "automatic_transfer": False,
    }
    atomic_json(status_path, state)
    try:
        for task in plan["tasks"]:
            validate_plan(plan, repo)
            state["current_task"] = task["task_id"]
            state["phase"] = "resource_wait"
            atomic_json(status_path, state)
            wait_start = time.monotonic()
            while True:
                ready, info = gpu_ready(plan["minimum_free_gpu_mib"])
                state["gpu"] = info
                atomic_json(status_path, state)
                if ready:
                    break
                if (
                    time.monotonic() - wait_start
                    > float(plan["busy_wait_minutes"]) * 60
                ):
                    raise RuntimeError(
                        "GPU remained occupied; no other process was stopped"
                    )
                if time.time() >= deadline:
                    raise TimeoutError("queue deadline during resource wait")
                time.sleep(20)
            if Path(task["output_dir"]).exists():
                raise FileExistsError("fresh task output already exists")
            state["phase"] = "train"
            atomic_json(status_path, state)
            execute(
                task["train_command"],
                log=logs / (task["task_id"] + "_train.log"),
                repo=repo,
                state=state,
                status_path=status_path,
                deadline=deadline,
            )
            validate_plan(plan, repo)
            state["phase"] = "verify"
            atomic_json(status_path, state)
            execute(
                task["verify_command"],
                log=logs / (task["task_id"] + "_verify.log"),
                repo=repo,
                state=state,
                status_path=status_path,
                deadline=deadline,
            )
            verification = Path(task["verification_file"])
            if not verification.is_file():
                raise RuntimeError("worker did not write independent verification")
            record = json.loads(verification.read_text())
            if record.get("status") != task["expected_verification_status"]:
                raise RuntimeError("independent verification status was not accepted")
            state["completed"].append(task["task_id"])
            atomic_json(status_path, state)
        validate_plan(plan, repo)
        state["status"] = "completed"
        state["phase"] = "source_postflight"
        state["current_task"] = None
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        files = []
        for task in plan["tasks"]:
            output = Path(task["output_dir"])
            for f in sorted(output.rglob("*")):
                if f.is_symlink():
                    raise ValueError("task output contains symlink")
                if f.is_file():
                    files.append(
                        {
                            "path": str(f.relative_to(root)),
                            "bytes": f.stat().st_size,
                            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                        }
                    )
        atomic_json(
            results / "postflight.json",
            {
                "status": "verified_development_train_val_only",
                "plan_digest": plan["plan_digest"],
                "tasks": state["completed"],
                "files": files,
                "test_evaluated": False,
                "source_retained": True,
            },
        )
        atomic_json(status_path, state)
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = str(error)
        state["failed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, state)
        (results / "failure.txt").write_text(traceback.format_exc(), encoding="utf8")
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan)
