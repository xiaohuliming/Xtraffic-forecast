"""Run only CUDA engineering admission after the pinned stage-one postflight."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_state(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    os.replace(temp, path)


def require_sources(plan):
    if (
        canonical_digest({k: v for k, v in plan.items() if k != "plan_digest"})
        != plan["plan_digest"]
    ):
        raise ValueError("calibration plan identity mismatch")
    root = Path(plan["repo"]).resolve()
    for relative, digest in plan["source_files"].items():
        path = root / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
            or file_sha(path) != digest
        ):
            raise ValueError("calibration source changed: " + relative)
    if (
        plan["purpose"] != "engineering_calibration_only"
        or plan["test_evaluated"]
        or plan["automatic_training"]
        or plan["automatic_shutdown"]
        or plan["automatic_transfer"]
    ):
        raise ValueError(
            "this controller admits no formal training, test, shutdown or transfer"
        )


def predecessor_admission(root, expected_digest, plan_path=None):
    """A partial run, observation failure or stale status never starts a new job."""
    queue = json.loads((root / "results/queue_status.json").read_text())
    if queue["plan_digest"] != expected_digest:
        raise ValueError("predecessor plan identity mismatch")
    if queue["status"] in ("failed", "stopped", "stopped_resumable"):
        raise RuntimeError(
            "predecessor is terminal incomplete; audit its original outputs"
        )
    if queue["status"] != "completed":
        return False, queue
    postflight = json.loads((root / "results/postflight.json").read_text())
    if (
        postflight["status"] != "verified_development_train_val_only"
        or postflight["plan_digest"] != expected_digest
        or postflight["test_evaluated"]
        or len(postflight["tasks"]) != 8
        or postflight["tasks"] != queue["completed"]
    ):
        raise ValueError("predecessor has no complete eight-task no-test postflight")
    declared = set()
    for row in postflight["files"]:
        relative = PurePosixPath(row["path"].replace("\\", "/"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "tasks"
        ):
            raise ValueError("unsafe predecessor inventory path")
        path = root.joinpath(*relative.parts)
        if (
            str(relative) in declared
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or file_sha(path) != row["sha256"]
        ):
            raise ValueError("predecessor task bytes do not match source postflight")
        declared.add(str(relative))
    actual = {
        p.relative_to(root).as_posix()
        for p in (root / "tasks").rglob("*")
        if p.is_file()
    }
    if actual != declared:
        raise ValueError("predecessor task tree differs from source inventory")
    if plan_path is not None:
        from scripts.analyze_ndr_development import admit_matrix

        declared_plan = json.loads(Path(plan_path).read_text(encoding="utf8"))
        admit_matrix(root, declared_plan)
    return True, queue


def process_live(pid):
    if not pid:
        return False
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return bool(ok and code.value == 259)


def gpu_ready(minimum_free):
    rows = (
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
    if len(rows) != 1:
        raise ValueError("single requested RTX 5080 is required")
    name, free, util = [v.strip() for v in rows[0].split(",")]
    if "5080" not in name:
        raise ValueError("authorized RTX 5080 is absent")
    state = {"name": name, "free_mib": float(free), "utilization": float(util)}
    return float(free) >= minimum_free and float(util) < 65, state


def run(plan_path):
    if os.name != "nt":
        raise RuntimeError("scheduled calibration is for Windows RTX5080")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    require_sources(plan)
    sys.path.insert(0, plan["repo"])
    root = Path(plan["run_root"])
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    logs.mkdir()
    status = root / "status.json"
    ctypes.windll.kernel32.GetConsoleWindow.restype = ctypes.c_void_p
    if ctypes.windll.kernel32.GetConsoleWindow():
        raise RuntimeError("calibration must use scheduled pythonw without a console")
    state = {
        "status": "waiting_verified_predecessor",
        "pid": os.getpid(),
        "worker_pid": None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed": [],
        "plan_digest": plan["plan_digest"],
        "test_evaluated": False,
        "automatic_training": False,
    }
    write_state(status, state)
    overall_deadline = datetime.fromisoformat(plan["absolute_deadline_utc"]).timestamp()
    try:
        while True:
            if time.time() >= overall_deadline:
                raise TimeoutError("frozen calibration admission deadline expired")
            try:
                admitted, previous = predecessor_admission(
                    Path(plan["predecessor_root"]),
                    plan["predecessor_plan_digest"],
                    plan["predecessor_plan_path"],
                )
                state["predecessor_status"] = previous["status"]
                state["predecessor_pid"] = previous.get("pid")
                state["predecessor_live"] = process_live(previous.get("pid"))
                state["predecessor_worker_live"] = process_live(
                    previous.get("worker_pid")
                )
                state.pop("observation_error", None)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as observation:
                state["observation_error"] = type(observation).__name__
                write_state(status, state)
                time.sleep(20)
                continue
            write_state(status, state)
            if (
                admitted
                and not state["predecessor_live"]
                and not state["predecessor_worker_live"]
            ):
                break
            time.sleep(60)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        for job in plan["jobs"]:
            require_sources(plan)
            wait_started = time.monotonic()
            while True:
                ready, gpu = gpu_ready(plan["minimum_free_gpu_mib"])
                state["gpu"] = gpu
                write_state(status, state)
                if ready:
                    break
                if (
                    time.time() >= overall_deadline
                    or time.monotonic() - wait_started > plan["busy_wait_minutes"] * 60
                ):
                    raise TimeoutError(
                        "GPU busy before calibration; no other process was stopped"
                    )
                time.sleep(20)
            state["status"] = "running_engineering_calibration"
            state["current_job"] = job["job_id"]
            write_state(status, state)
            with (logs / (job["job_id"] + ".log")).open("x", encoding="utf8") as log:
                child = subprocess.Popen(
                    job["command"],
                    cwd=plan["repo"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=0x08000000,
                )
                state["worker_pid"] = child.pid
                write_state(status, state)
                job_deadline = min(
                    overall_deadline, time.time() + plan["per_job_seconds"]
                )
                while child.poll() is None:
                    if time.time() >= job_deadline:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        state["worker_pid"] = None
                        write_state(status, state)
                        raise TimeoutError("owned engineering worker deadline reached")
                    time.sleep(5)
                state["worker_pid"] = None
                if child.returncode:
                    raise RuntimeError(
                        job["job_id"] + " exited " + str(child.returncode)
                    )
            result = json.loads(Path(job["result_file"]).read_text())
            if result["status"] != job["expected_status"] or result.get(
                "test_evaluated", False
            ):
                raise ValueError(
                    "engineering result status or no-test contract differs"
                )
            state["completed"].append(job["job_id"])
            write_state(status, state)
        require_sources(plan)
        state["status"] = "completed_engineering_only"
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_state(status, state)
        files = [
            {
                "path": p.relative_to(root).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": file_sha(p),
            }
            for p in root.rglob("*")
            if p.is_file() and p.name != "engineering_postflight.json"
        ]
        write_state(
            root / "engineering_postflight.json",
            {
                "status": "verified_engineering_only_not_performance",
                "jobs": state["completed"],
                "plan_digest": plan["plan_digest"],
                "files": files,
                "test_evaluated": False,
                "formal_training_started": False,
            },
        )
    except BaseException as error:
        state["status"] = "failed_engineering_admission"
        state["error"] = str(error)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_state(status, state)
        (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf8")
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan)
