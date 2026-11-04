"""Recover the pinned six-job calibration watcher only before any job has run.

The original controller, plan, worker argv and absolute deadline remain frozen.
An independently preserved exact old run tree and dead old process handles are
required before publishing a new live status. This wrapper reuses the verified
original predecessor/source admission and a separately SHA-bound GPU owner probe.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Callable
import uuid

CONTROLLER = "scripts/remote/windows/ndr_transfer_calibration.py"
GPU_PROVIDER_SHA256 = "23c999a6e4321da1d3298d6eb9e9682798e112648b5e2f6751a3850a9e052015"
INCIDENT_SCHEMA = "ndr_transfer_calibration_recovery_incident_v1"
JOBS = (
    "source_graph_h64_cost", "source_graph_h240_cost",
    "source_graph_h64_cuda_checkpoint_smoke", "source_graph_h64_cuda_checkpoint_smoke_verify",
    "source_graph_h240_cuda_checkpoint_smoke", "source_graph_h240_cuda_checkpoint_smoke_verify",
)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("recovery metadata must be a regular nonsymlink file")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("recovery metadata must be a JSON object")
    return value


@dataclass(frozen=True)
class FrozenAuthority:
    """Production authority is pinned; explicit fixture authorities are test-only."""

    plan_digest: str = "a1e2553da1b19dc04cb39bdb74dd701f12749444a5c31c6f4a7093b47cda708b"
    controller_sha256: str = "3f36a351a70991499f0026db152009bea77be7005431ba04093fd32bdd194c36"
    predecessor_digest: str = "ad530e29fdcf2bd0cdabdd739349814e805f12e11cfab6341e9180651807672b"
    jobs: tuple[str, ...] = JOBS
    gpu_provider_sha256: str = GPU_PROVIDER_SHA256


def require_gpu_provider(path: Path, expected_sha: str) -> None:
    if path.is_symlink() or not path.is_file() or file_sha(path) != expected_sha:
        raise ValueError("pinned GPU owner provider bytes differ")


def load_gpu_provider(path: Path, authority: FrozenAuthority):
    require_gpu_provider(path, authority.gpu_provider_sha256)
    name = "_pinned_calibration_gpu_owner_" + uuid.uuid4().hex
    specification = importlib.util.spec_from_file_location(name, path)
    provider = importlib.util.module_from_spec(specification)
    sys.modules[name] = provider
    try:
        specification.loader.exec_module(provider)
        require_gpu_provider(path, authority.gpu_provider_sha256)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return provider


def gpu_ready(context: dict, minimum_free: float) -> tuple[bool, dict]:
    require_gpu_provider(context["gpu_provider_path"], context["authority"].gpu_provider_sha256)
    gpu = context["gpu_probes"].gpu()
    require_gpu_provider(context["gpu_provider_path"], context["authority"].gpu_provider_sha256)
    if "5080" not in gpu["name"]:
        raise ValueError("authorized RTX 5080 is absent")
    return (not gpu["training_python_pids"] and not gpu["unknown_gpu_owner_pids"]
            and gpu["free_mib"] >= minimum_free and gpu["utilization"] < 65), gpu


def load_frozen_controller(plan: dict, authority: FrozenAuthority):
    if (plan.get("plan_digest") != authority.plan_digest
            or canonical_digest({k: v for k, v in plan.items() if k != "plan_digest"}) != authority.plan_digest
            or plan.get("predecessor_plan_digest") != authority.predecessor_digest
            or tuple(job.get("job_id") for job in plan.get("jobs", [])) != authority.jobs
            or plan.get("source_files", {}).get(CONTROLLER) != authority.controller_sha256):
        raise ValueError("recovery is restricted to the original fixed six-job plan and predecessor")
    repo = Path(plan["repo"]).resolve()
    path = repo / CONTROLLER
    if path.is_symlink() or not path.resolve().is_relative_to(repo) or file_sha(path) != authority.controller_sha256:
        raise ValueError("original frozen calibration controller bytes differ")
    specification = importlib.util.spec_from_file_location("_pinned_six_job_calibration", path)
    controller = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(controller)
    if file_sha(path) != authority.controller_sha256:
        raise RuntimeError("original controller changed during import")
    controller.require_sources(plan)
    return controller


def _manifest_files(manifest: dict) -> dict[str, dict]:
    declared = {}
    for row in manifest.get("files", []):
        relative = PurePosixPath(row["path"].replace("\\", "/"))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or ":" in str(relative):
            raise ValueError("preservation inventory contains an unsafe relative path")
        name = relative.as_posix()
        if (name in declared or type(row.get("bytes")) is not int or row["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))):
            raise ValueError("preservation inventory has duplicate or invalid file metadata")
        declared[name] = row
    if "status.json" not in declared:
        raise ValueError("preservation inventory omits the original calibration status")
    return declared


def _verify_tree(root: Path, declared: dict[str, dict]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("preserved/source run root must be a real directory")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("run tree contains a symlink or escaped file")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            actual.add(name)
            row = declared.get(name)
            if row is None or path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
                raise ValueError("run tree bytes differ from the complete preserved inventory")
    if actual != set(declared):
        raise ValueError("run tree file set differs from the complete preserved inventory")


def admit_empty_recovery(
    plan_path: Path, incident_path: Path, *, authority: FrozenAuthority | None = None,
    process_is_live: Callable[[int | None], bool] | None = None, now: float | None = None,
    gpu_provider_path: Path | None = None,
) -> dict[str, Any]:
    """Pure admission performs no status writes, child launches or GPU probes."""
    authority = authority or FrozenAuthority()
    plan = read_json(plan_path)
    controller = load_frozen_controller(plan, authority)
    gpu_provider_path = gpu_provider_path or Path(__file__).with_name("ndr_research_recovery.py")
    provider = load_gpu_provider(gpu_provider_path, authority)
    incident = read_json(incident_path)
    if (incident.get("schema_version") != INCIDENT_SCHEMA or incident.get("plan_digest") != authority.plan_digest
            or incident.get("original_jobs_executed") is not False):
        raise ValueError("recovery incident must bind the original plan and explicit zero-job history")
    deadline = datetime.fromisoformat(plan["absolute_deadline_utc"])
    if deadline.tzinfo is None or (time.time() if now is None else now) >= deadline.timestamp():
        raise TimeoutError("original fixed calibration deadline has expired")
    manifest_path = Path(incident["preservation_manifest_path"])
    if file_sha(manifest_path) != incident["preservation_manifest_sha256"]:
        raise ValueError("preservation manifest bytes differ from incident")
    manifest = read_json(manifest_path)
    root, archive = Path(plan["run_root"]), Path(manifest["archive_root"])
    if (Path(manifest["source_root"]).resolve() != root.resolve() or archive.resolve().is_relative_to(root.resolve())
            or root.resolve().is_relative_to(archive.resolve()) or incident_path.resolve().is_relative_to(root.resolve())
            or manifest_path.resolve().is_relative_to(root.resolve())):
        raise ValueError("preservation archive/incident must be independent of the original run tree")
    declared = _manifest_files(manifest)
    _verify_tree(root, declared)
    _verify_tree(archive, declared)
    status_sha = declared["status.json"]["sha256"]
    if status_sha != incident["old_status_sha256"]:
        raise ValueError("incident does not bind the exact preserved original status bytes")
    old = read_json(root / "status.json")
    if (old.get("status") not in ("waiting_verified_predecessor", "failed_engineering_admission")
            or old.get("plan_digest") != authority.plan_digest or old.get("completed") != []
            or old.get("worker_pid") != incident.get("old_worker_pid")
            or old.get("pid") != incident.get("old_supervisor_pid")
            or type(old.get("pid")) is not int or old["pid"] <= 0
            or old.get("current_job") is not None or old.get("test_evaluated") is not False
            or old.get("automatic_training") is not False):
        raise PermissionError("original calibration state is not a preserved inactive zero-job admission")
    worker = old.get("worker_pid")
    if worker is not None and (type(worker) is not int or worker <= 0):
        raise ValueError("old worker PID must be absent or a positive integer")
    live = process_is_live or controller.process_live
    if live(old["pid"]) or live(worker):
        raise PermissionError("original supervisor or worker is still live; no process was stopped")
    for job in plan["jobs"]:
        result = Path(job["result_file"])
        if not result.resolve().is_relative_to(root.resolve()):
            raise ValueError("original calibration job output escapes the frozen run root")
        if result.exists() or result.parent.exists() or (root / "logs" / (job["job_id"] + ".log")).exists():
            raise PermissionError("calibration job artifacts already exist; zero-job recovery is forbidden")
    return {"plan": plan, "controller": controller, "authority": authority, "root": root,
            "incident": incident, "incident_sha256": file_sha(incident_path), "manifest": manifest,
            "manifest_sha256": file_sha(manifest_path), "old": old, "deadline": deadline.timestamp(),
            "gpu_provider_path": gpu_provider_path, "gpu_provider": provider, "gpu_probes": provider.WindowsProbes()}


def _stop_owned(child) -> None:
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def execute_recovery(context: dict, *, now=time.time, sleep=time.sleep, popen=subprocess.Popen) -> dict:
    """Call only the original six frozen argv after the original full admission."""
    plan, controller, root = context["plan"], context["controller"], context["root"]
    deadline, child = context["deadline"], None
    if now() >= deadline:
        raise TimeoutError("original fixed calibration deadline expired before recovery")
    require_gpu_provider(context["gpu_provider_path"], context["authority"].gpu_provider_sha256)
    lock = root / "recovery_owner.lock"
    recovery_id = uuid.uuid4().hex
    with lock.open("x", encoding="utf8") as stream:
        json.dump({"pid": os.getpid(), "recovery_id": recovery_id, "plan_digest": plan["plan_digest"]}, stream)
    record_root = root / "recovery" / recovery_id
    logs = root / "logs"
    state = {"status": "waiting_verified_predecessor", "pid": os.getpid(), "worker_pid": None,
             "started_utc": datetime.now(timezone.utc).isoformat(), "completed": [],
             "plan_digest": plan["plan_digest"], "test_evaluated": False, "automatic_training": False,
             "automatic_transfer": False, "automatic_shutdown": False,
             "absolute_deadline_utc": plan["absolute_deadline_utc"], "recovery_id": recovery_id,
             "original_supervisor_pid": context["old"]["pid"], "original_status": context["old"]["status"],
             "recovery_controller_sha256": file_sha(Path(__file__)),
             "frozen_controller_sha256": context["authority"].controller_sha256,
             "gpu_provider_sha256": context["authority"].gpu_provider_sha256}

    def persist():
        controller.write_state(root / "status.json", state)
        controller.write_state(record_root / "state.json", state)

    try:
        record_root.mkdir(parents=True, exist_ok=False)
        logs.mkdir(exist_ok=True)
        controller.require_sources(plan)
        controller.write_state(record_root / "provenance.json", {
            "schema_version": INCIDENT_SCHEMA, "plan_digest": plan["plan_digest"],
            "incident_sha256": context["incident_sha256"], "preservation_manifest_sha256": context["manifest_sha256"],
            "preserved_archive_root": context["manifest"]["archive_root"],
            "old_status_sha256": context["incident"]["old_status_sha256"], "old_status": context["old"],
            "original_jobs_executed": False, "worker_argv_unchanged": True,
            "absolute_deadline_utc_unchanged": plan["absolute_deadline_utc"],
            "recovery_controller_sha256": state["recovery_controller_sha256"],
            "frozen_controller_sha256": state["frozen_controller_sha256"],
            "gpu_provider_path": str(context["gpu_provider_path"].resolve()),
            "gpu_provider_sha256": state["gpu_provider_sha256"]})
        persist()
        while True:
            if now() >= deadline:
                raise TimeoutError("original fixed calibration admission deadline expired")
            controller.require_sources(plan)
            require_gpu_provider(context["gpu_provider_path"], context["authority"].gpu_provider_sha256)
            try:
                admitted, previous = controller.predecessor_admission(
                    Path(plan["predecessor_root"]), plan["predecessor_plan_digest"], plan["predecessor_plan_path"])
                state.update(predecessor_status=previous["status"], predecessor_pid=previous.get("pid"),
                             predecessor_live=controller.process_live(previous.get("pid")),
                             predecessor_worker_live=controller.process_live(previous.get("worker_pid")))
                state.pop("observation_error", None)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
                state["observation_error"] = type(error).__name__
                persist()
                sleep(20)
                continue
            persist()
            if admitted and not state["predecessor_live"] and not state["predecessor_worker_live"]:
                break
            sleep(60)
        for job in plan["jobs"]:
            controller.require_sources(plan)
            waiting = now()
            while True:
                if now() >= deadline:
                    raise TimeoutError("original calibration deadline reached before a job")
                ready, gpu = gpu_ready(context, plan["minimum_free_gpu_mib"])
                state["gpu"] = gpu
                persist()
                if ready:
                    break
                if now() - waiting > plan["busy_wait_minutes"] * 60:
                    raise TimeoutError("GPU remained busy; no unrelated process was stopped")
                sleep(20)
            state.update(status="running_engineering_calibration", current_job=job["job_id"])
            persist()
            with (logs / (job["job_id"] + ".log")).open("x", encoding="utf8") as log:
                child = popen(job["command"], cwd=plan["repo"], stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)
                state["worker_pid"] = child.pid
                persist()
                job_deadline = min(deadline, now() + plan["per_job_seconds"])
                while child.poll() is None:
                    if now() >= job_deadline:
                        _stop_owned(child)
                        raise TimeoutError("owned engineering worker reached the unchanged deadline")
                    sleep(5)
                state["worker_pid"] = None
                if child.returncode:
                    raise RuntimeError(f"original engineering job {job['job_id']} exited {child.returncode}")
            child = None
            result = read_json(Path(job["result_file"]))
            if result.get("status") != job["expected_status"] or result.get("test_evaluated", False):
                raise ValueError("original engineering result status or no-test scope differs")
            state["completed"].append(job["job_id"])
            persist()
        controller.require_sources(plan)
        require_gpu_provider(context["gpu_provider_path"], context["authority"].gpu_provider_sha256)
        state.update(status="completed_engineering_only", finished_utc=datetime.now(timezone.utc).isoformat())
        state.pop("current_job", None)
        persist()
        lock.unlink()
        files = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": file_sha(p)}
                 for p in sorted(root.rglob("*")) if p.is_file() and p.name != "engineering_postflight.json"]
        controller.write_state(root / "engineering_postflight.json", {
            "status": "verified_engineering_only_not_performance", "jobs": state["completed"],
            "plan_digest": plan["plan_digest"], "files": files, "test_evaluated": False,
            "formal_training_started": False, "recovery_id": recovery_id,
            "absolute_deadline_utc_unchanged": plan["absolute_deadline_utc"]})
        return state
    except BaseException as error:
        _stop_owned(child)
        state.update(status="failed_engineering_admission", error=str(error), worker_pid=None,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        persist()
        (record_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf8")
        raise
    finally:
        if lock.exists():
            lock.unlink()


def run(plan_path: Path, incident_path: Path, gpu_provider_path: Path | None = None) -> dict:
    if os.name != "nt":
        raise RuntimeError("calibration recovery is only for the authorized Windows RTX5080")
    ctypes.windll.kernel32.GetConsoleWindow.restype = ctypes.c_void_p
    if ctypes.windll.kernel32.GetConsoleWindow():
        raise RuntimeError("calibration recovery requires scheduled pythonw without a console")
    context = admit_empty_recovery(plan_path, incident_path, gpu_provider_path=gpu_provider_path)
    sys.path.insert(0, context["plan"]["repo"])
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        return execute_recovery(context)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--gpu-provider", type=Path, help="SHA-pinned GPU owner provider; defaults to the wrapper's sibling")
    arguments = parser.parse_args()
    run(arguments.plan, arguments.incident, arguments.gpu_provider)
