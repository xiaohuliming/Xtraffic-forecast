#!/usr/bin/env python3
"""Serial calibrate/train/verify queue for frozen matched-budget NDR plans.

The controller is an execution shell. It admits a frozen plan read-only,
waits for the GPU to report enough free memory, then runs each job as an
owned child process, one at a time, inside the plan's absolute deadline and
total budget. Any child failure or timeout stops the queue without retry.
Only the standard library is used so the file can run on the Windows
workstation without importing the research checkout.

Usage: python ndr_matched_budget_serial_queue.py --plan plan.json [--admit-only]
Exit codes: 0 admitted or completed, 1 failed_stopped, 2 admission rejected,
3 stopped_deadline_or_budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import time
from typing import Any, Callable
import uuid

SCHEMA_VERSION = "ndr_matched_budget_serial_plan_v1"
STATE_SCHEMA_VERSION = "ndr_matched_budget_serial_state_v1"
JOURNAL_SCHEMA_VERSION = "ndr_matched_budget_owned_process_journal_v1"
ROLES = ("calibrate", "train", "verify")
REQUIRED_KEYS = frozenset({
    "schema_version", "purpose", "plan_digest", "budget_frozen", "absolute_deadline_utc",
    "total_budget_seconds", "busy_wait_seconds", "minimum_free_gpu_mib", "python", "run_root", "jobs",
})
OPTIONAL_KEYS = frozenset({"state_path", "journal_path", "metadata"})
JOB_KEYS = frozenset({"job_id", "role", "command", "timeout_seconds", "depends_on", "success_artifact", "output_dir"})
TRAIN_SUCCESS_STATUS = "training_completed_validation_pending_independent_verification"
VERIFY_SUCCESS_PREFIX = "verified_"
CALIBRATE_SUCCESS_PREFIX = "calibrated_"
# train_ndr_research.py --mode calibrate writes no cuda_budget_eligible key.
NDR_RESEARCH_CALIBRATION_STATUS = "calibrated_engineering_only"
TRAINER_ENTRIES = frozenset({"train_ndr_strong_reference.py", "train_ndr_research.py"})
TRAINER_DEFAULT_DEVICE = "cuda:0"
CHECKED_FLAGS = ("--mode", "--output-dir", "--checkpoint", "--resume", "--device")
MAX_BUSY_WAIT_SECONDS = 600
MIN_FREE_GPU_MIB = 12000
LOG_DIR_NAME = "controller_logs"
DEFAULT_STATE_NAME = "controller_state.json"
DEFAULT_JOURNAL_NAME = "owned_processes.jsonl"
CREATE_NO_WINDOW = 0x08000000
JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
NVIDIA_SMI_COMMAND = ("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REJECTED = 2
EXIT_BOUND = 3

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed_all_jobs"
STATUS_FAILED = "failed_stopped"
STATUS_BOUND = "stopped_deadline_or_budget"


class AdmissionError(ValueError):
    """The plan was rejected before any child process existed."""


class _QueueStop(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def plan_digest(plan: dict[str, Any]) -> str:
    return canonical_digest({key: value for key, value in plan.items() if key != "plan_digest"})


def seal_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the plan carrying its canonical digest."""
    sealed = {key: value for key, value in plan.items() if key != "plan_digest"}
    sealed["plan_digest"] = canonical_digest(sealed)
    return sealed


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def parse_deadline(text: Any) -> datetime:
    if not isinstance(text, str) or not text:
        raise AdmissionError("absolute_deadline_utc must be an ISO 8601 string")
    value = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        instant = datetime.fromisoformat(value)
    except ValueError as error:
        raise AdmissionError(f"absolute_deadline_utc is not ISO 8601: {text}") from error
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise AdmissionError("absolute_deadline_utc must carry an explicit UTC offset")
    return instant


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _reject_constant(name: str) -> Any:
    raise AdmissionError(f"plan JSON contains the non-finite constant {name}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdmissionError(f"plan JSON repeats the key {key!r}")
        result[key] = value
    return result


def load_plan_bytes(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8-sig")
        plan = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except AdmissionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdmissionError(f"plan is not UTF-8 JSON: {error}") from error
    if not isinstance(plan, dict):
        raise AdmissionError("plan must be a JSON object")
    return plan


def _norm(path: Path | str) -> Path:
    return Path(os.path.normcase(str(Path(os.path.abspath(path)).resolve())))


def _within(child: Path | str, parent: Path | str) -> bool:
    child_path, parent_path = _norm(child), _norm(parent)
    return child_path == parent_path or child_path.is_relative_to(parent_path)


def _same(left: Path | str, right: Path | str) -> bool:
    return _norm(left) == _norm(right)


def _absolute(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AdmissionError(f"{name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        raise AdmissionError(f"{name} must be an absolute path on this host: {value}")
    return path


def _relative_artifact(value: Any, job_id: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AdmissionError(f"job {job_id} success_artifact must be a non-empty relative path")
    for flavour in (PurePosixPath(value), PureWindowsPath(value)):
        if flavour.is_absolute() or flavour.drive or flavour.root or ".." in flavour.parts or not flavour.parts:
            raise AdmissionError(f"job {job_id} success_artifact must stay inside output_dir: {value}")
    return value


def _flag_values(command: list[str], flag: str) -> list[str | None]:
    values: list[str | None] = []
    for index, item in enumerate(command):
        if item == flag:
            values.append(command[index + 1] if index + 1 < len(command) else None)
        elif item.startswith(flag + "="):
            values.append(item[len(flag) + 1:])
    return values


def _single_flag(command: list[str], flag: str, job_id: str) -> str | None:
    values = _flag_values(command, flag)
    if len(values) > 1 or None in values or "" in values:
        raise AdmissionError(f"job {job_id} repeats, truncates or empties {flag}")
    return values[0] if values else None


def _trainer_entry(command: list[str]) -> str | None:
    for item in command[1:]:
        name = re.split(r"[\\/]", item)[-1]
        if name in TRAINER_ENTRIES:
            return name
    return None


def requested_device(command: list[str]) -> str:
    values = [value for value in _flag_values(command, "--device") if value]
    return values[-1] if values else TRAINER_DEFAULT_DEVICE


def _is_cuda(device: str) -> bool:
    return device.split(":", 1)[0].strip().lower() == "cuda"


def _creatable_directory(path: Path) -> bool:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return probe.is_dir() and os.access(probe, os.W_OK)


def _check_job_flags(job: dict[str, Any], jobs_by_id: dict[str, dict[str, Any]]) -> None:
    """Bind trainer flags to the plan fields that success detection relies on."""
    job_id, role, command = job["job_id"], job["role"], job["command"]
    entry = _trainer_entry(command)
    if entry is not None:
        for item in command:
            name = item.split("=", 1)[0]
            if name.startswith("--") and len(name) > 2 and any(flag != name and flag.startswith(name) for flag in CHECKED_FLAGS):
                raise AdmissionError(f"job {job_id} abbreviates a checked trainer flag: {name}")
    mode = _single_flag(command, "--mode", job_id)
    output = _single_flag(command, "--output-dir", job_id)
    checkpoint = _single_flag(command, "--checkpoint", job_id)
    resume = _single_flag(command, "--resume", job_id)
    _single_flag(command, "--device", job_id)
    if entry is not None and mode is None:
        raise AdmissionError(f"job {job_id} must pass an explicit --mode to {entry}")
    if mode is not None and mode != role:
        raise AdmissionError(f"job {job_id} declares role {role} but passes --mode {mode}")
    if output is not None and not _same(output, job["output_dir"]):
        raise AdmissionError(f"job {job_id} --output-dir differs from its plan output_dir")
    if resume is not None:
        if role != "train" or re.split(r"[\\/]", resume)[-1] != "last.pt" or not _same(Path(resume).parent, job["output_dir"]):
            raise AdmissionError(f"job {job_id} --resume must be output_dir/last.pt of a train job")
    if role == "verify":
        train = jobs_by_id[job["depends_on"][0]]
        if checkpoint is not None and not _same(Path(checkpoint).parent, train["output_dir"]):
            raise AdmissionError(f"job {job_id} --checkpoint is not inside its train job output_dir")
        if entry is not None and checkpoint is None:
            raise AdmissionError(f"job {job_id} must pass --checkpoint to {entry}")
        if entry is not None and output is None and not _same(Path(checkpoint).parent, job["output_dir"]):
            raise AdmissionError(f"job {job_id} without --output-dir writes next to its checkpoint")
    elif checkpoint is not None:
        raise AdmissionError(f"job {job_id} passes --checkpoint outside verify")
    elif entry is not None and output is None and resume is None:
        raise AdmissionError(f"job {job_id} must pass --output-dir to {entry}")


def _validate_jobs(plan: dict[str, Any], python: str, run_root: Path) -> list[dict[str, Any]]:
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise AdmissionError("jobs must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != JOB_KEYS:
            raise AdmissionError(f"job #{index} must have exactly the keys {sorted(JOB_KEYS)}")
        job_id = job["job_id"]
        if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
            raise AdmissionError(f"job #{index} job_id must match {JOB_ID_PATTERN.pattern}")
        if job_id in by_id:
            raise AdmissionError(f"duplicate job_id {job_id}")
        if job["role"] not in ROLES:
            raise AdmissionError(f"job {job_id} role must be one of {ROLES}")
        command = job["command"]
        if (not isinstance(command, list) or not command
                or any(not isinstance(item, str) or "\x00" in item for item in command)):
            raise AdmissionError(f"job {job_id} command must be a non-empty list of strings")
        if command[0] != python:
            raise AdmissionError(f"job {job_id} command must start with the plan python")
        timeout = job["timeout_seconds"]
        if not _finite_number(timeout) or timeout <= 0:
            raise AdmissionError(f"job {job_id} timeout_seconds must be finite and positive")
        depends = job["depends_on"]
        if not isinstance(depends, list) or any(not isinstance(item, str) for item in depends) or len(set(depends)) != len(depends):
            raise AdmissionError(f"job {job_id} depends_on must be a list of distinct job ids")
        for dependency in depends:
            if dependency not in by_id:
                raise AdmissionError(f"job {job_id} depends on {dependency}, which is not an earlier job")
        if job["role"] == "verify" and (len(depends) != 1 or by_id[depends[0]]["role"] != "train"):
            raise AdmissionError(f"verify job {job_id} must depend on exactly one earlier train job")
        artifact = _relative_artifact(job["success_artifact"], job_id)
        output = _absolute(job["output_dir"], f"job {job_id} output_dir")
        if _same(output, run_root) or not _within(output, run_root):
            raise AdmissionError(f"job {job_id} output_dir must be strictly inside run_root")
        by_id[job_id] = job | {"_output": output, "_artifact": output / artifact}
    return [by_id[job["job_id"]] for job in jobs]


def admit(plan_path: Path | str, *, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Validate a frozen plan without creating, writing or spawning anything."""
    path = Path(plan_path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AdmissionError(f"plan file is unreadable: {error}") from error
    plan = load_plan_bytes(raw)
    missing, unknown = REQUIRED_KEYS - set(plan), set(plan) - REQUIRED_KEYS - OPTIONAL_KEYS
    if missing or unknown:
        raise AdmissionError(f"plan keys differ from the schema: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if plan["schema_version"] != SCHEMA_VERSION:
        raise AdmissionError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(plan["purpose"], str) or not plan["purpose"].strip():
        raise AdmissionError("purpose must be a non-empty string")
    if plan["budget_frozen"] is not True:
        raise AdmissionError("budget_frozen must be true; draft plans authorize no worker")
    if "metadata" in plan and not isinstance(plan["metadata"], dict):
        raise AdmissionError("metadata must be a JSON object when present")
    if not isinstance(plan["plan_digest"], str) or not DIGEST_PATTERN.fullmatch(plan["plan_digest"]):
        raise AdmissionError("plan_digest must be 64 lowercase hex characters")
    if plan["plan_digest"] != plan_digest(plan):
        raise AdmissionError("plan_digest does not match the canonical plan content")
    deadline = parse_deadline(plan["absolute_deadline_utc"]).timestamp()
    if deadline <= now():
        raise AdmissionError("absolute_deadline_utc has already passed")
    budget = plan["total_budget_seconds"]
    if not _finite_number(budget) or budget <= 0:
        raise AdmissionError("total_budget_seconds must be finite and positive")
    busy = plan["busy_wait_seconds"]
    if not _finite_number(busy) or not 0 < busy <= MAX_BUSY_WAIT_SECONDS:
        raise AdmissionError(f"busy_wait_seconds must satisfy 0 < x <= {MAX_BUSY_WAIT_SECONDS}")
    minimum = plan["minimum_free_gpu_mib"]
    if not _finite_number(minimum) or minimum < MIN_FREE_GPU_MIB:
        raise AdmissionError(f"minimum_free_gpu_mib must be finite and >= {MIN_FREE_GPU_MIB}")
    python = _absolute(plan["python"], "python")
    if not python.is_file():
        raise AdmissionError(f"python does not exist on this host: {python}")
    run_root = _absolute(plan["run_root"], "run_root")
    if run_root.is_symlink() or (run_root.exists() and not run_root.is_dir()):
        raise AdmissionError("run_root must be a real directory")
    if not run_root.exists() and not _creatable_directory(run_root):
        raise AdmissionError("run_root does not exist and cannot be created")
    state_path = _absolute(plan.get("state_path", str(run_root / DEFAULT_STATE_NAME)), "state_path")
    journal_path = _absolute(plan.get("journal_path", str(run_root / DEFAULT_JOURNAL_NAME)), "journal_path")
    log_dir = run_root / LOG_DIR_NAME
    for name, value in (("state_path", state_path), ("journal_path", journal_path)):
        if _same(value, run_root) or not _within(value, run_root) or _within(value, log_dir):
            raise AdmissionError(f"{name} must be a file inside run_root outside {LOG_DIR_NAME}")
        if value.exists() or value.is_symlink():
            raise AdmissionError(f"{name} already exists; a plan executes into a fresh state")
    if _same(state_path, journal_path):
        raise AdmissionError("state_path and journal_path must differ")
    jobs = _validate_jobs(plan, plan["python"], run_root)
    if sum(job["timeout_seconds"] for job in jobs) > budget:
        raise AdmissionError("the sum of job timeout_seconds exceeds total_budget_seconds")
    by_id = {job["job_id"]: job for job in jobs}
    producers = [job for job in jobs if job["role"] != "verify"]
    for job in jobs:
        _check_job_flags(job, by_id)
        output, artifact = job["_output"], job["_artifact"]
        if _within(output, log_dir) or _within(log_dir, output):
            raise AdmissionError(f"job {job['job_id']} output_dir overlaps {LOG_DIR_NAME}")
        if any(_within(item, output) for item in (state_path, journal_path)):
            raise AdmissionError(f"job {job['job_id']} output_dir would contain controller state or journal")
        if output.is_symlink() or artifact.exists() or artifact.is_symlink():
            raise AdmissionError(f"job {job['job_id']} output_dir is a symlink or its success artifact already exists")
        resume = _single_flag(job["command"], "--resume", job["job_id"]) is not None
        if job["role"] != "verify" and output.exists() != resume:
            raise AdmissionError(f"job {job['job_id']} output_dir must be new, or exist only for --resume")
        allowed = {job["depends_on"][0]} if job["role"] == "verify" else {job["job_id"]}
        for other in producers:
            if other["job_id"] not in allowed and (_within(output, other["_output"]) or (job["role"] != "verify" and _within(other["_output"], output))):
                raise AdmissionError(f"job {job['job_id']} output_dir overlaps job {other['job_id']}")
        for stream in ("stdout", "stderr"):
            if (log_dir / f"{job['job_id']}.{stream}.log").exists():
                raise AdmissionError(f"job {job['job_id']} {stream} log already exists")
    artifacts = [_norm(job["_artifact"]) for job in jobs]
    if len(set(artifacts)) != len(artifacts):
        raise AdmissionError("two jobs share one success artifact path")
    return {
        "admitted": True, "plan": plan, "plan_path": str(path.resolve()),
        "plan_sha256": hashlib.sha256(raw).hexdigest(), "deadline": deadline,
        "run_root": run_root, "state_path": state_path, "journal_path": journal_path,
        "log_dir": log_dir, "jobs": jobs,
    }


def nvidia_smi_free_mib(*, run: Callable[..., Any] = subprocess.run, timeout_seconds: float = 30.0) -> float:
    """Return the smallest free memory over all visible GPUs, in MiB."""
    options: dict[str, Any] = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}
    completed = run(list(NVIDIA_SMI_COMMAND), capture_output=True, text=True, timeout=timeout_seconds, check=True, **options)
    values = [float(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("nvidia-smi returned no finite free-memory reading")
    return min(values)


def _probe_free_mib(provider: Callable[[], Any]) -> tuple[float | None, str | None]:
    try:
        value = provider()
    except Exception as error:  # a failed provider means "not idle"
        return None, f"{type(error).__name__}: {error}"
    if not _finite_number(value) or value < 0:
        return None, f"provider returned an invalid reading: {value!r}"
    return float(value), None


def _atomic_write_json(path: Path, value: Any, sleep: Callable[[float], None]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                sleep(0.05)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _append_journal(path: Path, entry: dict[str, Any]) -> None:
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} is missing or not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def evaluate_success_artifact(job: dict[str, Any]) -> dict[str, Any]:
    """Return the recorded artifact fields, or raise ValueError when the job did not succeed."""
    role, artifact = job["role"], _read_json_object(job["_artifact"])
    status = artifact.get("status")
    if not isinstance(status, str):
        raise ValueError("success artifact has no string status")
    detail: dict[str, Any] = {"path": str(job["_artifact"]), "status": status, "claim_scope": artifact.get("claim_scope")}
    if role == "train":
        if status != TRAIN_SUCCESS_STATUS:
            raise ValueError(f"train artifact status is {status}, not {TRAIN_SUCCESS_STATUS}")
        detail.update(completed_epochs=artifact.get("completed_epochs"), config_digest=artifact.get("config_digest"))
    elif role == "verify":
        if not status.startswith(VERIFY_SUCCESS_PREFIX):
            raise ValueError(f"verify artifact status {status} does not start with {VERIFY_SUCCESS_PREFIX}")
        detail.update(checkpoint_sha256=artifact.get("checkpoint_sha256"),
                      artifact_training_process_id=artifact.get("training_process_id"),
                      artifact_verification_process_id=artifact.get("verification_process_id"))
    else:
        if not status.startswith(CALIBRATE_SUCCESS_PREFIX):
            raise ValueError(f"calibration artifact status {status} does not start with {CALIBRATE_SUCCESS_PREFIX}")
        device = requested_device(job["command"])
        eligible = artifact.get("cuda_budget_eligible")
        detail.update(requested_device=device, cuda_budget_eligible=eligible)
        if _is_cuda(device):
            if "cuda_budget_eligible" in artifact:
                if eligible is not True:
                    raise ValueError("CUDA calibration is not cuda_budget_eligible")
                detail["cuda_evidence"] = "calibration_cuda_budget_eligible"
            else:
                config = _read_json_object(job["_output"] / "config.json")
                runtime = config.get("runtime")
                if status != NDR_RESEARCH_CALIBRATION_STATUS or not isinstance(runtime, dict) or runtime.get("device_type") != "cuda":
                    raise ValueError("CUDA calibration lacks cuda_budget_eligible and a CUDA runtime in config.json")
                detail["cuda_evidence"] = "config_runtime_device_type_only_trainer_reports_no_eligibility"
    return detail


def _stop_owned_child(child: Any, grace_seconds: float) -> tuple[int | None, str]:
    """Stop only the Popen child this controller created; never any other process."""
    if child.poll() is None:
        try:
            child.terminate()
        except OSError:
            pass
    try:
        return child.wait(timeout=grace_seconds), "terminate"
    except subprocess.TimeoutExpired:
        pass
    try:
        child.kill()
    except OSError:
        pass
    try:
        return child.wait(timeout=grace_seconds), "kill"
    except subprocess.TimeoutExpired:
        return None, "kill_unconfirmed"


def execute(context: dict[str, Any], *, gpu_free_mib: Callable[[], Any] | None = None,
            popen: Callable[..., Any] = subprocess.Popen, now: Callable[[], float] = time.time,
            monotonic: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
            poll_interval_seconds: float = 1.0, terminate_grace_seconds: float = 30.0,
            heartbeat_interval_seconds: float = 60.0) -> dict[str, Any]:
    """Run the admitted jobs serially; return the final state dictionary."""
    if context.get("admitted") is not True:
        raise AdmissionError("execute requires a context returned by admit")
    plan, jobs = context["plan"], context["jobs"]
    run_root, state_path, journal_path, log_dir = (context[key] for key in ("run_root", "state_path", "journal_path", "log_dir"))
    provider = gpu_free_mib if gpu_free_mib is not None else nvidia_smi_free_mib
    began = monotonic()
    budget, deadline = plan["total_budget_seconds"], context["deadline"]
    try:
        current_sha = hashlib.sha256(Path(context["plan_path"]).read_bytes()).hexdigest()
    except OSError as error:
        raise AdmissionError(f"plan file is unreadable at execution: {error}") from error
    if current_sha != context["plan_sha256"]:
        raise AdmissionError("plan file changed after admission")
    if deadline <= now():
        raise AdmissionError("absolute_deadline_utc passed before execution started")
    if state_path.exists() or journal_path.exists():
        raise AdmissionError("controller state or journal already exists; refusing to execute twice")
    run_root.mkdir(parents=True, exist_ok=True)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(journal_path, "x", encoding="utf-8"):
            pass
    except FileExistsError as error:
        raise AdmissionError("another controller already owns this journal") from error
    log_dir.mkdir(exist_ok=True)

    records = {job["job_id"]: {
        "job_id": job["job_id"], "role": job["role"], "depends_on": list(job["depends_on"]),
        "output_dir": job["output_dir"], "success_artifact": job["success_artifact"],
        "timeout_seconds": job["timeout_seconds"], "status": "pending", "pid": None,
        "started_at": None, "ended_at": None, "returncode": None, "wall_seconds": None,
        "terminal_reason": None, "stdout_log": str(log_dir / f"{job['job_id']}.stdout.log"),
        "stderr_log": str(log_dir / f"{job['job_id']}.stderr.log"),
    } for job in jobs}
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION, "status": STATUS_RUNNING, "plan_digest": plan["plan_digest"],
        "plan_path": context["plan_path"], "plan_sha256": context["plan_sha256"], "controller_pid": os.getpid(),
        "absolute_deadline_utc": plan["absolute_deadline_utc"], "total_budget_seconds": budget,
        "started_at": utc_iso(now()), "updated_at": None, "heartbeat_at": None, "ended_at": None,
        "elapsed_budget_seconds": 0.0, "current_job": None, "stop_reason": None, "error": None,
        "journal_path": str(journal_path), "jobs": [records[job["job_id"]] for job in jobs],
    }

    def persist() -> None:
        state["updated_at"] = utc_iso(now())
        state["elapsed_budget_seconds"] = monotonic() - began
        _atomic_write_json(state_path, state, sleep)

    def remaining() -> float:
        return min(budget - (monotonic() - began), deadline - now())

    def journal(event: str, job: dict[str, Any], **fields: Any) -> None:
        _append_journal(journal_path, {"schema_version": JOURNAL_SCHEMA_VERSION, "event": event,
                                       "plan_digest": plan["plan_digest"], "controller_pid": os.getpid(),
                                       "job_id": job["job_id"], "role": job["role"], **fields})

    def wait_for_gpu(job: dict[str, Any], record: dict[str, Any]) -> None:
        gate = {"polls": 0, "last_free_mib": None, "last_error": None, "required_free_mib": plan["minimum_free_gpu_mib"],
                "started_at": utc_iso(now()), "admitted_at": None}
        record.update(status="waiting_for_gpu", gpu_wait=gate)
        persist()
        while True:
            if remaining() <= 0:
                raise _QueueStop(STATUS_BOUND, f"deadline or total budget expired while job {job['job_id']} waited for GPU idle")
            free, error = _probe_free_mib(provider)
            gate.update(polls=gate["polls"] + 1, last_free_mib=free, last_error=error, last_poll_at=utc_iso(now()))
            if free is not None and free >= plan["minimum_free_gpu_mib"]:
                gate["admitted_at"] = utc_iso(now())
                persist()
                return
            persist()
            left = remaining()
            if left <= 0:
                raise _QueueStop(STATUS_BOUND, f"deadline or total budget expired while job {job['job_id']} waited for GPU idle")
            sleep(min(plan["busy_wait_seconds"], left))

    def supervise(child: Any, job: dict[str, Any], started: float) -> tuple[str, int | None, str | None]:
        last_beat = monotonic()
        while True:
            returncode = child.poll()
            if returncode is not None:
                return "exited", returncode, None
            elapsed = monotonic() - started
            if elapsed >= job["timeout_seconds"]:
                code, method = _stop_owned_child(child, terminate_grace_seconds)
                return "timeout", code, method
            left = remaining()
            if left <= 0:
                code, method = _stop_owned_child(child, terminate_grace_seconds)
                return "deadline_or_budget", code, method
            if monotonic() - last_beat >= heartbeat_interval_seconds:
                state["heartbeat_at"] = utc_iso(now())
                persist()
                last_beat = monotonic()
            sleep(max(0.0, min(poll_interval_seconds, job["timeout_seconds"] - elapsed, left)))

    child: Any = None
    active: dict[str, Any] | None = None
    active_started = 0.0
    try:
        persist()
        for job in jobs:
            record = records[job["job_id"]]
            for dependency in job["depends_on"]:
                if records[dependency]["status"] != "succeeded":
                    record.update(status="failed_dependency", terminal_reason=f"dependency {dependency} did not succeed")
                    raise _QueueStop(STATUS_FAILED, f"job {job['job_id']} dependency {dependency} did not succeed")
            wait_for_gpu(job, record)
            if remaining() <= 0:
                raise _QueueStop(STATUS_BOUND, f"deadline or total budget expired before job {job['job_id']} started")
            resume = _single_flag(job["command"], "--resume", job["job_id"]) is not None
            if job["_artifact"].exists() or (job["role"] != "verify" and job["_output"].exists() != resume):
                record.update(status="failed_preflight", terminal_reason="stale success artifact or unexpected output_dir state")
                raise _QueueStop(STATUS_FAILED, f"job {job['job_id']} found a stale artifact or output_dir before spawn")
            with open(record["stdout_log"], "xb") as stdout, open(record["stderr_log"], "xb") as stderr:
                options: dict[str, Any] = {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}
                active_started, started_wall = monotonic(), now()
                try:
                    child = popen(list(job["command"]), cwd=str(run_root), stdin=subprocess.DEVNULL,
                                  stdout=stdout, stderr=stderr, **options)
                except OSError as error:
                    record.update(status="failed_spawn", terminal_reason=f"{type(error).__name__}: {error}")
                    raise _QueueStop(STATUS_FAILED, f"job {job['job_id']} could not be spawned") from error
                active = job
                journal("spawned", job, pid=child.pid, command=list(job["command"]), started_at=utc_iso(started_wall),
                        cwd=str(run_root), creationflags=options.get("creationflags", 0))
                record.update(status="running", pid=child.pid, started_at=utc_iso(started_wall))
                if job["role"] == "train":
                    record["training_pid"] = child.pid
                elif job["role"] == "verify":
                    training_pid = records[job["depends_on"][0]]["pid"]
                    record.update(training_pid=training_pid, verification_pid=child.pid,
                                  pid_distinct_from_training=child.pid != training_pid)
                state["current_job"] = job["job_id"]
                persist()
                terminal, returncode, stop_method = supervise(child, job, active_started)
                ended_wall = now()
                wall = monotonic() - active_started
                journal("exited", job, pid=child.pid, returncode=returncode, ended_at=utc_iso(ended_wall),
                        terminal=terminal, stop_method=stop_method, wall_seconds=wall)
                child, active = None, None
            record.update(returncode=returncode, ended_at=utc_iso(ended_wall), wall_seconds=wall,
                          stop_method=stop_method)
            state["current_job"] = None
            if terminal == "timeout":
                record.update(status="failed_timeout", terminal_reason=f"exceeded timeout_seconds {job['timeout_seconds']}")
                raise _QueueStop(STATUS_FAILED, f"job {job['job_id']} timed out and was stopped")
            if terminal == "deadline_or_budget":
                record.update(status="terminated_deadline_or_budget", terminal_reason="absolute deadline or total budget expired")
                raise _QueueStop(STATUS_BOUND, f"deadline or total budget expired while job {job['job_id']} ran")
            if returncode != 0:
                record.update(status="failed_nonzero_exit", terminal_reason=f"returncode {returncode}")
                raise _QueueStop(STATUS_FAILED, f"job {job['job_id']} exited with returncode {returncode}")
            try:
                record["artifact"] = evaluate_success_artifact(job)
            except (OSError, ValueError) as error:
                record.update(status="failed_success_artifact", terminal_reason=str(error))
                raise _QueueStop(STATUS_FAILED, f"job {job['job_id']} success artifact rejected: {error}") from error
            record.update(status="succeeded", terminal_reason="returncode 0 and success artifact accepted")
            persist()
        state.update(status=STATUS_COMPLETED, stop_reason=None, ended_at=utc_iso(now()))
        persist()
        return state
    except BaseException as error:
        if isinstance(error, _QueueStop):
            state.update(status=error.status, stop_reason=error.reason)
        else:
            state.update(status=STATUS_FAILED, stop_reason="controller error", error=f"{type(error).__name__}: {error}")
        if child is not None and active is not None:
            code, method = _stop_owned_child(child, terminate_grace_seconds)
            record = records[active["job_id"]]
            record.update(status="failed_controller_error", returncode=code, stop_method=method,
                          ended_at=utc_iso(now()), wall_seconds=monotonic() - active_started,
                          terminal_reason="controller error stopped its own child")
            try:
                journal("exited", active, pid=child.pid, returncode=code, ended_at=record["ended_at"],
                        terminal="controller_error", stop_method=method, wall_seconds=record["wall_seconds"])
            except OSError as journal_error:
                state["journal_error"] = str(journal_error)
        for record in records.values():
            if record["status"] in ("pending", "waiting_for_gpu"):
                record["status"] = "not_started"
        state.update(current_job=None, ended_at=utc_iso(now()))
        try:
            persist()
        except OSError as persist_error:
            state["persist_error"] = str(persist_error)
        if not isinstance(error, Exception):
            raise
        return state


def exit_code(status: str) -> int:
    return {STATUS_COMPLETED: EXIT_OK, STATUS_BOUND: EXIT_BOUND}.get(status, EXIT_FAILED)


def main(argv: list[str] | None = None, **execute_options: Any) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--admit-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        context = admit(args.plan, now=execute_options.get("now", time.time))
        if args.admit_only:
            print(json.dumps({"status": "admitted_no_worker", "plan_digest": context["plan"]["plan_digest"]}, sort_keys=True), flush=True)
            return EXIT_OK
        state = execute(context, **execute_options)
    except AdmissionError as error:
        print(json.dumps({"status": "admission_rejected", "reason": str(error)}, sort_keys=True), flush=True)
        return EXIT_REJECTED
    print(json.dumps({"status": state["status"], "plan_digest": state["plan_digest"], "state_path": str(context["state_path"]),
                      "stop_reason": state["stop_reason"]}, sort_keys=True), flush=True)
    return exit_code(state["status"])


if __name__ == "__main__":
    sys.exit(main())
