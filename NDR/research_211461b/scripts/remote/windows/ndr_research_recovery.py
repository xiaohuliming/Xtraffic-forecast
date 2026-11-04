"""One audited continuation of the fixed eight-task Windows NDR development run.

The original plan, trainer and failed controller are immutable. Admission needs
an incident receipt and a complete byte-preserved copy of the old result tree.
Only the interrupted task uses last.pt. All later tasks use the original plan.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping


EXPECTED_PLAN = "ad530e29fdcf2bd0cdabdd739349814e805f12e11cfab6341e9180651807672b"
EXPECTED_DEADLINE = "2026-10-15T08:10:22.442895+00:00"
TASK_IDS = (
    "alameda_baseline_seed42",
    "contracosta_baseline_seed42",
    "alameda_mask_signed_seed42",
    "contracosta_mask_signed_seed42",
    "alameda_sanitized_control_seed42",
    "contracosta_sanitized_control_seed42",
    "alameda_projected_oracle_seed42",
    "contracosta_projected_oracle_seed42",
)
BUNDLE = "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5"
INCIDENT_SCHEMA = "ndr_stage1_recovery_incident_v1"
PRESERVATION_SCHEMA = "ndr_stage1_preservation_v1"
DESKTOP_PROCESS_NAMES = frozenset(
    {
        "dwm.exe",
        "startmenuexperiencehost.exe",
        "searchhost.exe",
        "explorer.exe",
        "nvidia overlay.exe",
        "nvcontainer.exe",
        "lockapp.exe",
        "armourycrate.exe",
        "phoneexperiencehost.exe",
        "textinputhost.exe",
        "steamwebhelper.exe",
        "msedgewebview2.exe",
        "baidunetdiskunite.exe",
        "flow.launcher.exe",
        "mythcool.exe",
        "shellexperiencehost.exe",
        "promecefpluginhost.exe",
        "wallpaper32.exe",
        "hipsdaemon.exe",
        "runtimebroker.exe",
        "applicationframehost.exe",
        "shellhost.exe",
        "chrome.exe",
        "msedge.exe",
        "taskmgr.exe",
        "nvcpui.exe",
    }
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def utc_seconds(value: str) -> float:
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None or instant.utcoffset().total_seconds() != 0:
        raise ValueError("recovery timestamps must be explicit UTC instants")
    return instant.timestamp()


def safe_relative(value: str) -> str:
    relative = PurePosixPath(value.replace("\\", "/"))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or ":" in str(relative)
    ):
        raise ValueError("unsafe preservation relative path")
    return str(relative)


def read_bound_json(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or file_sha(path) != expected_sha:
        raise ValueError("frozen recovery receipt byte identity differs")
    return json.loads(path.read_text(encoding="utf8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class Policy:
    plan_digest: str = EXPECTED_PLAN
    deadline_utc: str = EXPECTED_DEADLINE
    task_ids: tuple[str, ...] = TASK_IDS
    job_seconds: float = 43200.0


def validate_plan(plan: dict[str, Any], policy: Policy) -> None:
    if (
        digest({key: value for key, value in plan.items() if key != "plan_digest"})
        != plan["plan_digest"]
        or plan["plan_digest"] != policy.plan_digest
    ):
        raise ValueError("recovery cannot change the frozen original plan")
    if (
        tuple(task["task_id"] for task in plan["tasks"]) != policy.task_ids
        or len(policy.task_ids) != 8
        or plan["wall_hours"] != 48
        or plan["cache_bundle_digest"] != BUNDLE
        or plan["test_evaluated"]
        or plan["automatic_transfer"]
        or plan["automatic_shutdown"]
    ):
        raise ValueError("recovery only accepts the fixed eight-task no-test design")
    root, repo = Path(plan["run_root"]).resolve(), Path(plan["repo"]).resolve()
    for name, expected in plan["source_files"].items():
        path = repo / safe_relative(name)
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(repo)
            or file_sha(path) != expected
        ):
            raise ValueError("original deployed training source changed")
    for task in plan["tasks"]:
        if not Path(task["output_dir"]).resolve().is_relative_to(root):
            raise ValueError("task output escapes the original result tree")
        for kind in ("train_command", "verify_command"):
            if task[kind][0] != plan["python"]:
                raise ValueError("recovery must retain the original Python executable")
        if (
            float(option_value(task["train_command"], "--max-wallclock-seconds"))
            != policy.job_seconds
        ):
            raise ValueError(
                "original per-task training budget is not the frozen twelve hours"
            )


def option_value(command: list[str], option: str) -> str:
    positions = [index for index, token in enumerate(command) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError("frozen command lacks one unambiguous " + option)
    return command[positions[0] + 1]


def check_preservation(
    plan: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    root = Path(plan["run_root"]).resolve()
    backup = Path(manifest["backup_root"]).resolve()
    if (
        manifest.get("schema_version") != PRESERVATION_SCHEMA
        or manifest.get("snapshot_complete") is not True
        or Path(manifest["source_root"]).resolve() != root
        or backup.is_relative_to(root)
    ):
        raise PermissionError(
            "old outputs need a complete separate preserved result tree"
        )
    declared = {}
    for row in manifest["files"]:
        relative = safe_relative(row["path"])
        if relative in declared or type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("duplicate or invalid preservation inventory")
        for directory in (root, backup):
            path = directory / relative
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(directory)
                or not path.is_file()
                or path.stat().st_size != row["bytes"]
                or file_sha(path) != row["sha256"]
            ):
                raise PermissionError(
                    "old result bytes are not completely and identically preserved"
                )
        declared[relative] = row
    if manifest.get("inventory_sha256") != digest(manifest["files"]):
        raise ValueError("preservation inventory digest differs")
    for directory in (root, backup):
        actual = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise PermissionError("preservation tree contains a symlink")
            if path.is_file():
                actual.add(path.relative_to(directory).as_posix())
        if actual != set(declared):
            raise PermissionError("preservation omits or adds old result files")
    if "results/queue_status.json" not in declared or not any(
        name.startswith("logs/") for name in declared
    ):
        raise PermissionError("failed queue and original logs are not preserved")
    return declared


def audit_checkpoint(
    path: Path, plan: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    """CPU-only deserialization; the original worker repeats live cache/runtime admission."""
    import torch

    marker = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if (
        path.stat().st_size != marker["size_bytes"]
        or file_sha(path) != marker["sha256"]
    ):
        raise ValueError("last checkpoint and its committed sidecar differ")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    required = {
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "loaders",
        "progress",
        "data_identity",
        "model_factory",
    }
    if checkpoint.get("schema_version") != 1 or not required <= checkpoint.keys():
        raise ValueError("current task has no complete original resumable checkpoint")
    if (
        digest(config) != checkpoint["config_digest"]
        or marker["config_digest"] != checkpoint["config_digest"]
    ):
        raise ValueError("last checkpoint frozen configuration differs")
    if any(
        config[name] != task[name] for name in ("region", "variant", "seed", "epochs")
    ):
        raise ValueError("last checkpoint belongs to a different frozen task")
    if (
        config["claim_scope"] != "validation_development_only"
        or config["max_train_batches"] is not None
        or config["max_val_batches"] is not None
        or config["test_evaluation"]
        or config["early_stopping"]
        or config["precision"] != "FP32"
        or config["data_identity"] != checkpoint["data_identity"]
        or checkpoint["data_identity"]["cache_bundle_digest"] != BUNDLE
        or config["runtime"]["device_type"] != "cuda"
    ):
        raise ValueError(
            "recovery checkpoint is not the original complete no-test CUDA recipe"
        )
    source = config["source_snapshot"]
    if digest(source["files"]) != source["files_digest"] or any(
        row["sha256"] != plan["source_files"].get(name)
        for name, row in source["files"].items()
    ):
        raise ValueError("last checkpoint source differs from the pinned deployment")
    progress = checkpoint["progress"]
    train_count = checkpoint["data_identity"]["split_counts"]["train"]
    batches = math.ceil(train_count / config["batch_size"])
    cursor = checkpoint["loaders"]["train"]["next_batch"]
    if (
        type(progress["epoch"]) is not int
        or not 0 <= progress["epoch"] < task["epochs"]
        or progress["stage"] not in ("start", "train", "validate")
        or len(progress["history"]) != progress["epoch"]
        or type(cursor) is not int
        or not 0 <= cursor <= batches
        or checkpoint["scheduler"]["T_max"] != task["epochs"] * batches
        or checkpoint["scheduler"]["last_epoch"] != progress["updates"]
    ):
        raise ValueError(
            "checkpoint epoch, scheduler or committed cursor is inconsistent"
        )
    if progress["stage"] != "start" and (
        progress["partial_epoch"]["batches"] != cursor
        or progress["stage"] == "validate"
        and cursor != batches
    ):
        raise ValueError(
            "checkpoint partial optimizer progress differs from its loader cursor"
        )
    for split in ("train", "val"):
        loader = checkpoint["loaders"][split]
        order = loader["order"]
        if (
            loader["split"] != split
            or loader["batch_size"] != config["batch_size"]
            or loader["drop_last"] is not False
            or len(order) != checkpoint["data_identity"]["split_counts"][split]
            or len(torch.unique(order)) != len(order)
            or "rng" not in loader
        ):
            raise ValueError(
                "checkpoint loader order/RNG does not describe the complete original split"
            )
    if not {"python", "numpy", "torch_cpu", "torch_cuda"} <= checkpoint["rng"].keys():
        raise ValueError("checkpoint lacks the original full RNG state")
    if any(
        not torch.isfinite(value).all()
        for value in checkpoint["model"].values()
        if value.is_floating_point()
    ):
        raise ValueError("last checkpoint model state is nonfinite")
    return {
        "sha256": marker["sha256"],
        "config_digest": checkpoint["config_digest"],
        "progress": progress,
        "runtime": config["runtime"],
        "data_identity": checkpoint["data_identity"],
        "source_files_digest": source["files_digest"],
    }


def budget_remaining(
    incident: dict[str, Any], checkpoint: dict[str, Any], policy: Policy
) -> float:
    accounting = incident["current_task_budget"]
    start, stopped = (
        utc_seconds(accounting["started_at_utc"]),
        utc_seconds(accounting["stopped_at_utc"]),
    )
    if (
        accounting.get("includes_uncommitted_optimizer_work") is not True
        or stopped < start
        or not accounting.get("evidence_paths")
        or accounting.get("original_limit_seconds") != policy.job_seconds
    ):
        raise PermissionError(
            "current budget must include the interrupted epoch's unsaved work"
        )
    minimum_elapsed = max(
        stopped - start, float(checkpoint["progress"]["elapsed_seconds"])
    )
    elapsed = float(accounting["accounted_elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < minimum_elapsed or minimum_elapsed < 0:
        raise ValueError(
            "frozen task budget does not cover its full original worker lifetime"
        )
    remaining = policy.job_seconds - elapsed
    if remaining <= 0:
        raise TimeoutError("the original task budget is exhausted")
    return remaining


def known_desktop_process(process: str) -> bool:
    path = process.replace("\\", "/").casefold()
    while "//" in path:
        path = path.replace("//", "/")
    filename = path.rsplit("/", 1)[-1]
    if filename in DESKTOP_PROCESS_NAMES:
        return True
    # Signed Tencent WebHelper/Recorder identities were inspected in this incident.
    # Restrict these generic names to their observed product paths.
    return path.endswith("/wegame/qbblinktrial/browser.exe") or (
        "/wegameapps/rail_apps/" in path
        and path.endswith("/aclos/cross/recorder-release/recorder.exe")
    )


class WindowsProbes:
    def live(self, pid: int) -> bool:
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if kernel.GetLastError() == 87:
                return False
            raise PermissionError("original process handle cannot be inspected")
        code = ctypes.c_ulong()
        try:
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise PermissionError("original process exit state cannot be inspected")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)

    def gpu(self) -> dict[str, Any]:
        output = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                errors="replace",
                timeout=15,
            )
            .strip()
            .splitlines()
        )
        if len(output) != 1:
            raise PermissionError("recovery requires the single original RTX 5080")
        name, free, usage = [item.strip() for item in output[0].split(",")]
        apps = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                errors="replace",
                timeout=15,
            )
            .strip()
            .splitlines()
        )
        records, python_pids, unknown_pids = [], [], []
        for row in csv.reader(apps):
            if row:
                if len(row) != 2:
                    raise ValueError("GPU process identity cannot be interpreted")
                pid, process = int(row[0].strip()), row[1].strip()
                records.append({"pid": pid, "process_name": process})
                filename = process.replace("\\", "/").rsplit("/", 1)[-1].lower()
                if (
                    filename in {"python", "python.exe", "pythonw.exe"}
                    and pid != os.getpid()
                ):
                    python_pids.append(pid)
                elif pid != os.getpid() and not known_desktop_process(process):
                    unknown_pids.append(pid)
        return {
            "name": name,
            "free_mib": float(free),
            "utilization": float(usage),
            "processes": records,
            "training_python_pids": sorted(set(python_pids)),
            "unknown_gpu_owner_pids": unknown_pids,
            "wddm_desktop_processes_do_not_imply_cuda_competition": True,
        }


def require_idle(
    probes: Any, incident: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    for process in incident["original_processes"]:
        if process["alive_at_snapshot"] is not False or probes.live(process["pid"]):
            raise PermissionError(
                "an original supervisor or worker is alive; recovery is forbidden"
            )
    gpu = probes.gpu()
    if (
        "5080" not in gpu["name"]
        or gpu["training_python_pids"]
        or gpu["unknown_gpu_owner_pids"]
        or gpu["free_mib"] < plan["minimum_free_gpu_mib"]
        or gpu["utilization"] >= 65
    ):
        raise PermissionError(
            "the GPU has a competing or unverified owner; none will be stopped"
        )
    return gpu


def setup_paths(repo: str) -> Path:
    """Keep new admission helpers outside the immutable deployed trainer repo."""
    recovery_code = Path(__file__).resolve().parents[3]
    for entry in (str(recovery_code), str(Path(repo).resolve())):
        if entry in sys.path:
            sys.path.remove(entry)
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.path.insert(0, str(recovery_code))
    return recovery_code


def completed_verifier(
    root: Path, inventory: dict[str, Any], plan: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    setup_paths(plan["repo"])
    from scripts.retrieve_ndr_development import verify_task_tree

    return verify_task_tree(root, inventory, plan, task)


def admission(
    plan: dict[str, Any],
    incident: dict[str, Any],
    manifest: dict[str, Any],
    *,
    now: float,
    probes: Any,
    policy: Policy = Policy(),
    task_verifier: Callable = completed_verifier,
) -> dict[str, Any]:
    validate_plan(plan, policy)
    queue = incident["queue_snapshot"]
    if (
        incident.get("schema_version") != INCIDENT_SCHEMA
        or incident.get("plan_digest") != policy.plan_digest
        or incident.get("absolute_deadline_utc") != policy.deadline_utc
        or queue.get("plan_digest") != policy.plan_digest
        or queue.get("absolute_deadline_utc") != policy.deadline_utc
        or queue.get("status") != "failed"
        or queue.get("completed") != list(policy.task_ids[:2])
        or queue.get("current_task") != policy.task_ids[2]
        or queue.get("test_evaluated") is not False
    ):
        raise PermissionError(
            "incident is not the frozen failed two-of-eight predecessor"
        )
    if now >= utc_seconds(policy.deadline_utc):
        raise TimeoutError(
            "the original absolute forty-eight-hour deadline has expired"
        )
    if (
        {row["role"] for row in incident["original_processes"]}
        != {"supervisor", "worker"}
        or len(incident["original_processes"]) != 2
        or any(
            type(row["pid"]) is not int or row["pid"] <= 0
            for row in incident["original_processes"]
        )
        or {row["role"]: row["pid"] for row in incident["original_processes"]}
        != {"supervisor": queue["pid"], "worker": incident["interrupted_worker_pid"]}
    ):
        raise ValueError("incident does not bind the original process handles")
    rows = check_preservation(plan, manifest)
    old_queue = json.loads(
        (Path(manifest["backup_root"]) / "results/queue_status.json").read_text()
    )
    if old_queue != queue:
        raise ValueError("incident queue differs from the preserved failed status")
    current = plan["tasks"][2]
    prefix = "tasks/" + current["task_id"] + "/"
    if (
        not {
            prefix + name
            for name in (
                "last.pt",
                "last.pt.sha256.json",
                "config.json",
                "data_identity.json",
                "training_diagnostics.jsonl",
                "run_state.json",
            )
        }
        <= rows.keys()
    ):
        raise PermissionError(
            "interrupted task checkpoint and original history are not preserved"
        )
    verified = []
    for task in plan["tasks"][:2]:
        task_prefix = "tasks/" + task["task_id"] + "/"
        inventory = {
            "task_id": task["task_id"],
            "plan_digest": policy.plan_digest,
            "files": [
                {**entry, "path": name[len(task_prefix) :]}
                for name, entry in rows.items()
                if name.startswith(task_prefix)
            ],
        }
        verified.append(task_verifier(Path(task["output_dir"]), inventory, plan, task))
    checkpoint = audit_checkpoint(
        Path(current["output_dir"]) / "last.pt", plan, current
    )
    if (
        checkpoint["sha256"] != incident["last_checkpoint_sha256"]
        or checkpoint["config_digest"] != incident["current_config_digest"]
    ):
        raise ValueError(
            "incident does not bind the committed last checkpoint and configuration"
        )
    accounting = incident["current_task_budget"]
    log_path = "logs/" + current["task_id"] + "_train.log"
    if (
        set(accounting["evidence_paths"]) != {log_path, "results/queue_status.json"}
        or rows[log_path].get("created_at_utc") != accounting["started_at_utc"]
        or queue.get("failed_at") != accounting["stopped_at_utc"]
    ):
        raise ValueError(
            "task budget is not bound to preserved original launch/exit evidence"
        )
    remaining = budget_remaining(incident, checkpoint, policy)
    gpu = require_idle(probes, incident, plan)
    return {
        "completed": list(policy.task_ids[:2]),
        "current_task": current["task_id"],
        "checkpoint": checkpoint,
        "remaining_task_seconds": remaining,
        "gpu": gpu,
        "verified_completed_tasks": verified,
    }


def recovery_command(
    task: dict[str, Any], *, resume: bool, remaining_seconds: float
) -> list[str]:
    command = list(task["train_command"])
    position = command.index("--max-wallclock-seconds")
    command[position + 1] = str(remaining_seconds)
    if "--resume" in command:
        raise ValueError("the original frozen command unexpectedly contains resume")
    if resume:
        command += ["--resume", str(Path(task["output_dir"]) / "last.pt")]
    return command


def execute_owned(
    command: list[str],
    *,
    repo: Path,
    log: Path,
    hard_deadline: float,
    state: dict[str, Any],
    status_path: Path,
    now: Callable = time.time,
    sleep: Callable = time.sleep,
    popen: Callable = subprocess.Popen,
) -> None:
    if now() >= hard_deadline:
        raise TimeoutError("no worker may start at or after its retained deadline")
    with log.open("x", encoding="utf8") as stream:
        child = popen(
            command,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        try:
            state.update(worker_pid=child.pid, worker_command=command)
            atomic_json(status_path, state)
            while child.poll() is None:
                if now() >= hard_deadline:
                    raise TimeoutError(
                        "retained absolute or per-task budget expired; only the owned worker stopped"
                    )
                sleep(min(1.0, max(0.0, hard_deadline - now())))
            if child.returncode:
                raise RuntimeError(
                    "owned recovery worker failed; no competing process was changed"
                )
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            state.update(worker_pid=None, worker_exit_code=child.returncode)
            atomic_json(status_path, state)


def task_inventory(
    root: Path, task: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    directory = Path(task["output_dir"])
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("recovery task output contains unsafe paths")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha(path),
                }
            )
    return {
        "task_id": task["task_id"],
        "plan_digest": plan["plan_digest"],
        "files": files,
    }


def require_completed_unchanged(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    prefixes = tuple("tasks/" + task["task_id"] + "/" for task in plan["tasks"][:2])
    expected = {
        safe_relative(row["path"]): row
        for row in manifest["files"]
        if safe_relative(row["path"]).startswith(prefixes)
    }
    actual = set()
    for task in plan["tasks"][:2]:
        for path in Path(task["output_dir"]).rglob("*"):
            if path.is_symlink():
                raise ValueError("an immutable completed task acquired a symlink")
            if path.is_file():
                actual.add(path.relative_to(Path(plan["run_root"])).as_posix())
    if actual != expected.keys():
        raise ValueError(
            "the two reused completed tasks changed their preserved file set"
        )
    for name, row in expected.items():
        path = Path(plan["run_root"]) / name
        if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError("the two reused completed tasks changed preserved bytes")


def run(args: argparse.Namespace) -> None:
    if os.name != "nt":
        raise PermissionError(
            "actual recovery runs only on the authorized Windows host"
        )
    plan = json.loads(args.plan.read_text())
    incident = read_bound_json(args.incident, args.incident_sha256)
    manifest = read_bound_json(args.preservation, args.preservation_sha256)
    if incident.get("preservation_sha256") != args.preservation_sha256:
        raise ValueError("incident does not bind its preservation receipt")
    recovery_code = setup_paths(plan["repo"])
    recovery_files = {
        str(path): file_sha(path)
        for path in (
            Path(__file__).resolve(),
            recovery_code / "scripts/retrieve_ndr_development.py",
            recovery_code / "scripts/analyze_ndr_development.py",
        )
    }
    probes = WindowsProbes()
    accepted = admission(plan, incident, manifest, now=time.time(), probes=probes)
    control = args.control_root.resolve()
    if control.is_relative_to(Path(plan["run_root"]).resolve()):
        raise ValueError(
            "independent recovery control must live outside the original results"
        )
    control.mkdir(parents=True, exist_ok=False)
    lock = Path(plan["run_root"]) / "results/recovery.lock"
    with lock.open("x") as handle:
        json.dump({"pid": os.getpid(), "incident_sha256": args.incident_sha256}, handle)
    recovery_id = uuid.uuid4().hex
    state = {
        "status": "running",
        "plan_digest": plan["plan_digest"],
        "absolute_deadline_utc": EXPECTED_DEADLINE,
        "started_at": incident["queue_snapshot"]["started_at"],
        "pid": os.getpid(),
        "worker_pid": None,
        "completed": accepted["completed"],
        "current_task": accepted["current_task"],
        "phase": "recovery_admitted",
        "test_evaluated": False,
        "automatic_shutdown": False,
        "automatic_transfer": False,
        "recovery_id": recovery_id,
        "incident_sha256": args.incident_sha256,
        "preservation_sha256": args.preservation_sha256,
        "recovery_controller_sha256": file_sha(Path(__file__)),
        "original_supervisor_pid": incident["queue_snapshot"]["pid"],
        "original_deadline_retained": True,
    }
    status = Path(plan["run_root"]) / "results/queue_status.json"
    atomic_json(
        control / "admission.json",
        {key: value for key, value in accepted.items() if key != "checkpoint"},
    )
    atomic_json(control / "original_incident.json", incident)
    atomic_json(
        control / "source_provenance.json",
        {
            "controller_sha256": state["recovery_controller_sha256"],
            "original_plan_digest": plan["plan_digest"],
            "original_sources": plan["source_files"],
            "recovery_sources": recovery_files,
            "namespace_overlay": str(recovery_code),
        },
    )
    absolute = utc_seconds(EXPECTED_DEADLINE)
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise PermissionError(
            "the original keep-on idle-sleep policy could not be established"
        )
    try:
        for index, task in enumerate(plan["tasks"][2:], start=2):
            if any(
                file_sha(Path(name)) != expected
                for name, expected in recovery_files.items()
            ):
                raise ValueError("independent recovery/admission helper source changed")
            validate_plan(plan, Policy())
            require_completed_unchanged(plan, manifest)
            require_idle(probes, incident, plan)
            if time.time() >= absolute:
                raise TimeoutError("original absolute queue deadline expired")
            remaining = (
                accepted["remaining_task_seconds"]
                if index == 2
                else Policy().job_seconds
            )
            if index > 2 and Path(task["output_dir"]).exists():
                raise FileExistsError(
                    "a later frozen task already has outputs; fresh execution is forbidden"
                )
            train_started = time.time()
            hard = min(absolute, train_started + remaining)
            state.update(
                current_task=task["task_id"],
                phase="train",
                retained_task_seconds=remaining,
                task_hard_deadline_utc=datetime.fromtimestamp(
                    hard, timezone.utc
                ).isoformat(),
            )
            atomic_json(
                control / (task["task_id"] + "_budget.json"),
                {
                    "original_limit_seconds": Policy().job_seconds,
                    "previously_consumed_seconds": Policy().job_seconds - remaining,
                    "remaining_seconds": remaining,
                    "started_at_utc": datetime.fromtimestamp(
                        train_started, timezone.utc
                    ).isoformat(),
                    "hard_deadline_utc": state["task_hard_deadline_utc"],
                    "original_absolute_deadline_utc": EXPECTED_DEADLINE,
                },
            )
            execute_owned(
                recovery_command(
                    task,
                    resume=index == 2,
                    remaining_seconds=max(0.001, hard - train_started),
                ),
                repo=Path(plan["repo"]),
                log=Path(plan["run_root"])
                / "logs"
                / (task["task_id"] + "_recovery_" + recovery_id + "_train.log"),
                hard_deadline=hard,
                state=state,
                status_path=status,
            )
            summary = json.loads(
                (Path(task["output_dir"]) / "summary.json").read_text()
            )
            if (
                summary.get("completed_epochs") != task["epochs"]
                or summary.get("status")
                != "training_completed_validation_pending_independent_verification"
                or summary.get("test_evaluated") is not False
            ):
                raise RuntimeError(
                    "resumable or partial return is not completed training"
                )
            validate_plan(plan, Policy())
            require_idle(probes, incident, plan)
            state["phase"] = "verify"
            execute_owned(
                task["verify_command"],
                repo=Path(plan["repo"]),
                log=Path(plan["run_root"])
                / "logs"
                / (task["task_id"] + "_recovery_" + recovery_id + "_verify.log"),
                hard_deadline=absolute,
                state=state,
                status_path=status,
            )
            completed_verifier(
                Path(task["output_dir"]),
                task_inventory(Path(plan["run_root"]), task, plan),
                plan,
                task,
            )
            state["completed"].append(task["task_id"])
            atomic_json(status, state)
        validate_plan(plan, Policy())
        require_completed_unchanged(plan, manifest)
        if any(
            file_sha(Path(name)) != expected
            for name, expected in recovery_files.items()
        ):
            raise ValueError("independent recovery source changed before postflight")
        files = [
            {**row, "path": "tasks/" + task["task_id"] + "/" + row["path"]}
            for task in plan["tasks"]
            for row in task_inventory(Path(plan["run_root"]), task, plan)["files"]
        ]
        atomic_json(
            Path(plan["run_root"]) / "results/postflight.json",
            {
                "status": "verified_development_train_val_only",
                "plan_digest": plan["plan_digest"],
                "tasks": state["completed"],
                "files": files,
                "test_evaluated": False,
                "source_retained": True,
                "incident_sha256": args.incident_sha256,
                "recovery_controller_sha256": state["recovery_controller_sha256"],
            },
        )
        state.update(
            status="completed",
            phase="source_postflight",
            current_task=None,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(status, state)
        setup_paths(plan["repo"])
        from scripts.analyze_ndr_development import admit_matrix

        atomic_json(
            control / "completed_matrix_admission.json",
            admit_matrix(Path(plan["run_root"]), plan),
        )
    except BaseException as error:
        state.update(
            status="failed",
            error=str(error),
            failed_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(status, state)
        atomic_json(control / "failure.json", state)
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--incident-sha256", required=True)
    parser.add_argument("--preservation", type=Path, required=True)
    parser.add_argument("--preservation-sha256", required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
