"""Synthetic incident/control tests; no remote, traffic or GPU work is launched."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

import pytest
import torch

from scripts.remote.windows import ndr_research_recovery as recovery


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class Probes:
    def __init__(self, live=False, competing=False):
        self.alive, self.competing = live, competing
        self.checked = []

    def live(self, pid):
        self.checked.append(pid)
        return self.alive

    def gpu(self):
        return {
            "name": "NVIDIA GeForce RTX 5080",
            "free_mib": 15000,
            "utilization": 0,
            "training_python_pids": [1234] if self.competing else [],
            "unknown_gpu_owner_pids": [],
        }


@pytest.fixture
def incident_tree(tmp_path):
    """Formal-shaped control fixture is not an accepted research result."""
    repo, root, backup = tmp_path / "repo", tmp_path / "run", tmp_path / "backup"
    repo.mkdir()
    (repo / "frozen.py").write_text("FROZEN = True\n")
    (root / "results").mkdir(parents=True)
    (root / "logs").mkdir()
    original_start = 1000.0
    stop_time = 1800.0
    deadline = 60000.0
    tasks = []
    for index, tid in enumerate(recovery.TASK_IDS):
        region = "Alameda" if index % 2 == 0 else "ContraCosta"
        variant = ("baseline", "mask_signed", "sanitized_control", "projected_oracle")[
            index // 2
        ]
        tasks.append(
            {
                "task_id": tid,
                "region": region,
                "variant": variant,
                "seed": 42,
                "epochs": 30,
                "output_dir": str(root / "tasks" / tid),
                "train_command": [
                    "FROZEN_PYTHON",
                    "train.py",
                    "--mode",
                    "train",
                    "--max-wallclock-seconds",
                    "43200",
                ],
                "verify_command": ["FROZEN_PYTHON", "train.py", "--mode", "verify"],
            }
        )
    plan = {
        "repo": str(repo),
        "run_root": str(root),
        "python": "FROZEN_PYTHON",
        "wall_hours": 48,
        "cache_bundle_digest": recovery.BUNDLE,
        "test_evaluated": False,
        "automatic_transfer": False,
        "automatic_shutdown": False,
        "minimum_free_gpu_mib": 6000,
        "tasks": tasks,
        "source_files": {"frozen.py": recovery.file_sha(repo / "frozen.py")},
    }
    plan["plan_digest"] = recovery.digest(plan)
    policy = recovery.Policy(plan["plan_digest"], utc(deadline))
    queue = {
        "status": "failed",
        "phase": "train",
        "plan_digest": plan["plan_digest"],
        "absolute_deadline_utc": utc(deadline),
        "started_at": utc(10),
        "failed_at": utc(stop_time),
        "pid": 6232,
        "worker_pid": None,
        "completed": list(recovery.TASK_IDS[:2]),
        "current_task": recovery.TASK_IDS[2],
        "test_evaluated": False,
    }
    (root / "results/queue_status.json").write_text(json.dumps(queue))
    (root / "results/failure.txt").write_text("synthetic interrupted worker failure\n")
    for task in tasks[:2]:
        Path(task["output_dir"]).mkdir(parents=True)
        (Path(task["output_dir"]) / "immutable.txt").write_text(task["task_id"])
    current = tasks[2]
    directory = Path(current["output_dir"])
    directory.mkdir()
    log_name = "logs/" + current["task_id"] + "_train.log"
    (root / log_name).write_text("epoch24 batch900, synthetic control receipt only\n")
    identity = {
        "cache_bundle_digest": recovery.BUNDLE,
        "split_counts": {"train": 4, "val": 2},
    }
    source = {
        "files": {
            "frozen.py": {
                "sha256": plan["source_files"]["frozen.py"],
                "size_bytes": (repo / "frozen.py").stat().st_size,
            }
        }
    }
    source["files_digest"] = recovery.digest(source["files"])
    config = {key: current[key] for key in ("region", "variant", "seed", "epochs")}
    config.update(
        claim_scope="validation_development_only",
        max_train_batches=None,
        max_val_batches=None,
        test_evaluation=False,
        early_stopping=False,
        precision="FP32",
        data_identity=identity,
        runtime={"device_type": "cuda", "device_name": "synthetic_single_5080"},
        source_snapshot=source,
        batch_size=2,
    )

    def loader(split, count):
        return {
            "split": split,
            "batch_size": 2,
            "drop_last": False,
            "next_batch": count // 2,
            "order": torch.arange(count),
            "rng": {"synthetic_state": 1},
        }

    checkpoint = {
        "schema_version": 1,
        "config": config,
        "config_digest": recovery.digest(config),
        "model": {"weight": torch.ones(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"T_max": 60, "last_epoch": 2},
        "data_identity": identity,
        "model_factory": {},
        "loaders": {"train": loader("train", 4), "val": loader("val", 2)},
        "rng": {
            "python": random.getstate(),
            "numpy": {},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
        },
        "progress": {
            "epoch": 1,
            "stage": "start",
            "updates": 2,
            "history": [{}],
            "partial_epoch": {"batches": 2},
            "elapsed_seconds": 700.0,
        },
    }
    cp = directory / "last.pt"
    torch.save(checkpoint, cp)
    cp.with_suffix(".pt.sha256.json").write_text(
        json.dumps(
            {
                "sha256": recovery.file_sha(cp),
                "size_bytes": cp.stat().st_size,
                "config_digest": checkpoint["config_digest"],
            }
        )
    )
    for name, obj in (
        ("config.json", config),
        ("data_identity.json", identity),
        ("run_state.json", {"status": "failed"}),
    ):
        (directory / name).write_text(json.dumps(obj))
    (directory / "training_diagnostics.jsonl").write_text("{}\n")
    shutil.copytree(root, backup)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            row = {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": recovery.file_sha(path),
            }
            if row["path"] == log_name:
                row["created_at_utc"] = utc(original_start)
            files.append(row)
    manifest = {
        "schema_version": recovery.PRESERVATION_SCHEMA,
        "source_root": str(root),
        "backup_root": str(backup),
        "snapshot_complete": True,
        "files": files,
        "inventory_sha256": recovery.digest(files),
    }
    incident = {
        "schema_version": recovery.INCIDENT_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "absolute_deadline_utc": utc(deadline),
        "queue_snapshot": queue,
        "interrupted_worker_pid": 48172,
        "original_processes": [
            {"role": "supervisor", "pid": 6232, "alive_at_snapshot": False},
            {"role": "worker", "pid": 48172, "alive_at_snapshot": False},
        ],
        "last_checkpoint_sha256": recovery.file_sha(cp),
        "current_config_digest": checkpoint["config_digest"],
        "current_task_budget": {
            "started_at_utc": utc(original_start),
            "stopped_at_utc": utc(stop_time),
            "original_limit_seconds": 43200.0,
            "accounted_elapsed_seconds": 800.0,
            "includes_uncommitted_optimizer_work": True,
            "evidence_paths": [log_name, "results/queue_status.json"],
        },
    }
    calls = []

    def verifier(directory, inventory, actual_plan, task):
        calls.append(task["task_id"])
        assert directory == Path(task["output_dir"])
        assert inventory["plan_digest"] == actual_plan["plan_digest"]
        return {
            "status": "synthetic_verified_completed_task",
            "task_id": task["task_id"],
        }

    return plan, incident, manifest, policy, verifier, calls, checkpoint


def admit(fixture, **kwargs):
    plan, incident, manifest, policy, verifier, _, _ = fixture
    return recovery.admission(
        plan,
        incident,
        manifest,
        now=2000,
        probes=Probes(),
        policy=policy,
        task_verifier=verifier,
        **kwargs,
    )


def test_preserved_incident_admission_reuses_two_and_reserves_unsaved_work(
    incident_tree,
):
    result = admit(incident_tree)
    assert result["completed"] == list(recovery.TASK_IDS[:2])
    assert incident_tree[5] == list(recovery.TASK_IDS[:2])
    assert result["remaining_task_seconds"] == 42400
    assert (
        result["remaining_task_seconds"]
        < 43200 - incident_tree[6]["progress"]["elapsed_seconds"]
    )
    plan, _, _, _, _, _, _ = incident_tree
    before = list(plan["tasks"][2]["train_command"])
    command = recovery.recovery_command(
        plan["tasks"][2],
        resume=True,
        remaining_seconds=result["remaining_task_seconds"],
    )
    assert command[-2:] == [
        "--resume",
        str(Path(plan["tasks"][2]["output_dir"]) / "last.pt"),
    ]
    assert recovery.option_value(command, "--max-wallclock-seconds") == "42400.0"
    assert plan["tasks"][2]["train_command"] == before
    fresh = recovery.recovery_command(
        plan["tasks"][3], resume=False, remaining_seconds=43200
    )
    assert "--resume" not in fresh


@pytest.mark.parametrize(
    "changed", ["source", "backup", "new_source_file", "no_complete_copy"]
)
def test_changed_or_partial_preservation_blocks_before_any_worker(
    incident_tree, changed
):
    plan, _, manifest, _, _, _, _ = incident_tree
    relative = "results/failure.txt"
    if changed == "source":
        (Path(plan["run_root"]) / relative).write_text("changed")
    elif changed == "backup":
        (Path(manifest["backup_root"]) / relative).write_text("changed")
    elif changed == "new_source_file":
        (Path(plan["run_root"]) / "unpreserved.txt").write_text("new")
    else:
        manifest["snapshot_complete"] = False
    with pytest.raises(PermissionError):
        admit(incident_tree)


def test_wrong_checkpoint_binding_and_elapsed_reset_are_rejected(incident_tree):
    incident = incident_tree[1]
    incident["last_checkpoint_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="committed last"):
        admit(incident_tree)
    incident["last_checkpoint_sha256"] = recovery.file_sha(
        Path(incident_tree[0]["tasks"][2]["output_dir"]) / "last.pt"
    )
    incident["current_task_budget"]["accounted_elapsed_seconds"] = 700
    with pytest.raises(ValueError, match="full original worker lifetime"):
        admit(incident_tree)


@pytest.mark.parametrize("live,competing", [(True, False), (False, True)])
def test_original_handle_or_gpu_competitor_blocks_recovery(
    incident_tree, live, competing
):
    plan, incident, manifest, policy, verify, _, _ = incident_tree
    with pytest.raises(PermissionError):
        recovery.admission(
            plan,
            incident,
            manifest,
            now=2000,
            probes=Probes(live, competing),
            policy=policy,
            task_verifier=verify,
        )


def test_original_deadline_and_task_budget_cannot_restart(incident_tree):
    plan, incident, manifest, policy, verify, _, checkpoint = incident_tree
    with pytest.raises(TimeoutError, match="absolute"):
        recovery.admission(
            plan,
            incident,
            manifest,
            now=recovery.utc_seconds(policy.deadline_utc),
            probes=Probes(),
            policy=policy,
            task_verifier=verify,
        )
    incident["current_task_budget"]["accounted_elapsed_seconds"] = 43200
    with pytest.raises(TimeoutError, match="task budget"):
        recovery.budget_remaining(incident, checkpoint, policy)
    incident["absolute_deadline_utc"] = utc(90000)
    with pytest.raises(PermissionError, match="frozen failed"):
        admit(incident_tree)


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Worker:
    pid = 999

    def __init__(self, done=False):
        self.returncode = 0 if done else None
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_hard_deadline_stops_only_created_worker_and_never_changes_old_logs(tmp_path):
    clock, worker = Clock(), Worker()
    original = tmp_path / "original.log"
    original.write_text("preserved original bytes")
    old_sha = recovery.file_sha(original)
    created = []

    def popen(command, **kwargs):
        created.append(command)
        return worker

    state = {}
    with pytest.raises(TimeoutError, match="retained absolute"):
        recovery.execute_owned(
            ["only_owned"],
            repo=tmp_path,
            log=tmp_path / "new.log",
            hard_deadline=3,
            state=state,
            status_path=tmp_path / "status.json",
            now=clock.now,
            sleep=clock.sleep,
            popen=popen,
        )
    assert created == [["only_owned"]] and worker.terminated == 1 and not worker.killed
    assert clock.value == 3 and state["worker_pid"] is None
    assert recovery.file_sha(original) == old_sha


def test_existing_log_or_expired_start_never_spawns(tmp_path):
    path = tmp_path / "existing.log"
    path.write_text("old")

    def forbidden(*args, **kwargs):
        pytest.fail("a prohibited worker was spawned")

    with pytest.raises(FileExistsError):
        recovery.execute_owned(
            ["never"],
            repo=tmp_path,
            log=path,
            hard_deadline=10,
            state={},
            status_path=tmp_path / "state",
            now=lambda: 0,
            popen=forbidden,
        )
    with pytest.raises(TimeoutError):
        recovery.execute_owned(
            ["never"],
            repo=tmp_path,
            log=tmp_path / "fresh",
            hard_deadline=10,
            state={},
            status_path=tmp_path / "state",
            now=lambda: 10,
            popen=forbidden,
        )


def test_controller_exception_does_not_orphan_owned_worker(tmp_path):
    worker = Worker()

    def fail_sleep(seconds):
        raise RuntimeError("synthetic controller interruption")

    with pytest.raises(RuntimeError, match="controller interruption"):
        recovery.execute_owned(
            ["owned"],
            repo=tmp_path,
            log=tmp_path / "new.log",
            hard_deadline=10,
            state={},
            status_path=tmp_path / "state.json",
            now=lambda: 0,
            sleep=fail_sleep,
            popen=lambda *a, **k: worker,
        )
    assert worker.terminated == 1


@pytest.mark.parametrize("extra", ["desktop", "python", "unknown"])
def test_wddm_gui_and_cpu_observers_are_not_cuda_owners(monkeypatch, extra):
    calls = []
    rows = (
        "2076, C:\\Windows\\System32\\dwm.exe\n23752, D:\\��\\BaiduNetdiskUnite.exe\n"
    )
    if extra == "python":
        rows += "999123, C:\\Python313\\python.exe\n"
    elif extra == "unknown":
        rows += "999124, C:\\Unknown\\cuda_worker.exe\n"

    def capture(command, **kwargs):
        calls.append(command)
        assert kwargs["errors"] == "replace"
        if command[0] != "nvidia-smi":
            pytest.fail(
                "CPU-only Python process inventory was treated as CUDA ownership"
            )
        if "--query-gpu=name,memory.free,utilization.gpu" in command:
            return "NVIDIA GeForce RTX 5080, 15000, 6\n"
        return rows

    monkeypatch.setattr(recovery.subprocess, "check_output", capture)
    gpu = recovery.WindowsProbes().gpu()
    assert len(calls) == 2
    assert gpu["training_python_pids"] == ([999123] if extra == "python" else [])
    assert gpu["unknown_gpu_owner_pids"] == ([999124] if extra == "unknown" else [])
    assert gpu["wddm_desktop_processes_do_not_imply_cuda_competition"]


def test_reused_task_bytes_cannot_change_after_admission(incident_tree):
    plan, _, manifest, _, _, _, _ = incident_tree
    recovery.require_completed_unchanged(plan, manifest)
    path = Path(plan["tasks"][0]["output_dir"]) / "immutable.txt"
    path.write_text("changed after admission")
    with pytest.raises(ValueError, match="preserved bytes"):
        recovery.require_completed_unchanged(plan, manifest)


def test_observed_tencent_wddm_paths_are_not_generic_cuda_exceptions(monkeypatch):
    rows = (
        "20856, D:\\\\Program Files (x86)\\\\WeGame\\\\QBBlinkTrial\\\\browser.exe\n"
        "28420, D:\\\\WeGameApps\\\\rail_apps\\\\��\\\\ACLOS\\\\Cross\\\\recorder-release\\\\recorder.exe\n"
        "28424, C:\\Unknown\\browser.exe\n"
        "28428, C:\\Unknown\\recorder.exe\n"
    )

    def capture(command, **kwargs):
        if "--query-gpu=name,memory.free,utilization.gpu" in command:
            return "NVIDIA GeForce RTX 5080, 14476, 6\n"
        return rows

    monkeypatch.setattr(recovery.subprocess, "check_output", capture)
    gpu = recovery.WindowsProbes().gpu()
    assert gpu["training_python_pids"] == []
    assert gpu["unknown_gpu_owner_pids"] == [28424, 28428]
