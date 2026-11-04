"""Synthetic continuation-control tests; no remote, traffic or GPU work."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts.remote.windows import ndr_research_continuation as continuation
from scripts.remote.windows import ndr_research_recovery as provider


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


class Probes:
    def __init__(self, snapshots=None, live=False):
        self.snapshots = list(snapshots or [self.idle()])
        self.alive = live
        self.live_calls = []
        self.gpu_calls = 0

    @staticmethod
    def idle(**changes):
        return {
            "name": "NVIDIA GeForce RTX 5080",
            "free_mib": 15000,
            "utilization": 0,
            "training_python_pids": [],
            "unknown_gpu_owner_pids": [],
            **changes,
        }

    def live(self, pid):
        self.live_calls.append(pid)
        return self.alive

    def gpu(self):
        result = self.snapshots[min(self.gpu_calls, len(self.snapshots) - 1)]
        self.gpu_calls += 1
        return deepcopy(result)


class Clock:
    def __init__(self, value=2000):
        self.value = float(value)
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


def freeze(fixture):
    """Bind deliberately synthetic files to exact independent preserved copies."""
    root = Path(fixture.plan["run_root"])
    write_json(root / "results/queue_status.json", fixture.incident["queue_snapshot"])
    current = Path(fixture.plan["tasks"][3]["output_dir"])
    fixture.incident.update(
        current_summary_sha256=provider.file_sha(current / "summary.json"),
        current_best_checkpoint_sha256=provider.file_sha(current / "best.pt"),
        current_last_checkpoint_sha256=provider.file_sha(current / "last.pt"),
        current_config_digest=fixture.config_digest,
    )
    for directory in (fixture.backup, fixture.local_copy):
        if directory.exists():
            shutil.rmtree(directory)
        shutil.copytree(root, directory)
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": provider.file_sha(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    fixture.manifest = {
        "schema_version": "ndr_stage1_continuation_preservation_v1",
        "source_root": str(root),
        "backup_root": str(fixture.backup),
        "snapshot_complete": True,
        "files": files,
        "inventory_sha256": provider.digest(files),
    }
    fixture.local_receipt = {
        "schema_version": "ndr_stage1_continuation_local_preservation_v1",
        "status": "verified_exact_local_copy",
        "source_root": str(root),
        "local_copy_root": str(fixture.local_copy),
        "files": deepcopy(files),
        "inventory_sha256": provider.digest(files),
        "verified_at_utc": utc(1900),
    }


@pytest.fixture
def incident_tree(tmp_path):
    repo, root = tmp_path / "repo", tmp_path / "run"
    repo.mkdir()
    (repo / "frozen.py").write_text("FROZEN = True\n")
    (root / "results").mkdir(parents=True)
    (root / "logs").mkdir()
    tasks = []
    for index, task_id in enumerate(continuation.TASK_IDS):
        region = "Alameda" if index % 2 == 0 else "ContraCosta"
        variant = ("baseline", "mask_signed", "sanitized_control", "projected_oracle")[
            index // 2
        ]
        directory = root / "tasks" / task_id
        tasks.append(
            {
                "task_id": task_id,
                "region": region,
                "variant": variant,
                "seed": 42,
                "epochs": 30,
                "output_dir": str(directory),
                "train_command": [
                    "FROZEN_PYTHON",
                    "train.py",
                    "--mode",
                    "train",
                    "--region",
                    region,
                    "--variant",
                    variant,
                    "--output-dir",
                    str(directory),
                    "--device",
                    "cuda:0",
                    "--epochs",
                    "30",
                    "--seed",
                    "42",
                    "--max-wallclock-seconds",
                    "43200",
                ],
                "verify_command": [
                    "FROZEN_PYTHON",
                    "train.py",
                    "--mode",
                    "verify",
                    "--checkpoint",
                    str(directory / "best.pt"),
                    "--device",
                    "cuda:0",
                ],
                "verification_file": str(directory / "verification.json"),
                "expected_verification_status": "verified_checkpoint_complete_validation",
            }
        )
    plan = {
        "schema_version": 1,
        "protocol_version": "protocol_v3_noleak_dedup",
        "cache_schema_version": 3,
        "repo": str(repo),
        "run_root": str(root),
        "python": "FROZEN_PYTHON",
        "wall_hours": 48,
        "cache_bundle_digest": provider.BUNDLE,
        "test_evaluated": False,
        "automatic_transfer": False,
        "automatic_shutdown": False,
        "minimum_free_gpu_mib": 6000,
        "busy_wait_minutes": 30,
        "tasks": tasks,
        "source_files": {"frozen.py": provider.file_sha(repo / "frozen.py")},
    }
    plan["plan_digest"] = provider.digest(plan)
    policy = continuation.Policy(
        plan_digest=plan["plan_digest"], deadline_utc=utc(60000)
    )
    queue = {
        "status": "failed",
        "phase": "train",
        "error": "the GPU has a competing or unverified owner; none will be stopped",
        "plan_digest": plan["plan_digest"],
        "absolute_deadline_utc": policy.deadline_utc,
        "started_at": utc(10),
        "failed_at": utc(1800),
        "pid": 6232,
        "worker_pid": None,
        "worker_exit_code": 0,
        "completed": list(continuation.TASK_IDS[:3]),
        "current_task": continuation.TASK_IDS[3],
        "test_evaluated": False,
        "automatic_shutdown": False,
        "automatic_transfer": False,
    }
    incident = {
        "schema_version": "ndr_stage1_continuation_incident_v1",
        "plan_digest": plan["plan_digest"],
        "absolute_deadline_utc": policy.deadline_utc,
        "current_task_id": continuation.TASK_IDS[3],
        "queue_snapshot": queue,
        "previous_train_worker_pid": 30768,
        "original_processes": [
            {"role": "supervisor", "pid": 6232, "alive_at_snapshot": False},
            {"role": "train_worker", "pid": 30768, "alive_at_snapshot": False},
        ],
        "fourth_verification_absent": True,
        "failure_boundary": "post_train_gpu_admission",
    }
    for task in tasks[:3]:
        directory = Path(task["output_dir"])
        directory.mkdir(parents=True)
        (directory / "immutable.txt").write_text(task["task_id"])
    current = Path(tasks[3]["output_dir"])
    current.mkdir()
    source_files_digest = provider.digest(plan["source_files"])
    runtime = {"device_type": "cuda", "device_name": "synthetic_single_5080"}
    config = {key: tasks[3][key] for key in ("region", "variant", "seed", "epochs")}
    config_digest = provider.digest(config)
    write_json(current / "config.json", config)
    write_json(current / "data_identity.json", {"cache_bundle_digest": provider.BUNDLE})
    write_json(current / "source_snapshot.json", {"files_digest": source_files_digest})
    write_json(current / "run_state.json", {"status": "training_completed"})
    write_json(
        current / "summary.json",
        {
            "status": "training_completed_validation_pending_independent_verification",
            "completed_epochs": 30,
            "test_evaluated": False,
            "config_digest": config_digest,
        },
    )
    (current / "training_diagnostics.jsonl").write_text("{}\n")
    for checkpoint in ("best.pt", "last.pt"):
        path = current / checkpoint
        path.write_bytes(("synthetic " + checkpoint).encode())
        write_json(
            path.with_suffix(".pt.sha256.json"),
            {
                "sha256": provider.file_sha(path),
                "size_bytes": path.stat().st_size,
                "config_digest": config_digest,
            },
        )
    (root / "logs" / (tasks[3]["task_id"] + "_train.log")).write_text(
        "synthetic train exit 0 after epoch 30\n"
    )
    fixture = SimpleNamespace(
        plan=plan,
        incident=incident,
        policy=policy,
        backup=tmp_path / "backup",
        local_copy=tmp_path / "local_copy",
        config_digest=config_digest,
        calls=[],
        audit_calls=[],
        runtime=runtime,
        source_files_digest=source_files_digest,
    )

    def verifier(directory, inventory, actual_plan, task):
        fixture.calls.append(task["task_id"])
        assert directory == Path(task["output_dir"])
        assert inventory["plan_digest"] == actual_plan["plan_digest"]
        return {"status": "synthetic_verified", "task_id": task["task_id"]}

    def auditor(directory, actual_plan, task, actual_provider):
        fixture.audit_calls.append(task["task_id"])
        assert actual_provider is provider
        summary = json.loads((directory / "summary.json").read_text())
        return {
            "config_digest": fixture.config_digest,
            "best_checkpoint_sha256": provider.file_sha(directory / "best.pt"),
            "last_checkpoint_sha256": provider.file_sha(directory / "last.pt"),
            "completed_epochs": summary["completed_epochs"],
            "runtime": fixture.runtime,
            "source_files_digest": fixture.source_files_digest,
        }

    fixture.verifier, fixture.auditor = verifier, auditor
    freeze(fixture)
    return fixture


def admit(fixture, *, probes=None, **kwargs):
    return continuation.admit_continuation(
        fixture.plan,
        fixture.incident,
        fixture.manifest,
        fixture.local_receipt,
        provider=provider,
        now=2000,
        policy=fixture.policy,
        task_verifier=fixture.verifier,
        trained_auditor=fixture.auditor,
        probes=probes or Probes(),
        **kwargs,
    )


def test_exact_preserved_three_of_eight_admits_verify_only_fourth(incident_tree):
    accepted = admit(incident_tree)
    assert accepted["completed"] == list(continuation.TASK_IDS[:3])
    assert accepted["current_task"] == continuation.TASK_IDS[3]
    assert accepted["trained_fourth"]["completed_epochs"] == 30
    assert incident_tree.calls == list(continuation.TASK_IDS[:3])
    assert incident_tree.audit_calls == [continuation.TASK_IDS[3]]


@pytest.mark.parametrize(
    "field",
    [
        "current_summary_sha256",
        "current_best_checkpoint_sha256",
        "current_last_checkpoint_sha256",
        "current_config_digest",
    ],
)
def test_incident_must_bind_each_trained_artifact_identity(incident_tree, field):
    incident_tree.incident[field] = "0" * 64
    with pytest.raises((ValueError, PermissionError)):
        admit(incident_tree)


def test_frozen_source_drift_is_refused_before_task_or_checkpoint_audit(incident_tree):
    (Path(incident_tree.plan["repo"]) / "frozen.py").write_text("FROZEN = False\n")
    with pytest.raises(ValueError, match="source changed"):
        admit(incident_tree)
    assert incident_tree.calls == []
    assert incident_tree.audit_calls == []


def test_admission_rejects_live_original_pid_and_does_not_query_gpu(incident_tree):
    probes = Probes(live=True)
    with pytest.raises(PermissionError, match="PID is live"):
        admit(incident_tree, probes=probes)
    assert probes.gpu_calls == 0
    assert incident_tree.calls == []


def test_admission_defers_competing_gpu_to_wait_without_repeating_training(
    incident_tree,
):
    probes = Probes([Probes.idle(training_python_pids=[999])])
    accepted = admit(incident_tree, probes=probes)
    assert accepted["fourth_training_will_be_repeated"] is False
    assert probes.gpu_calls == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", list(continuation.TASK_IDS[:2])),
        ("completed", list(reversed(continuation.TASK_IDS[:3]))),
        ("current_task", continuation.TASK_IDS[4]),
        ("phase", "verify"),
        ("worker_exit_code", 1),
        ("error", "different failure"),
    ],
)
def test_wrong_frozen_failure_boundary_is_refused(incident_tree, field, value):
    incident_tree.incident["queue_snapshot"][field] = value
    freeze(incident_tree)
    with pytest.raises((ValueError, PermissionError)):
        admit(incident_tree)


def test_resealed_partial_training_does_not_reenter_as_verify_only(incident_tree):
    path = Path(incident_tree.plan["tasks"][3]["output_dir"]) / "summary.json"
    summary = json.loads(path.read_text())
    summary["completed_epochs"] = 29
    write_json(path, summary)
    freeze(incident_tree)
    with pytest.raises((ValueError, PermissionError, RuntimeError)):
        admit(incident_tree)


def test_existing_fourth_verification_is_not_silently_replaced(incident_tree):
    path = Path(incident_tree.plan["tasks"][3]["verification_file"])
    write_json(path, {"status": "already_exists"})
    freeze(incident_tree)
    with pytest.raises((ValueError, PermissionError, FileExistsError)):
        admit(incident_tree)


@pytest.mark.parametrize("copy", ["source", "backup"])
def test_changed_preserved_bytes_are_refused(incident_tree, copy):
    root = {
        "source": Path(incident_tree.plan["run_root"]),
        "backup": incident_tree.backup,
    }[copy]
    path = root / "tasks" / continuation.TASK_IDS[0] / "immutable.txt"
    path.write_text("different preserved bytes")
    with pytest.raises((ValueError, PermissionError)):
        admit(incident_tree)


@pytest.mark.parametrize(
    "change", ["manifest_digest", "local_digest", "local_files", "local_status"]
)
def test_invalid_inventory_or_local_receipt_is_refused(incident_tree, change):
    if change == "manifest_digest":
        incident_tree.manifest["inventory_sha256"] = "a" * 64
    elif change == "local_digest":
        incident_tree.local_receipt["inventory_sha256"] = "b" * 64
    elif change == "local_files":
        incident_tree.local_receipt["files"] = incident_tree.local_receipt["files"][1:]
        incident_tree.local_receipt["inventory_sha256"] = provider.digest(
            incident_tree.local_receipt["files"]
        )
    else:
        incident_tree.local_receipt["status"] = "not_verified"
    with pytest.raises((ValueError, PermissionError)):
        admit(incident_tree)


@pytest.mark.parametrize("index", [4, 5, 6, 7])
def test_later_empty_task_directory_blocks_fresh_execution(incident_tree, index):
    Path(incident_tree.plan["tasks"][index]["output_dir"]).mkdir()
    with pytest.raises((ValueError, PermissionError, FileExistsError)):
        admit(incident_tree)


def test_reused_first_three_tasks_remain_byte_immutable(incident_tree):
    continuation.require_preserved_tasks_unchanged(
        incident_tree.plan, incident_tree.manifest, provider
    )
    path = Path(incident_tree.plan["tasks"][2]["output_dir"]) / "immutable.txt"
    path.write_text("changed after admission")
    with pytest.raises((ValueError, PermissionError)):
        continuation.require_preserved_tasks_unchanged(
            incident_tree.plan, incident_tree.manifest, provider
        )


def context(fixture, tmp_path, probes=None):
    provider_path = tmp_path / "immutable_provider.py"
    shutil.copyfile(Path(provider.__file__), provider_path)
    control = tmp_path / "continuation_control"
    control.mkdir()
    return {
        "provider": provider,
        "provider_path": provider_path,
        "plan": fixture.plan,
        "incident": fixture.incident,
        "manifest": fixture.manifest,
        "accepted": admit(fixture),
        "helper_files": {str(provider_path): provider.file_sha(provider_path)},
        "recovery_id": "synthetic_continuation",
        "control": control,
        "status_path": Path(fixture.plan["run_root"]) / "results/queue_status.json",
        "state": {
            "status": "running",
            "phase": "waiting_for_gpu",
            "completed": list(continuation.TASK_IDS[:3]),
            "current_task": continuation.TASK_IDS[3],
            "worker_pid": None,
            "absolute_deadline_utc": fixture.policy.deadline_utc,
            "incident_sha256": "a" * 64,
            "continuation_controller_sha256": "b" * 64,
        },
        "probes": probes or Probes(),
        "policy": fixture.policy,
        "task_verifier": fixture.verifier,
    }


def test_provider_is_sha_bound_before_import(tmp_path):
    path = tmp_path / "immutable_provider.py"
    shutil.copyfile(Path(provider.__file__), path)
    assert provider.file_sha(path) == continuation.PROVIDER_SHA256
    loaded = continuation.load_provider(path)
    assert loaded.TASK_IDS == provider.TASK_IDS
    path.write_text(path.read_text() + "\nDRIFT = True\n")
    with pytest.raises((ValueError, PermissionError)):
        continuation.load_provider(path)


def test_two_consecutive_idle_probes_are_required_and_busy_resets_count(
    incident_tree, tmp_path
):
    clock = Clock()
    probes = Probes(
        [
            Probes.idle(),
            Probes.idle(training_python_pids=[999]),
            Probes.idle(),
            Probes.idle(),
        ]
    )
    state = context(incident_tree, tmp_path, probes)
    continuation.wait_for_idle(
        state, now=clock.now, sleep=clock.sleep, probe_interval_seconds=15
    )
    assert probes.gpu_calls == 4
    assert clock.sleeps == [15, 15, 15]
    assert state["state"]["worker_pid"] is None


def test_wrong_gpu_is_refused_without_waiting_or_stopping_owner(
    incident_tree, tmp_path
):
    clock = Clock()
    probes = Probes([Probes.idle(name="NVIDIA GeForce RTX 5090")])
    state = context(incident_tree, tmp_path, probes)
    with pytest.raises((ValueError, PermissionError)):
        continuation.wait_for_idle(
            state, now=clock.now, sleep=clock.sleep, probe_interval_seconds=15
        )
    assert clock.sleeps == []
    assert state["state"]["worker_pid"] is None


def test_expired_deadline_and_provider_drift_block_before_idle_wait(
    incident_tree, tmp_path
):
    state = context(incident_tree, tmp_path)
    clock = Clock(provider.utc_seconds(incident_tree.policy.deadline_utc))
    with pytest.raises(TimeoutError):
        continuation.wait_for_idle(state, now=clock.now, sleep=clock.sleep)
    assert state["probes"].gpu_calls == 0
    assert clock.sleeps == []
    path = state["provider_path"]
    path.write_text(path.read_text() + "\nDRIFT = True\n")
    with pytest.raises((ValueError, PermissionError)):
        continuation.wait_for_idle(state, now=lambda: 2000, sleep=clock.sleep)
    assert state["probes"].gpu_calls == 0


def test_competing_owner_wait_expires_without_creating_or_stopping_worker(
    incident_tree, tmp_path
):
    clock = Clock()
    probes = Probes([Probes.idle(unknown_gpu_owner_pids=[888])])
    state = context(incident_tree, tmp_path, probes)
    with pytest.raises(TimeoutError):
        continuation.wait_for_idle(
            state, now=clock.now, sleep=clock.sleep, probe_interval_seconds=15
        )
    assert clock.value == 2000 + 30 * 60
    assert probes.gpu_calls == 120
    assert state["state"]["worker_pid"] is None
    assert state["state"]["completed"] == list(continuation.TASK_IDS[:3])


def test_wall_clock_rollback_cannot_extend_original_resource_wait(
    incident_tree, tmp_path
):
    clock = Clock()
    state = context(
        incident_tree, tmp_path, Probes([Probes.idle(training_python_pids=[999])])
    )
    with pytest.raises(TimeoutError):
        continuation.wait_for_idle(
            state,
            now=lambda: 4000 - clock.now(),
            monotonic=clock.now,
            sleep=clock.sleep,
            probe_interval_seconds=15,
        )
    assert clock.value == 3800
    assert state["state"]["worker_pid"] is None


def test_gpu_query_error_is_fatal_without_busy_wait(incident_tree, tmp_path):
    clock = Clock()
    state = context(incident_tree, tmp_path)

    def broken_query():
        raise RuntimeError("synthetic nvidia-smi query failed")

    state["probes"].gpu = broken_query
    with pytest.raises(RuntimeError, match="query failed"):
        continuation.wait_for_idle(state, now=clock.now, sleep=clock.sleep)
    assert clock.sleeps == []
    assert state["state"]["worker_pid"] is None


@pytest.mark.parametrize(
    "changes",
    [{"free_mib": -1}, {"free_mib": float("nan")}, {"utilization": 101}],
)
def test_malformed_gpu_probe_is_fatal_without_busy_wait(
    incident_tree, tmp_path, changes
):
    clock = Clock()
    state = context(incident_tree, tmp_path, Probes([Probes.idle(**changes)]))
    with pytest.raises((ValueError, PermissionError)):
        continuation.wait_for_idle(state, now=clock.now, sleep=clock.sleep)
    assert clock.sleeps == []
    assert state["state"]["worker_pid"] is None


def execute_stub(fixture, clock, calls, *, partial_index=None, failure=None):
    def execute(command, **kwargs):
        mode = provider.option_value(command, "--mode")
        task = next(
            task for task in fixture.plan["tasks"] if command == task[mode + "_command"]
        )
        calls.append(
            {
                "task_id": task["task_id"],
                "mode": mode,
                "command": list(command),
                "started_at": clock.now(),
                **kwargs,
            }
        )
        if failure is not None:
            raise failure
        with kwargs["log"].open("x") as stream:
            stream.write("synthetic owned command only\n")
        directory = Path(task["output_dir"])
        if mode == "train":
            assert not directory.exists()
            directory.mkdir(parents=True)
            write_json(
                directory / "summary.json",
                {
                    "completed_epochs": 29
                    if partial_index is not None
                    and task == fixture.plan["tasks"][partial_index]
                    else 30,
                    "status": continuation.TRAINED_STATUS,
                    "test_evaluated": False,
                },
            )
        else:
            assert directory.exists()
            write_json(
                directory / "verification.json", {"status": "synthetic_verified"}
            )
        clock.value += 1
        kwargs["state"].update(worker_pid=None, worker_exit_code=0)
        provider.atomic_json(kwargs["status_path"], kwargs["state"])

    return execute


def test_fourth_verify_then_original_train_verify_commands_complete_frozen_queue(
    incident_tree, tmp_path, monkeypatch
):
    clock, calls = Clock(), []
    state = context(incident_tree, tmp_path)
    before = deepcopy(incident_tree.plan)
    matrix_calls = []

    def matrix(actual):
        matrix_calls.append(actual)
        assert actual["state"]["completed"] == list(continuation.TASK_IDS)
        return {"status": "synthetic_full_matrix", "tasks": list(continuation.TASK_IDS)}

    monkeypatch.setattr(continuation, "complete_matrix", matrix)
    result = continuation.execute_continuation(
        state,
        now=clock.now,
        sleep=clock.sleep,
        execute_owned=execute_stub(incident_tree, clock, calls),
    )
    expected = [(continuation.TASK_IDS[3], "verify")] + [
        (task_id, mode)
        for task_id in continuation.TASK_IDS[4:]
        for mode in ("train", "verify")
    ]
    assert [(call["task_id"], call["mode"]) for call in calls] == expected
    assert incident_tree.plan == before
    for call in calls:
        assert "--resume" not in call["command"]
        if call["mode"] == "train":
            assert (
                provider.option_value(call["command"], "--max-wallclock-seconds")
                == "43200"
            )
            assert call["hard_deadline"] == min(
                provider.utc_seconds(incident_tree.policy.deadline_utc),
                call["started_at"] + 43200,
            )
        else:
            assert call["hard_deadline"] == provider.utc_seconds(
                incident_tree.policy.deadline_utc
            )
    assert result["status"] == "completed"
    assert result["completed"] == list(continuation.TASK_IDS)
    assert result["current_task"] is None
    assert len(matrix_calls) == 1
    postflight = json.loads(
        (Path(incident_tree.plan["run_root"]) / "results/postflight.json").read_text()
    )
    assert postflight["tasks"] == list(continuation.TASK_IDS)
    assert postflight["test_evaluated"] is False
    continuation.require_preserved_tasks_unchanged(
        incident_tree.plan, incident_tree.manifest, provider, count=4
    )
    for row in incident_tree.manifest["files"]:
        if row["path"].startswith("logs/"):
            assert (
                provider.file_sha(Path(incident_tree.plan["run_root"]) / row["path"])
                == row["sha256"]
            )


def test_partial_later_train_does_not_proceed_to_verify_or_matrix_admission(
    incident_tree, tmp_path, monkeypatch
):
    clock, calls = Clock(), []
    state = context(incident_tree, tmp_path)

    def forbidden(*args):
        pytest.fail("a partial task reached complete-matrix admission")

    monkeypatch.setattr(continuation, "complete_matrix", forbidden)
    with pytest.raises(RuntimeError, match="partial training"):
        continuation.execute_continuation(
            state,
            now=clock.now,
            sleep=clock.sleep,
            execute_owned=execute_stub(incident_tree, clock, calls, partial_index=4),
        )
    assert [(call["task_id"], call["mode"]) for call in calls] == [
        (continuation.TASK_IDS[3], "verify"),
        (continuation.TASK_IDS[4], "train"),
    ]
    assert state["state"]["status"] == "failed"
    assert state["state"]["completed"] == list(continuation.TASK_IDS[:4])
    assert not Path(incident_tree.plan["tasks"][4]["verification_file"]).exists()
    assert not Path(incident_tree.plan["tasks"][5]["output_dir"]).exists()
    assert not (
        Path(incident_tree.plan["run_root"]) / "results/postflight.json"
    ).exists()


def test_expired_continuation_never_calls_owned_executor(incident_tree, tmp_path):
    state = context(incident_tree, tmp_path)
    clock = Clock(provider.utc_seconds(incident_tree.policy.deadline_utc))

    def forbidden(*args, **kwargs):
        pytest.fail("an owned worker was created after the retained deadline")

    with pytest.raises(TimeoutError):
        continuation.execute_continuation(
            state, now=clock.now, sleep=clock.sleep, execute_owned=forbidden
        )
    assert state["state"]["status"] == "failed"
    assert state["state"]["completed"] == list(continuation.TASK_IDS[:3])
    assert state["state"]["worker_pid"] is None


def test_fourth_verification_created_after_admission_is_not_overwritten(
    incident_tree, tmp_path
):
    clock = Clock()
    state = context(incident_tree, tmp_path)
    path = Path(incident_tree.plan["tasks"][3]["verification_file"])
    write_json(path, {"status": "external_completed_verification"})
    before = provider.file_sha(path)

    def forbidden(*args, **kwargs):
        pytest.fail("an existing fourth verification was overwritten")

    with pytest.raises((PermissionError, FileExistsError, ValueError)):
        continuation.execute_continuation(
            state, now=clock.now, sleep=clock.sleep, execute_owned=forbidden
        )
    assert provider.file_sha(path) == before
    assert state["state"]["completed"] == list(continuation.TASK_IDS[:3])


def test_owned_executor_failure_stops_queue_before_accepting_fourth(
    incident_tree, tmp_path
):
    clock, calls = Clock(), []
    state = context(incident_tree, tmp_path)
    with pytest.raises(TimeoutError, match="only owned"):
        continuation.execute_continuation(
            state,
            now=clock.now,
            sleep=clock.sleep,
            execute_owned=execute_stub(
                incident_tree, clock, calls, failure=TimeoutError("only owned")
            ),
        )
    assert [(call["task_id"], call["mode"]) for call in calls] == [
        (continuation.TASK_IDS[3], "verify")
    ]
    assert state["state"]["worker_pid"] is None
    assert state["state"]["status"] == "failed"
    assert state["state"]["completed"] == list(continuation.TASK_IDS[:3])
    continuation.require_preserved_tasks_unchanged(
        incident_tree.plan, incident_tree.manifest, provider, count=4
    )
