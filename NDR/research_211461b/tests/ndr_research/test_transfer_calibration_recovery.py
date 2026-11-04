"""Synthetic-only empty-watcher recovery, with no real processes or GPU calls."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import shutil

import pytest

from scripts.remote.windows import ndr_transfer_calibration_recovery as recovery
from tests.ndr_research.test_development_admission import fixture_matrix, write

ROOT = Path(__file__).resolve().parents[2]


def fixture(tmp_path, *, old_status="waiting_verified_predecessor"):
    repo, run, archive = tmp_path / "repo", tmp_path / "cal", tmp_path / "archive"
    controller = repo / recovery.CONTROLLER
    controller.parent.mkdir(parents=True)
    # This is exactly the frozen six-job source, not an alternative controller.
    shutil.copyfile(ROOT / recovery.CONTROLLER, controller)
    gpu_provider = tmp_path / "recovery_code" / "ndr_research_recovery.py"
    gpu_provider.parent.mkdir()
    shutil.copyfile(ROOT / "scripts/remote/windows/ndr_research_recovery.py", gpu_provider)
    run.mkdir()
    (run / "logs").mkdir()
    old = {"status": old_status, "pid": 41776, "worker_pid": None, "completed": [],
           "plan_digest": "pending", "test_evaluated": False, "automatic_training": False}
    stage = tmp_path / "stage1"
    stage_plan = fixture_matrix(stage)
    stage_plan_path = tmp_path / "stage1_plan.json"
    write(stage_plan_path, stage_plan)
    jobs = [{"job_id": name, "command": ["synthetic-python", "--frozen-job", name],
             "result_file": str(run / "tasks" / name / "calibration.json"), "expected_status": "synthetic_engineering_complete"}
            for name in recovery.JOBS]
    plan = {"repo": str(repo), "run_root": str(run), "source_files": {recovery.CONTROLLER: recovery.file_sha(controller)},
            "predecessor_root": str(stage), "predecessor_plan_path": str(stage_plan_path),
            "predecessor_plan_digest": stage_plan["plan_digest"], "purpose": "engineering_calibration_only",
            "test_evaluated": False, "automatic_training": False, "automatic_shutdown": False, "automatic_transfer": False,
            "jobs": jobs, "absolute_deadline_utc": "2099-01-01T00:00:00+00:00",
            "minimum_free_gpu_mib": 6000, "busy_wait_minutes": 30, "per_job_seconds": 3600}
    plan["plan_digest"] = recovery.canonical_digest(plan)
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    old["plan_digest"] = plan["plan_digest"]
    write(run / "status.json", old)
    if old_status == "failed_engineering_admission":
        (run / "failure.txt").write_text("synthetic predecessor failure before any job")
    shutil.copytree(run, archive)
    entries = [{"path": p.relative_to(run).as_posix(), "bytes": p.stat().st_size, "sha256": recovery.file_sha(p)}
               for p in sorted(run.rglob("*")) if p.is_file()]
    manifest_path = tmp_path / "preservation_manifest.json"
    write(manifest_path, {"source_root": str(run), "archive_root": str(archive), "files": entries})
    incident = {"schema_version": recovery.INCIDENT_SCHEMA, "plan_digest": plan["plan_digest"],
                "preservation_manifest_path": str(manifest_path), "preservation_manifest_sha256": recovery.file_sha(manifest_path),
                "old_status_sha256": recovery.file_sha(run / "status.json"), "old_supervisor_pid": 41776,
                "old_worker_pid": None, "original_jobs_executed": False}
    incident_path = tmp_path / "incident.json"
    write(incident_path, incident)
    authority = recovery.FrozenAuthority(plan["plan_digest"], recovery.file_sha(controller), stage_plan["plan_digest"])
    return plan_path, incident_path, authority, run, archive, stage, gpu_provider


def admit(items, **kwargs):
    return recovery.admit_empty_recovery(items[0], items[1], authority=items[2], process_is_live=lambda _: False,
                                         now=0, gpu_provider_path=items[6], **kwargs)


@pytest.mark.parametrize("old_status", ["waiting_verified_predecessor", "failed_engineering_admission"])
def test_admission_accepts_exact_preserved_dead_zero_job_state_without_writes(tmp_path, old_status):
    items = fixture(tmp_path, old_status=old_status)
    before = recovery.file_sha(items[3] / "status.json")
    context = admit(items)
    assert context["old"]["pid"] == 41776 and context["old"]["completed"] == []
    assert context["plan"]["absolute_deadline_utc"] == "2099-01-01T00:00:00+00:00"
    assert recovery.file_sha(items[3] / "status.json") == before
    assert not (items[3] / "recovery").exists()
    assert not (items[3] / "tasks").exists()


def test_live_original_handle_refuses_recovery_and_never_stops_a_process(tmp_path):
    items = fixture(tmp_path)
    called = []
    with pytest.raises(PermissionError, match="still live"):
        recovery.admit_empty_recovery(items[0], items[1], authority=items[2], now=0,
                                      gpu_provider_path=items[6],
                                      process_is_live=lambda pid: called.append(pid) or pid == 41776)
    assert called == [41776] and not (items[3] / "recovery").exists()


@pytest.mark.parametrize("which", ["source", "archive", "unlisted"])
def test_preservation_requires_exact_source_and_archive_bytes(tmp_path, which):
    items = fixture(tmp_path)
    if which == "unlisted":
        (items[3] / "extra.json").write_text("{}")
    else:
        (items[3 if which == "source" else 4] / "status.json").write_text("{}")
    with pytest.raises(ValueError, match="preserved inventory"):
        admit(items)


def test_deadline_plan_or_original_controller_changes_are_rejected(tmp_path):
    items = fixture(tmp_path)
    with pytest.raises(TimeoutError, match="deadline"):
        recovery.admit_empty_recovery(items[0], items[1], authority=items[2], now=5e9, process_is_live=lambda _: False)
    plan = json.loads(items[0].read_text())
    plan["absolute_deadline_utc"] = "2100-01-01T00:00:00+00:00"
    write(items[0], plan)
    with pytest.raises(ValueError, match="fixed six-job"):
        admit(items)
    plan["absolute_deadline_utc"] = "2099-01-01T00:00:00+00:00"
    write(items[0], plan)
    controller = Path(plan["repo"]) / recovery.CONTROLLER
    controller.write_text(controller.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="controller bytes"):
        admit(items)


@pytest.mark.parametrize("artifact", ["job_directory", "job_log", "completed", "current_job"])
def test_any_job_history_blocks_empty_watcher_recovery(tmp_path, artifact):
    items = fixture(tmp_path)
    plan = json.loads(items[0].read_text())
    job = plan["jobs"][0]
    if artifact == "job_directory":
        Path(job["result_file"]).parent.mkdir(parents=True)
    elif artifact == "job_log":
        (items[3] / "logs" / (job["job_id"] + ".log")).write_text("")
    else:
        old = json.loads((items[3] / "status.json").read_text())
        old[artifact] = [job["job_id"]] if artifact == "completed" else job["job_id"]
        write(items[3] / "status.json", old)
        shutil.rmtree(items[4])
        shutil.copytree(items[3], items[4])
        manifest = json.loads(Path(json.loads(items[1].read_text())["preservation_manifest_path"]).read_text())
        for row in manifest["files"]:
            p = items[3] / row["path"]
            row.update(bytes=p.stat().st_size, sha256=recovery.file_sha(p))
        manifest_path = Path(json.loads(items[1].read_text())["preservation_manifest_path"])
        write(manifest_path, manifest)
        incident = json.loads(items[1].read_text())
        incident.update(old_status_sha256=recovery.file_sha(items[3] / "status.json"),
                        preservation_manifest_sha256=recovery.file_sha(manifest_path))
        write(items[1], incident)
    with pytest.raises((ValueError, PermissionError), match="zero-job|state|inventory|artifacts"):
        admit(items)


def completed_fake_worker(context, calls):
    plan = context["plan"]

    class Done:
        pid, returncode = 81234, 0

        def poll(self):
            return 0

    def create(command, **kwargs):
        job = next(job for job in plan["jobs"] if job["command"] == command)
        calls.append(copy.deepcopy(command))
        assert kwargs["creationflags"] == 0x08000000 and kwargs["cwd"] == plan["repo"]
        result = Path(job["result_file"])
        result.parent.mkdir(parents=True)
        write(result, {"status": job["expected_status"], "test_evaluated": False})
        return Done()

    return create


def test_new_dynamic_predecessor_pid_and_full_admission_gate_precede_gpu_jobs(tmp_path):
    items = fixture(tmp_path)
    context = admit(items)
    controller = context["controller"]
    queue_path = items[5] / "results/queue_status.json"
    queue = json.loads(queue_path.read_text())
    queue.update(pid=92345, worker_pid=None)
    write(queue_path, queue)
    live, sleeps, calls, gpu_checks = {92345}, [], [], []
    controller.process_live = lambda pid: pid in live
    controller.gpu_ready = lambda _: pytest.fail("unbound old GPU gate was used")
    context["gpu_probes"].gpu = lambda: gpu_checks.append(1) or {
        "name": "synthetic RTX 5080", "free_mib": 10000, "utilization": 0,
        "training_python_pids": [], "unknown_gpu_owner_pids": []}

    def sleep(seconds):
        assert not calls and not gpu_checks
        sleeps.append(seconds)
        live.clear()

    result = recovery.execute_recovery(context, now=lambda: 0, sleep=sleep, popen=completed_fake_worker(context, calls))
    assert sleeps == [60] and result["predecessor_pid"] == 92345
    assert calls == [job["command"] for job in context["plan"]["jobs"]]
    assert result["completed"] == list(recovery.JOBS) and result["worker_pid"] is None
    assert result["absolute_deadline_utc"] == context["plan"]["absolute_deadline_utc"]
    assert recovery.file_sha(items[4] / "status.json") == context["incident"]["old_status_sha256"]
    postflight = json.loads((items[3] / "engineering_postflight.json").read_text())
    assert postflight["formal_training_started"] is False and postflight["test_evaluated"] is False
    assert postflight["jobs"] == list(recovery.JOBS)
    assert any(row["path"].startswith("recovery/") for row in postflight["files"])
    assert not (items[3] / "recovery_owner.lock").exists()


def test_failed_or_unverified_predecessor_never_reaches_gpu_or_spawn(tmp_path):
    items = fixture(tmp_path)
    context = admit(items)
    queue_path = items[5] / "results/queue_status.json"
    queue = json.loads(queue_path.read_text())
    queue["status"] = "failed"
    write(queue_path, queue)
    context["gpu_probes"].gpu = lambda: pytest.fail("failed predecessor queried GPU")
    with pytest.raises(RuntimeError, match="terminal incomplete"):
        recovery.execute_recovery(context, now=lambda: 0, sleep=lambda _: None, popen=lambda *_a, **_k: pytest.fail("spawned"))
    assert json.loads((items[3] / "status.json").read_text())["status"] == "failed_engineering_admission"
    assert not (items[3] / "tasks").exists()


def test_expired_fixed_deadline_and_exclusive_owner_block_launch(tmp_path):
    items = fixture(tmp_path)
    context = admit(items)
    with pytest.raises(TimeoutError, match="deadline"):
        recovery.execute_recovery(context, now=lambda: context["deadline"])
    (items[3] / "recovery_owner.lock").write_text("synthetic live owner")
    with pytest.raises(FileExistsError):
        recovery.execute_recovery(context, now=lambda: 0)
    assert not (items[3] / "tasks").exists()


def test_production_authority_has_no_cli_override_for_plan_or_controller():
    authority = dataclasses.asdict(recovery.FrozenAuthority())
    assert authority["plan_digest"] == "a1e2553da1b19dc04cb39bdb74dd701f12749444a5c31c6f4a7093b47cda708b"
    assert authority["controller_sha256"] == recovery.file_sha(ROOT / recovery.CONTROLLER)
    assert authority["gpu_provider_sha256"] == recovery.GPU_PROVIDER_SHA256
    assert authority["gpu_provider_sha256"] == recovery.file_sha(ROOT / "scripts/remote/windows/ndr_research_recovery.py")
    assert authority["jobs"] == recovery.JOBS


@pytest.mark.parametrize("process, permitted", [
    (r"C:\Windows\System32\dwm.exe", True),
    (r"D:\Tencent\WeGame\QBBLinkTrial\Browser.exe", True),
    (r"D:\Tencent\WeGameApps\rail_apps\game\aclos\cross\recorder-release\Recorder.exe", True),
    (r"C:\Python\python.exe", False),
    (r"C:\Python\pythonw.exe", False),
    (r"C:\unknown\Browser.exe", False),
    (r"C:\unknown\Recorder.exe", False),
    (r"C:\unknown\custom_cuda_worker.exe", False),
])
def test_low_utilization_high_free_memory_still_requires_classified_gpu_owner(tmp_path, monkeypatch, process, permitted):
    context = admit(fixture(tmp_path))
    queried = []

    def check_output(command, **kwargs):
        queried.append(command)
        if command[1].startswith("--query-gpu="):
            return "NVIDIA GeForce RTX 5080, 14000, 0\n"
        assert command[1] == "--query-compute-apps=pid,process_name"
        return f"76543, {process}\n"

    monkeypatch.setattr(context["gpu_provider"].subprocess, "check_output", check_output)
    ready, gpu = recovery.gpu_ready(context, 6000)
    assert ready is permitted
    assert gpu["free_mib"] == 14000 and gpu["utilization"] == 0
    assert len(queried) == 2
    assert bool(gpu["training_python_pids"] or gpu["unknown_gpu_owner_pids"]) is not permitted


@pytest.mark.parametrize("process", [r"C:\Python\python.exe", r"C:\unknown\custom_cuda_worker.exe"])
def test_competing_gpu_owner_never_spawns_or_stops_any_worker(tmp_path, monkeypatch, process):
    items = fixture(tmp_path)
    context = admit(items)
    clock = [0]
    context["controller"].process_live = lambda _: False

    def check_output(command, **kwargs):
        return "NVIDIA GeForce RTX 5080, 14000, 0\n" if command[1].startswith("--query-gpu=") else f"76543, {process}\n"

    monkeypatch.setattr(context["gpu_provider"].subprocess, "check_output", check_output)

    def sleep(_seconds):
        clock[0] += context["plan"]["busy_wait_minutes"] * 60 + 1

    with pytest.raises(TimeoutError, match="GPU remained busy"):
        recovery.execute_recovery(context, now=lambda: clock[0], sleep=sleep,
                                  popen=lambda *_a, **_k: pytest.fail("competing owner spawned a worker"))
    assert not (items[3] / "tasks").exists()
    state = json.loads((items[3] / "status.json").read_text())
    assert state["completed"] == [] and state["worker_pid"] is None
    assert state["gpu"]["training_python_pids"] or state["gpu"]["unknown_gpu_owner_pids"]


def test_cpu_only_observers_are_not_gpu_owners(tmp_path, monkeypatch):
    context = admit(fixture(tmp_path))

    def check_output(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return "NVIDIA GeForce RTX 5080, 14000, 0\n" if command[1].startswith("--query-gpu=") else ""

    monkeypatch.setattr(context["gpu_provider"].subprocess, "check_output", check_output)
    # Live CPU observers are absent from the CUDA owner inventory.
    context["gpu_probes"].live = lambda _: True
    ready, gpu = recovery.gpu_ready(context, 6000)
    assert ready and gpu["processes"] == []


@pytest.mark.parametrize("when", ["admission", "before_execute", "while_waiting"])
def test_gpu_provider_hash_drift_refuses_launch(tmp_path, when):
    items = fixture(tmp_path)
    context = admit(items) if when != "admission" else None

    def drift():
        items[6].write_text(items[6].read_text() + "\n# synthetic drift\n")

    if when == "admission":
        drift()
        with pytest.raises(ValueError, match="GPU owner provider bytes"):
            admit(items)
    else:
        if when == "before_execute":
            drift()
        else:
            queue_path = items[5] / "results/queue_status.json"
            queue = json.loads(queue_path.read_text())
            queue.update(pid=92345, worker_pid=None)
            write(queue_path, queue)
            context["controller"].process_live = lambda pid: pid == 92345
        context["gpu_probes"].gpu = lambda: pytest.fail("changed provider queried GPU")
        with pytest.raises(ValueError, match="GPU owner provider bytes"):
            recovery.execute_recovery(context, now=lambda: 0, sleep=lambda _: drift(),
                                      popen=lambda *_a, **_k: pytest.fail("changed provider spawned"))
    assert not (items[3] / "tasks").exists()
    assert not (items[3] / "recovery_owner.lock").exists()
