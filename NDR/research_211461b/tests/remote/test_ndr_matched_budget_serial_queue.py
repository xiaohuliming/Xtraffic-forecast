"""CPU tests for the matched-budget serial queue; no GPU, data, SSH or real trainer."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from scripts.remote.windows import ndr_matched_budget_serial_queue as queue

CONTROLLER = Path(queue.__file__).resolve()
TRAIN_STATUS = queue.TRAIN_SUCCESS_STATUS
IDLE = 16000.0


def iso_at(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def job_command(job_id: str, output_dir: Path, artifact: str | None, payload: dict | None, order_file: Path, *,
                exit_code: int = 0, require: Path | None = None, sleep_seconds: float = 0.0,
                new_output_dir: bool = True, extra_files: dict | None = None) -> list[str]:
    """A tiny child that behaves like a trainer: own output dir, one artifact, a pid record."""
    extras = extra_files or {}
    body = f"""
        import json, os, pathlib, sys, time
        out = pathlib.Path({str(output_dir)!r})
        require = {str(require) if require else None!r}
        if require is not None and not pathlib.Path(require).is_file():
            sys.exit(5)
        time.sleep({sleep_seconds!r})
        out.mkdir(parents=True, exist_ok={not new_output_dir!r})
        with open({str(order_file)!r}, "a") as stream:
            stream.write(json.dumps({{"job_id": {job_id!r}, "pid": os.getpid()}}) + "\\n")
        for name, value in {extras!r}.items():
            (out / name).write_text(json.dumps(value))
        payload = {payload!r}
        if payload is not None:
            (out / {artifact!r}).write_text(json.dumps(payload))
        sys.exit({exit_code!r})
    """
    return [sys.executable, "-c", textwrap.dedent(body)]


def example_plan(tmp_path: Path, *, base_time: float | None = None) -> dict:
    """The three-job calibrate, train, verify example used across these tests."""
    base_time = time.time() if base_time is None else base_time
    run_root = tmp_path / "run"
    order = tmp_path / "order.jsonl"
    calibrate_dir = run_root / "tasks" / "alameda_bist_ndr_calibrate"
    train_dir = run_root / "tasks" / "alameda_bist_ndr_train"
    verify_dir = train_dir / "independent_verify"
    calibration = {"status": "calibrated_cuda_full_effective_batches", "claim_scope": "calibration_only",
                   "cuda_budget_eligible": True}
    summary = {"status": TRAIN_STATUS, "claim_scope": "validation_development_only", "completed_epochs": 100,
               "config_digest": "0" * 64, "test_evaluated": False}
    verification = {"status": "verified_complete_source_validation_development", "claim_scope": "validation_development_only",
                    "checkpoint_sha256": "1" * 64, "test_evaluated": False}
    jobs = [
        {"job_id": "alameda_bist_ndr_calibrate", "role": "calibrate",
         "command": job_command("alameda_bist_ndr_calibrate", calibrate_dir, "calibration.json", calibration, order),
         "timeout_seconds": 600, "depends_on": [], "success_artifact": "calibration.json", "output_dir": str(calibrate_dir)},
        {"job_id": "alameda_bist_ndr_train", "role": "train",
         "command": job_command("alameda_bist_ndr_train", train_dir, "summary.json", summary, order),
         "timeout_seconds": 1200, "depends_on": [], "success_artifact": "summary.json", "output_dir": str(train_dir)},
        {"job_id": "alameda_bist_ndr_verify", "role": "verify",
         "command": job_command("alameda_bist_ndr_verify", verify_dir, "verification.json", verification, order,
                                require=train_dir / "summary.json", new_output_dir=False),
         "timeout_seconds": 600, "depends_on": ["alameda_bist_ndr_train"], "success_artifact": "verification.json",
         "output_dir": str(verify_dir)},
    ]
    return {
        "schema_version": queue.SCHEMA_VERSION,
        "purpose": "cpu_test_matched_budget_serial_queue",
        "budget_frozen": True,
        "absolute_deadline_utc": iso_at(base_time + 3600),
        "total_budget_seconds": 3000,
        "busy_wait_seconds": 600,
        "minimum_free_gpu_mib": 12000,
        "python": sys.executable,
        "run_root": str(run_root),
        "metadata": {"note": "synthetic CPU fixture"},
        "jobs": jobs,
    }


def trainer_shaped_plan(tmp_path: Path, *, base_time: float | None = None) -> dict:
    """Real trainer argv shapes; admitted only, never executed."""
    base_time = time.time() if base_time is None else base_time
    run_root = tmp_path / "run"
    repo = tmp_path / "repo"
    strong, research = str(repo / "scripts/train_ndr_strong_reference.py"), str(repo / "scripts/train_ndr_research.py")
    prefix = [sys.executable, "-X", "utf8", "-u"]
    data = ["--data-dir", str(tmp_path / "cache"), "--graph-dir", str(tmp_path / "graphs")]
    bist = ["--bist-checkout", str(tmp_path / "bist")]
    calibrate_dir = run_root / "tasks/alameda_bist_ndr_calibrate"
    train_dir = run_root / "tasks/alameda_bist_ndr_train"
    ndr_dir = run_root / "tasks/alameda_ndr_net_train"
    jobs = [
        {"job_id": "alameda_bist_ndr_calibrate", "role": "calibrate", "timeout_seconds": 1800, "depends_on": [],
         "command": prefix + [strong, "--mode", "calibrate", "--method", "bist_ndr", "--recipe", "native", "--region", "Alameda",
                              "--epochs", "100", "--batch-size", "64", "--microbatch-size", "64", "--warmup-batches", "3",
                              "--timed-batches", "20", "--device", "cuda:0", *data, *bist, "--output-dir", str(calibrate_dir)],
         "success_artifact": "calibration.json", "output_dir": str(calibrate_dir)},
        {"job_id": "alameda_bist_ndr_train", "role": "train", "timeout_seconds": 21600, "depends_on": [],
         "command": prefix + [strong, "--mode", "train", "--method", "bist_ndr", "--recipe", "native", "--region", "Alameda",
                              "--epochs", "100", "--seed", "42", "--batch-size", "64", "--microbatch-size", "64",
                              "--threads", "3", "--device", "cuda:0", *data, *bist, "--output-dir", str(train_dir)],
         "success_artifact": "summary.json", "output_dir": str(train_dir)},
        {"job_id": "alameda_bist_ndr_verify", "role": "verify", "timeout_seconds": 600, "depends_on": ["alameda_bist_ndr_train"],
         "command": prefix + [strong, "--mode", "verify", "--device", "cuda:0", *data, *bist,
                              "--checkpoint", str(train_dir / "best.pt"), "--output-dir", str(train_dir / "independent_verify")],
         "success_artifact": "verification.json", "output_dir": str(train_dir / "independent_verify")},
        {"job_id": "alameda_ndr_net_train", "role": "train", "timeout_seconds": 64800, "depends_on": [],
         "command": prefix + [research, "--mode", "train", "--variant", "baseline", "--region", "Alameda", "--epochs", "100",
                              "--seed", "42", "--batch-size", "16", "--microbatch-size", "8", "--threads", "3",
                              "--device", "cuda:0", *data, "--output-dir", str(ndr_dir)],
         "success_artifact": "summary.json", "output_dir": str(ndr_dir)},
        {"job_id": "alameda_ndr_net_verify", "role": "verify", "timeout_seconds": 600, "depends_on": ["alameda_ndr_net_train"],
         "command": prefix + [research, "--mode", "verify", "--device", "cuda:0", *data, "--checkpoint", str(ndr_dir / "best.pt")],
         "success_artifact": "verification.json", "output_dir": str(ndr_dir)},
    ]
    return {"schema_version": queue.SCHEMA_VERSION, "purpose": "matched_budget_trainer_argv_shape", "budget_frozen": True,
            "absolute_deadline_utc": iso_at(base_time + 172800), "total_budget_seconds": 172800, "busy_wait_seconds": 600,
            "minimum_free_gpu_mib": 12000, "python": sys.executable, "run_root": str(run_root), "jobs": jobs}


def write_plan(path: Path, plan: dict, *, seal: bool = True) -> Path:
    path.write_text(json.dumps(queue.seal_plan(plan) if seal else plan, indent=2))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def order_rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "order.jsonl"
    return read_jsonl(path) if path.exists() else []


class FakeClock:
    def __init__(self, wall: float) -> None:
        self.wall, self.mono, self.sleeps = wall, 5000.0, []

    def now(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.wall += seconds
        self.mono += seconds


class HangingChild:
    """Owned child that never exits on its own."""

    def __init__(self, pid: int) -> None:
        self.pid, self.returncode, self.terminated, self.killed = pid, None, 0, 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        self.returncode = -15

    def kill(self) -> None:
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class FakePopen:
    def __init__(self) -> None:
        self.calls, self.children = [], []

    def __call__(self, command, **kwargs):
        child = HangingChild(9100 + len(self.children))
        self.calls.append({"command": command, "kwargs": kwargs})
        self.children.append(child)
        return child


class RecordingPopen:
    """Delegates to the real subprocess.Popen and records each owned spawn."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, command, **kwargs):
        child = subprocess.Popen(command, **kwargs)
        self.calls.append({"command": command, "kwargs": kwargs, "pid": child.pid})
        return child


def run_real(tmp_path: Path, plan: dict, **options):
    path = write_plan(tmp_path / "plan.json", plan)
    context = queue.admit(path)
    popen = RecordingPopen()
    state = queue.execute(context, gpu_free_mib=lambda: IDLE, popen=popen, poll_interval_seconds=0.02, **options)
    return context, state, popen


def records(state: dict) -> dict:
    return {row["job_id"]: row for row in state["jobs"]}


# Admission


def test_valid_plan_admits_read_only_and_cli_spawns_nothing(tmp_path):
    plan = queue.seal_plan(example_plan(tmp_path))
    path = write_plan(tmp_path / "plan.json", plan, seal=False)
    context = queue.admit(path)
    assert context["plan"]["plan_digest"] == plan["plan_digest"] == queue.plan_digest(plan)
    assert [job["job_id"] for job in context["jobs"]] == [job["job_id"] for job in plan["jobs"]]
    assert context["state_path"] == Path(plan["run_root"]) / "controller_state.json"
    assert context["journal_path"] == Path(plan["run_root"]) / "owned_processes.jsonl"
    completed = subprocess.run([sys.executable, str(CONTROLLER), "--plan", str(path), "--admit-only"],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.strip()) == {"status": "admitted_no_worker", "plan_digest": plan["plan_digest"]}
    assert not Path(plan["run_root"]).exists()
    assert not (tmp_path / "order.jsonl").exists()


def test_trainer_shaped_plan_admits(tmp_path):
    path = write_plan(tmp_path / "plan.json", trainer_shaped_plan(tmp_path))
    context = queue.admit(path)
    assert [job["role"] for job in context["jobs"]] == ["calibrate", "train", "verify", "train", "verify"]
    assert not (tmp_path / "run").exists()


def test_existing_run_root_is_accepted_but_old_state_is_not(tmp_path):
    plan = example_plan(tmp_path)
    Path(plan["run_root"]).mkdir()
    queue.admit(write_plan(tmp_path / "plan.json", plan))
    (Path(plan["run_root"]) / "controller_state.json").write_text("{}")
    with pytest.raises(queue.AdmissionError, match="state_path already exists"):
        queue.admit(tmp_path / "plan.json")


def test_digest_tampering_is_rejected(tmp_path, capsys):
    plan = queue.seal_plan(example_plan(tmp_path))
    plan["jobs"][1]["timeout_seconds"] = 1300
    path = write_plan(tmp_path / "plan.json", plan, seal=False)
    with pytest.raises(queue.AdmissionError, match="plan_digest does not match"):
        queue.admit(path)
    assert queue.main(["--plan", str(path), "--admit-only"]) == queue.EXIT_REJECTED
    output = json.loads(capsys.readouterr().out.strip())
    assert output["status"] == "admission_rejected" and "plan_digest" in output["reason"]
    forged = queue.seal_plan(example_plan(tmp_path)) | {"plan_digest": "0" * 64}
    with pytest.raises(queue.AdmissionError, match="plan_digest does not match"):
        queue.admit(write_plan(tmp_path / "forged.json", forged, seal=False))


def test_expired_deadline_is_rejected(tmp_path):
    plan = example_plan(tmp_path) | {"absolute_deadline_utc": iso_at(time.time() - 1)}
    with pytest.raises(queue.AdmissionError, match="already passed"):
        queue.admit(write_plan(tmp_path / "plan.json", plan))
    clock = FakeClock(time.time() + 7200)
    with pytest.raises(queue.AdmissionError, match="already passed"):
        queue.admit(write_plan(tmp_path / "plan.json", example_plan(tmp_path)), now=clock.now)


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_unfrozen_budget_is_rejected(tmp_path, value):
    plan = example_plan(tmp_path) | {"budget_frozen": value}
    with pytest.raises(queue.AdmissionError, match="budget_frozen must be true"):
        queue.admit(write_plan(tmp_path / "plan.json", plan))


@pytest.mark.parametrize("depends_on", [[], ["alameda_bist_ndr_calibrate"], ["alameda_bist_ndr_calibrate", "alameda_bist_ndr_train"]])
def test_verify_needs_exactly_one_train_dependency(tmp_path, depends_on):
    plan = example_plan(tmp_path)
    plan["jobs"][2]["depends_on"] = depends_on
    with pytest.raises(queue.AdmissionError, match="exactly one earlier train job"):
        queue.admit(write_plan(tmp_path / "plan.json", plan))


def test_verify_before_its_train_is_rejected(tmp_path):
    plan = example_plan(tmp_path)
    plan["jobs"] = [plan["jobs"][0], plan["jobs"][2], plan["jobs"][1]]
    with pytest.raises(queue.AdmissionError, match="not an earlier job"):
        queue.admit(write_plan(tmp_path / "plan.json", plan))


def _set_job(index, key, value):
    def change(plan, tmp_path):
        plan["jobs"][index][key] = value(plan, tmp_path) if callable(value) else value
    return change


def _set(key, value):
    def change(plan, tmp_path):
        plan[key] = value
    return change


def _drop_job_key(plan, tmp_path):
    del plan["jobs"][0]["timeout_seconds"]


def _existing_train_dir(plan, tmp_path):
    Path(plan["jobs"][1]["output_dir"]).mkdir(parents=True)


def _duplicate_job(plan, tmp_path):
    plan["jobs"][1]["job_id"] = plan["jobs"][0]["job_id"]


@pytest.mark.parametrize(("change", "message"), [
    (_set("schema_version", "ndr_strong4_development_serial_plan_v1"), "schema_version"),
    (_set("purpose", " "), "purpose"),
    (_set("unexpected", 1), "unknown"),
    (_set("absolute_deadline_utc", "2099-01-01T00:00:00"), "explicit UTC offset"),
    (_set("absolute_deadline_utc", "tomorrow"), "not ISO 8601"),
    (_set("total_budget_seconds", 0), "total_budget_seconds"),
    (_set("total_budget_seconds", 2000), "sum of job timeout_seconds"),
    (_set("total_budget_seconds", True), "total_budget_seconds"),
    (_set("busy_wait_seconds", 0), "busy_wait_seconds"),
    (_set("busy_wait_seconds", 601), "busy_wait_seconds"),
    (_set("minimum_free_gpu_mib", 11999), "minimum_free_gpu_mib"),
    (_set("python", "python3"), "absolute path"),
    (_set("run_root", "relative/run"), "absolute path"),
    (_set("jobs", []), "non-empty list"),
    (_duplicate_job, "duplicate job_id"),
    (_drop_job_key, "exactly the keys"),
    (_set_job(0, "job_id", "../escape"), "job_id must match"),
    (_set_job(0, "role", "test"), "role must be one of"),
    (_set_job(0, "timeout_seconds", 0), "timeout_seconds"),
    (_set_job(0, "timeout_seconds", -5), "timeout_seconds"),
    (_set_job(0, "timeout_seconds", "600"), "timeout_seconds"),
    (_set_job(0, "timeout_seconds", True), "timeout_seconds"),
    (_set_job(0, "command", []), "non-empty list of strings"),
    (_set_job(0, "command", "python train.py"), "non-empty list of strings"),
    (_set_job(0, "command", [sys.executable, 3]), "non-empty list of strings"),
    (_set_job(0, "command", ["/usr/bin/other-python", "-c", "pass"]), "start with the plan python"),
    (_set_job(0, "depends_on", ["alameda_bist_ndr_calibrate"]), "not an earlier job"),
    (_set_job(0, "success_artifact", "../calibration.json"), "inside output_dir"),
    (_set_job(0, "success_artifact", "/tmp/calibration.json"), "inside output_dir"),
    (_set_job(0, "success_artifact", "C:/calibration.json"), "inside output_dir"),
    (_set_job(0, "output_dir", "tasks/relative"), "absolute path"),
    (_set_job(0, "output_dir", lambda plan, tmp_path: str(tmp_path / "elsewhere")), "strictly inside run_root"),
    (_set_job(0, "output_dir", lambda plan, tmp_path: plan["run_root"]), "strictly inside run_root"),
    (_set_job(0, "output_dir", lambda plan, tmp_path: str(Path(plan["run_root"]) / "controller_logs" / "x")), "overlaps controller_logs"),
    (_set_job(0, "output_dir", lambda plan, tmp_path: plan["jobs"][1]["output_dir"]), "overlaps job"),
    (_set_job(2, "output_dir", lambda plan, tmp_path: plan["jobs"][0]["output_dir"]), "overlaps job"),
    (_set_job(2, "success_artifact", "../summary.json"), "inside output_dir"),
    (_existing_train_dir, "must be new"),
    (_set("state_path", "controller_state.json"), "absolute path"),
    (_set_job(0, "command", lambda plan, tmp_path: plan["jobs"][0]["command"] + ["--mode", "train"]), "passes --mode train"),
    (_set_job(1, "command", lambda plan, tmp_path: plan["jobs"][1]["command"] + ["--output-dir", str(tmp_path / "x")]), "--output-dir differs"),
])
def test_schema_violations_are_rejected(tmp_path, change, message):
    plan = example_plan(tmp_path)
    change(plan, tmp_path)
    with pytest.raises(queue.AdmissionError, match=message.replace("(", r"\(")):
        queue.admit(write_plan(tmp_path / "plan.json", plan))


def _replace_arg(index, old, new):
    def change(plan, tmp_path):
        command = plan["jobs"][index]["command"]
        command[command.index(old) + 1] = new(plan, tmp_path) if callable(new) else new
    return change


def _drop_flag(index, flag):
    def change(plan, tmp_path):
        command = plan["jobs"][index]["command"]
        position = command.index(flag)
        del command[position:position + 2]
    return change


def _abbreviate(index, flag, short):
    def change(plan, tmp_path):
        command = plan["jobs"][index]["command"]
        command[command.index(flag)] = short
    return change


@pytest.mark.parametrize(("change", "message"), [
    (_replace_arg(1, "--mode", "verify"), "passes --mode verify"),
    (_drop_flag(1, "--mode"), "explicit --mode"),
    (_drop_flag(1, "--output-dir"), "must pass --output-dir"),
    (_abbreviate(1, "--output-dir", "--output"), "abbreviates"),
    (_replace_arg(2, "--checkpoint", lambda plan, tmp_path: str(tmp_path / "other" / "best.pt")), "--checkpoint is not inside"),
    (_drop_flag(2, "--checkpoint"), "must pass --checkpoint"),
    (_replace_arg(4, "--checkpoint", lambda plan, tmp_path: plan["jobs"][3]["output_dir"] + "/sub/best.pt"), "--checkpoint is not inside"),
])
def test_trainer_flags_must_match_plan_fields(tmp_path, change, message):
    plan = trainer_shaped_plan(tmp_path)
    change(plan, tmp_path)
    with pytest.raises(queue.AdmissionError, match=message):
        queue.admit(write_plan(tmp_path / "plan.json", plan))


@pytest.mark.parametrize("raw", [
    b'{"schema_version": "a", "schema_version": "b"}',
    b'{"total_budget_seconds": NaN}',
    b'{"total_budget_seconds": Infinity}',
    b'[1, 2]',
    b'not json',
])
def test_ambiguous_or_non_finite_plan_json_is_rejected(tmp_path, raw):
    path = tmp_path / "plan.json"
    path.write_bytes(raw)
    with pytest.raises(queue.AdmissionError):
        queue.admit(path)


# Execution


def test_serial_order_and_separate_training_and_verification_pids(tmp_path):
    plan = example_plan(tmp_path)
    context, state, popen = run_real(tmp_path, plan)
    assert state["status"] == queue.STATUS_COMPLETED
    expected = [job["job_id"] for job in plan["jobs"]]
    rows = order_rows(tmp_path)
    assert [row["job_id"] for row in rows] == expected
    assert [call["command"] for call in popen.calls] == [job["command"] for job in plan["jobs"]]
    by_job = records(state)
    train, verify = by_job["alameda_bist_ndr_train"], by_job["alameda_bist_ndr_verify"]
    assert train["training_pid"] == train["pid"] == popen.calls[1]["pid"]
    assert verify["verification_pid"] == verify["pid"] == popen.calls[2]["pid"]
    assert verify["training_pid"] == train["pid"] != verify["verification_pid"]
    assert verify["pid_distinct_from_training"] is True
    assert [row["pid"] for row in rows] == [call["pid"] for call in popen.calls]
    assert all(row["status"] == "succeeded" and row["returncode"] == 0 for row in state["jobs"])
    assert by_job["alameda_bist_ndr_calibrate"]["artifact"]["cuda_budget_eligible"] is True
    assert by_job["alameda_bist_ndr_calibrate"]["artifact"]["cuda_evidence"] == "calibration_cuda_budget_eligible"
    for call in popen.calls:
        assert call["kwargs"]["cwd"] == str(context["run_root"]) and call["kwargs"]["stdin"] == subprocess.DEVNULL
    for job_id in expected:
        assert (context["log_dir"] / f"{job_id}.stdout.log").is_file()
        assert (context["log_dir"] / f"{job_id}.stderr.log").is_file()
    assert json.loads(context["state_path"].read_text()) == state
    assert queue.main(["--plan", str(tmp_path / "plan.json")], gpu_free_mib=lambda: IDLE) == queue.EXIT_REJECTED


def test_verify_waits_for_train_success_artifact(tmp_path):
    plan = example_plan(tmp_path)
    train_dir = Path(plan["jobs"][1]["output_dir"])
    plan["jobs"][1]["command"] = job_command("alameda_bist_ndr_train", train_dir, "summary.json",
                                             {"status": "stopped_resumable", "completed_epochs": 3}, tmp_path / "order.jsonl")
    _, state, popen = run_real(tmp_path, plan)
    by_job = records(state)
    assert state["status"] == queue.STATUS_FAILED
    assert by_job["alameda_bist_ndr_train"]["status"] == "failed_success_artifact"
    assert by_job["alameda_bist_ndr_verify"]["status"] == "not_started"
    assert len(popen.calls) == 2
    assert (train_dir / "summary.json").is_file()


def test_nonzero_exit_stops_queue_and_keeps_files(tmp_path):
    plan = example_plan(tmp_path)
    train_dir = Path(plan["jobs"][1]["output_dir"])
    plan["jobs"][1]["command"] = job_command("alameda_bist_ndr_train", train_dir, "partial.json", {"status": "partial"},
                                             tmp_path / "order.jsonl", exit_code=7)
    context, state, popen = run_real(tmp_path, plan)
    by_job = records(state)
    assert state["status"] == queue.STATUS_FAILED and "returncode 7" in state["stop_reason"]
    assert by_job["alameda_bist_ndr_train"]["status"] == "failed_nonzero_exit"
    assert by_job["alameda_bist_ndr_train"]["returncode"] == 7
    assert by_job["alameda_bist_ndr_verify"]["status"] == "not_started"
    assert len(popen.calls) == 2
    assert (train_dir / "partial.json").is_file()
    assert (Path(plan["jobs"][0]["output_dir"]) / "calibration.json").is_file()
    events = [row["event"] for row in read_jsonl(context["journal_path"])]
    assert events == ["spawned", "exited", "spawned", "exited"]


def test_missing_success_artifact_is_a_failure(tmp_path):
    plan = example_plan(tmp_path)
    calibrate_dir = Path(plan["jobs"][0]["output_dir"])
    plan["jobs"][0]["command"] = job_command("alameda_bist_ndr_calibrate", calibrate_dir, None, None, tmp_path / "order.jsonl")
    _, state, popen = run_real(tmp_path, plan)
    assert state["status"] == queue.STATUS_FAILED
    assert records(state)["alameda_bist_ndr_calibrate"]["status"] == "failed_success_artifact"
    assert len(popen.calls) == 1


def test_timeout_records_terminal_state_and_stops_only_owned_child(tmp_path):
    clock = FakeClock(time.time())
    plan = example_plan(tmp_path, base_time=clock.wall)
    plan["jobs"][0]["timeout_seconds"] = 5
    context = queue.admit(write_plan(tmp_path / "plan.json", plan), now=clock.now)
    popen = FakePopen()
    state = queue.execute(context, gpu_free_mib=lambda: IDLE, popen=popen, now=clock.now,
                          monotonic=clock.monotonic, sleep=clock.sleep)
    by_job = records(state)
    assert state["status"] == queue.STATUS_FAILED and "timed out" in state["stop_reason"]
    calibrate = by_job["alameda_bist_ndr_calibrate"]
    assert calibrate["status"] == "failed_timeout" and calibrate["returncode"] == -15
    assert calibrate["stop_method"] == "terminate" and calibrate["wall_seconds"] == pytest.approx(5)
    assert [child.terminated for child in popen.children] == [1] and popen.children[0].killed == 0
    assert by_job["alameda_bist_ndr_train"]["status"] == "not_started"
    assert len(popen.calls) == 1
    exited = read_jsonl(context["journal_path"])[-1]
    assert exited["event"] == "exited" and exited["terminal"] == "timeout" and exited["pid"] == popen.children[0].pid
    assert sum(clock.sleeps) == pytest.approx(5)


def test_timeout_terminates_a_real_owned_child(tmp_path):
    plan = example_plan(tmp_path)
    calibrate_dir = Path(plan["jobs"][0]["output_dir"])
    plan["jobs"][0]["timeout_seconds"] = 1
    plan["jobs"][0]["command"] = job_command("alameda_bist_ndr_calibrate", calibrate_dir, "calibration.json",
                                             {"status": "calibrated_cuda_full_effective_batches"}, tmp_path / "order.jsonl",
                                             sleep_seconds=60)
    started = time.monotonic()
    _, state, popen = run_real(tmp_path, plan, terminate_grace_seconds=10)
    assert time.monotonic() - started < 30
    calibrate = records(state)["alameda_bist_ndr_calibrate"]
    assert state["status"] == queue.STATUS_FAILED and calibrate["status"] == "failed_timeout"
    assert calibrate["returncode"] is not None and calibrate["stop_method"] in {"terminate", "kill"}
    assert len(popen.calls) == 1 and not (calibrate_dir / "calibration.json").exists()


@pytest.mark.parametrize(("deadline_offset", "budget", "expected_sleeps"), [(1500, 3000, [600, 600, 300]), (7200, 1700, [600, 600, 500])])
def test_insufficient_gpu_memory_waits_then_stops_at_deadline_or_budget(tmp_path, capsys, deadline_offset, budget, expected_sleeps):
    clock = FakeClock(time.time())
    plan = example_plan(tmp_path, base_time=clock.wall)
    plan["absolute_deadline_utc"] = iso_at(clock.wall + deadline_offset)
    plan["total_budget_seconds"] = budget
    plan["jobs"][1]["timeout_seconds"] = 500
    path = write_plan(tmp_path / "plan.json", plan)
    readings = []

    def busy_gpu():
        readings.append(4096.0)
        if len(readings) == 2:
            raise RuntimeError("nvidia-smi failed")
        return 4096.0

    popen = FakePopen()
    code = queue.main(["--plan", str(path)], gpu_free_mib=busy_gpu, popen=popen, now=clock.now,
                      monotonic=clock.monotonic, sleep=clock.sleep)
    assert code == queue.EXIT_BOUND
    assert json.loads(capsys.readouterr().out.strip())["status"] == queue.STATUS_BOUND
    state = json.loads((Path(plan["run_root"]) / "controller_state.json").read_text())
    assert state["status"] == queue.STATUS_BOUND and "waited for GPU idle" in state["stop_reason"]
    gate = records(state)["alameda_bist_ndr_calibrate"]["gpu_wait"]
    assert gate["polls"] == len(readings) == 3 and gate["admitted_at"] is None
    assert all(row["status"] == "not_started" for row in state["jobs"])
    assert popen.calls == [] and clock.sleeps == expected_sleeps
    assert (Path(plan["run_root"]) / "owned_processes.jsonl").read_text() == ""


def test_failed_gpu_provider_is_not_idle_until_it_reports_memory(tmp_path):
    plan = example_plan(tmp_path)
    plan["jobs"] = plan["jobs"][:1]
    readings = iter([RuntimeError("driver"), float("nan"), 11999.0, 12000.0])

    def provider():
        value = next(readings)
        if isinstance(value, Exception):
            raise value
        return value

    path = write_plan(tmp_path / "plan.json", plan)
    sleeps = []
    state = queue.execute(queue.admit(path), gpu_free_mib=provider, popen=RecordingPopen(), sleep=lambda s: sleeps.append(s) or time.sleep(min(s, 0.01)),
                          poll_interval_seconds=0.02)
    assert state["status"] == queue.STATUS_COMPLETED
    gate = state["jobs"][0]["gpu_wait"]
    assert gate["polls"] == 4 and gate["last_free_mib"] == 12000.0 and gate["admitted_at"] is not None
    assert sleeps[:3] == [600, 600, 600]


def test_deadline_during_running_job_stops_own_child(tmp_path):
    clock = FakeClock(time.time())
    plan = example_plan(tmp_path, base_time=clock.wall)
    plan["absolute_deadline_utc"] = iso_at(clock.wall + 100)
    context = queue.admit(write_plan(tmp_path / "plan.json", plan), now=clock.now)
    popen = FakePopen()
    state = queue.execute(context, gpu_free_mib=lambda: IDLE, popen=popen, now=clock.now,
                          monotonic=clock.monotonic, sleep=clock.sleep, poll_interval_seconds=7)
    calibrate = records(state)["alameda_bist_ndr_calibrate"]
    assert state["status"] == queue.STATUS_BOUND and calibrate["status"] == "terminated_deadline_or_budget"
    assert popen.children[0].terminated == 1 and len(popen.calls) == 1
    assert sum(clock.sleeps) == pytest.approx(100) and max(clock.sleeps) <= 7
    assert records(state)["alameda_bist_ndr_train"]["status"] == "not_started"


def test_journal_is_append_only_and_status_writes_are_atomic(tmp_path, monkeypatch):
    plan = example_plan(tmp_path)
    run_root = Path(plan["run_root"])
    journal_path, state_path = run_root / "owned_processes.jsonl", run_root / "controller_state.json"
    real_replace, replacements, snapshots = os.replace, [], []

    def spy(source, target):
        source, target = Path(source), Path(target)
        if target == state_path:
            assert source.parent == target.parent and source.name.startswith(".controller_state.json.") and source.suffix == ".tmp"
            json.loads(source.read_text())
            replacements.append(source.name)
            snapshots.append(journal_path.read_text())
        return real_replace(source, target)

    monkeypatch.setattr(queue.os, "replace", spy)
    context, state, popen = run_real(tmp_path, plan)
    assert state["status"] == queue.STATUS_COMPLETED
    assert len(replacements) >= 2 * len(plan["jobs"]) + 2 and len(set(replacements)) == len(replacements)
    assert all(later.startswith(earlier) for earlier, later in zip(snapshots, snapshots[1:]))
    assert not [path for path in run_root.rglob("*") if path.name.endswith(".tmp")]
    rows = read_jsonl(journal_path)
    spawned = [row for row in rows if row["event"] == "spawned"]
    exited = [row for row in rows if row["event"] == "exited"]
    assert [row["event"] for row in rows] == ["spawned", "exited"] * len(plan["jobs"])
    assert [row["job_id"] for row in spawned] == [job["job_id"] for job in plan["jobs"]]
    assert [row["pid"] for row in spawned] == [call["pid"] for call in popen.calls] == [row["pid"] for row in exited]
    for row, job in zip(spawned, plan["jobs"]):
        assert row["command"] == job["command"] and row["plan_digest"] == state["plan_digest"]
        assert datetime.fromisoformat(row["started_at"]).utcoffset().total_seconds() == 0
    for row in exited:
        assert row["returncode"] == 0 and row["terminal"] == "exited" and datetime.fromisoformat(row["ended_at"])


def test_calibration_must_be_cuda_budget_eligible_on_cuda(tmp_path):
    plan = example_plan(tmp_path)
    plan["jobs"] = plan["jobs"][:1]
    calibrate_dir = Path(plan["jobs"][0]["output_dir"])
    plan["jobs"][0]["command"] = job_command("alameda_bist_ndr_calibrate", calibrate_dir, "calibration.json",
                                             {"status": "calibrated_cpu_engineering_only", "cuda_budget_eligible": False},
                                             tmp_path / "order.jsonl")
    _, state, _ = run_real(tmp_path, plan)
    assert state["status"] == queue.STATUS_FAILED
    assert "not cuda_budget_eligible" in state["jobs"][0]["terminal_reason"]


def test_ndr_research_calibration_needs_cuda_runtime_evidence(tmp_path):
    def run(base: Path, device_type: str) -> dict:
        base.mkdir()
        plan = example_plan(base)
        plan["jobs"] = plan["jobs"][:1]
        calibrate_dir = Path(plan["jobs"][0]["output_dir"])
        plan["jobs"][0]["command"] = job_command(
            "alameda_ndr_net_calibrate", calibrate_dir, "calibration.json",
            {"status": queue.NDR_RESEARCH_CALIBRATION_STATUS, "claim_scope": "calibration_only"},
            base / "order.jsonl", extra_files={"config.json": {"runtime": {"device_type": device_type}}})
        return run_real(base, plan)[1]

    accepted = run(tmp_path / "cuda", "cuda")
    assert accepted["status"] == queue.STATUS_COMPLETED
    artifact = accepted["jobs"][0]["artifact"]
    assert artifact["cuda_budget_eligible"] is None
    assert artifact["cuda_evidence"] == "config_runtime_device_type_only_trainer_reports_no_eligibility"
    rejected = run(tmp_path / "cpu", "cpu")
    assert rejected["status"] == queue.STATUS_FAILED


def test_plan_changed_after_admission_spawns_nothing(tmp_path):
    path = write_plan(tmp_path / "plan.json", example_plan(tmp_path))
    context = queue.admit(path)
    path.write_text(path.read_text() + "\n")
    popen = FakePopen()
    with pytest.raises(queue.AdmissionError, match="changed after admission"):
        queue.execute(context, gpu_free_mib=lambda: IDLE, popen=popen)
    assert popen.calls == [] and not (tmp_path / "run").exists()


def test_default_gpu_provider_parses_nvidia_smi_and_fails_closed():
    calls = []

    def missing_nvidia_smi(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="15800\n 13000 \n", stderr="")

    assert queue.nvidia_smi_free_mib(run=fake_run) == 13000.0
    assert calls[0][0] == list(queue.NVIDIA_SMI_COMMAND) and calls[0][1]["timeout"] == 30.0
    with pytest.raises(ValueError):
        queue.nvidia_smi_free_mib(run=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""))
    free, error = queue._probe_free_mib(lambda: queue.nvidia_smi_free_mib(run=missing_nvidia_smi))
    assert free is None and "FileNotFoundError" in error


def test_controller_is_stdlib_only_and_never_signals_foreign_processes():
    source = CONTROLLER.read_text()
    for forbidden in ("os.kill", "taskkill", "psutil", "killpg", "TerminateProcess", "Stop-Process"):
        assert forbidden not in source
    probe = ("import sys; before = set(sys.modules); sys.path.insert(0, sys.argv[1]); "
             "import ndr_matched_budget_serial_queue; "
             "extra = {name.split('.')[0] for name in set(sys.modules) - before} - set(sys.stdlib_module_names); "
             "extra.discard('ndr_matched_budget_serial_queue'); print(sorted(extra))")
    completed = subprocess.run([sys.executable, "-I", "-c", probe, str(CONTROLLER.parent)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
