from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = (
    REPO_ROOT
    / "scripts/remote/windows/protocol_v3_validate_repartition_marker.py"
)
WAITER = (
    REPO_ROOT
    / "scripts/remote/windows/protocol_v3_repartition_5080_27_waiter.ps1"
)
WAITER_WRAPPER = (
    REPO_ROOT
    / "scripts/remote/windows/protocol_v3_repartition_5080_27_waiter.cmd"
)
LINUX_TRANSITION = (
    REPO_ROOT
    / "scripts/remote/linux/protocol_v3_start_5090_shard_after_boundary.sh"
)


def valid_marker() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "ready",
        "plan": "27_18",
        "boundary_task": "039_fullwindow_staeformer_base_alameda_seed42",
        "forbidden_task": "040_fullwindow_graphwavenet_base_contracosta_seed42",
        "boundary_status_sha256": "1" * 64,
        "boundary_summary_sha256": "2" * 64,
        "boundary_state_sha256": "3" * 64,
        "old_runner_pid": 1001,
        "old_finalizer_pid": 1002,
        "generated_at": "2026-07-22T12:00:00Z",
        "extra_provenance": {"allowed": True},
    }


def run_validator(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--marker", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def read_waiter() -> str:
    return WAITER.read_text(encoding="utf-8")


def test_waiter_defaults_to_the_isolated_27_18_ops_root():
    assert r"C:\Users\ASUS\ops_27_18" in read_waiter()


def test_validator_accepts_the_complete_boundary_attestation(tmp_path: Path):
    marker = tmp_path / "boundary.json"
    marker.write_text(json.dumps(valid_marker()), encoding="utf-8")

    completed = run_validator(marker)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "BOUNDARY_MARKER_OK"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("boundary_status_sha256", "a" * 63),
        ("boundary_summary_sha256", "B" * 64),
        ("boundary_state_sha256", None),
    ],
)
def test_validator_rejects_invalid_boundary_digests(
    tmp_path: Path,
    field: str,
    bad_value: object,
):
    payload = valid_marker()
    payload[field] = bad_value
    marker = tmp_path / "boundary.json"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    completed = run_validator(marker)

    assert completed.returncode == 2
    assert "BOUNDARY_MARKER_INVALID" in completed.stderr


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("old_runner_pid", 0),
        ("old_finalizer_pid", True),
        ("generated_at", "  "),
    ],
)
def test_validator_rejects_incomplete_boundary_provenance(
    tmp_path: Path,
    field: str,
    bad_value: object,
):
    payload = valid_marker()
    payload[field] = bad_value
    marker = tmp_path / "boundary.json"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    completed = run_validator(marker)

    assert completed.returncode == 2
    assert "BOUNDARY_MARKER_INVALID" in completed.stderr


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", "1"),
        ("status", "waiting"),
        ("plan", "24_21"),
        ("boundary_task", "040_wrong_boundary"),
        ("forbidden_task", "041_wrong_forbidden"),
    ],
)
def test_validator_rejects_a_wrong_repartition_contract(
    tmp_path: Path,
    field: str,
    bad_value: object,
):
    payload = valid_marker()
    payload[field] = bad_value
    marker = tmp_path / "boundary.json"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    completed = run_validator(marker)

    assert completed.returncode == 2
    assert "BOUNDARY_MARKER_INVALID" in completed.stderr


@pytest.mark.parametrize(
    "content",
    [
        "{not-json",
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
    ],
)
def test_validator_rejects_non_strict_json(tmp_path: Path, content: str):
    marker = tmp_path / "boundary.json"
    marker.write_text(content, encoding="utf-8")

    completed = run_validator(marker)

    assert completed.returncode == 2
    assert "BOUNDARY_MARKER_INVALID" in completed.stderr


def test_waiter_only_waits_for_an_absent_atomic_relay_marker():
    script = read_waiter()

    assert (
        "/control/repartition_27_18_boundary_ready.json"
        in script
    )
    assert r"C:\Users\ASUS\.ssh\xtraffic_relay_ed25519" in script
    assert "BatchMode=yes" in script
    assert "StrictHostKeyChecking=accept-new" in script
    assert "test ! -L" in script
    assert "BOUNDARY_ABSENT" in script
    assert 'Write-WaiterLog "WAIT_BOUNDARY' in script
    assert "Start-Sleep -Seconds $PollSeconds" in script
    assert "protocol_v3_validate_repartition_marker.py" in script
    assert "BOUNDARY_MARKER_INVALID" in script
    assert "invalid boundary marker" in script


def test_waiter_rechecks_the_local_24_task_boundary_before_any_launch():
    script = read_waiter()

    assert "[int]$OldFinalizerPid = 12524" in script
    assert "Get-CimInstance Win32_Process" in script
    assert "protocol_v3_finalize_5080_shard.ps1" in script
    assert "protocol_v3_finalize_5080_shard.cmd" in script
    assert 'queue.state -ne "completed"' in script
    assert "[int]$queue.total_tasks -ne 24" in script
    assert "[int]$queue.completed_tasks -ne 24" in script
    assert "040_fullwindow_graphwavenet_base_contracosta_seed42" in script
    assert "forbidden task040 formal result already exists" in script

    main = script[script.rindex("\ntry {") :]
    assert main.index("Assert-LocalRepartitionBoundary") < main.index(
        "Start-RepartitionProcesses"
    )


def test_waiter_only_stops_pid_12524_after_exact_command_line_attestation():
    script = read_waiter()

    assert "Get-CimInstance Win32_Process -Filter" in script
    assert "ProcessId = $OldFinalizerPid" in script
    assert r"C:\Users\ASUS\protocol_v3_finalize_5080_shard.ps1" in script
    assert r"C:\Users\ASUS\protocol_v3_finalize_5080_shard.cmd" in script
    assert "old finalizer PID command line does not match" in script
    assert "Get-DescendantProcessIds -RootProcessId $OldFinalizerPid" in script
    assert "Stop-ProcessTree -ProcessId $OldFinalizerPid" in script
    assert "Get-Process -Id $treeProcessId" in script
    assert "foreach ($treeProcessId in $oldTreeProcessIds)" in script
    assert "old 5080 finalizer processes remain alive" in script

    main = script[script.rindex("\ntry {") :]
    first_boundary = main.index("Assert-LocalRepartitionBoundary")
    stop_old = main.index("Stop-ExactOldFinalizer")
    second_boundary = main.index(
        "Assert-LocalRepartitionBoundary",
        first_boundary + 1,
    )
    launch = main.index("Start-RepartitionProcesses")
    assert first_boundary < stop_old < second_boundary < launch


def test_waiter_bounds_every_ssh_call_and_kills_the_timed_out_process_tree():
    script = read_waiter()

    assert "[int]$SshWallClockTimeoutSeconds" in script
    assert "$process.WaitForExit($TimeoutSeconds * 1000)" in script
    assert "$process.WaitForExit()" in script
    assert "$process.Refresh()" in script
    assert "$process.Kill($true)" in script
    assert '"/PID", ([string]$ProcessId), "/T", "/F"' in script
    assert "TimedOut = $timedOut" in script
    assert "if ($result.TimedOut)" in script
    assert 'Write-WaiterLog "SSH_TIMEOUT' in script
    assert 'Write-WaiterLog "WAIT_BOUNDARY_SSH reason=timeout"' in script
    assert 'Write-WaiterLog "WAIT_BOUNDARY_SSH reason=exit_code_' in script
    assert "boundary marker SSH retry limit reached" in script
    assert '"ServerAliveInterval=5"' in script
    assert '"ServerAliveCountMax=3"' in script
    assert "Invoke-SshCapture -Command $command" in script
    assert "TimeoutSeconds = $SshWallClockTimeoutSeconds" in script


def test_waiter_starts_runner_then_finalizer_only_after_the_27_task_queue_exists():
    script = read_waiter()

    assert "protocol_v3_launch_5080_shard.cmd" in script
    assert "protocol_v3_finalize_5080_shard.cmd" in script
    assert "Start-CmdDetached -Path $RunnerLauncher" in script
    assert "Wait-ForNewRunnerContract" in script
    assert "[int]$queue.total_tasks -eq 27" in script
    assert '$queue.state -in @("preflight", "running", "completed")' in script
    assert "new 27-task runner entered failed state" in script
    assert "new 27-task runner exited before publishing its queue" in script
    assert "Start-CmdDetached -Path $FinalizerLauncher" in script
    assert "new 27-task finalizer exited during startup" in script
    assert "REPARTITION_STARTED runner_pid=" in script

    start_function = script[
        script.index("function Start-RepartitionProcesses") :
        script.index("\ntry {", script.index("function Start-RepartitionProcesses"))
    ]
    runner_start = start_function.index("Start-CmdDetached -Path $RunnerLauncher")
    queue_gate = start_function.index("Wait-ForNewRunnerContract")
    finalizer_start = start_function.index(
        "Start-CmdDetached -Path $FinalizerLauncher"
    )
    assert runner_start < queue_gate < finalizer_start


def test_waiter_wrapper_uses_an_independent_durable_log():
    wrapper = WAITER_WRAPPER.read_text(encoding="utf-8")

    assert "powershell.exe -NoProfile -ExecutionPolicy Bypass" in wrapper
    assert "protocol_v3_repartition_5080_27_waiter.ps1" in wrapper
    assert "protocol_v3_5080_repartition27_waiter_console.log 2>&1" in wrapper


def test_linux_publisher_and_windows_validator_share_the_marker_contract(
    tmp_path: Path,
):
    transition = LINUX_TRANSITION.read_text(encoding="utf-8")
    published_fields = set(
        re.findall(r'^\s*"([a-z][a-z0-9_]*)"\s*:', transition, re.MULTILINE)
    )
    payload = valid_marker()
    payload.pop("extra_provenance")

    assert set(payload) <= published_fields

    marker = tmp_path / "cross_host_boundary.json"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    completed = run_validator(marker)
    assert completed.returncode == 0, completed.stderr
