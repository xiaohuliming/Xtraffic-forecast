from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FINALIZER = (
    REPO_ROOT
    / "scripts/remote/windows/protocol_v3_finalize_5080_shard.ps1"
)
FINALIZER_WRAPPER = (
    REPO_ROOT
    / "scripts/remote/windows/protocol_v3_finalize_5080_shard.cmd"
)
RUNNER_WRAPPER = (
    REPO_ROOT
    / "scripts/remote/windows/protocol_v3_launch_5080_shard.cmd"
)


def read_finalizer() -> str:
    return FINALIZER.read_text(encoding="utf-8")


def test_5080_repartition_controls_default_to_an_isolated_ops_root():
    isolated_ops_root = r"C:\Users\ASUS\ops_27_18"

    assert isolated_ops_root in read_finalizer()
    assert isolated_ops_root in RUNNER_WRAPPER.read_text(encoding="utf-8")


def test_finalizer_waits_for_the_exact_completed_5080_shard_contract():
    script = read_finalizer()

    assert 'State -eq "completed"' in script
    assert "$totalTasks -eq 27" in script
    assert "$completedTasks -eq 27" in script
    assert "$skippedTasks -ge 24" in script
    assert "$skippedTasks -le 27" in script
    assert "$totalTasks -eq 24" not in script
    assert "$completedTasks -eq 24" not in script
    assert "RunnerExitCode" in script
    assert 'runner exit code is not zero' in script
    assert 'QUEUE_OK completed=27/27' in script
    assert 'QUEUE_OK completed=24/24' not in script


def test_finalizer_uses_the_external_postflight_export_allowlist():
    script = read_finalizer()

    assert "protocol_v3_shard_postflight.py" in script
    assert '"--shard", "5080"' in script
    assert 'report.status -ne "verified"' in script
    assert 'report.shard -ne "5080"' in script
    assert "report.export_paths" in script
    assert "report.files" in script
    assert "report.actual_skipped_valid_tasks" in script
    assert "inherited skipped task identity is missing" in script
    assert "postflight task state summary is inconsistent" in script
    assert "export path set differs from hashed file records" in script
    assert "Get-FileHash" in script
    assert "source file hash differs from postflight report" in script


def test_finalizer_requires_the_24_inherited_results_but_allows_safe_restarts():
    script = read_finalizer()

    assert "[int]$report.validated_tasks -ne 27" in script
    assert "[int]$report.expected_skipped_valid_tasks -ne 24" in script
    assert "[int]$report.minimum_skipped_valid_tasks -ne 24" in script
    assert "$actualSkipped -lt 24" in script
    assert "$actualSkipped -gt 27" in script
    assert "$expectedSkipIds.Count -ne 24" in script
    assert "$tasks.Count -ne 27" in script
    assert "$completedStateCount + $skippedStateCount -ne 27" in script
    assert "[int]$report.queue_status.total_tasks -ne 27" in script
    assert "[int]$report.queue_status.completed_tasks -ne 27" in script
    assert "minimum_skipped_valid_tasks = 24" in script
    assert "validated_tasks = 27" in script


def test_finalizer_serializes_upload_and_waits_for_relay_capacity():
    script = read_finalizer()

    assert r'C:\Users\ASUS\.ssh\xtraffic_relay_ed25519' in script
    assert 'RemoteName = "rtx5080_27"' in script
    assert 'DependencyName = "rtx5090_18"' in script
    assert 'rtx5080_24' not in script
    assert 'rtx5090_21' not in script
    assert '/.${RemoteName}.incoming' in script
    assert 'distributed_45_5080_5090/shards' in script
    assert 'Wait-ForDependencyTransfer' in script
    assert '_TRANSFER_COMPLETE.json' in script
    assert "$requiredCapacity = $exports.SourceBytes + $SpaceMarginBytes" in script
    assert 'Write-FinalizerLog "WAIT_SPACE' in script
    assert "Start-Sleep -Seconds $SpaceRetrySeconds" in script


def test_repartition_uses_new_local_attestation_and_runner_artifacts():
    script = read_finalizer()
    runner = RUNNER_WRAPPER.read_text(encoding="utf-8")
    wrapper = FINALIZER_WRAPPER.read_text(encoding="utf-8")

    expected_names = (
        "protocol_v3_5080_shard27.exitcode",
        "protocol_v3_5080_shard27_finalize.pid",
        "protocol_v3_5080_shard27_postflight.json",
        "protocol_v3_5080_shard27_transfer.sha256",
        "protocol_v3_5080_shard27_files.txt",
        "protocol_v3_5080_shard27_complete.json",
        "protocol_v3_5080_shard27_finalize.log",
    )
    assert all(name in script for name in expected_names)
    assert "protocol_v3_5080_shard27.exitcode" in runner
    assert "protocol_v3_5080_shard27.pid" in runner
    assert "protocol_v3_5080_shard27.log" in runner
    assert "protocol_v3_5080_shard27_finalize_console.log" in wrapper
    assert "remote_runs\\minimal_45_5080" in script
    assert "remote_runs\\minimal_45_5080" in runner

    legacy_artifacts = (
        "protocol_v3_5080_shard.exitcode",
        "protocol_v3_5080_shard_postflight.json",
        "protocol_v3_5080_shard_transfer.sha256",
        "protocol_v3_5080_shard_files.txt",
        "protocol_v3_5080_shard_complete.json",
        "protocol_v3_5080_shard_finalize.log",
    )
    assert all(name not in script for name in legacy_artifacts)
    assert "protocol_v3_5080_shard.exitcode" not in runner


def test_new_runner_and_finalizer_publish_distinct_liveness_pid_files():
    script = read_finalizer()
    runner = RUNNER_WRAPPER.read_text(encoding="utf-8")

    assert "[string]$PidFile" in script
    assert "[string]$PID" in script
    assert "finally {" in script
    assert "Remove-Item -LiteralPath $PidFile -Force" in script
    assert 'set "PID_FILE=C:\\Users\\ASUS\\protocol_v3_5080_shard27.pid"' in runner
    assert "[System.IO.File]::WriteAllText('%PID_FILE%'" in runner
    assert 'if exist "%PID_FILE%" del /f /q "%PID_FILE%"' in runner


def test_runner_refuses_to_mutate_the_reused_results_root_until_old_finalizer_stops():
    runner = RUNNER_WRAPPER.read_text(encoding="utf-8")

    guard = "Get-CimInstance Win32_Process"
    first_mutation = 'if exist "%EXIT_CODE%" del /f /q "%EXIT_CODE%"'
    assert guard in runner
    assert "protocol_v3_finalize_5080_shard.ps1" in runner
    assert "ProcessId -ne $PID" in runner
    assert "exit 41" in runner
    assert "if errorlevel 1 exit /b 41" in runner
    assert runner.index(guard) < runner.index(first_mutation)


def test_finalizer_never_powers_off_the_5080():
    script = read_finalizer().lower()

    forbidden = (
        "shutdown.exe",
        "stop-computer",
        "restart-computer",
        "shutdown -",
        "shutdown /",
    )
    assert all(token not in script for token in forbidden)
    assert "finalize_success machine_stays_on" in script


def test_each_export_is_hash_checked_resumable_and_remotely_promoted():
    script = read_finalizer()

    assert "foreach ($record in $exports.Records)" in script
    assert "Upload-RemoteFileByHash" in script
    assert '"$RemotePath.part"' in script
    assert 'printf \'%s\\n\' MATCH' in script
    assert 'printf \'%s\\n\' UPLOAD' in script
    assert "TRANSFER_SKIP sha256=" in script
    assert "TRANSFER_FILE_OK sha256=" in script
    assert "sha256sum -c ../metadata/transfer.sha256" in script
    assert "Test-RemoteVerifiedTree" in script
    assert "cmp `\"`$tmp/actual.files`\" `\"`$tmp/expected.files`\"" in script
    assert "cmp `\"`$tmp/actual.dirs`\" `\"`$tmp/expected.dirs`\"" in script
    assert "postflight.json transfer.files transfer.sha256" in script
    assert "mv '$RemoteIncoming' '$RemoteFinal'" in script
    assert "$RemoteIncoming/_TRANSFER_COMPLETE.json.tmp" in script
    assert '-Root $RemoteFinal' in script
    assert '-MarkerName "_TRANSFER_COMPLETE.json"' in script


def test_existing_destination_requires_stable_full_attestation_recheck():
    script = read_finalizer()

    assert "source_completed_at" in script
    assert "postflight_files_digest" in script
    assert "file_list_sha256" in script
    assert "markerDigest" in script
    assert "relay destination exists but full verification failed" in script
    assert "REMOTE_ALREADY_VERIFIED" in script


def test_all_operational_errors_end_in_a_fail_closed_exit():
    script = read_finalizer()

    assert '$ErrorActionPreference = "Stop"' in script
    assert "Set-StrictMode -Version Latest" in script
    assert 'Write-FinalizerLog "FINALIZE_BLOCKED' in script
    assert "catch {" in script
    assert "exit 1" in script


def test_all_finalizer_ssh_and_scp_calls_have_keepalives_and_tree_kill_timeouts():
    script = read_finalizer()

    assert "[int]$SshWallClockTimeoutSeconds" in script
    assert "[int]$RemoteVerificationWallClockTimeoutSeconds" in script
    assert "[int]$ScpWallClockTimeoutSeconds" in script
    assert "[int]$PostflightWallClockTimeoutSeconds" in script
    assert "$process.WaitForExit($TimeoutSeconds * 1000)" in script
    assert "$process.WaitForExit()" in script
    assert "$process.Refresh()" in script
    assert "$process.Kill($true)" in script
    assert '"/PID", ([string]$ProcessId), "/T", "/F"' in script
    assert "TimedOut = $timedOut" in script
    assert script.count('"ServerAliveInterval=5"') >= 2
    assert script.count('"ServerAliveCountMax=3"') >= 2
    assert "TimeoutSeconds = $SshWallClockTimeoutSeconds" in script
    assert "Invoke-SshCapture -Command $command -TimeoutSeconds $RemoteVerificationWallClockTimeoutSeconds" in script
    assert script.count("Test-RemoteVerifiedTree") >= 4
    assert "TimeoutSeconds = $ScpWallClockTimeoutSeconds" in script
    assert "TimeoutSeconds = $PostflightWallClockTimeoutSeconds" in script
    assert "native process timed out" in script


def test_wrapper_runs_the_powershell_finalizer_with_a_durable_log():
    wrapper = FINALIZER_WRAPPER.read_text(encoding="utf-8")

    assert "powershell.exe -NoProfile -ExecutionPolicy Bypass" in wrapper
    assert "-File C:\\Users\\ASUS\\protocol_v3_finalize_5080_shard.ps1" in wrapper
    assert "protocol_v3_5080_shard27_finalize_console.log 2>&1" in wrapper
