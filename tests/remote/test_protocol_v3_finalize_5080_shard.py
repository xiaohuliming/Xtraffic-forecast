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


def read_finalizer() -> str:
    return FINALIZER.read_text(encoding="utf-8")


def test_finalizer_waits_for_the_exact_completed_5080_shard_contract():
    script = read_finalizer()

    assert 'State -eq "completed"' in script
    assert "$totalTasks -eq 24" in script
    assert "$completedTasks -eq 24" in script
    assert "$skippedTasks -ge 4" in script
    assert "$skippedTasks -le 24" in script
    assert "RunnerExitCode" in script
    assert 'runner exit code is not zero' in script


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


def test_finalizer_serializes_upload_and_waits_for_relay_capacity():
    script = read_finalizer()

    assert r'C:\Users\ASUS\.ssh\xtraffic_relay_ed25519' in script
    assert 'RemoteName = "rtx5080_24"' in script
    assert 'DependencyName = "rtx5090_21"' in script
    assert '/.${RemoteName}.incoming' in script
    assert 'distributed_45_5080_5090/shards' in script
    assert 'Wait-ForDependencyTransfer' in script
    assert '_TRANSFER_COMPLETE.json' in script
    assert "$requiredCapacity = $exports.SourceBytes + $SpaceMarginBytes" in script
    assert 'Write-FinalizerLog "WAIT_SPACE' in script
    assert "Start-Sleep -Seconds $SpaceRetrySeconds" in script


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


def test_wrapper_runs_the_powershell_finalizer_with_a_durable_log():
    wrapper = FINALIZER_WRAPPER.read_text(encoding="utf-8")

    assert "powershell.exe -NoProfile -ExecutionPolicy Bypass" in wrapper
    assert "-File C:\\Users\\ASUS\\protocol_v3_finalize_5080_shard.ps1" in wrapper
    assert "protocol_v3_5080_shard_finalize_console.log 2>&1" in wrapper
