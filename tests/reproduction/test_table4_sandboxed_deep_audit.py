import hashlib
import sys

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4 import sandboxed_deep_audit
from scripts.reproduction.igstgnn_table4 import deep_audit_worker
from scripts.reproduction.igstgnn_table4 import sandbox_probe_worker
from scripts.reproduction.igstgnn_table4.sandboxed_deep_audit import (
    run_sandboxed_deep_audit,
    verify_macos_sandbox_boundary,
)


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="This security boundary uses the macOS sandbox.",
)


def _valid_sample():
    ticks = 72 + np.arange(24)
    x_data = np.zeros((12, 4, 3), dtype=np.float32)
    x_data[..., 1] = (ticks[:12] % 288)[:, None] / 288
    x_data[..., 2] = (ticks[:12] // 288)[:, None] / 7
    y_data = np.zeros((12, 4, 3), dtype=np.float32)
    y_data[..., 1] = (ticks[12:] % 288)[:, None] / 288
    y_data[..., 2] = (ticks[12:] // 288)[:, None] / 7
    return {
        "x_data": x_data,
        "y_data": y_data,
        "incident_features": {
            "Incident Time": 0.0,
            "Description": 0,
            "Type": 0,
            "Holiday": 0,
        },
        "incident_position": 0,
        "incident_distances": np.ones((4, 3), dtype=np.float32),
        "durations": np.ones(12, dtype=np.float32),
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sandbox_boundary_negative_controls_all_fail_closed(tmp_path):
    probe = verify_macos_sandbox_boundary(
        work_directory=tmp_path,
        timeout_seconds=30,
    )

    assert probe["status"] == "sandbox_boundary_verified"
    assert probe["allowed_input_read"] is True
    assert probe["credential_read_denied"] is True
    assert probe["neighbor_read_denied"] is True
    assert probe["network_denied"] is True
    assert probe["fork_denied"] is True
    assert probe["spawn_denied"] is True
    assert probe["assertion_mode"] == "exact_key_and_value_match"
    assert len(probe["controller_sha256"]) == 64
    assert len(probe["controller_profile_sha256"]) == 64
    assert len(probe["rendered_profile_sha256"]) == 64
    assert len(probe["worker_report_sha256"]) == 64


def test_sandbox_boundary_rejects_a_symlink_work_directory(tmp_path):
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    symlink_directory = tmp_path / "link"
    symlink_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ValueError, match="owned, non-symlink directory"):
        verify_macos_sandbox_boundary(
            work_directory=symlink_directory,
            timeout_seconds=30,
        )


def test_sandbox_boundary_rejects_extra_negative_control_fields(
    tmp_path,
    monkeypatch,
):
    original_read = sandboxed_deep_audit._read_bounded_json

    def read_with_extra_field(path, **kwargs):
        report = original_read(path, **kwargs)
        report["unexpected"] = True
        return report

    monkeypatch.setattr(
        sandboxed_deep_audit,
        "_read_bounded_json",
        read_with_extra_field,
    )

    with pytest.raises(RuntimeError, match="invalid report"):
        verify_macos_sandbox_boundary(
            work_directory=tmp_path,
            timeout_seconds=30,
        )


def test_sandboxed_deep_audit_writes_a_verified_manifest(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    incident_file = input_dir / "incident_all.npy"
    np.save(incident_file, np.asarray([_valid_sample()], dtype=object))
    output = output_dir / "deep_audit.json"

    audit = run_sandboxed_deep_audit(
        incident_file=incident_file,
        expected_sha256=_sha256(incident_file),
        output=output,
        region="Synthetic",
        node_count=4,
        expected_sample_count=1,
        description_count=1,
        type_count=1,
        timeout_seconds=30,
    )

    assert audit["status"] == "records_structurally_verified"
    assert audit["records_validated"] == 1
    worker_verification = audit["worker_verification"]
    assert worker_verification["input_sha256"] == _sha256(incident_file)
    assert len(worker_verification["deep_audit_sha256"]) == 64
    assert len(worker_verification["restricted_npy_sha256"]) == 64
    boundary = audit["security_boundary"]
    assert len(boundary["controller_sha256"]) == 64
    assert len(boundary["controller_profile_sha256"]) == 64
    assert len(boundary["rendered_profile_sha256"]) == 64
    assert (
        boundary["negative_controls"]["assertion_mode"] == "exact_key_and_value_match"
    )
    assert output.is_file()


def test_sandbox_blocks_pickle_from_reading_an_unapproved_file(tmp_path):
    secret_dir = tmp_path / "secret"
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    secret_dir.mkdir()
    input_dir.mkdir()
    output_dir.mkdir()
    secret = secret_dir / "credential.txt"
    secret.write_text("must-not-be-readable", encoding="utf-8")

    class SecretReader:
        def __reduce__(self):
            expression = f"open({str(secret)!r}, encoding='utf-8').read()"
            return eval, (expression,)

    incident_file = input_dir / "incident_all.npy"
    np.save(
        incident_file,
        np.asarray([SecretReader()], dtype=object),
    )
    output = output_dir / "deep_audit.json"

    with pytest.raises(
        RuntimeError,
        match=r"Forbidden pickle global: builtins\.eval",
    ):
        run_sandboxed_deep_audit(
            incident_file=incident_file,
            expected_sha256=_sha256(incident_file),
            output=output,
            region="Synthetic",
            node_count=4,
            expected_sample_count=1,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
        )

    assert not output.exists()


def test_sandboxed_deep_audit_rejects_an_unexpected_input_hash(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    incident_file = input_dir / "incident_all.npy"
    np.save(incident_file, np.asarray([_valid_sample()], dtype=object))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_sandboxed_deep_audit(
            incident_file=incident_file,
            expected_sha256="0" * 64,
            output=output_dir / "deep_audit.json",
            region="Synthetic",
            node_count=4,
            expected_sample_count=1,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
        )


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("--expected-sha256", "Incident input SHA-256 mismatch"),
        ("--expected-deep-audit-sha256", "deep_audit module SHA-256 mismatch"),
    ],
)
def test_worker_independently_verifies_parent_pinned_bytes(
    tmp_path,
    monkeypatch,
    argument,
    message,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    incident_file = input_dir / "incident_all.npy"
    np.save(incident_file, np.asarray([_valid_sample()], dtype=object))
    output = output_dir / "deep_audit.json"
    original_command = sandboxed_deep_audit._sandbox_command

    def command_with_wrong_worker_hash(**kwargs):
        command = original_command(**kwargs)
        index = command.index(argument)
        command[index + 1] = "0" * 64
        return command

    monkeypatch.setattr(
        sandboxed_deep_audit,
        "_sandbox_command",
        command_with_wrong_worker_hash,
    )

    with pytest.raises(RuntimeError, match=message):
        run_sandboxed_deep_audit(
            incident_file=incident_file,
            expected_sha256=_sha256(incident_file),
            output=output,
            region="Synthetic",
            node_count=4,
            expected_sample_count=1,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "worker_module",
    [deep_audit_worker, sandbox_probe_worker],
)
def test_worker_output_identity_is_checked_before_truncation(
    tmp_path,
    worker_module,
):
    output = tmp_path / "worker-output.json"
    output.touch(mode=0o600)
    original_identity = worker_module._stable_output_identity(output.lstat())
    output.unlink()
    output.write_bytes(b"racing-owner\n")
    output.chmod(0o600)

    with pytest.raises(RuntimeError, match="identity changed before writing"):
        worker_module._write_precreated_output(
            output,
            b"{}\n",
            expected_identity=original_identity,
        )

    assert output.read_bytes() == b"racing-owner\n"


def test_parent_rejects_worker_json_larger_than_one_mib(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    incident_file = input_dir / "incident_all.npy"
    np.save(incident_file, np.asarray([_valid_sample()], dtype=object))
    output = output_dir / "deep_audit.json"
    original_create = sandboxed_deep_audit._create_staging_file

    def create_oversized_staging(destination):
        staging = original_create(destination)
        staging.write_bytes(b"{" + b" " * (1024 * 1024))
        return staging

    monkeypatch.setattr(
        sandboxed_deep_audit,
        "_create_staging_file",
        create_oversized_staging,
    )
    monkeypatch.setattr(
        sandboxed_deep_audit,
        "verify_macos_sandbox_boundary",
        lambda **kwargs: {"status": "sandbox_boundary_verified"},
    )
    monkeypatch.setattr(
        sandboxed_deep_audit.subprocess,
        "Popen",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        sandboxed_deep_audit,
        "_bounded_child_output",
        lambda *args, **kwargs: (0, "", "", 0),
    )

    with pytest.raises(RuntimeError, match="exceeds 1048576 bytes"):
        run_sandboxed_deep_audit(
            incident_file=incident_file,
            expected_sha256=_sha256(incident_file),
            output=output,
            region="Synthetic",
            node_count=4,
            expected_sample_count=1,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
        )

    assert not output.exists()


def test_final_publish_never_clobbers_or_cleans_up_a_racing_destination(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    incident_file = input_dir / "incident_all.npy"
    np.save(incident_file, np.asarray([_valid_sample()], dtype=object))
    output = output_dir / "deep_audit.json"
    original_publish = sandboxed_deep_audit._write_final_manifest

    def race_publish(staging, destination, audit, **kwargs):
        destination.write_text("racing-owner\n", encoding="utf-8")
        return original_publish(staging, destination, audit, **kwargs)

    monkeypatch.setattr(
        sandboxed_deep_audit,
        "_write_final_manifest",
        race_publish,
    )

    with pytest.raises(FileExistsError):
        run_sandboxed_deep_audit(
            incident_file=incident_file,
            expected_sha256=_sha256(incident_file),
            output=output,
            region="Synthetic",
            node_count=4,
            expected_sample_count=1,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
        )

    assert output.read_text(encoding="utf-8") == "racing-owner\n"
