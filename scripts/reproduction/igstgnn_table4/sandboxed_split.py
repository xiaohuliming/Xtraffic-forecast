from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .release_audit import _audit_safe_region_structure, read_npy_header
from .sandboxed_deep_audit import (
    _DEFAULT_MAX_RSS_BYTES,
    _EXPECTED_DYLD_PROFILE_SHA256,
    _EXPECTED_SANDBOX_EXEC_SHA256,
    _PROJECT_ROOT,
    _RESTRICTED_NPY,
    _SANDBOX_EXEC,
    _bounded_child_output,
    _empty_environment,
    _file_identity,
    _require_owned_regular_file,
    _sha256_file,
    _verify_boundary_files,
)


_MODULE_DIR = Path(__file__).resolve().parent
_WORKER = _MODULE_DIR / "split_worker.py"
_POSTFLIGHT_WORKER = _MODULE_DIR / "split_postflight_worker.py"
_PROBE_WORKER = _MODULE_DIR / "sandbox_probe_worker.py"
_DEEP_AUDIT = _MODULE_DIR / "deep_audit.py"
_OFFICIAL_SPLIT = _MODULE_DIR / "official_split.py"
_SPLIT_ARTIFACTS = _MODULE_DIR / "split_artifacts.py"
_FINAL_FILENAMES = {
    "train": "incident_train.npy",
    "val": "incident_val.npy",
    "test": "incident_test.npy",
}
_SUPPORT_FILENAMES = frozenset(
    {
        "adj_matrix.npy",
        "incident_stats.npz",
        "sensors.csv",
        "desc_mapping.json",
        "type_mapping.json",
    }
)
_RELEASE_FILENAMES = _SUPPORT_FILENAMES | {"incident_all.npy"}
_FINAL_BUNDLE_FILENAMES = (
    frozenset(_FINAL_FILENAMES.values()) | _SUPPORT_FILENAMES | {"split_manifest.json"}
)
_MAX_JSON_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_JSON_POSTFLIGHT_BYTES = 512 * 1024
_AT_FDCWD = -2
_RENAME_EXCL = 0x00000004


def _split_profile() -> str:
    return """\
(version 1)
(deny default)
(import "dyld-support.sb")

(define python-exe (param "PYTHON_EXE"))
(define python-lib (param "PYTHON_LIB"))
(define worker-file (param "WORKER"))
(define deep-audit-file (param "DEEP_AUDIT"))
(define official-split-file (param "OFFICIAL_SPLIT"))
(define restricted-npy-file (param "RESTRICTED_NPY"))
(define split-artifacts-file (param "SPLIT_ARTIFACTS"))
(define input-file (param "INPUT"))
(define train-file (param "TRAIN"))
(define val-file (param "VAL"))
(define test-file (param "TEST"))
(define manifest-file (param "MANIFEST"))
(define ssh-dir (param "SSH_DIR"))
(define data-ssh-dir (param "DATA_SSH_DIR"))
(define lab-hosts-file (param "LAB_HOSTS"))
(define data-lab-hosts-file (param "DATA_LAB_HOSTS"))

(deny network*)
(deny process-fork)
(deny file-read*
  (subpath ssh-dir)
  (subpath data-ssh-dir)
  (literal lab-hosts-file)
  (literal data-lab-hosts-file))

(allow process-exec (literal python-exe))
(allow file-read* file-test-existence file-map-executable
  (literal python-exe)
  (subpath python-lib))
(allow file-read* file-test-existence
  (literal worker-file)
  (literal deep-audit-file)
  (literal official-split-file)
  (literal restricted-npy-file)
  (literal split-artifacts-file)
  (literal input-file))
(allow file-read-metadata file-test-existence
  (path-ancestors worker-file)
  (path-ancestors deep-audit-file)
  (path-ancestors official-split-file)
  (path-ancestors restricted-npy-file)
  (path-ancestors split-artifacts-file)
  (path-ancestors input-file)
  (path-ancestors train-file)
  (path-ancestors val-file)
  (path-ancestors test-file)
  (path-ancestors manifest-file)
  (literal train-file)
  (literal val-file)
  (literal test-file)
  (literal manifest-file))
(allow file-write* file-test-existence
  (literal train-file)
  (literal val-file)
  (literal test-file)
  (literal manifest-file))
(allow file-read* file-test-existence
  (literal train-file)
  (literal val-file)
  (literal test-file))

(allow file-read* file-test-existence
  (literal "/dev/null")
  (literal "/dev/urandom")
  (literal "/dev/random"))
(allow file-write-data (literal "/dev/null"))

(allow sysctl-read
  (sysctl-name
    "kern.ostype"
    "kern.hostname"
    "kern.osrelease"
    "kern.version"
    "hw.machine"))
"""


def _postflight_profile() -> str:
    return """\
(version 1)
(deny default)
(import "dyld-support.sb")

(define python-exe (param "PYTHON_EXE"))
(define python-lib (param "PYTHON_LIB"))
(define worker-file (param "WORKER"))
(define deep-audit-file (param "DEEP_AUDIT"))
(define restricted-npy-file (param "RESTRICTED_NPY"))
(define train-file (param "TRAIN"))
(define val-file (param "VAL"))
(define test-file (param "TEST"))
(define output-file (param "OUTPUT"))
(define ssh-dir (param "SSH_DIR"))
(define data-ssh-dir (param "DATA_SSH_DIR"))
(define lab-hosts-file (param "LAB_HOSTS"))
(define data-lab-hosts-file (param "DATA_LAB_HOSTS"))

(deny network*)
(deny process-fork)
(deny file-read*
  (subpath ssh-dir)
  (subpath data-ssh-dir)
  (literal lab-hosts-file)
  (literal data-lab-hosts-file))

(allow process-exec (literal python-exe))
(allow file-read* file-test-existence file-map-executable
  (literal python-exe)
  (subpath python-lib))
(allow file-read* file-test-existence
  (literal worker-file)
  (literal deep-audit-file)
  (literal restricted-npy-file)
  (literal train-file)
  (literal val-file)
  (literal test-file))
(allow file-read-metadata file-test-existence
  (path-ancestors worker-file)
  (path-ancestors deep-audit-file)
  (path-ancestors restricted-npy-file)
  (path-ancestors train-file)
  (path-ancestors val-file)
  (path-ancestors test-file)
  (path-ancestors output-file)
  (literal output-file))
(allow file-write* file-test-existence (literal output-file))

(allow file-read* file-test-existence
  (literal "/dev/null")
  (literal "/dev/urandom")
  (literal "/dev/random"))
(allow file-write-data (literal "/dev/null"))

(allow sysctl-read
  (sysctl-name
    "kern.ostype"
    "kern.hostname"
    "kern.osrelease"
    "kern.version"
    "hw.machine"))
"""


def _profile_probe_command(
    *,
    profile: str,
    profile_name: str,
    python: Path,
    input_file: Path,
    output_file: Path,
    credential_file: Path,
    neighbor_file: Path,
) -> list[str]:
    home = Path.home()
    lab_hosts = _PROJECT_ROOT / "config" / "lab_hosts.env"
    common = {
        "PYTHON_EXE": python,
        "PYTHON_LIB": python.parent.parent / "lib",
        "WORKER": _PROBE_WORKER,
        "DEEP_AUDIT": _DEEP_AUDIT,
        "RESTRICTED_NPY": _RESTRICTED_NPY,
        "SSH_DIR": home / ".ssh",
        "DATA_SSH_DIR": f"/System/Volumes/Data{home / '.ssh'}",
        "LAB_HOSTS": lab_hosts,
        "DATA_LAB_HOSTS": f"/System/Volumes/Data{lab_hosts}",
    }
    if profile_name == "writer":
        values = {
            **common,
            "OFFICIAL_SPLIT": _OFFICIAL_SPLIT,
            "SPLIT_ARTIFACTS": _SPLIT_ARTIFACTS,
            "INPUT": input_file,
            "TRAIN": output_file,
            "VAL": output_file,
            "TEST": output_file,
            "MANIFEST": output_file,
        }
    elif profile_name == "postflight":
        values = {
            **common,
            "TRAIN": input_file,
            "VAL": input_file,
            "TEST": input_file,
            "OUTPUT": output_file,
        }
    else:
        raise ValueError(f"Unknown sandbox profile: {profile_name}")
    command = [str(_SANDBOX_EXEC)]
    for key, value in values.items():
        command.extend(("-D", f"{key}={value}"))
    command.extend(
        (
            "-p",
            profile,
            str(python),
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            str(_PROBE_WORKER),
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--credential-file",
            str(credential_file),
            "--neighbor-file",
            str(neighbor_file),
        )
    )
    return command


def _run_actual_profile_probe(
    *,
    profile: str,
    profile_name: str,
    work_directory: Path,
    python: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    credential_file = _PROJECT_ROOT / "config" / "lab_hosts.env"
    neighbor_file = _PROJECT_ROOT / "CLAUDE.md"
    if not credential_file.is_file() or not neighbor_file.is_file():
        raise RuntimeError("Sandbox negative-control source files are unavailable")
    probe_input = work_directory / f".{profile_name}-sandbox-input.{uuid.uuid4().hex}"
    probe_output = work_directory / f".{profile_name}-sandbox-output.{uuid.uuid4().hex}"
    _precreate(probe_input)
    _precreate(probe_output)
    input_flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        input_flags |= os.O_NOFOLLOW
    descriptor = os.open(probe_input, input_flags)
    try:
        os.write(descriptor, b"allowed-probe-input\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    try:
        process = subprocess.Popen(
            _profile_probe_command(
                profile=profile,
                profile_name=profile_name,
                python=python,
                input_file=probe_input,
                output_file=probe_output,
                credential_file=credential_file,
                neighbor_file=neighbor_file,
            ),
            cwd="/",
            env=_empty_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        return_code, _, stderr, peak_rss = _bounded_child_output(
            process,
            timeout_seconds=timeout_seconds,
            max_rss_bytes=512 * 1024 * 1024,
        )
        if return_code != 0:
            raise RuntimeError(
                f"{profile_name} sandbox negative controls failed with "
                f"exit code {return_code}: {stderr[-4000:].strip()}"
            )
        report, _ = _read_bounded_json(
            probe_output,
            max_bytes=64 * 1024,
            label=f"{profile_name} sandbox probe",
        )
        required_true = {
            "allowed_input_read",
            "credential_read_denied",
            "neighbor_read_denied",
            "network_denied",
            "fork_denied",
            "spawn_denied",
        }
        if report.get("status") != "sandbox_boundary_verified" or any(
            report.get(key) is not True for key in required_true
        ):
            raise RuntimeError(f"{profile_name} sandbox negative controls are invalid")
        report["profile"] = profile_name
        report["sandbox_profile_sha256"] = hashlib.sha256(
            profile.encode("utf-8")
        ).hexdigest()
        report["peak_rss_bytes"] = peak_rss
        report["probe_worker_sha256"] = _sha256_file(_PROBE_WORKER)
        return report
    finally:
        probe_input.unlink(missing_ok=True)
        probe_output.unlink(missing_ok=True)


def _verify_actual_split_profiles(
    *,
    work_directory: Path,
    python: Path,
    timeout_seconds: int,
) -> dict[str, dict[str, Any]]:
    profiles = {
        "writer": _split_profile(),
        "postflight": _postflight_profile(),
    }
    return {
        name: _run_actual_profile_probe(
            profile=profile,
            profile_name=name,
            work_directory=work_directory,
            python=python,
            timeout_seconds=timeout_seconds,
        )
        for name, profile in profiles.items()
    }


def _precreate(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    os.close(descriptor)


def _atomic_publish_directory_no_clobber(
    staging: Path,
    destination: Path,
) -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "No reviewed atomic no-clobber directory publish is available"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameatx_np = libc.renameatx_np
    except AttributeError as exc:
        raise RuntimeError(
            "renameatx_np is unavailable; refusing a non-atomic publish"
        ) from exc
    renameatx_np.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameatx_np.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameatx_np(
        _AT_FDCWD,
        os.fsencode(staging),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_EXCL,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"Prepared split directory already exists: {destination}",
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _descriptor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_descriptor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _rewrite_owned_json_file(path: Path, value: dict[str, Any]) -> None:
    candidate = Path(path)
    _require_owned_regular_file(candidate, label="Final split manifest")
    before = candidate.lstat()
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise ValueError("Final split manifest must have mode 0600")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_JSON_MANIFEST_BYTES:
        raise ValueError("Final split manifest exceeded its fixed size limit")
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        if _descriptor_identity(os.fstat(descriptor)) != _descriptor_identity(before):
            raise RuntimeError("Final split manifest identity changed")
        os.ftruncate(descriptor, 0)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            remaining = remaining[written:]
        os.fsync(descriptor)
        if _stable_descriptor_identity(
            os.fstat(descriptor)
        ) != _stable_descriptor_identity(before):
            raise RuntimeError("Final split manifest identity changed")
    finally:
        os.close(descriptor)


def _read_bounded_json(
    path: Path,
    *,
    max_bytes: int,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    candidate = Path(path)
    _require_owned_regular_file(candidate, label=label)
    before = candidate.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        if _descriptor_identity(opened) != _descriptor_identity(before):
            raise RuntimeError(f"{label} identity changed before reading")
        if opened.st_size <= 0 or opened.st_size > max_bytes:
            raise ValueError(f"{label} must be between 1 and {max_bytes} bytes")
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != opened.st_size or len(payload) > max_bytes:
            raise ValueError(f"{label} exceeded its fixed size limit")
        if _descriptor_identity(os.fstat(descriptor)) != _descriptor_identity(opened):
            raise RuntimeError(f"{label} identity changed while reading")
    finally:
        os.close(descriptor)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return decoded, observed_sha256


def _validate_release_binding(
    *,
    release_manifest_file: Path,
    expected_release_manifest_sha256: str,
    source: Path,
    expected_sha256: str,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
    support_files: Mapping[str, tuple[Path, str]] | None,
) -> tuple[dict[str, tuple[Path, str]], dict[str, Any]]:
    if support_files is None or set(support_files) != _SUPPORT_FILENAMES:
        raise ValueError("Support files must contain the five official runtime files")
    release_manifest_path = Path(release_manifest_file)
    release_manifest, observed_manifest_sha256 = _read_bounded_json(
        release_manifest_path,
        max_bytes=_MAX_JSON_MANIFEST_BYTES,
        label="Locked release manifest",
        expected_sha256=expected_release_manifest_sha256,
    )
    if (
        release_manifest.get("status") != "safe_structure_verified"
        or release_manifest.get("protocol_id") != "igstgnn_table4_official_v1"
    ):
        raise ValueError("Locked release manifest is not formally verified")
    region_manifest = release_manifest.get("regions", {}).get(region)
    if not isinstance(region_manifest, dict):
        raise ValueError("Region is absent from the locked release manifest")
    mappings = region_manifest.get("safe_structure", {}).get("mappings", {})
    if (
        region_manifest.get("node_count") != node_count
        or region_manifest.get("sample_count") != expected_sample_count
        or mappings.get("description_count") != description_count
        or mappings.get("type_count") != type_count
    ):
        raise ValueError(
            "Runtime region contract differs from the locked release manifest"
        )
    locked_files = region_manifest.get("files")
    if not isinstance(locked_files, dict) or set(locked_files) != _RELEASE_FILENAMES:
        raise ValueError(
            "Locked release manifest must bind all six official region files"
        )
    if locked_files["incident_all.npy"].get("sha256") != expected_sha256:
        raise ValueError("Incident SHA-256 differs from the locked release manifest")

    normalized = {}
    for name in sorted(_SUPPORT_FILENAMES):
        value = support_files[name]
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[1], str)
        ):
            raise ValueError(f"Invalid support file binding for {name}")
        supplied_path, supplied_sha256 = value
        expected_path = source.parent / name
        candidate = Path(os.path.abspath(Path(supplied_path)))
        if candidate != expected_path:
            raise ValueError(
                f"Support input must be the fixed same-region file: {name}"
            )
        locked_sha256 = locked_files[name].get("sha256")
        if supplied_sha256 != locked_sha256:
            raise ValueError(
                f"Support SHA-256 is not bound by the release manifest: {name}"
            )
        normalized[name] = (candidate, locked_sha256)

    return normalized, {
        "path": str(release_manifest_path),
        "sha256": observed_manifest_sha256,
        "region": region,
    }


def _copy_support_files(
    *,
    support_files: Mapping[str, tuple[Path, str]],
    staging: Path,
    destination: Path,
) -> dict[str, Any]:
    if set(support_files) != _SUPPORT_FILENAMES:
        raise ValueError("Support files must contain the five official runtime files")
    manifest = {}
    for name in sorted(_SUPPORT_FILENAMES):
        source_path, expected_sha256 = support_files[name]
        source = Path(source_path)
        _require_owned_regular_file(source, label="Support input")
        source = source.resolve()
        before = source.lstat()
        source_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            source_flags |= os.O_NOFOLLOW
        source_descriptor = os.open(source, source_flags)
        target = staging / name
        target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            target_flags |= os.O_NOFOLLOW
        target_descriptor = os.open(target, target_flags, 0o600)
        digest = hashlib.sha256()
        copied = 0
        try:
            if _descriptor_identity(os.fstat(source_descriptor)) != (
                _descriptor_identity(before)
            ):
                raise RuntimeError(f"Support input identity changed: {source}")
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                copied += len(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(target_descriptor, view)
                    view = view[written:]
            os.fsync(target_descriptor)
            if _descriptor_identity(os.fstat(source_descriptor)) != (
                _descriptor_identity(before)
            ):
                raise RuntimeError(f"Support input identity changed: {source}")
        finally:
            os.close(source_descriptor)
            os.close(target_descriptor)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(f"Support file SHA-256 mismatch: {source}")
        manifest[name] = {
            "path": str(destination / name),
            "bytes": copied,
            "sha256": observed_sha256,
        }
    return manifest


def _command(
    *,
    python: Path,
    source: Path,
    outputs: dict[str, Path],
    manifest_output: Path,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
    expected_sha256: str,
    expected_content_digest: str,
) -> list[str]:
    home = Path.home()
    lab_hosts = _PROJECT_ROOT / "config" / "lab_hosts.env"
    values = {
        "PYTHON_EXE": python,
        "PYTHON_LIB": python.parent.parent / "lib",
        "WORKER": _WORKER,
        "DEEP_AUDIT": _DEEP_AUDIT,
        "OFFICIAL_SPLIT": _OFFICIAL_SPLIT,
        "RESTRICTED_NPY": _RESTRICTED_NPY,
        "SPLIT_ARTIFACTS": _SPLIT_ARTIFACTS,
        "INPUT": source,
        "TRAIN": outputs["train"],
        "VAL": outputs["val"],
        "TEST": outputs["test"],
        "MANIFEST": manifest_output,
        "SSH_DIR": home / ".ssh",
        "DATA_SSH_DIR": f"/System/Volumes/Data{home / '.ssh'}",
        "LAB_HOSTS": lab_hosts,
        "DATA_LAB_HOSTS": f"/System/Volumes/Data{lab_hosts}",
    }
    command = [str(_SANDBOX_EXEC)]
    for key, value in values.items():
        command.extend(("-D", f"{key}={value}"))
    command.extend(
        (
            "-p",
            _split_profile(),
            str(python),
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            str(_WORKER),
            "--incident-file",
            str(source),
            "--expected-sha256",
            expected_sha256,
            "--train-output",
            str(outputs["train"]),
            "--val-output",
            str(outputs["val"]),
            "--test-output",
            str(outputs["test"]),
            "--manifest-output",
            str(manifest_output),
            "--region",
            region,
            "--node-count",
            str(node_count),
            "--expected-sample-count",
            str(expected_sample_count),
            "--description-count",
            str(description_count),
            "--type-count",
            str(type_count),
            "--expected-content-digest",
            expected_content_digest,
        )
    )
    return command


def _postflight_command(
    *,
    python: Path,
    outputs: dict[str, Path],
    report_output: Path,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
    expected_content_digest: str,
    split_content_digests: dict[str, str],
) -> list[str]:
    home = Path.home()
    lab_hosts = _PROJECT_ROOT / "config" / "lab_hosts.env"
    values = {
        "PYTHON_EXE": python,
        "PYTHON_LIB": python.parent.parent / "lib",
        "WORKER": _POSTFLIGHT_WORKER,
        "DEEP_AUDIT": _DEEP_AUDIT,
        "RESTRICTED_NPY": _RESTRICTED_NPY,
        "TRAIN": outputs["train"],
        "VAL": outputs["val"],
        "TEST": outputs["test"],
        "OUTPUT": report_output,
        "SSH_DIR": home / ".ssh",
        "DATA_SSH_DIR": f"/System/Volumes/Data{home / '.ssh'}",
        "LAB_HOSTS": lab_hosts,
        "DATA_LAB_HOSTS": f"/System/Volumes/Data{lab_hosts}",
    }
    command = [str(_SANDBOX_EXEC)]
    for key, value in values.items():
        command.extend(("-D", f"{key}={value}"))
    command.extend(
        (
            "-p",
            _postflight_profile(),
            str(python),
            "-I",
            "-S",
            "-B",
            "-X",
            "utf8",
            str(_POSTFLIGHT_WORKER),
            "--train",
            str(outputs["train"]),
            "--val",
            str(outputs["val"]),
            "--test",
            str(outputs["test"]),
            "--output",
            str(report_output),
            "--region",
            region,
            "--node-count",
            str(node_count),
            "--expected-sample-count",
            str(expected_sample_count),
            "--description-count",
            str(description_count),
            "--type-count",
            str(type_count),
            "--expected-content-digest",
            expected_content_digest,
            "--expected-train-digest",
            split_content_digests["train"],
            "--expected-val-digest",
            split_content_digests["val"],
            "--expected-test-digest",
            split_content_digests["test"],
        )
    )
    return command


def _run_independent_postflight(
    *,
    python: Path,
    outputs: dict[str, Path],
    report_output: Path,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
    expected_content_digest: str,
    split_content_digests: dict[str, str],
    timeout_seconds: int,
    max_rss_bytes: int,
) -> tuple[dict[str, Any], int]:
    process = subprocess.Popen(
        _postflight_command(
            python=python,
            outputs=outputs,
            report_output=report_output,
            region=region,
            node_count=node_count,
            expected_sample_count=expected_sample_count,
            description_count=description_count,
            type_count=type_count,
            expected_content_digest=expected_content_digest,
            split_content_digests=split_content_digests,
        ),
        cwd="/",
        env=_empty_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    return_code, _, stderr, peak_rss = _bounded_child_output(
        process,
        timeout_seconds=timeout_seconds,
        max_rss_bytes=max_rss_bytes,
    )
    if return_code != 0:
        raise RuntimeError(
            "independent split postflight failed with exit code "
            f"{return_code}: {stderr[-4000:].strip()}"
        )
    report, _ = _read_bounded_json(
        report_output,
        max_bytes=_MAX_JSON_POSTFLIGHT_BYTES,
        label="Independent split postflight",
    )
    if (
        report.get("status") != "official_splits_independently_reverified"
        or report.get("recombined_content_digest") != expected_content_digest
        or report.get("records_revalidated") != expected_sample_count
    ):
        raise RuntimeError("Independent split postflight is invalid")
    return report, peak_rss


def run_sandboxed_official_split(
    *,
    incident_file: Path,
    expected_sha256: str,
    expected_content_digest: str,
    output_directory: Path,
    raw_release_root: Path,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
    timeout_seconds: int,
    support_files: Mapping[str, tuple[Path, str]],
    release_manifest_file: Path,
    expected_release_manifest_sha256: str,
    python_executable: Path | None = None,
    max_rss_bytes: int = _DEFAULT_MAX_RSS_BYTES,
) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("The reviewed macOS sandbox is unavailable")
    _verify_boundary_files()
    unresolved_raw_root = Path(raw_release_root)
    raw_root_stat = unresolved_raw_root.lstat()
    if (
        unresolved_raw_root.is_symlink()
        or not stat.S_ISDIR(raw_root_stat.st_mode)
        or raw_root_stat.st_uid != os.getuid()
    ):
        raise ValueError("Raw release root is not an owned directory")
    raw_root = unresolved_raw_root.resolve()
    source = Path(incident_file)
    _require_owned_regular_file(source, label="Incident input")
    source = source.resolve()
    if not source.is_relative_to(raw_root):
        raise ValueError("Incident input is outside the raw release root")
    destination = Path(output_directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Prepared split directory already exists: {destination}")
    unresolved_parent = destination.parent
    parent_stat = unresolved_parent.lstat()
    if (
        unresolved_parent.is_symlink()
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.getuid()
    ):
        raise ValueError("Prepared split parent is not an owned directory")
    parent = unresolved_parent.resolve()
    destination = parent / destination.name
    if destination.is_relative_to(raw_root):
        raise ValueError("Prepared output must stay outside the raw release")
    python = Path(python_executable or sys.executable).resolve()
    for path in (
        _WORKER,
        _POSTFLIGHT_WORKER,
        _PROBE_WORKER,
        _DEEP_AUDIT,
        _OFFICIAL_SPLIT,
        _RESTRICTED_NPY,
        _SPLIT_ARTIFACTS,
    ):
        _require_owned_regular_file(path, label="Split code")
    support_files, release_binding = _validate_release_binding(
        release_manifest_file=release_manifest_file,
        expected_release_manifest_sha256=expected_release_manifest_sha256,
        source=source,
        expected_sha256=expected_sha256,
        region=region,
        node_count=node_count,
        expected_sample_count=expected_sample_count,
        description_count=description_count,
        type_count=type_count,
        support_files=support_files,
    )
    source_identity = _file_identity(source)
    if _sha256_file(source) != expected_sha256:
        raise ValueError("Incident input SHA-256 mismatch")
    if _file_identity(source) != source_identity:
        raise RuntimeError("Incident input changed while it was being hashed")
    sandbox_probes = _verify_actual_split_profiles(
        work_directory=parent,
        timeout_seconds=min(timeout_seconds, 30),
        python=python,
    )

    staging = parent / f".{destination.name}.{uuid.uuid4().hex}.stage"
    staging.mkdir(mode=0o700)
    outputs = {
        split: staging / filename for split, filename in _FINAL_FILENAMES.items()
    }
    manifest_output = staging / "split_manifest.json"
    postflight_output = staging / ".postflight.json"
    for path in (*outputs.values(), manifest_output, postflight_output):
        _precreate(path)

    try:
        process = subprocess.Popen(
            _command(
                python=python,
                source=source,
                outputs=outputs,
                manifest_output=manifest_output,
                region=region,
                node_count=node_count,
                expected_sample_count=expected_sample_count,
                description_count=description_count,
                type_count=type_count,
                expected_sha256=expected_sha256,
                expected_content_digest=expected_content_digest,
            ),
            cwd="/",
            env=_empty_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        return_code, _, stderr, peak_rss = _bounded_child_output(
            process,
            timeout_seconds=timeout_seconds,
            max_rss_bytes=max_rss_bytes,
        )
        if return_code != 0:
            raise RuntimeError(
                "sandboxed official split failed with exit code "
                f"{return_code}: {stderr[-4000:].strip()}"
            )
        if _file_identity(source) != source_identity:
            raise RuntimeError("Incident input changed during split generation")
        manifest, _ = _read_bounded_json(
            manifest_output,
            max_bytes=_MAX_JSON_MANIFEST_BYTES,
            label="Split worker manifest",
        )
        if (
            manifest.get("status") != "official_splits_written_and_reloaded"
            or manifest.get("region") != region
            or manifest.get("deep_content_digest") != expected_content_digest
            or manifest.get("recombined_matches_input") is not True
        ):
            raise RuntimeError("Split worker returned an invalid manifest")

        expected_counts = {
            "train": int(expected_sample_count * 0.70),
            "val": int(expected_sample_count * 0.15),
        }
        expected_counts["test"] = (
            expected_sample_count - expected_counts["train"] - expected_counts["val"]
        )
        if manifest.get("counts") != expected_counts:
            raise RuntimeError("Split worker returned incorrect split counts")
        for split, path in outputs.items():
            header = read_npy_header(path)
            if header != {
                "shape": [expected_counts[split]],
                "fortran_order": False,
                "dtype": "object",
            }:
                raise RuntimeError(f"{split} NPY header is invalid")
            observed_hash = _sha256_file(path)
            if manifest["files"][split]["sha256"] != observed_hash:
                raise RuntimeError(f"{split} file hash changed after writing")
            manifest["files"][split]["path"] = str(
                destination / _FINAL_FILENAMES[split]
            )

        postflight, postflight_peak_rss = _run_independent_postflight(
            python=python,
            outputs=outputs,
            report_output=postflight_output,
            region=region,
            node_count=node_count,
            expected_sample_count=expected_sample_count,
            description_count=description_count,
            type_count=type_count,
            expected_content_digest=expected_content_digest,
            split_content_digests=manifest["split_content_digests"],
            timeout_seconds=timeout_seconds,
            max_rss_bytes=max_rss_bytes,
        )
        for key in (
            "duplicate_full_record_count",
            "duplicate_xy_window_count",
            "duplicate_model_consumed_record_count",
            "model_consumed_record_multiplicity",
            "model_consumed_digest",
            "cross_split_full_record_intersections",
            "cross_split_xy_window_intersections",
            "cross_split_model_consumed_record_intersections",
        ):
            if postflight[key] != manifest[key]:
                raise RuntimeError(f"Independent split postflight disagrees on {key}")
        postflight_output.unlink()
        manifest["independent_postflight"] = postflight
        manifest["support_files"] = _copy_support_files(
            support_files=support_files,
            staging=staging,
            destination=destination,
        )
        safe_structure = _audit_safe_region_structure(
            staging,
            spec={
                "nodes": node_count,
                "description_count": description_count,
                "type_count": type_count,
            },
        )
        manifest["status"] = "official_splits_prepared"
        manifest["schema_version"] = 2
        manifest["protocol_id"] = "igstgnn_table4_official_v1"
        manifest["record_structure_status"] = "records_structurally_verified"
        overlap_counts = (
            postflight["cross_split_full_record_intersections"],
            postflight["cross_split_xy_window_intersections"],
            postflight["cross_split_model_consumed_record_intersections"],
        )
        has_cross_split_overlap = any(
            int(count) > 0 for counts in overlap_counts for count in counts.values()
        )
        if has_cross_split_overlap:
            manifest["split_integrity_status"] = "cross_split_duplicates_detected"
            manifest["validity_label"] = "official_protocol_with_known_overlap"
        else:
            manifest["split_integrity_status"] = "no_cross_split_duplicates_detected"
            manifest["validity_label"] = "official_protocol_without_detected_overlap"
        manifest["support_structure_status"] = "safe_structure_verified"
        manifest["safe_structure"] = safe_structure
        manifest["release_manifest_binding"] = release_binding
        manifest["artifact_allowlist"] = sorted(_FINAL_BUNDLE_FILENAMES)
        manifest["source_incident_sha256"] = expected_sha256
        manifest["split_semantics"] = {
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "ordered": True,
            "shuffled": False,
            "renormalized": False,
            "deduplicated": False,
        }
        manifest["security_boundary"] = {
            "mode": "restricted_numpy_unpickler_plus_macos_sandbox",
            "negative_controls": sandbox_probes,
            "peak_rss_bytes": peak_rss,
            "postflight_peak_rss_bytes": postflight_peak_rss,
            "max_rss_bytes": max_rss_bytes,
            "sandbox_exec_sha256": _EXPECTED_SANDBOX_EXEC_SHA256,
            "dyld_support_profile_sha256": (_EXPECTED_DYLD_PROFILE_SHA256),
            "split_profile_sha256": hashlib.sha256(
                _split_profile().encode("utf-8")
            ).hexdigest(),
            "postflight_profile_sha256": hashlib.sha256(
                _postflight_profile().encode("utf-8")
            ).hexdigest(),
            "split_worker_sha256": _sha256_file(_WORKER),
            "postflight_worker_sha256": _sha256_file(_POSTFLIGHT_WORKER),
            "split_artifacts_sha256": _sha256_file(_SPLIT_ARTIFACTS),
        }
        _rewrite_owned_json_file(manifest_output, manifest)
        observed_artifacts = {path.name for path in staging.iterdir()}
        if observed_artifacts != _FINAL_BUNDLE_FILENAMES:
            raise RuntimeError(
                "Prepared split bundle differs from its artifact allowlist"
            )
        staging_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        _atomic_publish_directory_no_clobber(staging, destination)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
