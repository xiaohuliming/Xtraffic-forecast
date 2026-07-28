from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import selectors
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import psutil


_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[2]
_CONTROLLER = Path(__file__).resolve()
_WORKER = _MODULE_DIR / "deep_audit_worker.py"
_PROBE_WORKER = _MODULE_DIR / "sandbox_probe_worker.py"
_DEEP_AUDIT = _MODULE_DIR / "deep_audit.py"
_RESTRICTED_NPY = _MODULE_DIR / "restricted_npy.py"
_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_DYLD_PROFILE = Path("/System/Library/Sandbox/Profiles/dyld-support.sb")
_EXPECTED_SANDBOX_EXEC_SHA256 = (
    "8290e4be7387a0df83cd1559e86afd880464f269450573d012795761fe298f16"
)
_EXPECTED_DYLD_PROFILE_SHA256 = (
    "06215a5d32689aefe395c29710e182eb54ba22162f50df8b4842290f8a19bf1c"
)
_MAX_CHILD_OUTPUT_BYTES = 64 * 1024
_MAX_JSON_BYTES = 1024 * 1024
_DEFAULT_MAX_RSS_BYTES = 24 * 1024 * 1024 * 1024
_AT_FDCWD = -2
_RENAME_EXCL = 0x00000004


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sandbox_profile() -> str:
    return """\
(version 1)
(deny default)
(import "dyld-support.sb")

(define python-exe (param "PYTHON_EXE"))
(define python-lib (param "PYTHON_LIB"))
(define worker-file (param "WORKER"))
(define deep-audit-file (param "DEEP_AUDIT"))
(define restricted-npy-file (param "RESTRICTED_NPY"))
(define input-file (param "INPUT"))
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
  (literal input-file))
(allow file-read-metadata file-test-existence
  (path-ancestors worker-file)
  (path-ancestors deep-audit-file)
  (path-ancestors restricted-npy-file)
  (path-ancestors input-file)
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


def _sandbox_parameters(
    *,
    python: Path,
    worker: Path,
    input_file: Path,
    output_file: Path,
) -> dict[str, str]:
    python_lib = python.parent.parent / "lib"
    home = Path.home()
    lab_hosts = _PROJECT_ROOT / "config" / "lab_hosts.env"
    return {
        "PYTHON_EXE": str(python),
        "PYTHON_LIB": str(python_lib),
        "WORKER": str(worker),
        "DEEP_AUDIT": str(_DEEP_AUDIT),
        "RESTRICTED_NPY": str(_RESTRICTED_NPY),
        "INPUT": str(input_file),
        "OUTPUT": str(output_file),
        "SSH_DIR": str(home / ".ssh"),
        "DATA_SSH_DIR": f"/System/Volumes/Data{home / '.ssh'}",
        "LAB_HOSTS": str(lab_hosts),
        "DATA_LAB_HOSTS": f"/System/Volumes/Data{lab_hosts}",
    }


def _sandbox_prefix(parameters: dict[str, str]) -> list[str]:
    definitions: list[str] = []
    for key, value in sorted(parameters.items()):
        definitions.extend(["-D", f"{key}={value}"])
    return [
        str(_SANDBOX_EXEC),
        *definitions,
        "-p",
        _sandbox_profile(),
    ]


def _rendered_profile_sha256(parameters: dict[str, str]) -> str:
    rendered_identity = json.dumps(
        {
            "controller_profile": _sandbox_profile(),
            "parameters": parameters,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered_identity).hexdigest()


def _file_identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
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


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _stable_file_identity(path: Path) -> tuple[int, ...]:
    return _stable_stat_identity(path.lstat())


def _unlink_if_identity(
    path: Path,
    *,
    expected_identity: tuple[int, ...],
) -> None:
    try:
        current_identity = _stable_file_identity(path)
    except FileNotFoundError:
        return
    if current_identity == expected_identity:
        path.unlink()


def _require_owned_regular_file(path: Path, *, label: str) -> None:
    value = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_uid != os.getuid()
    ):
        raise ValueError(f"{label} must be an owned, single-link regular file: {path}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_bounded_json(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    value = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_uid != os.getuid()
        or value.st_size <= 0
        or value.st_size > max_bytes
    ):
        if value.st_size > max_bytes:
            raise RuntimeError(f"{label} exceeds {max_bytes} bytes")
        raise RuntimeError(f"{label} is not a valid bounded regular file")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for audit JSON")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        identity = _file_identity(path)
        if (
            before.st_dev != identity[0]
            or before.st_ino != identity[1]
            or before.st_size != identity[6]
        ):
            raise RuntimeError(f"{label} identity changed before reading")
        payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise RuntimeError(f"{label} exceeds {max_bytes} bytes")
        after = os.fstat(handle.fileno())
        if (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != identity:
            raise RuntimeError(f"{label} changed while being read")
    try:
        decoded = payload.decode("utf-8")
        result = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"{label} is not strict JSON") from error
    if type(result) is not dict:
        raise RuntimeError(f"{label} must contain a JSON object")
    return result


def _validated_paths(
    incident_file: Path,
    output: Path,
    python_executable: Path,
) -> tuple[Path, Path, Path]:
    source = Path(incident_file)
    destination = Path(output)
    python = Path(python_executable).resolve()
    _require_owned_regular_file(source, label="Incident input")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Deep audit output already exists: {destination}")
    parent_stat = destination.parent.lstat()
    if (
        destination.parent.is_symlink()
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.getuid()
    ):
        raise ValueError(
            "Output parent must be an owned, non-symlink directory: "
            f"{destination.parent}"
        )
    if not python.is_file() or python.is_symlink():
        raise FileNotFoundError(f"Resolved Python executable not found: {python}")
    for code_file in (
        _CONTROLLER,
        _WORKER,
        _PROBE_WORKER,
        _DEEP_AUDIT,
        _RESTRICTED_NPY,
    ):
        _require_owned_regular_file(code_file, label="Audit code")
    return source.resolve(), destination.resolve(), python


def _verify_boundary_files() -> None:
    if not _SANDBOX_EXEC.is_file() or not _DYLD_PROFILE.is_file():
        raise RuntimeError("The reviewed macOS sandbox boundary is unavailable")
    sandbox_hash = _sha256_file(_SANDBOX_EXEC)
    dyld_hash = _sha256_file(_DYLD_PROFILE)
    if sandbox_hash != _EXPECTED_SANDBOX_EXEC_SHA256:
        raise RuntimeError(
            "sandbox-exec changed since the security review; fail closed"
        )
    if dyld_hash != _EXPECTED_DYLD_PROFILE_SHA256:
        raise RuntimeError(
            "dyld-support.sb changed since the security review; fail closed"
        )


def _create_staging_file(destination: Path) -> Path:
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.stage")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(staging, flags, 0o600)
    try:
        created_identity = _stable_stat_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    value = staging.lstat()
    if (
        _stable_file_identity(staging) != created_identity
        or not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_size != 0
    ):
        _unlink_if_identity(
            staging,
            expected_identity=created_identity,
        )
        raise RuntimeError("Could not create a secure deep-audit staging file")
    return staging


def _kill_and_wait(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _bounded_child_output(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: int,
    max_rss_bytes: int,
) -> tuple[int, str, str, int]:
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Child output pipes were not created")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    peak_rss = 0
    child = psutil.Process(process.pid)

    try:
        while selector.get_map():
            if time.monotonic() >= deadline:
                _kill_and_wait(process)
                raise RuntimeError(
                    f"sandboxed deep audit exceeded {timeout_seconds} seconds"
                )
            try:
                current_rss = child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                current_rss = 0
            peak_rss = max(peak_rss, current_rss)
            if current_rss > max_rss_bytes:
                _kill_and_wait(process)
                raise RuntimeError("sandboxed deep audit exceeded its RSS limit")

            events = selector.select(timeout=0.2)
            for key, _ in events:
                stream = key.fileobj
                chunk = os.read(stream.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                buffer = buffers[key.data]
                if len(buffer) + len(chunk) > _MAX_CHILD_OUTPUT_BYTES:
                    _kill_and_wait(process)
                    raise RuntimeError("sandboxed deep audit exceeded its output limit")
                buffer.extend(chunk)
    finally:
        selector.close()

    return_code = process.wait(timeout=5)
    return (
        return_code,
        buffers["stdout"].decode("utf-8", errors="replace"),
        buffers["stderr"].decode("utf-8", errors="replace"),
        peak_rss,
    )


def _sandbox_command(
    *,
    python: Path,
    source: Path,
    staging: Path,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
    expected_sha256: str,
    expected_deep_audit_sha256: str,
    expected_restricted_npy_sha256: str,
) -> list[str]:
    parameters = _sandbox_parameters(
        python=python,
        worker=_WORKER,
        input_file=source,
        output_file=staging,
    )
    return [
        *_sandbox_prefix(parameters),
        str(python),
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        str(_WORKER),
        "--incident-file",
        str(source),
        "--output",
        str(staging),
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
        "--expected-sha256",
        expected_sha256,
        "--expected-deep-audit-sha256",
        expected_deep_audit_sha256,
        "--expected-restricted-npy-sha256",
        expected_restricted_npy_sha256,
    ]


def _sandbox_probe_command(
    *,
    python: Path,
    input_file: Path,
    output_file: Path,
    credential_file: Path,
    neighbor_file: Path,
) -> list[str]:
    parameters = _sandbox_parameters(
        python=python,
        worker=_PROBE_WORKER,
        input_file=input_file,
        output_file=output_file,
    )
    return [
        *_sandbox_prefix(parameters),
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
    ]


def _empty_environment() -> dict[str, str]:
    return {
        "HOME": "/var/empty",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONUTF8": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TMPDIR": "/var/empty",
    }


def verify_macos_sandbox_boundary(
    *,
    work_directory: Path,
    timeout_seconds: int,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("The reviewed macOS sandbox is unavailable")
    _verify_boundary_files()
    unresolved_work_dir = Path(work_directory)
    work_stat = unresolved_work_dir.lstat()
    if (
        unresolved_work_dir.is_symlink()
        or not stat.S_ISDIR(work_stat.st_mode)
        or work_stat.st_uid != os.getuid()
    ):
        raise ValueError(
            "Sandbox probe directory is not an owned, non-symlink directory"
        )
    work_dir = unresolved_work_dir.resolve()
    python = Path(python_executable or sys.executable).resolve()
    if not python.is_file() or python.is_symlink():
        raise FileNotFoundError(f"Python executable not found: {python}")
    _require_owned_regular_file(_PROBE_WORKER, label="Sandbox probe code")

    credential_file = _PROJECT_ROOT / "config" / "lab_hosts.env"
    neighbor_file = _PROJECT_ROOT / "CLAUDE.md"
    if not credential_file.is_file() or not neighbor_file.is_file():
        raise RuntimeError("Sandbox negative-control source files are unavailable")

    probe_input = work_dir / f".sandbox-input.{uuid.uuid4().hex}"
    probe_output = work_dir / f".sandbox-output.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    input_descriptor = os.open(probe_input, flags, 0o600)
    try:
        os.write(input_descriptor, b"allowed-probe-input\n")
        os.fsync(input_descriptor)
    finally:
        os.close(input_descriptor)
    output_descriptor = os.open(probe_output, flags, 0o600)
    os.close(output_descriptor)
    probe_input_identity = _stable_file_identity(probe_input)
    probe_output_identity = _stable_file_identity(probe_output)

    try:
        command = _sandbox_probe_command(
            python=python,
            input_file=probe_input,
            output_file=probe_output,
            credential_file=credential_file,
            neighbor_file=neighbor_file,
        )
        process = subprocess.Popen(
            command,
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
                "macOS sandbox negative controls failed with exit code "
                f"{return_code}: {stderr[-4000:].strip()}"
            )
        report = _read_bounded_json(
            probe_output,
            max_bytes=64 * 1024,
            label="macOS sandbox probe report",
        )
        expected_report = {
            "status": "sandbox_boundary_verified",
            "allowed_input_read": True,
            "credential_read_denied": True,
            "neighbor_read_denied": True,
            "network_denied": True,
            "fork_denied": True,
            "spawn_denied": True,
        }
        if report != expected_report:
            raise RuntimeError(
                "macOS sandbox negative controls returned an invalid report"
            )
        worker_report_sha256 = hashlib.sha256(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        probe_parameters = _sandbox_parameters(
            python=python,
            worker=_PROBE_WORKER,
            input_file=probe_input,
            output_file=probe_output,
        )
        report["assertion_mode"] = "exact_key_and_value_match"
        report["worker_report_sha256"] = worker_report_sha256
        report["peak_rss_bytes"] = peak_rss
        report["sandbox_exec_sha256"] = _EXPECTED_SANDBOX_EXEC_SHA256
        report["dyld_support_profile_sha256"] = _EXPECTED_DYLD_PROFILE_SHA256
        report["probe_worker_sha256"] = _sha256_file(_PROBE_WORKER)
        report["controller_sha256"] = _sha256_file(_CONTROLLER)
        report["controller_profile_sha256"] = hashlib.sha256(
            _sandbox_profile().encode("utf-8")
        ).hexdigest()
        report["rendered_profile_sha256"] = _rendered_profile_sha256(probe_parameters)
        return report
    finally:
        _unlink_if_identity(
            probe_input,
            expected_identity=probe_input_identity,
        )
        _unlink_if_identity(
            probe_output,
            expected_identity=probe_output_identity,
        )


def _write_final_manifest(
    staging: Path,
    destination: Path,
    audit: dict[str, Any],
    *,
    expected_staging_identity: tuple[int, ...],
) -> None:
    payload = (
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > _MAX_JSON_BYTES:
        raise RuntimeError(f"Final deep-audit manifest exceeds {_MAX_JSON_BYTES} bytes")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for audit publication")
    descriptor = os.open(
        staging,
        os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        current_identity = _stable_stat_identity(before)
        if current_identity != expected_staging_identity:
            raise RuntimeError("Audit staging identity changed before final rewrite")
        os.ftruncate(descriptor, 0)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Could not write final deep-audit manifest")
            remaining = remaining[written:]
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if _stable_stat_identity(after) != expected_staging_identity:
            raise RuntimeError("Audit staging identity changed while rewriting")
    finally:
        os.close(descriptor)

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameatx_np = libc.renameatx_np
    except AttributeError as error:
        raise RuntimeError(
            "renameatx_np is unavailable; refusing a non-atomic publish"
        ) from error
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
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                f"Deep audit output already exists: {destination}",
                str(destination),
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def run_sandboxed_deep_audit(
    *,
    incident_file: Path,
    expected_sha256: str,
    output: Path,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
    timeout_seconds: int,
    python_executable: Path | None = None,
    max_rss_bytes: int = _DEFAULT_MAX_RSS_BYTES,
) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("No approved object-NPY sandbox is available on this host")
    _verify_boundary_files()
    source, destination, python = _validated_paths(
        incident_file,
        output,
        python_executable or Path(sys.executable),
    )
    sandbox_probe = verify_macos_sandbox_boundary(
        work_directory=destination.parent,
        timeout_seconds=min(timeout_seconds, 30),
        python_executable=python,
    )
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected_sha256 must be lowercase hexadecimal")

    source_identity = _file_identity(source)
    observed_sha256 = _sha256_file(source)
    if _file_identity(source) != source_identity:
        raise RuntimeError("Incident input changed while it was being hashed")
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Incident input SHA-256 mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )

    code_identities = {
        path: _file_identity(path)
        for path in (_CONTROLLER, _WORKER, _DEEP_AUDIT, _RESTRICTED_NPY)
    }
    code_sha256 = {
        path: _sha256_file(path)
        for path in (_CONTROLLER, _WORKER, _DEEP_AUDIT, _RESTRICTED_NPY)
    }
    staging = _create_staging_file(destination)
    staging_identity = _stable_file_identity(staging)
    try:
        command = _sandbox_command(
            python=python,
            source=source,
            staging=staging,
            region=region,
            node_count=node_count,
            expected_sample_count=expected_sample_count,
            description_count=description_count,
            type_count=type_count,
            expected_sha256=expected_sha256,
            expected_deep_audit_sha256=code_sha256[_DEEP_AUDIT],
            expected_restricted_npy_sha256=code_sha256[_RESTRICTED_NPY],
        )
        process = subprocess.Popen(
            command,
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
                "sandboxed deep audit failed with exit code "
                f"{return_code}: {stderr[-4000:].strip()}"
            )
        if _file_identity(source) != source_identity:
            raise RuntimeError("Incident input changed during the sandboxed audit")
        if any(
            _file_identity(path) != identity
            for path, identity in code_identities.items()
        ):
            raise RuntimeError("Deep-audit code changed during execution")
        _require_owned_regular_file(staging, label="Audit staging output")
        if stat.S_IMODE(staging.stat().st_mode) != 0o600:
            raise RuntimeError("Audit staging output mode changed")
        audit = _read_bounded_json(
            staging,
            max_bytes=_MAX_JSON_BYTES,
            label="sandboxed deep-audit manifest",
        )
        if (
            audit.get("status") != "records_structurally_verified"
            or audit.get("region") != region
            or audit.get("records_validated") != expected_sample_count
        ):
            raise RuntimeError("sandboxed deep audit returned an invalid manifest")
        expected_worker_verification = {
            "input_sha256": expected_sha256,
            "deep_audit_sha256": code_sha256[_DEEP_AUDIT],
            "restricted_npy_sha256": code_sha256[_RESTRICTED_NPY],
            "input_open_mode": "O_NOFOLLOW_single_fd_hash_rewind_load",
            "code_load_mode": "O_NOFOLLOW_hash_then_exec_exact_bytes",
        }
        if audit.get("worker_verification") != expected_worker_verification:
            raise RuntimeError("sandboxed deep audit worker verification is invalid")
        audit["security_boundary"] = {
            "mode": "restricted_numpy_unpickler_plus_macos_sandbox",
            "input_sha256": observed_sha256,
            "network_access": False,
            "credential_paths_accessible": False,
            "process_fork_allowed": False,
            "sandbox_exec_sha256": _EXPECTED_SANDBOX_EXEC_SHA256,
            "dyld_support_profile_sha256": _EXPECTED_DYLD_PROFILE_SHA256,
            "python_executable": str(python),
            "python_sha256": _sha256_file(python),
            "controller_sha256": code_sha256[_CONTROLLER],
            "controller_profile_sha256": hashlib.sha256(
                _sandbox_profile().encode("utf-8")
            ).hexdigest(),
            "rendered_profile_sha256": _rendered_profile_sha256(
                _sandbox_parameters(
                    python=python,
                    worker=_WORKER,
                    input_file=source,
                    output_file=staging,
                )
            ),
            "worker_sha256": code_sha256[_WORKER],
            "deep_audit_sha256": code_sha256[_DEEP_AUDIT],
            "restricted_npy_sha256": code_sha256[_RESTRICTED_NPY],
            "peak_rss_bytes": peak_rss,
            "max_rss_bytes": max_rss_bytes,
            "timeout_seconds": timeout_seconds,
            "negative_controls": sandbox_probe,
        }
        _write_final_manifest(
            staging,
            destination,
            audit,
            expected_staging_identity=staging_identity,
        )
        return audit
    except BaseException:
        _unlink_if_identity(
            staging,
            expected_identity=staging_identity,
        )
        _unlink_if_identity(
            destination,
            expected_identity=staging_identity,
        )
        raise
