from __future__ import annotations

import argparse
import errno
import json
import os
import resource
import socket
import stat
from pathlib import Path
from typing import Any, Callable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--credential-file", required=True, type=Path)
    parser.add_argument("--neighbor-file", required=True, type=Path)
    return parser.parse_args()


def _apply_resource_limits() -> None:
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (1024 * 1024, 1024 * 1024),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _is_denied(action: Callable[[], Any]) -> bool:
    try:
        action()
    except OSError as exc:
        return exc.errno in {errno.EPERM, errno.EACCES}
    return False


def _read(path: Path) -> bytes:
    return path.read_bytes()


def _connect() -> None:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        connection.settimeout(0.5)
        connection.connect(("127.0.0.1", 9))
    finally:
        connection.close()


def _fork() -> None:
    child = os.fork()
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)


def _spawn() -> None:
    child = os.posix_spawn("/usr/bin/true", ["true"], {})
    os.waitpid(child, 0)


def _stable_output_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _write_precreated_output(
    path: Path,
    payload: bytes,
    *,
    expected_identity: tuple[int, ...],
) -> None:
    if len(payload) > 1024 * 1024:
        raise ValueError("Probe report exceeds its size limit")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for probe output")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if _stable_output_identity(before) != expected_identity or before.st_size != 0:
            raise RuntimeError("Probe output identity changed before writing")
        os.ftruncate(descriptor, 0)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Could not write sandbox probe report")
            remaining = remaining[written:]
        os.fsync(descriptor)
        if _stable_output_identity(os.fstat(descriptor)) != expected_identity:
            raise RuntimeError("Probe output identity changed while writing")
    finally:
        os.close(descriptor)


def main() -> None:
    args = _parse_args()
    _apply_resource_limits()
    output_stat = args.output.lstat()
    if (
        args.output.is_symlink()
        or not stat.S_ISREG(output_stat.st_mode)
        or output_stat.st_nlink != 1
        or output_stat.st_uid != os.getuid()
        or stat.S_IMODE(output_stat.st_mode) != 0o600
        or output_stat.st_size != 0
    ):
        raise ValueError("Probe output is not a secure staging file")
    output_identity = _stable_output_identity(output_stat)

    report = {
        "status": "sandbox_boundary_verified",
        "allowed_input_read": _read(args.input) == b"allowed-probe-input\n",
        "credential_read_denied": _is_denied(lambda: _read(args.credential_file)),
        "neighbor_read_denied": _is_denied(lambda: _read(args.neighbor_file)),
        "network_denied": _is_denied(_connect),
        "fork_denied": _is_denied(_fork),
        "spawn_denied": _is_denied(_spawn),
    }
    if not all(value is True for key, value in report.items() if key != "status"):
        raise RuntimeError(f"Sandbox negative control failed: {report!r}")
    payload = (
        json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    _write_precreated_output(
        args.output,
        payload,
        expected_identity=output_identity,
    )


if __name__ == "__main__":
    main()
