from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path

try:
    import resource
except ModuleNotFoundError:
    resource = None

_SITE_PACKAGES = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
sys.path.insert(0, str(_SITE_PACKAGES))

import psutil  # noqa: E402


_MAX_CODE_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 1024 * 1024


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
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


def _stable_output_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _open_nofollow(path: Path, flags: int) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for deep-audit files")
    try:
        return os.open(
            path,
            flags | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise ValueError(
            f"Deep-audit file could not be opened safely: {path}"
        ) from error


def _load_exact_module(
    name: str,
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[types.ModuleType, str]:
    if not path.is_absolute() or not _valid_sha256(expected_sha256):
        raise ValueError(f"{name} module identity is invalid")
    descriptor = _open_nofollow(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_size > _MAX_CODE_BYTES
        ):
            raise ValueError(f"{name} module is not an approved code file")
        identity = _descriptor_identity(before)
        source = handle.read(_MAX_CODE_BYTES + 1)
        if len(source) > _MAX_CODE_BYTES:
            raise ValueError(f"{name} module exceeds its size limit")
        if _descriptor_identity(os.fstat(handle.fileno())) != identity:
            raise RuntimeError(f"{name} module changed while being read")
    observed_sha256 = hashlib.sha256(source).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{name} module SHA-256 mismatch: expected "
            f"{expected_sha256}, got {observed_sha256}"
        )
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, observed_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incident-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--node-count", required=True, type=int)
    parser.add_argument("--expected-sample-count", required=True, type=int)
    parser.add_argument("--description-count", required=True, type=int)
    parser.add_argument("--type-count", required=True, type=int)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-deep-audit-sha256", required=True)
    parser.add_argument("--expected-restricted-npy-sha256", required=True)
    return parser.parse_args()


def _apply_resource_limits() -> None:
    if resource is None:
        raise RuntimeError("POSIX resource limits are required for deep audit")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (16 * 1024 * 1024, 16 * 1024 * 1024),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if sys.platform == "darwin":
        current_vms = psutil.Process(os.getpid()).memory_info().vms
        address_space_limit = current_vms + 20 * 1024 * 1024 * 1024
        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        if hard_limit != resource.RLIM_INFINITY and hard_limit < address_space_limit:
            raise RuntimeError(
                "Existing RLIMIT_AS cannot provide the required VMS headroom"
            )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (address_space_limit, address_space_limit),
        )


def _require_empty_output(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_size != 0
    ):
        raise ValueError(
            "Worker output must be an empty, owned, mode-0600 regular file"
        )
    return _stable_output_identity(value)


def _load_verified_incident_file(
    path: Path,
    *,
    expected_sha256: str,
    restricted_npy: types.ModuleType,
) -> tuple[object, str]:
    if not path.is_absolute() or not _valid_sha256(expected_sha256):
        raise ValueError("Incident input identity is invalid")
    descriptor = _open_nofollow(path, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
        ):
            raise ValueError(
                "Incident input must be an owned, single-link regular file"
            )
        identity = _descriptor_identity(before)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        if _descriptor_identity(os.fstat(handle.fileno())) != identity:
            raise RuntimeError("Incident input changed while being hashed")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                "Incident input SHA-256 mismatch: expected "
                f"{expected_sha256}, got {observed_sha256}"
            )
        handle.seek(0)
        samples = restricted_npy.load_restricted_object_npy_handle(
            handle,
            source_size=before.st_size,
        )
        if _descriptor_identity(os.fstat(handle.fileno())) != identity:
            raise RuntimeError("Incident input changed while being loaded")
    return samples, observed_sha256


def _write_precreated_output(
    path: Path,
    payload: bytes,
    *,
    expected_identity: tuple[int, ...],
) -> None:
    if len(payload) > _MAX_JSON_BYTES:
        raise ValueError("Deep-audit JSON exceeds its size limit")
    descriptor = _open_nofollow(path, os.O_WRONLY)
    try:
        before = os.fstat(descriptor)
        if _stable_output_identity(before) != expected_identity or before.st_size != 0:
            raise RuntimeError("Worker output identity changed before writing")
        os.ftruncate(descriptor, 0)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Could not write deep-audit JSON")
            remaining = remaining[written:]
        os.fsync(descriptor)
        if _stable_output_identity(os.fstat(descriptor)) != expected_identity:
            raise RuntimeError("Worker output identity changed while writing")
    finally:
        os.close(descriptor)


def main() -> None:
    args = _parse_args()
    _apply_resource_limits()
    if not args.incident_file.is_absolute() or not args.output.is_absolute():
        raise ValueError("Worker input and output paths must be absolute")
    output_identity = _require_empty_output(args.output)
    module_dir = Path(__file__).resolve().parent
    deep_audit, deep_audit_sha256 = _load_exact_module(
        "deep_audit",
        module_dir / "deep_audit.py",
        expected_sha256=args.expected_deep_audit_sha256,
    )
    restricted_npy, restricted_npy_sha256 = _load_exact_module(
        "restricted_npy",
        module_dir / "restricted_npy.py",
        expected_sha256=args.expected_restricted_npy_sha256,
    )
    samples, input_sha256 = _load_verified_incident_file(
        args.incident_file,
        expected_sha256=args.expected_sha256,
        restricted_npy=restricted_npy,
    )
    audit = deep_audit.audit_loaded_samples(
        samples,
        region=args.region,
        node_count=args.node_count,
        expected_sample_count=args.expected_sample_count,
        description_count=args.description_count,
        type_count=args.type_count,
    )
    audit["worker_verification"] = {
        "input_sha256": input_sha256,
        "deep_audit_sha256": deep_audit_sha256,
        "restricted_npy_sha256": restricted_npy_sha256,
        "input_open_mode": "O_NOFOLLOW_single_fd_hash_rewind_load",
        "code_load_mode": "O_NOFOLLOW_hash_then_exec_exact_bytes",
    }
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
    _write_precreated_output(
        args.output,
        payload,
        expected_identity=output_identity,
    )


if __name__ == "__main__":
    main()
