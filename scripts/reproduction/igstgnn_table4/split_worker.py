from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
import sys
import types
from pathlib import Path


_SITE_PACKAGES = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
sys.path.insert(0, str(_SITE_PACKAGES))

import psutil  # noqa: E402


def _load_exact_module(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    with path.open("rb") as handle:
        source = handle.read()
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


_MODULE_DIR = Path(__file__).resolve().parent
_DEEP_AUDIT = _load_exact_module("deep_audit", _MODULE_DIR / "deep_audit.py")
_OFFICIAL_SPLIT = _load_exact_module(
    "official_split",
    _MODULE_DIR / "official_split.py",
)
_RESTRICTED_NPY = _load_exact_module(
    "restricted_npy",
    _MODULE_DIR / "restricted_npy.py",
)
_SPLIT_ARTIFACTS = _load_exact_module(
    "split_artifacts",
    _MODULE_DIR / "split_artifacts.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incident-file", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--val-output", required=True, type=Path)
    parser.add_argument("--test-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--node-count", required=True, type=int)
    parser.add_argument("--expected-sample-count", required=True, type=int)
    parser.add_argument("--description-count", required=True, type=int)
    parser.add_argument("--type-count", required=True, type=int)
    parser.add_argument("--expected-content-digest", required=True)
    return parser.parse_args()


def _apply_resource_limits() -> None:
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024),
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


def _require_empty_output(path: Path) -> None:
    value = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_size != 0
    ):
        raise ValueError(f"Invalid precreated split output: {path}")


def _stable_output_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _require_precreated_manifest(path: Path) -> tuple[int, ...]:
    _require_empty_output(path)
    return _stable_output_identity(path.lstat())


def _write_precreated_manifest(
    path: Path,
    manifest: dict[str, object],
    *,
    expected_identity: tuple[int, ...] | None = None,
) -> None:
    identity = expected_identity or _require_precreated_manifest(path)
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > 1024 * 1024:
        raise ValueError("Split worker manifest exceeds its size limit")
    current = path.lstat()
    if _stable_output_identity(current) != identity or current.st_size != 0:
        raise RuntimeError("Split worker manifest identity changed")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for split outputs")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if _stable_output_identity(before) != identity or before.st_size != 0:
            raise RuntimeError("Split worker manifest identity changed")
        os.ftruncate(descriptor, 0)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Could not write split worker manifest")
            remaining = remaining[written:]
        os.fsync(descriptor)
        if _stable_output_identity(os.fstat(descriptor)) != identity:
            raise RuntimeError("Split worker manifest identity changed")
    finally:
        os.close(descriptor)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _load_verified_incident_file(
    path: Path,
    *,
    expected_sha256: str,
):
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected SHA-256 must be lowercase hexadecimal")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for incident inputs")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"Incident input could not be opened safely: {path}"
        ) from error

    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Incident input must be a regular file")
        identity = _file_identity(before)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        if _file_identity(os.fstat(handle.fileno())) != identity:
            raise RuntimeError("Incident input changed while being hashed")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Incident input SHA-256 mismatch")
        handle.seek(0)
        samples = _RESTRICTED_NPY.load_restricted_object_npy_handle(
            handle,
            source_size=before.st_size,
        )
        if _file_identity(os.fstat(handle.fileno())) != identity:
            raise RuntimeError("Incident input changed while being loaded")
        return samples


def main() -> None:
    args = _parse_args()
    _apply_resource_limits()
    split_outputs = {
        "train": args.train_output,
        "val": args.val_output,
        "test": args.test_output,
    }
    for path in (*split_outputs.values(), args.manifest_output):
        _require_empty_output(path)
    manifest_identity = _require_precreated_manifest(args.manifest_output)

    samples = _load_verified_incident_file(
        args.incident_file,
        expected_sha256=args.expected_sha256,
    )
    deep_audit = _DEEP_AUDIT.audit_loaded_samples(
        samples,
        region=args.region,
        node_count=args.node_count,
        expected_sample_count=args.expected_sample_count,
        description_count=args.description_count,
        type_count=args.type_count,
    )
    if deep_audit["content_digest"] != args.expected_content_digest:
        raise ValueError("Deep content digest changed before split generation")

    split_manifest = _SPLIT_ARTIFACTS.write_precreated_official_split_files(
        samples,
        outputs=split_outputs,
    )
    manifest = {
        **split_manifest,
        "region": args.region,
        "deep_content_digest": deep_audit["content_digest"],
        "duplicate_full_record_count": deep_audit["duplicate_full_record_count"],
        "duplicate_xy_window_count": deep_audit["duplicate_xy_window_count"],
        "duplicate_model_consumed_record_count": deep_audit[
            "duplicate_model_consumed_record_count"
        ],
        "model_consumed_record_multiplicity": deep_audit[
            "model_consumed_record_multiplicity"
        ],
        "model_consumed_digest": deep_audit["model_consumed_digest"],
        "cross_split_full_record_intersections": deep_audit[
            "cross_split_full_record_intersections"
        ],
        "cross_split_xy_window_intersections": deep_audit[
            "cross_split_xy_window_intersections"
        ],
        "cross_split_model_consumed_record_intersections": deep_audit[
            "cross_split_model_consumed_record_intersections"
        ],
    }
    _write_precreated_manifest(
        args.manifest_output,
        manifest,
        expected_identity=manifest_identity,
    )


if __name__ == "__main__":
    main()
