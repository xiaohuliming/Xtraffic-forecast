from __future__ import annotations

import argparse
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
_RESTRICTED_NPY = _load_exact_module(
    "restricted_npy",
    _MODULE_DIR / "restricted_npy.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--node-count", required=True, type=int)
    parser.add_argument("--expected-sample-count", required=True, type=int)
    parser.add_argument("--description-count", required=True, type=int)
    parser.add_argument("--type-count", required=True, type=int)
    parser.add_argument("--expected-content-digest", required=True)
    parser.add_argument("--expected-train-digest", required=True)
    parser.add_argument("--expected-val-digest", required=True)
    parser.add_argument("--expected-test-digest", required=True)
    return parser.parse_args()


def _apply_resource_limits() -> None:
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (7200, 7200))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (16 * 1024 * 1024, 16 * 1024 * 1024),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _require_precreated_report(path: Path) -> tuple[int, ...]:
    output_stat = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(output_stat.st_mode)
        or output_stat.st_nlink != 1
        or output_stat.st_uid != os.getuid()
        or stat.S_IMODE(output_stat.st_mode) != 0o600
        or output_stat.st_size != 0
    ):
        raise ValueError("Invalid postflight output staging file")
    return _stable_identity(output_stat)


def _write_precreated_report(
    path: Path,
    report: dict[str, object],
    *,
    expected_identity: tuple[int, ...] | None = None,
) -> None:
    identity = expected_identity or _require_precreated_report(path)
    current = path.lstat()
    if _stable_identity(current) != identity or current.st_size != 0:
        raise RuntimeError("Postflight output identity changed")
    payload = (
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if _stable_identity(os.fstat(descriptor)) != identity:
            raise RuntimeError("Postflight output identity changed")
        os.ftruncate(descriptor, 0)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        if _stable_identity(os.fstat(descriptor)) != identity:
            raise RuntimeError("Postflight output identity changed")
    finally:
        os.close(descriptor)


def main() -> None:
    args = _parse_args()
    _apply_resource_limits()
    output_identity = _require_precreated_report(args.output)

    split_paths = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }
    splits = {
        split: _RESTRICTED_NPY.load_restricted_object_npy(path)
        for split, path in split_paths.items()
    }
    counts = {split: len(value) for split, value in splits.items()}
    expected_counts = {
        "train": int(args.expected_sample_count * 0.70),
        "val": int(args.expected_sample_count * 0.15),
    }
    expected_counts["test"] = (
        args.expected_sample_count - expected_counts["train"] - expected_counts["val"]
    )
    if counts != expected_counts:
        raise ValueError("Postflight split counts do not match")

    split_digests = {
        split: _DEEP_AUDIT.sequence_content_digest(value)
        for split, value in splits.items()
    }
    expected_digests = {
        "train": args.expected_train_digest,
        "val": args.expected_val_digest,
        "test": args.expected_test_digest,
    }
    if split_digests != expected_digests:
        raise ValueError("Postflight split content digest mismatch")

    recombined = __import__("numpy").concatenate(
        [splits["train"], splits["val"], splits["test"]]
    )
    audit = _DEEP_AUDIT.audit_loaded_samples(
        recombined,
        region=args.region,
        node_count=args.node_count,
        expected_sample_count=args.expected_sample_count,
        description_count=args.description_count,
        type_count=args.type_count,
    )
    if audit["content_digest"] != args.expected_content_digest:
        raise ValueError("Postflight recombined content digest mismatch")

    report = {
        "status": "official_splits_independently_reverified",
        "records_revalidated": len(recombined),
        "counts": counts,
        "split_content_digests": split_digests,
        "recombined_content_digest": audit["content_digest"],
        "duplicate_full_record_count": audit["duplicate_full_record_count"],
        "duplicate_xy_window_count": audit["duplicate_xy_window_count"],
        "duplicate_model_consumed_record_count": audit[
            "duplicate_model_consumed_record_count"
        ],
        "model_consumed_record_multiplicity": audit[
            "model_consumed_record_multiplicity"
        ],
        "model_consumed_digest": audit["model_consumed_digest"],
        "cross_split_full_record_intersections": audit[
            "cross_split_full_record_intersections"
        ],
        "cross_split_xy_window_intersections": audit[
            "cross_split_xy_window_intersections"
        ],
        "cross_split_model_consumed_record_intersections": audit[
            "cross_split_model_consumed_record_intersections"
        ],
    }
    _write_precreated_report(
        args.output,
        report,
        expected_identity=output_identity,
    )


if __name__ == "__main__":
    main()
