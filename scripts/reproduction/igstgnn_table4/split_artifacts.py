from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .deep_audit import sequence_content_digest
    from .official_split import split_official_samples
    from .restricted_npy import load_restricted_object_npy
except ImportError:
    from deep_audit import sequence_content_digest
    from official_split import split_official_samples
    from restricted_npy import load_restricted_object_npy


_SPLIT_NAMES = ("train", "val", "test")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _require_precreated_output(path: Path) -> tuple[int, ...]:
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
            f"Split output must be an empty, owned, mode-0600 regular file: {path}"
        )
    return _stable_identity(value)


def _write_split_array(
    path: Path,
    split_array: np.ndarray,
    *,
    expected_identity: tuple[int, ...],
) -> None:
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if _stable_identity(os.fstat(descriptor)) != expected_identity:
            raise RuntimeError(f"Split output identity changed: {path}")
        os.ftruncate(descriptor, 0)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.save(handle, split_array, allow_pickle=True)
            handle.flush()
            os.fsync(handle.fileno())
        if _stable_identity(os.fstat(descriptor)) != expected_identity:
            raise RuntimeError(f"Split output identity changed: {path}")
    finally:
        os.close(descriptor)


def write_precreated_official_split_files(
    samples: np.ndarray,
    *,
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 1
        or samples.dtype != np.dtype("object")
    ):
        raise ValueError("Samples must be a one-dimensional object array")
    if set(outputs) != set(_SPLIT_NAMES):
        raise ValueError("Split outputs must contain train, val, and test")

    output_paths = {split: Path(outputs[split]) for split in _SPLIT_NAMES}
    identities = {
        split: _require_precreated_output(path) for split, path in output_paths.items()
    }
    splits = split_official_samples(samples)
    input_digest = sequence_content_digest(samples)

    for split in _SPLIT_NAMES:
        split_array = np.asarray(splits[split], dtype=object)
        _write_split_array(
            output_paths[split],
            split_array,
            expected_identity=identities[split],
        )

    reloaded = {
        split: load_restricted_object_npy(output_paths[split]) for split in _SPLIT_NAMES
    }
    counts = {split: len(reloaded[split]) for split in _SPLIT_NAMES}
    split_digests = {
        split: sequence_content_digest(reloaded[split]) for split in _SPLIT_NAMES
    }
    recombined_digest = sequence_content_digest(
        chain.from_iterable(reloaded[split] for split in _SPLIT_NAMES)
    )
    recombined_matches_input = recombined_digest == input_digest
    if not recombined_matches_input:
        raise RuntimeError("Reloaded split content does not match its input")

    return {
        "status": "official_splits_written_and_reloaded",
        "counts": counts,
        "input_content_digest": input_digest,
        "recombined_content_digest": recombined_digest,
        "recombined_matches_input": recombined_matches_input,
        "split_content_digests": split_digests,
        "files": {
            split: {
                "path": str(output_paths[split]),
                "bytes": output_paths[split].stat().st_size,
                "sha256": _sha256_file(output_paths[split]),
            }
            for split in _SPLIT_NAMES
        },
    }
