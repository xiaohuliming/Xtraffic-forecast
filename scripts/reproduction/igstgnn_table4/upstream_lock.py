from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
    secure_io_provenance,
    verify_secure_directory,
)


DEFAULT_LOCK_PATH = Path(__file__).with_name("upstreams.lock.json")
_MAX_GIT_OUTPUT_BYTES = 64 * 1024
_MAX_LOCK_BYTES = 1024 * 1024


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_upstream_lock(
    path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    source = Path(path)
    payload = read_verified_regular_file(
        source,
        purpose="Upstream lock",
        max_bytes=_MAX_LOCK_BYTES,
        require_nonempty=True,
    )
    lock = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    sources = lock.get("sources")
    if lock.get("schema_version") != 1 or not isinstance(sources, list) or not sources:
        raise ValueError("Invalid upstream lock schema")
    source_ids = [source_entry.get("source_id") for source_entry in sources]
    if any(not isinstance(source_id, str) for source_id in source_ids):
        raise ValueError("Every upstream source must have a source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Upstream source IDs must be unique")
    return lock


def upstream_lock_digest(lock: dict[str, Any]) -> str:
    payload = json.dumps(
        lock,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open_verified_regular_file(
        path,
        purpose="Locked upstream file",
        require_nonempty=True,
    ) as opened:
        for chunk in iter(lambda: opened.handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(checkout: Path, *args: str) -> str:
    environment = {
        "HOME": "/var/empty",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(checkout),
            *args,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
        env=environment,
    )
    if (
        len(result.stdout) > _MAX_GIT_OUTPUT_BYTES
        or len(result.stderr) > _MAX_GIT_OUTPUT_BYTES
    ):
        raise RuntimeError("Git verification output exceeded its limit")
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Git verification failed with exit code {result.returncode}: {error}"
        )
    return result.stdout.decode("utf-8", errors="strict").strip()


def verify_pristine_upstream_checkout(
    *,
    source_id: str,
    checkout: Path | str,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict[str, Any]:
    lock = load_upstream_lock(lock_path)
    source_by_id = {source["source_id"]: source for source in lock["sources"]}
    if source_id not in source_by_id:
        raise ValueError(f"Unknown upstream source_id: {source_id}")
    expected = source_by_id[source_id]

    resolved = verify_secure_directory(
        checkout,
        purpose="Upstream checkout",
    )
    verify_secure_directory(
        resolved / ".git",
        purpose="Upstream checkout .git",
    )
    top_level = Path(_run_git(resolved, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != resolved:
        raise ValueError("Upstream checkout is not the Git top-level")

    commit = _run_git(resolved, "rev-parse", "--verify", "HEAD")
    if commit != expected["commit"]:
        raise ValueError(
            f"Upstream HEAD mismatch for {source_id}: "
            f"expected {expected['commit']}, got {commit}"
        )
    tree = _run_git(resolved, "rev-parse", "--verify", "HEAD^{tree}")
    if tree != expected["tree"]:
        raise ValueError(
            f"Upstream tree mismatch for {source_id}: "
            f"expected {expected['tree']}, got {tree}"
        )
    status = _run_git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise ValueError(f"Upstream checkout is dirty: {source_id}")

    expected_license_sha256 = expected.get("license_sha256")
    license_path = resolved / "LICENSE"
    observed_license_sha256 = None
    if expected_license_sha256 is not None:
        try:
            observed_license_sha256 = _sha256_file(license_path)
        except (FileNotFoundError, OSError) as error:
            raise ValueError(f"Locked LICENSE is missing for {source_id}") from error
        if observed_license_sha256 != expected_license_sha256:
            raise ValueError(f"LICENSE hash mismatch for {source_id}")

    return {
        "status": "pristine_upstream_verified",
        "source_id": source_id,
        "checkout": str(resolved),
        "commit": commit,
        "tree": tree,
        "git_dirty": False,
        "license_spdx": expected.get("license_spdx"),
        "license_sha256": observed_license_sha256,
        "lock_digest": upstream_lock_digest(lock),
        "secure_io_boundary": secure_io_provenance(),
    }
