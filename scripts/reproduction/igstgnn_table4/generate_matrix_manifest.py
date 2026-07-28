from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from .manifest import METHODS, REGIONS, build_table4_manifest
from .profiles import load_profile_set
from .secure_io import read_verified_regular_file, verify_secure_directory
from .upstream_lock import load_upstream_lock, upstream_lock_digest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "profiles.public_material_v2.json"
)
DEFAULT_UPSTREAM_LOCK_PATH = Path(__file__).with_name("upstreams.lock.json")
_PROFILE_SET_ID = "igstgnn_table4_public_material_profiles_v2"
_PROTOCOL_ID = "igstgnn_table4_official_v1"
_MAX_INPUT_BYTES = 2 * 1024 * 1024
_EXPECTED_SOURCE_IDS = {
    **{method: "largest_b7c6ff5" for method in METHODS[:11]},
    "bist": "bist_dd94adf",
    "igstgnn": "igstgnn_64c6e24",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_strict_json_object(
    path: Path,
    *,
    purpose: str,
) -> tuple[dict[str, Any], str]:
    payload = read_verified_regular_file(
        path,
        purpose=purpose,
        max_bytes=_MAX_INPUT_BYTES,
        require_nonempty=True,
    )
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{purpose} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_profile_and_lock(
    *,
    profile_path: Path,
    upstream_lock_path: Path,
) -> tuple[dict[str, Any], str]:
    profile_set = load_profile_set(profile_path)
    if profile_set["profile_set_id"] != _PROFILE_SET_ID:
        raise ValueError("The matrix requires the current public profile v2")
    upstream_lock = load_upstream_lock(upstream_lock_path)
    if upstream_lock.get("protocol_id") != _PROTOCOL_ID:
        raise ValueError("Upstream lock protocol_id mismatch")
    locked_source_ids = {source.get("source_id") for source in upstream_lock["sources"]}
    for method, expected_source_id in _EXPECTED_SOURCE_IDS.items():
        if (
            profile_set["profiles"][method].get("source_id") != expected_source_id
            or expected_source_id not in locked_source_ids
        ):
            raise ValueError(f"{method} profile is not bound to its locked upstream")
    return profile_set, upstream_lock_digest(upstream_lock)


def _validate_release_manifest(
    path: Path,
) -> tuple[dict[str, Any], str]:
    release, release_sha256 = _read_strict_json_object(
        path,
        purpose="Official release manifest",
    )
    if (
        release.get("protocol_id") != _PROTOCOL_ID
        or release.get("status") != "safe_structure_verified"
    ):
        raise ValueError("Official release manifest status is invalid")
    _require_sha256(
        release.get("release_digest"),
        label="release_digest",
    )
    regions = release.get("regions")
    if not isinstance(regions, dict) or tuple(regions) != REGIONS:
        raise ValueError("Official release manifest must contain the ordered regions")
    return release, release_sha256


def _validate_split_manifests(
    *,
    split_root: Path,
    release: dict[str, Any],
    release_sha256: str,
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for region in REGIONS:
        path = split_root / region / "split_manifest.json"
        split, split_sha256 = _read_strict_json_object(
            path,
            purpose=f"{region} split manifest",
        )
        if (
            split.get("schema_version") != 2
            or split.get("protocol_id") != _PROTOCOL_ID
            or split.get("status") != "official_splits_prepared"
            or split.get("region") != region
            or split.get("validity_label") != "official_protocol_with_known_overlap"
            or split.get("record_structure_status") != "records_structurally_verified"
            or split.get("support_structure_status") != "safe_structure_verified"
            or split.get("independent_postflight", {}).get("status")
            != "official_splits_independently_reverified"
        ):
            raise ValueError(f"{region} split manifest status is invalid")
        release_binding = split.get("release_manifest_binding")
        if not isinstance(release_binding, dict) or (
            release_binding.get("region") != region
            or release_binding.get("sha256") != release_sha256
        ):
            raise ValueError(f"{region} split manifest release binding is invalid")
        release_region = release["regions"][region]
        expected_incident_sha256 = (
            release_region.get("files", {}).get("incident_all.npy", {}).get("sha256")
        )
        if (
            split.get("counts") != release_region.get("split_counts")
            or split.get("source_incident_sha256") != expected_incident_sha256
        ):
            raise ValueError(f"{region} split manifest does not match the release")
        _require_sha256(
            expected_incident_sha256,
            label=f"{region} source incident SHA-256",
        )
        bindings[region] = split_sha256
    return bindings


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive_fsynced(path: Path, payload: bytes) -> None:
    parent = verify_secure_directory(
        path.parent,
        purpose="Table 4 manifest output directory",
    )
    target = parent / path.name
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
    )
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if os.name != "nt":
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def create_table4_matrix_manifest(
    *,
    profile_path: Path | str,
    upstream_lock_path: Path | str,
    release_manifest_path: Path | str,
    split_root: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Table 4 matrix manifest already exists: {destination}")
    profile_set, source_lock_digest = _validate_profile_and_lock(
        profile_path=Path(profile_path),
        upstream_lock_path=Path(upstream_lock_path),
    )
    release, release_manifest_sha256 = _validate_release_manifest(
        Path(release_manifest_path)
    )
    split_bindings = _validate_split_manifests(
        split_root=Path(split_root),
        release=release,
        release_sha256=release_manifest_sha256,
    )

    bist_seed_by_region = {
        region: int(torch.randint(999999, (1,)).item()) for region in REGIONS
    }
    task_seeds_by_region = {
        region: {
            method: (
                bist_seed_by_region[region]
                if method == "bist"
                else profile_set["profiles"][method]["effective"]["seed"]
            )
            for method in METHODS
        }
        for region in REGIONS
    }
    manifest = build_table4_manifest(
        release_digest=release["release_digest"],
        source_lock_digest=source_lock_digest,
        profile_set_id=profile_set["profile_set_id"],
        profile_set_sha256=profile_set["profile_set_sha256"],
        config_hashes=profile_set["config_hashes"],
        profile_ids={
            method: profile_set["profiles"][method]["profile_id"] for method in METHODS
        },
        task_seeds_by_region=task_seeds_by_region,
        split_bundle_manifest_sha256=split_bindings,
    )
    payload = _json_bytes(manifest)
    _write_exclusive_fsynced(destination, payload)
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    return {
        "output_path": str(destination.resolve(strict=True)),
        "manifest_sha256": manifest_sha256,
        "bist_seed_by_region": bist_seed_by_region,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Create the immutable Table 4 v2 public-material matrix manifest")
    )
    parser.add_argument(
        "--profile-set",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
    )
    parser.add_argument(
        "--upstream-lock",
        type=Path,
        default=DEFAULT_UPSTREAM_LOCK_PATH,
    )
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = create_table4_matrix_manifest(
        profile_path=arguments.profile_set,
        upstream_lock_path=arguments.upstream_lock,
        release_manifest_path=arguments.release_manifest,
        split_root=arguments.split_root,
        output_path=arguments.output,
    )
    print(result["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
