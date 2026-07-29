from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import (
    _reject_constant,
    _require_sha256,
    _strict_object,
    load_table4_manifest,
)
from .secure_io import read_verified_regular_file


EXTENSION_REGIONS = ("Alameda", "Contra_Costa", "Orange")
EXTENSION_SEEDS = (42, 1, 2)
EXTENSION_PROFILE_PATH = "ndr_extension_profile_v1.json"
EXTENSION_PROFILE_ID = "ndr_net_table4_extension_v1"
EXTENSION_SOURCE_ID = "xtraffic_ndr_s0g"
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_PROFILE_BYTES = 256 * 1024


def _region_slug(region: str) -> str:
    return region.lower()


def _task_id(task_number: int, region: str, seed: int) -> str:
    return f"ndr{task_number:03d}_{_region_slug(region)}_seed{seed}"


def load_ndr_extension_profile(
    path: Path | str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="profile_sha256")
    payload = read_verified_regular_file(
        Path(path),
        purpose="Table 4 NDR extension profile",
        max_bytes=_MAX_PROFILE_BYTES,
        require_nonempty=True,
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Table 4 NDR extension profile SHA-256 mismatch")
    profile = json.loads(
        bytes(payload).decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(profile, dict):
        raise ValueError("Table 4 NDR extension profile must contain an object")
    required = {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "profile_id": EXTENSION_PROFILE_ID,
        "method": "ndr_net",
        "variant": "s0g",
        "source_id": EXTENSION_SOURCE_ID,
    }
    for key, expected in required.items():
        if profile.get(key) != expected:
            raise ValueError(f"Unexpected NDR extension profile field: {key}")
    effective = profile.get("effective")
    if not isinstance(effective, dict):
        raise ValueError("NDR extension profile effective config must be an object")
    if effective.get("seeds") != list(EXTENSION_SEEDS):
        raise ValueError("NDR extension profile must freeze seeds 42, 1, and 2")
    return profile


def build_ndr_extension_manifest(
    *,
    base_manifest: dict[str, Any],
    base_manifest_sha256: str,
    profile_sha256: str,
) -> dict[str, Any]:
    _require_sha256(base_manifest_sha256, label="base_manifest_sha256")
    _require_sha256(profile_sha256, label="profile_sha256")
    if base_manifest.get("matrix_id") != "igstgnn_table4_public_repro_39_v2":
        raise ValueError("NDR extension requires the immutable public 39-task matrix")
    if base_manifest.get("protocol_id") != "igstgnn_table4_official_v1":
        raise ValueError("NDR extension requires the official Table 4 protocol")
    split_hashes = base_manifest.get("split_bundle_manifest_sha256")
    if not isinstance(split_hashes, dict) or tuple(split_hashes) != EXTENSION_REGIONS:
        raise ValueError("NDR extension requires all three ordered split bindings")
    for region, digest in split_hashes.items():
        _require_sha256(digest, label=f"{region} split bundle hash")

    tasks = []
    for task_number, (region, seed) in enumerate(
        (
            (region, seed)
            for region in EXTENSION_REGIONS
            for seed in EXTENSION_SEEDS
        ),
        start=1,
    ):
        task_id = _task_id(task_number, region, seed)
        tasks.append(
            {
                "task_id": task_id,
                "method": "ndr_net",
                "region": region,
                "seed": seed,
                "source_id": EXTENSION_SOURCE_ID,
                "profile_id": EXTENSION_PROFILE_ID,
                "config_sha256": profile_sha256,
                "split_bundle_manifest_sha256": split_hashes[region],
                "output_dir": f"tasks/{task_id}",
            }
        )
    return {
        "schema_version": 1,
        "status": "planned_project_method_extension",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_ndr_extension_9_v1",
        "base_matrix_id": "igstgnn_table4_public_repro_39_v2",
        "base_manifest_sha256": base_manifest_sha256,
        "validity_label": "official_protocol_with_known_overlap",
        "claim_scope": "same_protocol_project_method_extension",
        "profile_path": EXTENSION_PROFILE_PATH,
        "profile_sha256": profile_sha256,
        "split_bundle_manifest_sha256": dict(split_hashes),
        "tasks": tasks,
    }


def load_ndr_extension_manifest(
    path: Path | str,
    *,
    expected_sha256: str,
    base_manifest_path: Path | str,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="expected_sha256")
    source = Path(path)
    payload = read_verified_regular_file(
        source,
        purpose="Table 4 NDR extension manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
        require_nonempty=True,
    )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("Table 4 NDR extension manifest SHA-256 mismatch")
    manifest = json.loads(
        bytes(payload).decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(manifest, dict):
        raise ValueError("Table 4 NDR extension manifest must contain an object")
    if manifest.get("profile_path") != EXTENSION_PROFILE_PATH:
        raise ValueError("Unexpected NDR extension profile path")
    profile_sha256 = manifest.get("profile_sha256")
    load_ndr_extension_profile(
        source.parent / EXTENSION_PROFILE_PATH,
        expected_sha256=profile_sha256,
    )
    base_manifest = load_table4_manifest(
        base_manifest_path,
        expected_sha256=expected_base_manifest_sha256,
    )
    rebuilt = build_ndr_extension_manifest(
        base_manifest=base_manifest,
        base_manifest_sha256=expected_base_manifest_sha256,
        profile_sha256=profile_sha256,
    )
    if manifest != rebuilt:
        raise ValueError(
            "Table 4 NDR extension manifest does not match the canonical matrix"
        )
    return {**manifest, "manifest_sha256": observed_sha256}
