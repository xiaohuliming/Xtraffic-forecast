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


NATIVE100_METHODS = ("bist", "ndr_net")
NATIVE100_REGIONS = ("Alameda", "Contra_Costa", "Orange")
NATIVE100_SEEDS = (42, 1, 2)
NATIVE100_PROFILE_PATH = "native100_compare_profile_v1.json"
NATIVE100_PROFILE_ID = "bist_ndr_native100_compare_v1"
NATIVE100_MATRIX_ID = "igstgnn_table4_bist_ndr_native100_18_v1"
NATIVE100_CLAIM_SCOPE = "same_table_native_recipe_100_epoch_supplement"
_MAX_JSON_BYTES = 256 * 1024


def _read_json(path: Path, *, purpose: str) -> tuple[dict[str, Any], str]:
    payload = read_verified_regular_file(
        path,
        purpose=purpose,
        max_bytes=_MAX_JSON_BYTES,
        require_nonempty=True,
    )
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{purpose} must contain an object")
    return value, hashlib.sha256(payload).hexdigest()


def load_native100_compare_profile(
    path: Path | str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="profile_sha256")
    profile, observed_sha256 = _read_json(
        Path(path),
        purpose="Table 4 native 100 epoch comparison profile",
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("Native 100 epoch profile SHA-256 mismatch")
    exact = {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "profile_id": NATIVE100_PROFILE_ID,
        "reported_columns": ["H3", "H6", "H12", "Average"],
        "seeds": list(NATIVE100_SEEDS),
    }
    for key, expected in exact.items():
        if profile.get(key) != expected:
            raise ValueError(f"Unexpected native 100 epoch profile field: {key}")
    methods = profile.get("methods")
    if not isinstance(methods, dict) or tuple(methods) != NATIVE100_METHODS:
        raise ValueError("Native 100 epoch method order is invalid")
    for method in NATIVE100_METHODS:
        entry = methods[method]
        if not isinstance(entry, dict):
            raise ValueError("Native 100 epoch method profile is invalid")
        _require_sha256(
            entry.get("base_config_sha256"),
            label=f"{method} base_config_sha256",
        )
        override = entry.get("override")
        effective = entry.get("effective")
        if (
            not isinstance(override, dict)
            or not isinstance(effective, dict)
            or override.get("max_epochs", {}).get("to") != 100
            or effective.get("max_epochs") != 100
        ):
            raise ValueError("Only the 100 epoch cap override is permitted")
    if methods["bist"]["effective"].get("patience") != 20:
        raise ValueError("BiST must retain its native patience 20")
    if methods["ndr_net"]["effective"].get("early_stopping") is not False:
        raise ValueError("NDR-Net must retain its fixed-epoch native recipe")
    return {**profile, "profile_sha256": observed_sha256}


def _task_id(number: int, *, method: str, region: str, seed: int) -> str:
    return f"n100{number:03d}_{method}_{region.lower()}_seed{seed}"


def build_native100_compare_manifest(
    *,
    base_manifest: dict[str, Any],
    base_manifest_sha256: str,
    profile: dict[str, Any],
    profile_sha256: str,
) -> dict[str, Any]:
    _require_sha256(base_manifest_sha256, label="base_manifest_sha256")
    _require_sha256(profile_sha256, label="profile_sha256")
    if base_manifest.get("matrix_id") != "igstgnn_table4_public_repro_39_v2":
        raise ValueError("Native 100 epoch comparison requires the public matrix")
    if base_manifest.get("protocol_id") != "igstgnn_table4_official_v1":
        raise ValueError("Native 100 epoch comparison requires Table 4 protocol")
    if profile.get("profile_id") != NATIVE100_PROFILE_ID:
        raise ValueError("Native 100 epoch profile identity is invalid")
    split_hashes = base_manifest.get("split_bundle_manifest_sha256")
    if not isinstance(split_hashes, dict) or tuple(split_hashes) != NATIVE100_REGIONS:
        raise ValueError("Native 100 epoch comparison requires all split bindings")
    for region, digest in split_hashes.items():
        _require_sha256(digest, label=f"{region} split bundle hash")

    tasks = []
    number = 0
    for method in NATIVE100_METHODS:
        method_profile = profile["methods"][method]
        for region in NATIVE100_REGIONS:
            for seed in NATIVE100_SEEDS:
                number += 1
                task_id = _task_id(
                    number,
                    method=method,
                    region=region,
                    seed=seed,
                )
                tasks.append(
                    {
                        "task_id": task_id,
                        "method": method,
                        "region": region,
                        "seed": seed,
                        "source_id": method_profile["source_id"],
                        "profile_id": NATIVE100_PROFILE_ID,
                        "config_sha256": profile_sha256,
                        "split_bundle_manifest_sha256": split_hashes[region],
                        "output_dir": f"tasks/{task_id}",
                    }
                )
    return {
        "schema_version": 1,
        "status": "planned_same_table_native_recipe_supplement",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": NATIVE100_MATRIX_ID,
        "base_matrix_id": "igstgnn_table4_public_repro_39_v2",
        "base_manifest_sha256": base_manifest_sha256,
        "validity_label": "official_protocol_with_known_overlap",
        "claim_scope": NATIVE100_CLAIM_SCOPE,
        "profile_path": NATIVE100_PROFILE_PATH,
        "profile_sha256": profile_sha256,
        "split_bundle_manifest_sha256": dict(split_hashes),
        "tasks": tasks,
    }


def load_native100_compare_manifest(
    path: Path | str,
    *,
    expected_sha256: str,
    base_manifest_path: Path | str,
    expected_base_manifest_sha256: str,
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="expected_sha256")
    source = Path(path)
    manifest, observed_sha256 = _read_json(
        source,
        purpose="Table 4 native 100 epoch comparison manifest",
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("Native 100 epoch manifest SHA-256 mismatch")
    if manifest.get("profile_path") != NATIVE100_PROFILE_PATH:
        raise ValueError("Unexpected native 100 epoch profile path")
    profile = load_native100_compare_profile(
        source.parent / NATIVE100_PROFILE_PATH,
        expected_sha256=manifest.get("profile_sha256"),
    )
    base_manifest = load_table4_manifest(
        base_manifest_path,
        expected_sha256=expected_base_manifest_sha256,
    )
    rebuilt = build_native100_compare_manifest(
        base_manifest=base_manifest,
        base_manifest_sha256=expected_base_manifest_sha256,
        profile=profile,
        profile_sha256=profile["profile_sha256"],
    )
    if manifest != rebuilt:
        raise ValueError("Native 100 epoch manifest is not canonical")
    return {
        **manifest,
        "manifest_sha256": observed_sha256,
        "profile": profile,
    }
