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


PAIR_METHODS = ("bist_aligned", "bist_ndr")
PAIR_REGIONS = ("Alameda", "Contra_Costa", "Orange")
PAIR_SEEDS = (42, 1, 2)
PAIR_PROFILE_PATH = "bist_ndr_pair_profile_v1.json"
PAIR_PROFILE_ID = "bist_ndr_matched_pair_v1"
PAIR_MATRIX_ID = "igstgnn_table4_bist_ndr_pair_18_v1"
PAIR_CLAIM_SCOPE = "same_protocol_project_method_extension"
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


def load_bist_ndr_pair_profile(
    path: Path | str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="profile_sha256")
    profile, observed_sha256 = _read_json(
        Path(path),
        purpose="Table 4 BiST-NDR paired profile",
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("Table 4 BiST-NDR paired profile SHA-256 mismatch")
    exact = {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "profile_id": PAIR_PROFILE_ID,
    }
    for key, expected in exact.items():
        if profile.get(key) != expected:
            raise ValueError(f"Unexpected BiST-NDR paired profile field: {key}")
    methods = profile.get("methods")
    if not isinstance(methods, dict) or tuple(methods) != PAIR_METHODS:
        raise ValueError("BiST-NDR paired profile method order is invalid")
    for method in PAIR_METHODS:
        entry = methods[method]
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("source_id"), str)
            or not entry["source_id"]
        ):
            raise ValueError("BiST-NDR paired method source binding is invalid")
    effective = profile.get("effective")
    if not isinstance(effective, dict):
        raise ValueError("BiST-NDR paired effective profile is missing")
    if effective.get("seeds") != list(PAIR_SEEDS):
        raise ValueError("BiST-NDR paired profile must freeze seeds 42, 1, and 2")
    if effective.get("loader_contract") != {
        "training_shuffle": "global_numpy_permutation",
        "partial_batch": "retain",
    }:
        raise ValueError("BiST-NDR paired profile must retain every partial batch")
    evaluation = effective.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get(
        "test_observation_set"
    ) != "identical_retained_official_split_samples_for_both_methods":
        raise ValueError("BiST-NDR paired test observation contract is invalid")
    return {**profile, "profile_sha256": observed_sha256}


def _task_id(
    task_number: int,
    *,
    method: str,
    region: str,
    seed: int,
) -> str:
    return (
        f"pair{task_number:03d}_{method}_{region.lower()}_seed{seed}"
    )


def build_bist_ndr_pair_manifest(
    *,
    base_manifest: dict[str, Any],
    base_manifest_sha256: str,
    profile: dict[str, Any],
    profile_sha256: str,
) -> dict[str, Any]:
    _require_sha256(base_manifest_sha256, label="base_manifest_sha256")
    _require_sha256(profile_sha256, label="profile_sha256")
    if base_manifest.get("matrix_id") != "igstgnn_table4_public_repro_39_v2":
        raise ValueError("BiST-NDR pairing requires the immutable public matrix")
    if base_manifest.get("protocol_id") != "igstgnn_table4_official_v1":
        raise ValueError("BiST-NDR pairing requires the official Table 4 protocol")
    if profile.get("profile_id") != PAIR_PROFILE_ID:
        raise ValueError("BiST-NDR paired profile identity is invalid")
    split_hashes = base_manifest.get("split_bundle_manifest_sha256")
    if not isinstance(split_hashes, dict) or tuple(split_hashes) != PAIR_REGIONS:
        raise ValueError("BiST-NDR pairing requires all ordered split bindings")
    for region, digest in split_hashes.items():
        _require_sha256(digest, label=f"{region} split bundle hash")

    tasks = []
    task_number = 0
    for method in PAIR_METHODS:
        method_profile = profile["methods"][method]
        for region in PAIR_REGIONS:
            for seed in PAIR_SEEDS:
                task_number += 1
                task_id = _task_id(
                    task_number,
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
                        "profile_id": PAIR_PROFILE_ID,
                        "config_sha256": profile_sha256,
                        "split_bundle_manifest_sha256": split_hashes[region],
                        "output_dir": f"tasks/{task_id}",
                    }
                )
    return {
        "schema_version": 1,
        "status": "planned_matched_project_method_experiment",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": PAIR_MATRIX_ID,
        "base_matrix_id": "igstgnn_table4_public_repro_39_v2",
        "base_manifest_sha256": base_manifest_sha256,
        "validity_label": "official_protocol_with_known_overlap",
        "claim_scope": PAIR_CLAIM_SCOPE,
        "profile_path": PAIR_PROFILE_PATH,
        "profile_sha256": profile_sha256,
        "split_bundle_manifest_sha256": dict(split_hashes),
        "tasks": tasks,
    }


def load_bist_ndr_pair_manifest(
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
        purpose="Table 4 BiST-NDR paired manifest",
    )
    if observed_sha256 != expected_sha256:
        raise ValueError("Table 4 BiST-NDR paired manifest SHA-256 mismatch")
    if manifest.get("profile_path") != PAIR_PROFILE_PATH:
        raise ValueError("Unexpected BiST-NDR paired profile path")
    profile = load_bist_ndr_pair_profile(
        source.parent / PAIR_PROFILE_PATH,
        expected_sha256=manifest.get("profile_sha256"),
    )
    base_manifest = load_table4_manifest(
        base_manifest_path,
        expected_sha256=expected_base_manifest_sha256,
    )
    rebuilt = build_bist_ndr_pair_manifest(
        base_manifest=base_manifest,
        base_manifest_sha256=expected_base_manifest_sha256,
        profile=profile,
        profile_sha256=profile["profile_sha256"],
    )
    if manifest != rebuilt:
        raise ValueError(
            "Table 4 BiST-NDR manifest does not match the canonical matrix"
        )
    return {
        **manifest,
        "manifest_sha256": observed_sha256,
        "profile": profile,
    }
