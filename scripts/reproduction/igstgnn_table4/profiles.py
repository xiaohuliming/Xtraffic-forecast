from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import METHODS


ALLOWED_AUTHORITIES = frozenset(
    {
        "official_table4",
        "upstream_default",
        "public_material_inference",
        "compatibility_patch",
    }
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_profile_set(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Profile set not found: {source}")
    raw_bytes = source.read_bytes()
    profile_set = json.loads(
        raw_bytes.decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if profile_set.get("schema_version") != 1:
        raise ValueError("Profile schema_version must be 1")
    if profile_set.get("protocol_id") != "igstgnn_table4_official_v1":
        raise ValueError("Profile protocol_id mismatch")
    if not isinstance(profile_set.get("shared"), dict):
        raise ValueError("Profile shared settings must be an object")
    profiles = profile_set.get("profiles")
    if not isinstance(profiles, dict) or tuple(profiles) != METHODS:
        raise ValueError("Profiles must contain the ordered Table 4 methods")

    profile_ids = set()
    for method, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"{method} profile must be an object")
        if not isinstance(profile.get("profile_id"), str):
            raise ValueError(f"{method} profile_id is missing")
        if profile["profile_id"] in profile_ids:
            raise ValueError("Profile IDs must be unique")
        profile_ids.add(profile["profile_id"])
        if not isinstance(profile.get("source_id"), str):
            raise ValueError(f"{method} source_id is missing")
        if not isinstance(profile.get("effective"), dict):
            raise ValueError(f"{method} effective settings are missing")
        if "seed" not in profile["effective"]:
            raise ValueError(f"{method} effective seed is missing")
        seed = profile["effective"]["seed"]
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError(f"{method} effective seed is invalid")
        if method == "bist":
            if (
                seed is not None
                or profile["effective"].get("seed_policy")
                != "upstream_runtime_random_draw"
            ):
                raise ValueError("bist effective seed policy is invalid")
        elif seed is None:
            raise ValueError(f"{method} effective seed must be fixed")
        loader_contract = profile["effective"].get("loader_contract")
        expected_partial_batch = "retain" if method == "igstgnn" else "drop"
        if loader_contract != {
            "training_shuffle": "global_numpy_permutation",
            "partial_batch": expected_partial_batch,
        }:
            raise ValueError(f"{method} loader_contract is invalid")
        authorities = profile.get("authority")
        if not isinstance(authorities, dict) or not authorities:
            raise ValueError(f"{method} authority map is missing")
        unknown = sorted(set(authorities.values()) - ALLOWED_AUTHORITIES)
        if unknown:
            raise ValueError(f"{method} has unknown authority labels: {unknown!r}")
        for field in authorities:
            if field not in profile["effective"]:
                raise ValueError(f"{method} authority field is not effective: {field}")
        if not isinstance(profile.get("public_material_gaps"), list):
            raise ValueError(f"{method} public_material_gaps must be a list")
        if not isinstance(profile.get("compatibility_patches"), list):
            raise ValueError(f"{method} compatibility_patches must be a list")

    config_hashes = {
        method: _canonical_sha256(
            {
                "protocol_id": profile_set["protocol_id"],
                "shared": profile_set["shared"],
                "profile": profile,
            }
        )
        for method, profile in profiles.items()
    }
    return {
        **profile_set,
        "profile_set_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "profile_set_canonical_sha256": _canonical_sha256(profile_set),
        "config_hashes": config_hashes,
    }
