from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .secure_io import read_verified_regular_file


METHODS = (
    "hl",
    "lstm",
    "dcrnn",
    "agcrn",
    "stgcn",
    "gwnet",
    "astgcn",
    "sttn",
    "dstagnn",
    "dgcrn",
    "d2stgnn",
    "bist",
    "igstgnn",
)
REGIONS = ("Alameda", "Contra_Costa", "Orange")
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024


def _region_slug(region: str) -> str:
    return region.lower()


def _source_id(method: str) -> str:
    if method == "igstgnn":
        return "igstgnn_64c6e24"
    if method == "bist":
        return "bist_dd94adf"
    return "largest_b7c6ff5"


def _require_sha256(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def build_table4_manifest(
    *,
    release_digest: str,
    source_lock_digest: str,
    profile_set_id: str,
    profile_set_sha256: str,
    config_hashes: Mapping[str, str],
    profile_ids: Mapping[str, str],
    task_seeds_by_region: Mapping[
        str,
        Mapping[str, int | None],
    ],
    split_bundle_manifest_sha256: Mapping[str, str],
) -> dict[str, Any]:
    _require_sha256(release_digest, label="release_digest")
    _require_sha256(source_lock_digest, label="source_lock_digest")
    _require_sha256(profile_set_sha256, label="profile_set_sha256")
    if not isinstance(profile_set_id, str) or not profile_set_id:
        raise ValueError("profile_set_id must be a non-empty string")
    if tuple(config_hashes) != METHODS:
        raise ValueError("config_hashes must contain the ordered Table 4 methods")
    if tuple(profile_ids) != METHODS:
        raise ValueError("profile_ids must contain the ordered Table 4 methods")
    if tuple(task_seeds_by_region) != REGIONS:
        raise ValueError(
            "task_seeds_by_region must contain the ordered Table 4 regions"
        )
    if tuple(split_bundle_manifest_sha256) != REGIONS:
        raise ValueError(
            "split bundle bindings must contain the ordered Table 4 regions"
        )
    for method, digest in config_hashes.items():
        _require_sha256(digest, label=f"{method} config hash")
    for method, profile_id in profile_ids.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError(f"{method} profile_id must be non-empty")
    for region, task_seeds in task_seeds_by_region.items():
        if tuple(task_seeds) != METHODS:
            raise ValueError(f"{region} task seeds must contain the ordered methods")
        for method, seed in task_seeds.items():
            if seed is not None and (
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            ):
                raise ValueError(
                    f"{region} {method} task seed must be null or "
                    "a non-negative integer"
                )
        bist_seed = task_seeds["bist"]
        if not isinstance(bist_seed, int) or not 0 <= bist_seed < 999999:
            raise ValueError(
                f"{region} BiST task seed must freeze torch.randint in [0, 999999)"
            )
    for region, digest in split_bundle_manifest_sha256.items():
        _require_sha256(digest, label=f"{region} split bundle hash")

    tasks = []
    for task_number, (region, method) in enumerate(
        ((region, method) for region in REGIONS for method in METHODS),
        start=1,
    ):
        seed = task_seeds_by_region[region][method]
        task_id = f"{task_number:03d}_{_region_slug(region)}_{method}"
        tasks.append(
            {
                "task_id": task_id,
                "method": method,
                "region": region,
                "seed": seed,
                "source_id": _source_id(method),
                "profile_id": profile_ids[method],
                "config_sha256": config_hashes[method],
                "split_bundle_manifest_sha256": (split_bundle_manifest_sha256[region]),
                "output_dir": f"tasks/{task_id}",
            }
        )
    return {
        "schema_version": 1,
        "status": "planned_public_material_reproduction",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_public_repro_39_v2",
        "validity_label": "official_protocol_with_known_overlap",
        "claim_scope": "public_material_maximum_fidelity_reproduction",
        "release_digest": release_digest,
        "source_lock_digest": source_lock_digest,
        "profile_set_id": profile_set_id,
        "profile_set_sha256": profile_set_sha256,
        "bist_seed_freezes": {
            "upstream_expression": "torch.randint(999999, (1,))",
            "upper_exclusive": 999999,
            "realized_seed_by_region": {
                region: task_seeds_by_region[region]["bist"] for region in REGIONS
            },
        },
        "split_bundle_manifest_sha256": dict(split_bundle_manifest_sha256),
        "tasks": tasks,
    }


def load_table4_manifest(
    path: Path | str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    _require_sha256(expected_sha256, label="expected_sha256")
    source = Path(path)
    payload = read_verified_regular_file(
        source,
        purpose="Table 4 manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
        require_nonempty=True,
    )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("Table 4 manifest SHA-256 mismatch")
    manifest = json.loads(
        bytes(payload).decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(manifest, dict):
        raise ValueError("Table 4 manifest must contain an object")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(METHODS) * len(REGIONS):
        raise ValueError("Table 4 manifest must contain exactly 39 tasks")
    first_region_tasks = tasks[: len(METHODS)]
    config_hashes = {
        method: task.get("config_sha256")
        for method, task in zip(METHODS, first_region_tasks, strict=True)
    }
    profile_ids = {
        method: task.get("profile_id")
        for method, task in zip(METHODS, first_region_tasks, strict=True)
    }
    task_seeds_by_region = {
        region: {
            method: task.get("seed")
            for method, task in zip(
                METHODS,
                tasks[region_index * len(METHODS) : (region_index + 1) * len(METHODS)],
                strict=True,
            )
        }
        for region_index, region in enumerate(REGIONS)
    }
    rebuilt = build_table4_manifest(
        release_digest=manifest.get("release_digest"),
        source_lock_digest=manifest.get("source_lock_digest"),
        profile_set_id=manifest.get("profile_set_id"),
        profile_set_sha256=manifest.get("profile_set_sha256"),
        config_hashes=config_hashes,
        profile_ids=profile_ids,
        task_seeds_by_region=task_seeds_by_region,
        split_bundle_manifest_sha256=manifest.get("split_bundle_manifest_sha256"),
    )
    if manifest != rebuilt:
        raise ValueError("Table 4 manifest does not match the canonical matrix")
    return {**manifest, "manifest_sha256": observed_sha256}
