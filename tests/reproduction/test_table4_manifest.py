import hashlib
import json
import os

import pytest

from scripts.reproduction.igstgnn_table4.manifest import (
    METHODS,
    REGIONS,
    build_table4_manifest,
    load_table4_manifest,
)


def _bundle_hashes():
    return {region: str(index) * 64 for index, region in enumerate(REGIONS, start=1)}


def _config_hashes():
    return {
        method: format(index, "x") * 64 for index, method in enumerate(METHODS, start=1)
    }


def _profile_ids():
    return {method: f"{method}_public_material_v2" for method in METHODS}


def _task_seeds_by_region():
    return {
        region: {
            **{method: 2023 for method in METHODS[:11]},
            "bist": 424242 + region_index,
            "igstgnn": 2025,
        }
        for region_index, region in enumerate(REGIONS)
    }


def test_table4_manifest_contains_exact_ordered_13_by_3_paper_matrix():
    config_hashes = _config_hashes()

    manifest = build_table4_manifest(
        release_digest="a" * 64,
        source_lock_digest="b" * 64,
        profile_set_id="public_profiles_v1",
        profile_set_sha256="c" * 64,
        config_hashes=config_hashes,
        profile_ids=_profile_ids(),
        task_seeds_by_region=_task_seeds_by_region(),
        split_bundle_manifest_sha256=_bundle_hashes(),
    )

    tasks = manifest["tasks"]
    assert manifest["status"] == "planned_public_material_reproduction"
    assert manifest["validity_label"] == ("official_protocol_with_known_overlap")
    assert len(tasks) == 39
    assert [(task["region"], task["method"]) for task in tasks] == [
        (region, method) for region in REGIONS for method in METHODS
    ]
    assert len({(task["region"], task["method"]) for task in tasks}) == 39
    assert [task["task_id"][:3] for task in tasks] == [
        f"{number:03d}" for number in range(1, 40)
    ]
    assert all(task["output_dir"] == f"tasks/{task['task_id']}" for task in tasks)
    assert {task["seed"] for task in tasks if task["method"] in METHODS[:11]} == {2023}
    assert {task["seed"] for task in tasks if task["method"] == "bist"} == {
        424242,
        424243,
        424244,
    }
    assert {task["seed"] for task in tasks if task["method"] == "igstgnn"} == {2025}
    assert manifest["bist_seed_freezes"] == {
        "upstream_expression": "torch.randint(999999, (1,))",
        "upper_exclusive": 999999,
        "realized_seed_by_region": {
            "Alameda": 424242,
            "Contra_Costa": 424243,
            "Orange": 424244,
        },
    }
    assert all(
        task["split_bundle_manifest_sha256"] == _bundle_hashes()[task["region"]]
        for task in tasks
    )


def test_table4_manifest_assigns_methods_to_locked_upstream_sources():
    manifest = build_table4_manifest(
        release_digest="a" * 64,
        source_lock_digest="b" * 64,
        profile_set_id="public_profiles_v1",
        profile_set_sha256="c" * 64,
        config_hashes=_config_hashes(),
        profile_ids=_profile_ids(),
        task_seeds_by_region=_task_seeds_by_region(),
        split_bundle_manifest_sha256=_bundle_hashes(),
    )

    source_by_method = {
        task["method"]: task["source_id"]
        for task in manifest["tasks"]
        if task["region"] == "Alameda"
    }

    assert set(source_by_method[method] for method in METHODS[:11]) == {
        "largest_b7c6ff5"
    }
    assert source_by_method["bist"] == "bist_dd94adf"
    assert source_by_method["igstgnn"] == "igstgnn_64c6e24"


def test_table4_manifest_rejects_missing_or_non_hex_data_bindings():
    invalid = _bundle_hashes()
    invalid.pop("Orange")

    try:
        build_table4_manifest(
            release_digest="a" * 64,
            source_lock_digest="b" * 64,
            profile_set_id="public_profiles_v1",
            profile_set_sha256="c" * 64,
            config_hashes=_config_hashes(),
            profile_ids=_profile_ids(),
            task_seeds_by_region=_task_seeds_by_region(),
            split_bundle_manifest_sha256=invalid,
        )
    except ValueError as exc:
        assert "regions" in str(exc)
    else:
        raise AssertionError("Missing region binding was accepted")


def test_table4_manifest_round_trips_through_strict_loader(tmp_path):
    manifest = build_table4_manifest(
        release_digest="a" * 64,
        source_lock_digest="b" * 64,
        profile_set_id="public_profiles_v1",
        profile_set_sha256="c" * 64,
        config_hashes=_config_hashes(),
        profile_ids=_profile_ids(),
        task_seeds_by_region=_task_seeds_by_region(),
        split_bundle_manifest_sha256=_bundle_hashes(),
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_table4_manifest(path, expected_sha256=digest)

    assert loaded["manifest_sha256"] == digest
    assert loaded["tasks"] == manifest["tasks"]


def test_table4_manifest_loads_through_the_windows_compatibility_boundary(
    tmp_path,
    monkeypatch,
):
    manifest = build_table4_manifest(
        release_digest="a" * 64,
        source_lock_digest="b" * 64,
        profile_set_id="public_profiles_v1",
        profile_set_sha256="c" * 64,
        config_hashes=_config_hashes(),
        profile_ids=_profile_ids(),
        task_seeds_by_region=_task_seeds_by_region(),
        split_bundle_manifest_sha256=_bundle_hashes(),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    loaded = load_table4_manifest(path, expected_sha256=digest)

    assert loaded["manifest_sha256"] == digest


@pytest.mark.parametrize("mutation", ["task", "duplicate_key", "hash"])
def test_table4_manifest_loader_rejects_tampering(tmp_path, mutation):
    manifest = build_table4_manifest(
        release_digest="a" * 64,
        source_lock_digest="b" * 64,
        profile_set_id="public_profiles_v1",
        profile_set_sha256="c" * 64,
        config_hashes=_config_hashes(),
        profile_ids=_profile_ids(),
        task_seeds_by_region=_task_seeds_by_region(),
        split_bundle_manifest_sha256=_bundle_hashes(),
    )
    path = tmp_path / "manifest.json"
    if mutation == "task":
        manifest["tasks"][0]["region"] = "Orange"
        payload = json.dumps(manifest)
    elif mutation == "duplicate_key":
        payload = '{"schema_version":1,"schema_version":1}'
    else:
        payload = json.dumps(manifest)
    path.write_text(payload, encoding="utf-8")
    expected = (
        "f" * 64
        if mutation == "hash"
        else hashlib.sha256(path.read_bytes()).hexdigest()
    )

    with pytest.raises(ValueError):
        load_table4_manifest(path, expected_sha256=expected)
