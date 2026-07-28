import json
from pathlib import Path

import pytest

from scripts.reproduction.igstgnn_table4.manifest import METHODS
from scripts.reproduction.igstgnn_table4.profiles import (
    load_profile_set,
)


PROFILE_PATH = "config/reproduction/igstgnn_table4/profiles.public_material_v2.json"


def test_public_material_profiles_cover_all_table4_methods_and_hash_cleanly():
    profile_set = load_profile_set(PROFILE_PATH)

    assert profile_set["protocol_id"] == "igstgnn_table4_official_v1"
    assert profile_set["profile_set_id"] == (
        "igstgnn_table4_public_material_profiles_v2"
    )
    assert tuple(profile_set["profiles"]) == METHODS
    assert set(profile_set["config_hashes"]) == set(METHODS)
    assert len(profile_set["profile_set_sha256"]) == 64
    assert len(profile_set["profile_set_canonical_sha256"]) == 64
    assert all(len(value) == 64 for value in profile_set["config_hashes"].values())
    assert profile_set["profiles"]["igstgnn"]["effective"]["batch_size_by_region"] == {
        "Alameda": 48,
        "Contra_Costa": 48,
        "Orange": 24,
    }
    assert (
        profile_set["profiles"]["igstgnn"]["authority"]["batch_size_by_region"]
        == "official_table4"
    )
    assert {
        profile_set["profiles"][method]["effective"]["seed"] for method in METHODS[:11]
    } == {2023}
    assert profile_set["profiles"]["bist"]["effective"]["seed"] is None
    assert profile_set["profiles"]["bist"]["effective"]["seed_policy"] == (
        "upstream_runtime_random_draw"
    )
    assert profile_set["profiles"]["igstgnn"]["effective"]["seed"] == 2025
    assert profile_set["profiles"]["d2stgnn"]["effective"]["batch_size_by_region"] == {
        "Alameda": 64,
        "Contra_Costa": 64,
        "Orange": 64,
    }
    assert profile_set["profiles"]["d2stgnn"]["effective"]["max_epochs"] == 100
    assert profile_set["profiles"]["d2stgnn"]["effective"]["patience"] == 30
    assert (
        profile_set["profiles"]["bist"]["effective"]["input_contract"][
            "time_slots_per_day"
        ]
        == 96
    )
    assert {
        profile_set["profiles"][method]["effective"]["loader_contract"]["partial_batch"]
        for method in METHODS[:12]
    } == {"drop"}
    assert (
        profile_set["profiles"]["igstgnn"]["effective"]["loader_contract"][
            "partial_batch"
        ]
        == "retain"
    )
    assert {
        profile["effective"]["loader_contract"]["training_shuffle"]
        for profile in profile_set["profiles"].values()
    } == {"global_numpy_permutation"}
    assert all(
        "windows_secure_io_compatibility_v1" in profile["compatibility_patches"]
        for profile in profile_set["profiles"].values()
    )
    assert all(
        profile["public_material_gaps"]
        for method, profile in profile_set["profiles"].items()
        if method not in {"hl", "igstgnn"}
    )


def test_profile_loader_rejects_unknown_authority_labels(tmp_path):
    profile = {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "profile_set_id": "invalid",
        "shared": {},
        "profiles": {
            method: {
                "profile_id": method,
                "source_id": "source",
                "effective": {
                    "seed": (
                        None
                        if method == "bist"
                        else 2025
                        if method == "igstgnn"
                        else 2023
                    ),
                    **(
                        {"seed_policy": "upstream_runtime_random_draw"}
                        if method == "bist"
                        else {}
                    ),
                    "loader_contract": {
                        "training_shuffle": "global_numpy_permutation",
                        "partial_batch": ("retain" if method == "igstgnn" else "drop"),
                    },
                },
                "authority": {"seed": "invented_authority"},
                "public_material_gaps": [],
                "compatibility_patches": [],
            }
            for method in METHODS
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="authority"):
        load_profile_set(path)


def test_profile_loader_rejects_non_source_loader_semantics(tmp_path):
    profile = json.loads(Path(PROFILE_PATH).read_text(encoding="utf-8"))
    profile["profiles"]["lstm"]["effective"]["loader_contract"]["partial_batch"] = (
        "retain"
    )
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="loader_contract"):
        load_profile_set(path)
