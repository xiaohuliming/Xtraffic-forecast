from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    NATIVE100_METHODS,
    NATIVE100_REGIONS,
    NATIVE100_SEEDS,
    load_native100_compare_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
BASE_MANIFEST = CONFIG_ROOT / "matrix.public_material_v2.json"
MANIFEST = CONFIG_ROOT / "matrix.native100_compare_v1.json"
BASE_MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native100_matrix_preserves_table4_recipes_and_changes_only_epoch_cap():
    matrix = load_native100_compare_manifest(
        MANIFEST,
        expected_sha256=_sha256(MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        expected_base_manifest_sha256=BASE_MANIFEST_SHA256,
    )

    assert matrix["matrix_id"] == "igstgnn_table4_bist_ndr_native100_18_v1"
    assert matrix["protocol_id"] == "igstgnn_table4_official_v1"
    assert matrix["validity_label"] == "official_protocol_with_known_overlap"
    assert matrix["claim_scope"] == "same_table_native_recipe_100_epoch_supplement"
    assert [
        (task["method"], task["region"], task["seed"])
        for task in matrix["tasks"]
    ] == [
        (method, region, seed)
        for method in NATIVE100_METHODS
        for region in NATIVE100_REGIONS
        for seed in NATIVE100_SEEDS
    ]

    profile = matrix["profile"]
    assert profile["reported_columns"] == ["H3", "H6", "H12", "Average"]
    assert profile["methods"]["bist"]["effective"]["max_epochs"] == 100
    assert profile["methods"]["bist"]["effective"]["patience"] == 20
    assert profile["methods"]["bist"]["effective"]["loader_contract"] == {
        "training_shuffle": "global_numpy_permutation",
        "partial_batch": "drop",
    }
    assert profile["methods"]["ndr_net"]["effective"]["max_epochs"] == 100
    assert profile["methods"]["ndr_net"]["effective"]["early_stopping"] is False
    assert profile["methods"]["ndr_net"]["effective"]["loader_contract"] == {
        "training_shuffle": "global_numpy_permutation",
        "partial_batch": "retain",
    }
    assert all(task["output_dir"].startswith("tasks/") for task in matrix["tasks"])
    assert len({task["output_dir"] for task in matrix["tasks"]}) == 18
