from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.reproduction.igstgnn_table4.bist_ndr_pair_manifest import (
    PAIR_METHODS,
    PAIR_REGIONS,
    PAIR_SEEDS,
    load_bist_ndr_pair_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
)
BASE_MANIFEST = CONFIG_ROOT / "matrix.public_material_v2.json"
PAIR_MANIFEST = CONFIG_ROOT / "matrix.bist_ndr_pair_v1.json"
BASE_MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bist_ndr_pair_matrix_is_a_matched_eighteen_task_experiment():
    assert _sha256(BASE_MANIFEST) == BASE_MANIFEST_SHA256

    matrix = load_bist_ndr_pair_manifest(
        PAIR_MANIFEST,
        expected_sha256=_sha256(PAIR_MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        expected_base_manifest_sha256=BASE_MANIFEST_SHA256,
    )

    tasks = matrix["tasks"]
    profile = matrix["profile"]
    assert matrix["matrix_id"] == "igstgnn_table4_bist_ndr_pair_18_v1"
    assert matrix["claim_scope"] == "same_protocol_project_method_extension"
    assert matrix["validity_label"] == "official_protocol_with_known_overlap"
    assert len(tasks) == 18
    assert [
        (task["method"], task["region"], task["seed"])
        for task in tasks
    ] == [
        (method, region, seed)
        for method in PAIR_METHODS
        for region in PAIR_REGIONS
        for seed in PAIR_SEEDS
    ]
    assert len({task["task_id"] for task in tasks}) == 18
    assert len({task["output_dir"] for task in tasks}) == 18
    assert profile["effective"]["seeds"] == list(PAIR_SEEDS)
    assert profile["effective"]["loader_contract"] == {
        "training_shuffle": "global_numpy_permutation",
        "partial_batch": "retain",
    }
    assert profile["effective"]["evaluation"]["test_observation_set"] == (
        "identical_retained_official_split_samples_for_both_methods"
    )
    assert set(profile["methods"]) == set(PAIR_METHODS)
