import hashlib
from pathlib import Path

from scripts.reproduction.igstgnn_table4.ndr_extension_manifest import (
    EXTENSION_REGIONS,
    EXTENSION_SEEDS,
    load_ndr_extension_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.public_material_v2.json"
)
EXTENSION_MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.ndr_extension_v1.json"
)
BASE_MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ndr_extension_is_nine_same_protocol_tasks_without_mutating_base_matrix():
    assert _sha256(BASE_MANIFEST) == BASE_MANIFEST_SHA256

    extension = load_ndr_extension_manifest(
        EXTENSION_MANIFEST,
        expected_sha256=_sha256(EXTENSION_MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        expected_base_manifest_sha256=BASE_MANIFEST_SHA256,
    )

    tasks = extension["tasks"]
    assert extension["protocol_id"] == "igstgnn_table4_official_v1"
    assert extension["matrix_id"] == "igstgnn_table4_ndr_extension_9_v1"
    assert extension["base_matrix_id"] == "igstgnn_table4_public_repro_39_v2"
    assert extension["base_manifest_sha256"] == BASE_MANIFEST_SHA256
    assert extension["validity_label"] == "official_protocol_with_known_overlap"
    assert extension["claim_scope"] == "same_protocol_project_method_extension"
    assert len(tasks) == 9
    assert [(task["region"], task["seed"]) for task in tasks] == [
        (region, seed) for region in EXTENSION_REGIONS for seed in EXTENSION_SEEDS
    ]
    assert {task["method"] for task in tasks} == {"ndr_net"}
    assert len({task["task_id"] for task in tasks}) == 9
    assert len({task["output_dir"] for task in tasks}) == 9
    assert all(task["output_dir"] == f"tasks/{task['task_id']}" for task in tasks)
    assert all(task["profile_id"] == "ndr_net_table4_extension_v1" for task in tasks)
    assert all(task["source_id"] == "xtraffic_ndr_s0g" for task in tasks)
    assert all(
        task["split_bundle_manifest_sha256"]
        == extension["split_bundle_manifest_sha256"][task["region"]]
        for task in tasks
    )
