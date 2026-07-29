import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.reproduction.igstgnn_table4.ndr_extension_distributed import (
    build_ndr_host_launch_specs,
    load_ndr_extension_plan,
    validate_ndr_extension_host_results,
)
from scripts.reproduction.igstgnn_table4.ndr_extension_manifest import (
    load_ndr_extension_manifest,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ndr_extension_plan_covers_each_task_once_on_its_region_host():
    manifest_path = CONFIG_ROOT / "matrix.ndr_extension_v1.json"
    base_manifest_path = CONFIG_ROOT / "matrix.public_material_v2.json"
    plan = load_ndr_extension_plan(
        CONFIG_ROOT / "distributed_ndr_extension_9gpu_v1.json",
        manifest_path=manifest_path,
        expected_manifest_sha256=_sha256(manifest_path),
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=_sha256(base_manifest_path),
    )

    task_ids = [
        task_id
        for shard in plan["shards"]
        for task_id in shard["task_ids"]
    ]
    assert len(task_ids) == 9
    assert len(set(task_ids)) == 9
    assert set(task_ids) == {
        f"ndr{index:03d}_{region}_seed{seed}"
        for index, region, seed in (
            (1, "alameda", 42),
            (2, "alameda", 1),
            (3, "alameda", 2),
            (4, "contra_costa", 42),
            (5, "contra_costa", 1),
            (6, "contra_costa", 2),
            (7, "orange", 42),
            (8, "orange", 1),
            (9, "orange", 2),
        )
    }
    assert all(
        shard["host_id"] == "table4_4090d5"
        for shard in plan["shards"]
        if any("_orange_" in task_id for task_id in shard["task_ids"])
    )
    assert all(
        shard["host_id"] == "table4_4090d4"
        for shard in plan["shards"]
        if not any("_orange_" in task_id for task_id in shard["task_ids"])
    )


def test_ndr_host_launch_specs_pin_one_shard_to_each_declared_gpu():
    manifest_path = CONFIG_ROOT / "matrix.ndr_extension_v1.json"
    base_manifest_path = CONFIG_ROOT / "matrix.public_material_v2.json"
    specs = build_ndr_host_launch_specs(
        plan_path=CONFIG_ROOT / "distributed_ndr_extension_9gpu_v1.json",
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=_sha256(base_manifest_path),
        host_id="table4_4090d4",
        splits_root=Path("/data/official_splits"),
        results_base=Path("/results/ndr_extension"),
        python_executable=Path("/venv/bin/python"),
    )

    assert len(specs) == 4
    assert [spec["gpu_index"] for spec in specs] == [0, 1, 2, 3]
    assert [spec["cuda_visible_devices"] for spec in specs] == ["0", "1", "2", "3"]
    assert all(
        spec["command"][:5]
        == [
            "/venv/bin/python",
            "-X",
            "utf8",
            "-m",
            "scripts.reproduction.igstgnn_table4.ndr_extension_matrix_runner",
        ]
        for spec in specs
    )
    assert all(
        spec["command"][spec["command"].index("--base-manifest-sha256") + 1]
        == _sha256(base_manifest_path)
        for spec in specs
    )


def test_ndr_host_postflight_revalidates_every_task_and_exact_export_file(tmp_path):
    manifest_path = CONFIG_ROOT / "matrix.ndr_extension_v1.json"
    base_manifest_path = CONFIG_ROOT / "matrix.public_material_v2.json"
    manifest_sha256 = _sha256(manifest_path)
    base_manifest_sha256 = _sha256(base_manifest_path)
    plan_path = CONFIG_ROOT / "distributed_ndr_extension_9gpu_v1.json"
    plan = load_ndr_extension_plan(
        plan_path,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    manifest = load_ndr_extension_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    results_base = tmp_path / "results"
    for shard in plan["shards"]:
        if shard["host_id"] != "table4_4090d5":
            continue
        shard_root = results_base / shard["results_root"]
        shard_root.mkdir(parents=True)
        status = {
            "schema_version": 1,
            "state": "completed",
            "protocol_id": manifest["protocol_id"],
            "matrix_id": manifest["matrix_id"],
            "manifest_sha256": manifest_sha256,
            "base_manifest_sha256": base_manifest_sha256,
            "selected_task_ids": shard["task_ids"],
            "completed": shard["task_ids"],
            "skipped": [],
            "failed": [],
            "current": None,
        }
        (shard_root / "queue_status.json").write_text(
            json.dumps(status),
            encoding="utf-8",
        )
        for task_id in shard["task_ids"]:
            checkpoint = tmp_path / f"{task_id}.pt"
            checkpoint.write_bytes(b"checkpoint")
            log = tmp_path / f"{task_id}.log"
            log.write_text("complete\n", encoding="utf-8")
            labels = np.ones((1, 12, 2), dtype=np.float32)
            write_completed_task_artifacts(
                target_dir=shard_root / "tasks" / task_id,
                task=tasks[task_id],
                predictions=labels + 1,
                labels=labels,
                checkpoint_source=checkpoint,
                train_log_source=log,
                provenance={
                    "protocol_id": "igstgnn_table4_official_v1",
                    "training_code_sha256": "a" * 64,
                },
                claim_scope="same_protocol_project_method_extension",
            )

    report_path = tmp_path / "postflight.json"
    report = validate_ndr_extension_host_results(
        plan_path=plan_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
        host_id="table4_4090d5",
        results_base=results_base,
        report_path=report_path,
    )

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 3
    assert report["validated_shards"] == 3
    assert report["file_count"] == 18
    assert report_path.is_file()
