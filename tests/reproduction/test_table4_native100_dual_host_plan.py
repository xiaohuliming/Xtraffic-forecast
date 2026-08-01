from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
MANIFEST = CONFIG_ROOT / "matrix.native100_compare_v1.json"
PLAN = CONFIG_ROOT / "distributed_native100_5080_5090_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native100_dual_host_plan_is_an_exact_region_partition():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["schema_version"] == 1
    assert plan["plan_id"] == "igstgnn_table4_native100_5080_5090_v1"
    assert plan["matrix_id"] == manifest["matrix_id"]
    assert plan["manifest_sha256"] == _sha256(MANIFEST)
    assert plan["execution_layout"] == "dual_host_region_stratified"

    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    assigned = [
        task_id
        for host in plan["hosts"]
        for task_id in host["task_ids"]
    ]
    assert len(assigned) == len(set(assigned)) == 18
    assert set(assigned) == set(task_by_id)

    host_by_task = {
        task_id: host["host_id"]
        for host in plan["hosts"]
        for task_id in host["task_ids"]
    }
    for task in manifest["tasks"]:
        expected_host = (
            "native100_rtx5090_orange"
            if task["region"] == "Orange"
            else "native100_rtx5080_alameda_contra"
        )
        assert host_by_task[task["task_id"]] == expected_host

    for region in {task["region"] for task in manifest["tasks"]}:
        assert len(
            {
                host_by_task[task["task_id"]]
                for task in manifest["tasks"]
                if task["region"] == region
            }
        ) == 1


def test_native100_dual_host_plan_excludes_stale_5080_orange_results():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    host = next(
        item
        for item in plan["hosts"]
        if item["host_id"] == "native100_rtx5080_alameda_contra"
    )

    assert host["result_selection"] == "declared_task_ids_only"
    assert host["excluded_existing_task_ids"] == [
        "n100007_bist_orange_seed42",
        "n100008_bist_orange_seed1",
        "n100009_bist_orange_seed2",
    ]
    assert not any("_orange_" in task_id for task_id in host["task_ids"])
