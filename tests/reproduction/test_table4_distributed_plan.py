from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reproduction.igstgnn_table4.distributed_plan import (
    load_distributed_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.public_material_v2.json"
)
PLAN = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "distributed_4090d_9gpu_v1.json"
)
MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357e"
    "b5320393417a83d344adb05151023a78"
)


def test_distributed_plan_covers_every_task_once_and_keeps_regions_on_one_host():
    plan = load_distributed_plan(
        PLAN,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    assert plan["plan_id"] == "igstgnn_table4_4090d_9gpu_v1"
    assert len(plan["hosts"]) == 2
    assert len(plan["shards"]) == 9
    assert sum(len(shard["task_ids"]) for shard in plan["shards"]) == 39

    task_host = {
        task_id: shard["host_id"]
        for shard in plan["shards"]
        for task_id in shard["task_ids"]
    }
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for region in {task["region"] for task in manifest["tasks"]}:
        assert {
            task_host[task["task_id"]]
            for task in manifest["tasks"]
            if task["region"] == region
        } == {
            next(
                host_id
                for host_id, regions in plan["region_host_assignment"].items()
                if region in regions
            )
        }


def test_distributed_plan_rejects_a_duplicate_task(tmp_path):
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    value["shards"][1]["task_ids"][0] = value["shards"][0]["task_ids"][0]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly once"):
        load_distributed_plan(
            broken,
            manifest_path=MANIFEST,
            expected_manifest_sha256=MANIFEST_SHA256,
        )
