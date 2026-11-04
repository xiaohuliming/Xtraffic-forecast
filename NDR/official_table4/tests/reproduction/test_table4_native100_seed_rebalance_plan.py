from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
MANIFEST = CONFIG_ROOT / "matrix.native100_compare_v1.json"
PLAN = CONFIG_ROOT / "distributed_native100_seed_rebalance_v2.json"
LAUNCHER = (
    PROJECT_ROOT
    / "scripts"
    / "remote"
    / "windows"
    / "igstgnn_table4_native100_5080_orange_seeds1_2.cmd"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_rebalance_is_complete_unique_and_pair_preserving() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["plan_id"] == "igstgnn_table4_native100_seed_rebalance_v2"
    assert plan["status"] == "rejected_after_live_speed_gate"
    assert plan["active_plan_id_after_rejection"] == (
        "igstgnn_table4_native100_5080_5090_v1"
    )
    assert plan["manifest_sha256"] == _sha256(MANIFEST)
    assert plan["execution_layout"] == "dual_host_region_seed_pair_stratified"
    assert plan["active_execution_layout_after_rejection"] == (
        "dual_host_region_stratified"
    )

    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    assigned = [task_id for host in plan["hosts"] for task_id in host["task_ids"]]
    assert len(assigned) == len(set(assigned)) == 18
    assert set(assigned) == set(tasks)
    host_by_task = {
        task_id: host["host_id"]
        for host in plan["hosts"]
        for task_id in host["task_ids"]
    }
    for region in ("Alameda", "Contra_Costa", "Orange"):
        for seed in (42, 1, 2):
            pair = [
                task for task in tasks.values()
                if task["region"] == region and task["seed"] == seed
            ]
            assert len(pair) == 2
            assert len({host_by_task[task["task_id"]] for task in pair}) == 1


def test_5080_recovery_launcher_selects_only_remaining_orange_ndr_seeds() -> None:
    content = LAUNCHER.read_text(encoding="utf-8")
    assert "AUDIT ARTIFACT ONLY. DO NOT RUN" in content
    assert "exit /b 90" in content
    task_ids = re.findall(r"--task-id (n100\d{3}_[a-z0-9_]+)", content)
    assert task_ids == [
        "n100017_ndr_net_orange_seed1",
        "n100018_ndr_net_orange_seed2",
    ]
    assert 'set "RUN_ROOT=D:\\t4\\native100_orange_5080_recovery"' in content
    assert "shutdown" not in content.lower()
