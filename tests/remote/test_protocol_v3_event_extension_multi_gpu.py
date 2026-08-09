from __future__ import annotations

from scripts.remote.linux.protocol_v3_event_extension_multi_gpu import (
    assign_tasks_lpt,
    build_plan,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    CANONICAL_MANIFEST,
    load_extension_manifest,
)


def test_extension_lpt_assignment_is_complete_and_deterministic() -> None:
    tasks = load_extension_manifest(CANONICAL_MANIFEST).tasks
    costs = {task.task_id: float(index + 1) for index, task in enumerate(tasks)}
    first = assign_tasks_lpt(tasks, gpu_count=4, task_seconds=costs)
    second = assign_tasks_lpt(tuple(reversed(tasks)), gpu_count=4, task_seconds=costs)
    assert first == second
    assert sorted(task.task_id for shard in first.values() for task in shard) == sorted(
        task.task_id for task in tasks
    )
    assert all(first[gpu] for gpu in first)


def test_extension_plan_binds_support_and_all_nine_tasks() -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    costs = {task.task_id: 1.0 for task in manifest.tasks}
    shards = assign_tasks_lpt(manifest.tasks, gpu_count=4, task_seconds=costs)
    plan = build_plan(
        manifest=manifest,
        shards=shards,
        matrix_id="v3_extensions_test",
        host_id="host_test",
        code_sha="1" * 40,
        cache_bundle_id="bundle-test",
        cache_bundle_digest="2" * 64,
        support_manifest_sha256="3" * 64,
        loading_audit_sha256="4" * 64,
        task_seconds=costs,
        gpu_count=4,
    )
    assert plan["task_count"] == 9
    assert plan["execution_layout"] == "single_host_multi_gpu_interleaved"
    assert plan["support_manifest_sha256"] == "3" * 64
    assert plan["loading_audit_sha256"] == "4" * 64
    assert plan["task_microbatch_sizes"] == {
        "v3x008_event_igstgnn_orange_seed42": 8,
        "v3x009_event_ndr_net_orange_seed42": 8,
    }
    assigned = [task for values in plan["assignments"].values() for task in values]
    assert sorted(assigned) == sorted(task.task_id for task in manifest.tasks)
    assert len(plan["plan_digest"]) == 64
