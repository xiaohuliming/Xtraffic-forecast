from __future__ import annotations

from pathlib import Path

from scripts.remote.linux.protocol_v3_event_orange_8gpu import (
    GWN_RECOVERY_GPU_METHODS,
    OOM_RECOVERY_GPU_METHODS,
    OOM_RECOVERY_MICROBATCH_SIZES,
    build_orange_gpu_shards,
    write_orange_shard_manifests,
)
from scripts.remote.windows.protocol_v3_baseline_runner import (
    BASELINE_METHODS,
    load_baseline_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_MANIFEST = (
    REPO_ROOT
    / "config"
    / "protocol_v3"
    / "table4_event_baselines_phase1_v1.json"
)


def test_orange_gpu_shards_cover_each_method_once_on_one_host() -> None:
    manifest = load_baseline_manifest(EVENT_MANIFEST)

    shards = build_orange_gpu_shards(manifest)

    assert tuple(shards) == tuple(range(8))
    tasks = [task for shard in shards.values() for task in shard]
    assert len(tasks) == 11
    assert len({task.task_id for task in tasks}) == 11
    assert {task.region for task in tasks} == {"Orange"}
    assert {task.model for task in tasks} == set(BASELINE_METHODS)
    assert [task.model for task in shards[0]] == ["dcrnn"]
    assert [task.model for task in shards[1]] == ["dgcrn"]
    assert [task.model for task in shards[2]] == ["d2stgnn"]
    assert [task.model for task in shards[3]] == ["dstagnn"]


def test_written_shard_manifests_are_individually_runner_valid(
    tmp_path: Path,
) -> None:
    manifest = load_baseline_manifest(EVENT_MANIFEST)

    paths = write_orange_shard_manifests(
        EVENT_MANIFEST,
        manifest,
        tmp_path,
    )

    assert tuple(paths) == tuple(range(8))
    task_ids: set[str] = set()
    for gpu, path in paths.items():
        shard = load_baseline_manifest(
            path,
            require_complete_matrix=False,
        )
        assert [task.model for task in shard.tasks] == [
            task.model for task in build_orange_gpu_shards(manifest)[gpu]
        ]
        assert not task_ids.intersection(task.task_id for task in shard.tasks)
        task_ids.update(task.task_id for task in shard.tasks)
    assert len(task_ids) == 11


def test_oom_recovery_shards_bind_three_failed_models_to_idle_gpus(
    tmp_path: Path,
) -> None:
    manifest = load_baseline_manifest(EVENT_MANIFEST)
    shards = build_orange_gpu_shards(
        manifest,
        assignments=OOM_RECOVERY_GPU_METHODS,
    )

    paths = write_orange_shard_manifests(
        EVENT_MANIFEST,
        manifest,
        tmp_path,
        assignments=OOM_RECOVERY_GPU_METHODS,
        microbatch_sizes=OOM_RECOVERY_MICROBATCH_SIZES,
    )

    assert {gpu: shard[0].model for gpu, shard in shards.items()} == {
        1: "dgcrn",
        2: "d2stgnn",
        4: "sttn",
    }
    assert {
        gpu: load_baseline_manifest(
            path,
            require_complete_matrix=False,
        ).training["microbatch_size"]
        for gpu, path in paths.items()
    } == {1: 8, 2: 8, 4: 16}


def test_graphwavenet_recovery_keeps_orange_followup_on_gpu6() -> None:
    manifest = load_baseline_manifest(EVENT_MANIFEST)

    shards = build_orange_gpu_shards(
        manifest,
        assignments=GWN_RECOVERY_GPU_METHODS,
    )

    assert tuple(shards) == (6,)
    assert [task.model for task in shards[6]] == ["graphwavenet", "agcrn"]
