from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.remote.linux.protocol_v3_event_multi_gpu import (
    PlanError,
    _parse_microbatch,
    plan_digest,
    aggregate_queue_status,
    assign_tasks_lpt,
    build_nas_plan,
    load_calibration,
    select_region_tasks,
    shard_load_units,
    shard_name,
    task_cost,
    write_shard_manifests,
)
from scripts.remote.windows.protocol_v3_baseline_runner import (
    BASELINE_METHODS,
    load_baseline_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVENT_MANIFEST = (
    REPO_ROOT / "config" / "protocol_v3" / "table4_event_baselines_phase1_v1.json"
)
CALIBRATION = (
    REPO_ROOT / "config" / "protocol_v3" / "event_baseline_gpu_calibration_v1.json"
)
PROJECTED_CALIBRATION = (
    REPO_ROOT
    / "config"
    / "protocol_v3"
    / "event_baseline_gpu_calibration_5090_projected_v1.json"
)
MEASURED_5090_CALIBRATION = (
    REPO_ROOT
    / "config"
    / "protocol_v3"
    / "event_baseline_gpu_calibration_5090x4_measured_v1.json"
)


@pytest.fixture(scope="module")
def manifest():
    return load_baseline_manifest(EVENT_MANIFEST)


@pytest.fixture(scope="module")
def calibration():
    return load_calibration(CALIBRATION)


def test_shipped_calibrations_cover_every_baseline_method_and_region() -> None:
    for path in (CALIBRATION, PROJECTED_CALIBRATION, MEASURED_5090_CALIBRATION):
        payload = load_calibration(path)
        assert set(payload["epoch_seconds"]) == set(BASELINE_METHODS)
        assert set(payload["region_scale"]) == {"Alameda", "ContraCosta", "Orange"}


def test_measured_5090_calibration_declares_its_microbatch_requirements() -> None:
    payload = load_calibration(MEASURED_5090_CALIBRATION)

    # Only the two Orange tasks exceeded 32 GB at batch 32, so the override is
    # keyed by task id and must not spill onto the same models in other regions.
    assert payload["required_microbatch"] == {
        "v3e033_event_d2stgnn_orange_seed42": 16,
        "v3e030_event_sttn_orange_seed42": 16,
    }
    assert payload["status"] == "partially_measured"
    assert payload["measured_on"]["dgcrn"] == "concurrent_4way"
    assert payload["measured_on"]["graphwavenet"] == "estimate_not_probed"
    for model, peak in payload["peak_allocated_bytes"].items():
        assert peak < 31.36 * 1024 ** 3, model


def test_region_selection_requires_every_method_of_each_region(manifest) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])

    assert len(tasks) == 22
    assert {task.region for task in tasks} == {"ContraCosta", "Orange"}
    for region in ("ContraCosta", "Orange"):
        models = {task.model for task in tasks if task.region == region}
        assert models == set(BASELINE_METHODS)


def test_region_selection_rejects_unknown_region(manifest) -> None:
    with pytest.raises(PlanError):
        select_region_tasks(manifest, ["Nowhere"])


def test_region_selection_rejects_incomplete_region(manifest) -> None:
    trimmed = tuple(
        task
        for task in manifest.tasks
        if not (task.region == "Orange" and task.model == "dgcrn")
    )
    partial = type(manifest)(
        manifest_version=manifest.manifest_version,
        protocol_version=manifest.protocol_version,
        cache_schema_version=manifest.cache_schema_version,
        default_results_root=manifest.default_results_root,
        default_cache_dir=manifest.default_cache_dir,
        default_graph_dir=manifest.default_graph_dir,
        training=manifest.training,
        tasks=trimmed,
        source_path=manifest.source_path,
        digest=manifest.digest,
    )

    with pytest.raises(PlanError):
        select_region_tasks(partial, ["Alameda", "Orange"])


def test_assignment_partitions_every_task_exactly_once(manifest, calibration) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])

    shards = assign_tasks_lpt(tasks, 4, calibration)

    assert tuple(sorted(shards)) == (0, 1, 2, 3)
    assigned = [task.task_id for shard in shards.values() for task in shard]
    assert sorted(assigned) == sorted(task.task_id for task in tasks)
    assert len(set(assigned)) == len(assigned)


def test_assignment_is_deterministic(manifest, calibration) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])

    first = assign_tasks_lpt(tasks, 4, calibration)
    second = assign_tasks_lpt(tuple(reversed(tasks)), 4, calibration)

    assert {gpu: [task.task_id for task in shard] for gpu, shard in first.items()} == {
        gpu: [task.task_id for task in shard] for gpu, shard in second.items()
    }


def test_assignment_beats_round_robin_makespan(manifest, calibration) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    gpu_count = 4

    shards = assign_tasks_lpt(tasks, gpu_count, calibration)
    lpt_makespan = max(shard_load_units(shards, calibration).values())
    round_robin = [0.0] * gpu_count
    for index, task in enumerate(tasks):
        round_robin[index % gpu_count] += task_cost(task, calibration)

    assert lpt_makespan <= max(round_robin)


def test_single_dominant_task_bounds_the_makespan(manifest, calibration) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    heaviest = max(task_cost(task, calibration) for task in tasks)

    shards = assign_tasks_lpt(tasks, 4, calibration)

    assert max(shard_load_units(shards, calibration).values()) >= heaviest


def test_assignment_rejects_more_gpus_than_tasks(manifest, calibration) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta"])

    with pytest.raises(PlanError):
        assign_tasks_lpt(tasks, len(tasks) + 1, calibration)


def test_written_shard_manifests_are_individually_runner_valid(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)

    paths = write_shard_manifests(
        EVENT_MANIFEST,
        shards,
        tmp_path,
        results_root_prefix="outputs/protocol_v3_noleak_dedup/remote_runs/probe",
    )

    seen: set[str] = set()
    for gpu, path in paths.items():
        shard = load_baseline_manifest(path, require_complete_matrix=False)
        assert [task.task_id for task in shard.tasks] == [
            task.task_id for task in shards[gpu]
        ]
        assert not seen.intersection(task.task_id for task in shard.tasks)
        seen.update(task.task_id for task in shard.tasks)
    assert len(seen) == 22


def test_microbatch_override_refuses_to_leak_onto_a_shared_shard(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)
    crowded = next(gpu for gpu, shard in shards.items() if len(shard) > 1)

    with pytest.raises(PlanError):
        write_shard_manifests(
            EVENT_MANIFEST,
            shards,
            tmp_path,
            results_root_prefix="outputs/probe",
            microbatch_sizes={shards[crowded][0].task_id: 8},
        )


def test_isolated_override_task_gets_a_dedicated_gpu(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    target = next(
        task.task_id
        for task in tasks
        if task.region == "ContraCosta" and task.model == "d2stgnn"
    )

    shards = assign_tasks_lpt(
        tasks, 4, calibration, microbatch_sizes={target: 8}
    )
    paths = write_shard_manifests(
        EVENT_MANIFEST,
        shards,
        tmp_path,
        results_root_prefix="outputs/probe",
        microbatch_sizes={target: 8},
    )

    solo = next(
        gpu
        for gpu, shard in shards.items()
        if [task.task_id for task in shard] == [target]
    )
    shard = load_baseline_manifest(paths[solo], require_complete_matrix=False)
    assert shard.training["microbatch_size"] == 8
    assigned = [task.task_id for value in shards.values() for task in value]
    assert sorted(assigned) == sorted(task.task_id for task in tasks)
    for gpu, value in shards.items():
        if gpu != solo:
            assert target not in {task.task_id for task in value}


def test_distinct_microbatch_groups_need_their_own_gpus(manifest, calibration) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    sizes = {task.task_id: 2 + index for index, task in enumerate(tasks[:5])}

    with pytest.raises(PlanError):
        assign_tasks_lpt(tasks, 4, calibration, microbatch_sizes=sizes)


def test_same_microbatch_tasks_may_share_a_gpu(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    targets = {
        task.task_id: 16
        for task in tasks
        if task.model in ("d2stgnn", "sttn")
    }

    shards = assign_tasks_lpt(tasks, 4, calibration, microbatch_sizes=targets)
    paths = write_shard_manifests(
        EVENT_MANIFEST,
        shards,
        tmp_path,
        results_root_prefix="outputs/probe",
        microbatch_sizes=targets,
    )

    for gpu, shard in shards.items():
        declared = {task.task_id in targets for task in shard}
        assert len(declared) == 1, f"GPU {gpu} mixes override and plain tasks"
        loaded = load_baseline_manifest(paths[gpu], require_complete_matrix=False)
        if declared == {True}:
            assert loaded.training["microbatch_size"] == 16
        else:
            assert "microbatch_size" not in loaded.training
    assigned = [t.task_id for shard in shards.values() for t in shard]
    assert sorted(assigned) == sorted(t.task_id for t in tasks)


def test_shard_manifest_refuses_to_mix_override_and_plain_tasks(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)
    crowded = next(gpu for gpu, shard in shards.items() if len(shard) > 1)

    with pytest.raises(PlanError, match="one training config"):
        write_shard_manifests(
            EVENT_MANIFEST,
            shards,
            tmp_path,
            results_root_prefix="outputs/probe",
            microbatch_sizes={shards[crowded][0].task_id: 16},
        )


def _write_queue(root: Path, gpu: int, payload: dict) -> None:
    path = root / shard_name(gpu) / "queue_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _aggregate(manifest, shards, root: Path, return_codes) -> dict:
    return aggregate_queue_status(
        manifest=manifest,
        shards=shards,
        results_root=root,
        return_codes=return_codes,
        cache_bundle_id="bundle-test",
        cache_bundle_digest="0" * 64,
        code_sha="a" * 40,
        total_tasks=sum(len(shard) for shard in shards.values()),
    )


def test_aggregate_reports_running_before_children_exit(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)
    for gpu in shards:
        _write_queue(
            tmp_path,
            gpu,
            {
                "state": "running",
                "completed_tasks": 1,
                "current_task": shards[gpu][0].task_id,
            },
        )

    status = _aggregate(manifest, shards, tmp_path, {gpu: None for gpu in shards})

    assert status["state"] == "running"
    assert status["failed_count"] == 0
    assert status["current_task"] is not None


def test_aggregate_completes_only_when_every_shard_completes(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)
    for gpu, shard in shards.items():
        _write_queue(
            tmp_path,
            gpu,
            {
                "state": "completed",
                "completed_tasks": len(shard),
                "skipped_valid_tasks": 0,
                "current_task": None,
            },
        )

    status = _aggregate(manifest, shards, tmp_path, {gpu: 0 for gpu in shards})

    assert status["state"] == "completed"
    assert status["total_tasks"] == 22
    assert status["completed_tasks"] == 22
    assert status["current_task"] is None
    assert status["failed_count"] == 0


def test_aggregate_fails_closed_on_a_single_failed_shard(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)
    for gpu, shard in shards.items():
        _write_queue(
            tmp_path,
            gpu,
            {
                "state": "completed" if gpu else "failed",
                "completed_tasks": len(shard) if gpu else 0,
                "current_task": None,
            },
        )

    status = _aggregate(
        manifest, shards, tmp_path, {gpu: (0 if gpu else 1) for gpu in shards}
    )

    assert status["state"] == "failed"
    assert status["failed_count"] >= 1


def test_aggregate_rejects_a_short_completed_count(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)
    for gpu, shard in shards.items():
        _write_queue(
            tmp_path,
            gpu,
            {
                "state": "completed",
                "completed_tasks": max(len(shard) - 1, 0),
                "current_task": None,
            },
        )

    status = _aggregate(manifest, shards, tmp_path, {gpu: 0 for gpu in shards})

    assert status["state"] == "failed"


def test_nas_plan_matches_the_host_postflight_contract(
    manifest, calibration, tmp_path: Path
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)

    plan = build_nas_plan(
        matrix_id="protocol_v3_event_cc_orange_5090x4_v1",
        host_id="event_rtx5090x4_cc_orange",
        manifest=manifest,
        canonical_manifest=EVENT_MANIFEST,
        canonical_manifest_sha256="f" * 64,
        shards=shards,
        results_root=tmp_path / "results",
        control_root=tmp_path / "control",
        cache_dir=tmp_path / "cache",
        graph_dir=tmp_path / "graphs",
        cache_bundle_id="bundle-test",
        cache_bundle_digest="0" * 64,
        code_sha="a" * 40,
        python_executable="/usr/bin/python3",
    )

    assert plan["plan_version"] == 1
    assert plan["expected_task_count"] == 22
    assert plan["completion_gate"]["expected_tasks"] == 22
    assert plan["completion_gate"]["runner_exit_code"].endswith("runner.exitcode")
    source_ids = [source["source_id"] for source in plan["sources"]]
    assert source_ids == sorted(source_ids)
    assert len(set(source_ids)) == len(source_ids)
    union: list[str] = []
    for source in plan["sources"]:
        assert source["task_ids"] == sorted(
            source["task_ids"],
            key=[task.task_id for task in manifest.tasks].index,
        )
        union.extend(source["task_ids"])
    assert len(set(union)) == 22


def test_calibration_loader_rejects_nonpositive_cost(tmp_path: Path) -> None:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    payload["epoch_seconds"]["dgcrn"] = 0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PlanError):
        load_calibration(path)


def test_task_cost_requires_a_calibrated_model(manifest, tmp_path: Path) -> None:
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    payload["epoch_seconds"].pop("dgcrn")
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    calibration = load_calibration(path)
    task = next(
        task
        for task in manifest.tasks
        if task.model == "dgcrn" and task.region == "Orange"
    )

    with pytest.raises(PlanError):
        task_cost(task, calibration)


def test_plan_digest_ignores_timestamps_but_tracks_decisions(
    manifest, calibration
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    shards = assign_tasks_lpt(tasks, 4, calibration)
    base = {
        "matrix_id": "m",
        "host_id": "h",
        "execution_layout": "single_host_multi_gpu_region_complete",
        "source_manifest_sha256": "a" * 64,
        "calibration_sha256": "b" * 64,
        "code_sha": "c" * 40,
        "cache_bundle_id": "bundle",
        "cache_bundle_digest": "d" * 64,
        "regions": ["ContraCosta", "Orange"],
        "gpu_count": 4,
        "task_count": 22,
        "microbatch_sizes": None,
        "assignments": {
            str(gpu): [task.task_id for task in shard]
            for gpu, shard in shards.items()
        },
        "created_at": "2026-08-07T00:00:00+00:00",
    }
    later = dict(base, created_at="2026-08-08T09:00:00+00:00")
    moved = dict(base, gpu_count=3)

    assert plan_digest(base) == plan_digest(later)
    assert plan_digest(base) != plan_digest(moved)


def test_microbatch_override_must_not_exceed_the_canonical_batch(
    manifest, calibration
) -> None:
    tasks = select_region_tasks(manifest, ["ContraCosta", "Orange"])
    target = tasks[0].task_id
    batch_size = manifest.training["batch_size"]

    assert _parse_microbatch([f"{target}={batch_size}"], tasks, batch_size) == {
        target: batch_size
    }
    with pytest.raises(PlanError):
        _parse_microbatch([f"{target}={batch_size + 1}"], tasks, batch_size)
    with pytest.raises(PlanError):
        _parse_microbatch([f"{target}=0"], tasks, batch_size)


def test_shard_names_sort_lexicographically_like_gpu_index() -> None:
    names = [shard_name(index) for index in range(12)]

    assert names == sorted(names)
    assert names[0] == "gpu00"
    assert names[10] == "gpu10"


def test_launch_script_cannot_be_satisfied_by_the_reviewed_plan() -> None:
    launcher = (
        REPO_ROOT / "scripts/remote/linux/protocol_v3_event_launch_host.sh"
    ).read_text(encoding="utf-8")

    # The reviewed plan is archived so its existence proves nothing about this run.
    assert "plan.reviewed.json" in launcher
    assert "rm -f '$REMOTE_CONTROL/controller.pid'" in launcher
    # The controller must be alive and its regenerated plan must match the review.
    assert "kill -0 \\$(cat '$REMOTE_CONTROL/controller.pid')" in launcher
    assert "--expect-plan-digest" in launcher
    assert 'EXPECTED_PLAN_DIGEST:?' in launcher
    assert '[ "$LIVE_DIGEST" = "$EXPECTED_PLAN_DIGEST" ]' in launcher
    assert '[ "$LIVE_STATE" = "running" ]' in launcher


def test_controller_refuses_to_start_on_a_changed_plan(tmp_path: Path) -> None:
    from scripts.remote.linux import protocol_v3_event_multi_gpu as ctl

    argv = [
        "--control-root", str(tmp_path / "control"),
        "--results-root", str(tmp_path / "results"),
        "--cache-dir", str(tmp_path / "cache"),
        "--graph-dir", str(tmp_path / "graphs"),
        "--cache-bundle-id", "bundle-test",
        "--cache-bundle-digest", "0" * 64,
        "--code-sha", "a" * 40,
        "--matrix-id", "m",
        "--host-id", "h",
        "--regions", "ContraCosta", "Orange",
        "--gpu-count", "4",
        "--plan-only",
    ]

    assert ctl.main(argv) == 0
    reviewed = json.loads(
        (tmp_path / "control" / "plan.json").read_text(encoding="utf-8")
    )["plan_digest"]
    assert ctl.main(argv + ["--expect-plan-digest", reviewed]) == 0
    with pytest.raises(SystemExit):
        ctl.main(argv + ["--expect-plan-digest", "d" * 64])


def test_changing_the_gpu_count_changes_the_plan_digest(tmp_path: Path) -> None:
    from scripts.remote.linux import protocol_v3_event_multi_gpu as ctl

    def digest_for(gpu_count: int, control: Path) -> str:
        ctl.main(
            [
                "--control-root", str(control),
                "--results-root", str(tmp_path / "results"),
                "--cache-dir", str(tmp_path / "cache"),
                "--graph-dir", str(tmp_path / "graphs"),
                "--cache-bundle-id", "bundle-test",
                "--cache-bundle-digest", "0" * 64,
                "--code-sha", "a" * 40,
                "--matrix-id", "m",
                "--host-id", "h",
                "--regions", "ContraCosta", "Orange",
                "--gpu-count", str(gpu_count),
                "--plan-only",
            ]
        )
        return json.loads((control / "plan.json").read_text(encoding="utf-8"))[
            "plan_digest"
        ]

    assert digest_for(3, tmp_path / "c3") != digest_for(4, tmp_path / "c4")


def test_alloc_conf_is_part_of_the_plan_contract(tmp_path: Path) -> None:
    from scripts.remote.linux import protocol_v3_event_multi_gpu as ctl

    def emit(control: Path, extra: list[str]) -> dict:
        ctl.main(
            [
                "--control-root", str(control),
                "--results-root", str(tmp_path / "results"),
                "--cache-dir", str(tmp_path / "cache"),
                "--graph-dir", str(tmp_path / "graphs"),
                "--cache-bundle-id", "bundle-test",
                "--cache-bundle-digest", "0" * 64,
                "--code-sha", "a" * 40,
                "--matrix-id", "m", "--host-id", "h",
                "--regions", "ContraCosta", "Orange",
                "--gpu-count", "4", "--plan-only",
            ]
            + extra
        )
        return json.loads((control / "plan.json").read_text(encoding="utf-8"))

    default = emit(tmp_path / "a", [])
    changed = emit(tmp_path / "b", ["--alloc-conf", "max_split_size_mb:128"])

    assert default["alloc_conf"] == "expandable_segments:True"
    assert changed["alloc_conf"] == "max_split_size_mb:128"
    assert default["plan_digest"] != changed["plan_digest"]


def test_launch_script_binds_the_finalizer_task_count_to_the_live_plan() -> None:
    launcher = (
        REPO_ROOT / "scripts/remote/linux/protocol_v3_event_launch_host.sh"
    ).read_text(encoding="utf-8")

    # A hardcoded task count would arm the finalizer for the wrong matrix size.
    assert "EXPECTED_TASKS=22" not in launcher
    assert 'EXPECTED_TASKS:?' in launcher
    assert '[ "$LIVE_TASKS" = "$EXPECTED_TASKS" ]' in launcher
    assert "EXPECTED_TASKS='$EXPECTED_TASKS'" in launcher
    assert "--alloc-conf '$ALLOC_CONF'" in launcher
