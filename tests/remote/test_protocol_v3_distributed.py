from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.remote.distributed import protocol_v3_distributed as distributed
from scripts.remote.distributed import protocol_v3_shard_postflight as postflight
from scripts.remote.windows import protocol_v3_runner as runner


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MANIFEST = (
    REPO_ROOT / "scripts/remote/windows/protocol_v3_manifest.json"
)


def canonical_manifest() -> runner.Manifest:
    return runner.load_manifest(
        CANONICAL_MANIFEST,
        require_production_matrix=True,
    )


def task_number(task: runner.Task) -> int:
    return int(task.task_id.split("_", 1)[0])


def test_fixed_shards_are_disjoint_complete_and_keep_comparisons_together():
    manifest = canonical_manifest()
    shards = distributed.build_shard_manifests(manifest)
    numbers = {
        name: {task_number(task) for task in shard.tasks}
        for name, shard in shards.items()
    }

    assert numbers == {
        "5080": {
            *range(1, 5),
            *range(7, 11),
            *range(13, 17),
            *range(19, 31),
            *range(40, 43),
        },
        "5090": {
            *range(5, 7),
            *range(11, 13),
            *range(17, 19),
            *range(31, 40),
            *range(43, 46),
        },
    }
    assert numbers["5080"].isdisjoint(numbers["5090"])
    assert numbers["5080"] | numbers["5090"] == set(range(1, 46))

    for comparison in distributed.DIRECT_COMPARISON_GROUPS:
        owners = {
            name
            for name, shard_numbers in numbers.items()
            if shard_numbers.intersection(comparison)
        }
        assert len(owners) == 1


def test_sharding_preserves_canonical_manifest_identity_and_config_hashes(
    tmp_path: Path,
):
    manifest = canonical_manifest()
    shards = distributed.build_shard_manifests(manifest)
    run_inputs = runner.ProtocolV3Inputs(
        results_root=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        graph_dir=tmp_path / "graphs",
        cache_bundle_id="bundle-test",
        cache_bundle_digest="a" * 64,
        code_sha="b" * 40,
        python_executable=sys.executable,
    )

    for shard in shards.values():
        assert shard.source_path == manifest.source_path
        assert shard.digest == manifest.digest
        assert shard.training == manifest.training
        for task in shard.tasks:
            canonical_hash = runner.derive_config_hash(
                task,
                manifest,
                run_inputs,
            )
            shard_hash = runner.derive_config_hash(
                task,
                shard,
                run_inputs,
            )
            assert shard_hash == canonical_hash


def test_shard_validation_rejects_a_complete_but_split_comparison():
    manifest = canonical_manifest()
    shards = distributed.build_shard_manifests(manifest)
    task_by_number = {
        task_number(task): task for task in manifest.tasks
    }
    bad_5080 = tuple(
        task_by_number[5] if task_number(task) == 2 else task
        for task in shards["5080"].tasks
    )
    bad_5090 = tuple(
        task_by_number[2] if task_number(task) == 5 else task
        for task in shards["5090"].tasks
    )

    with pytest.raises(distributed.ShardValidationError, match="comparison"):
        distributed.validate_shard_manifests(
            manifest,
            {
                "5080": replace(shards["5080"], tasks=bad_5080),
                "5090": replace(shards["5090"], tasks=bad_5090),
            },
        )


def test_shard_validation_rejects_complete_comparison_group_exchange():
    manifest = canonical_manifest()
    shards = distributed.build_shard_manifests(manifest)
    exchanged_numbers = {
        "5080": (
            set(distributed.SHARD_SPECS["5080"].task_numbers)
            - {40, 41, 42}
            | {43, 44, 45}
        ),
        "5090": (
            set(distributed.SHARD_SPECS["5090"].task_numbers)
            - {43, 44, 45}
            | {40, 41, 42}
        ),
    }
    exchanged = {
        name: replace(
            shards[name],
            tasks=tuple(
                task
                for task in manifest.tasks
                if task_number(task) in numbers
            ),
        )
        for name, numbers in exchanged_numbers.items()
    }

    with pytest.raises(
        distributed.ShardValidationError,
        match="fixed task assignment",
    ):
        distributed.validate_shard_manifests(manifest, exchanged)


def test_shard_validation_rejects_missing_migrated_task():
    manifest = canonical_manifest()
    shards = distributed.build_shard_manifests(manifest)
    incomplete_5080 = replace(
        shards["5080"],
        tasks=tuple(
            task
            for task in shards["5080"].tasks
            if task_number(task) != 42
        ),
    )

    with pytest.raises(
        distributed.ShardValidationError,
        match=r"missing=\[42\]",
    ):
        distributed.validate_shard_manifests(
            manifest,
            {
                "5080": incomplete_5080,
                "5090": shards["5090"],
            },
        )


@pytest.mark.parametrize(
    ("shard_name", "expected_count", "expected_skips"),
    [
        (
            "5080",
            27,
            {
                *range(1, 5),
                *range(7, 11),
                *range(13, 17),
                *range(19, 31),
            },
        ),
        (
            "5090",
            18,
            {
                *range(5, 7),
                *range(11, 13),
                *range(17, 19),
                *range(31, 40),
            },
        ),
    ],
)
def test_cli_loads_canonical_manifest_and_delegates_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shard_name: str,
    expected_count: int,
    expected_skips: set[int],
):
    captured: dict[str, object] = {}

    def fake_run_queue(
        manifest,
        run_inputs,
        *,
        repo_root,
        verify_git,
    ):
        captured.update(
            manifest=manifest,
            run_inputs=run_inputs,
            repo_root=repo_root,
            verify_git=verify_git,
        )
        return 17

    monkeypatch.setattr(distributed.runner, "run_queue", fake_run_queue)
    results_root = tmp_path / f"results-{shard_name}"
    rc = distributed.main(
        [
            "--shard",
            shard_name,
            "--manifest",
            str(CANONICAL_MANIFEST),
            "--repo-root",
            str(REPO_ROOT),
            "--results-root",
            str(results_root),
            "--cache-bundle-id",
            "bundle-test",
            "--cache-bundle-digest",
            "a" * 64,
            "--code-sha",
            "b" * 40,
            "--python",
            sys.executable,
        ]
    )

    selected = captured["manifest"]
    assert rc == 17
    assert len(selected.tasks) == expected_count
    assert distributed.SHARD_SPECS[shard_name].expected_skip_numbers == (
        expected_skips
    )
    assert captured["run_inputs"].results_root == results_root
    assert captured["repo_root"] == REPO_ROOT
    assert captured["verify_git"] is True


def test_default_result_roots_are_host_specific():
    assert (
        distributed.SHARD_SPECS["5080"].default_results_root
        != distributed.SHARD_SPECS["5090"].default_results_root
    )


def test_standalone_entrypoint_prefers_configured_repo_root(tmp_path: Path):
    shallow_script = tmp_path / "protocol_v3_distributed.py"

    assert distributed._discover_repo_root(
        script_path=shallow_script,
        environ={"XTRAFFIC_REPO_ROOT": str(REPO_ROOT)},
    ) == REPO_ROOT


def test_standalone_entrypoint_reports_clear_repo_discovery_failure(
    tmp_path: Path,
):
    shallow_script = tmp_path / "protocol_v3_distributed.py"

    with pytest.raises(RuntimeError, match="set XTRAFFIC_REPO_ROOT"):
        distributed._discover_repo_root(
            script_path=shallow_script,
            environ={},
        )


def postflight_inputs(tmp_path: Path) -> runner.ProtocolV3Inputs:
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    cache_dir.mkdir()
    graph_dir.mkdir()
    return runner.ProtocolV3Inputs(
        results_root=tmp_path / "results",
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id="bundle-test",
        cache_bundle_digest="a" * 64,
        code_sha="b" * 40,
        python_executable=sys.executable,
    )


def test_postflight_serializes_source_host_resolved_run_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    run_inputs = runner.ProtocolV3Inputs(
        results_root=Path("relative-results"),
        cache_dir=Path("relative-cache"),
        graph_dir=Path("relative-graphs"),
        cache_bundle_id="bundle-source-host",
        cache_bundle_digest="c" * 64,
        code_sha="d" * 40,
        python_executable="source-python",
    )

    assert postflight._serialize_run_inputs(run_inputs) == {
        "results_root": str((tmp_path / "relative-results").resolve()),
        "cache_dir": str((tmp_path / "relative-cache").resolve()),
        "graph_dir": str((tmp_path / "relative-graphs").resolve()),
        "cache_bundle_id": "bundle-source-host",
        "cache_bundle_digest": "c" * 64,
        "code_sha": "d" * 40,
        "python_executable": "source-python",
    }


def write_valid_shard_results(
    manifest: runner.Manifest,
    run_inputs: runner.ProtocolV3Inputs,
    shard_name: str,
    *,
    additional_skipped_numbers: set[int] | None = None,
) -> None:
    results_root = run_inputs.results_root
    results_root.mkdir(parents=True)
    additional_skipped_numbers = additional_skipped_numbers or set()
    expected_skip_ids = {
        task.task_id
        for task in manifest.tasks
        if (
            task_number(task)
            in distributed.SHARD_SPECS[shard_name].expected_skip_numbers
            or task_number(task) in additional_skipped_numbers
        )
    }
    queue = {
        "state": "completed",
        "updated_at": "2026-07-20T12:00:00+00:00",
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "manifest_digest": manifest.digest,
        "total_tasks": len(manifest.tasks),
        "completed_tasks": len(manifest.tasks),
        "skipped_valid_tasks": len(expected_skip_ids),
        "current_task": None,
    }
    (results_root / "queue_status.json").write_text(
        json.dumps(queue),
        encoding="utf-8",
    )

    for task in manifest.tasks:
        invocation = runner.build_production_invocation(
            task,
            manifest,
            run_inputs,
            results_root / ".staging" / task.task_id / "__TEST__",
        )
        config_hash = runner.derive_config_hash(
            task,
            manifest,
            run_inputs,
            invocation_config=invocation.config,
        )
        result_dir = results_root / task.output_dir
        result_dir.mkdir(parents=True)
        summary = {
            "task_id": task.task_id,
            "region": task.region,
            "model": task.model,
            "variant": task.variant,
            "seed": task.seed,
            "protocol_version": manifest.protocol_version,
            "sampling_protocol": task.sampling_protocol,
            "cache_schema_version": manifest.cache_schema_version,
            "cache_bundle_id": run_inputs.cache_bundle_id,
            "cache_bundle_digest": run_inputs.cache_bundle_digest,
            "code_sha": run_inputs.code_sha,
            "config_hash": config_hash,
            "all": 1.25,
            "affected": 1.5,
            "unaffected": 1.2,
            "valid_count": 120,
        }
        (result_dir / "summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        actual = np.full((1, 1, 120), 10.0, dtype=np.float32)
        predicted = actual.copy()
        predicted[..., :20] += 1.5
        predicted[..., 20:] += 1.2
        np.savez_compressed(
            result_dir / "test_predictions.npz",
            region_code=np.array(
                [
                    {
                        "Alameda": 0,
                        "ContraCosta": 1,
                        "Orange": 2,
                    }[task.region]
                ],
                dtype=np.int64,
            ),
            sample_start=np.array([100], dtype=np.int64),
            region_node_idx=np.arange(120, dtype=np.int64),
            pred_raw_flow=predicted,
            actual_future_flow=actual,
            y_mask_flow=np.ones_like(actual, dtype=np.bool_),
            affected_mask=np.concatenate(
                [
                    np.ones((1, 20), dtype=np.bool_),
                    np.zeros((1, 100), dtype=np.bool_),
                ],
                axis=1,
            ),
        )
        for artifact in invocation.required_artifacts:
            path = result_dir / artifact
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"artifact:{artifact}\n".encode())

        task_state = {
            "state": (
                "skipped_valid"
                if task.task_id in expected_skip_ids
                else "completed"
            ),
            "task_id": task.task_id,
            "config_hash": config_hash,
        }
        state_path = (
            results_root / "state/tasks" / f"{task.task_id}.json"
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(task_state), encoding="utf-8")


def patch_postflight_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postflight,
        "verify_environment",
        lambda repo_root, run_inputs: {
            "git": {"head": run_inputs.code_sha, "clean": True},
            "cache": {"status": "verified"},
        },
    )
    monkeypatch.setattr(
        postflight,
        "runtime_provenance",
        lambda run_inputs: {
            "hostname": "test-host",
            "python_executable": run_inputs.python_executable,
        },
    )


def valid_postflight_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shard_name: str = "5080",
):
    canonical = canonical_manifest()
    manifest = distributed.build_shard_manifests(canonical)[shard_name]
    run_inputs = postflight_inputs(tmp_path)
    write_valid_shard_results(manifest, run_inputs, shard_name)
    patch_postflight_environment(monkeypatch)
    return canonical, manifest, run_inputs


def test_postflight_validates_every_result_and_emits_per_file_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )

    report = postflight.build_postflight_report(
        shard_name="5080",
        canonical=canonical,
        manifest=manifest,
        repo_root=REPO_ROOT,
        run_inputs=run_inputs,
    )

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 27
    assert report["expected_skipped_valid_tasks"] == 24
    assert report["minimum_skipped_valid_tasks"] == 24
    assert report["actual_skipped_valid_tasks"] == 24
    assert report["task_state_counts"] == {
        "completed": 3,
        "skipped_valid": 24,
    }
    assert report["runtime"]["hostname"] == "test-host"
    assert report["run_inputs"] == {
        "results_root": str(run_inputs.results_root.resolve()),
        "cache_dir": str(run_inputs.cache_dir.resolve()),
        "graph_dir": str(run_inputs.graph_dir.resolve()),
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "python_executable": run_inputs.python_executable,
    }
    assert len(report["files"]) > 27
    summary_path = (
        run_inputs.results_root
        / manifest.tasks[0].output_dir
        / "summary.json"
    )
    summary_relpath = summary_path.relative_to(
        run_inputs.results_root
    ).as_posix()
    summary_record = next(
        item for item in report["files"] if item["path"] == summary_relpath
    )
    assert summary_record["sha256"] == hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    task_by_id = {task.task_id: task for task in manifest.tasks}
    for task_report in report["tasks"]:
        task = task_by_id[task_report["task_id"]]
        expected_invocation = runner.build_production_invocation(
            task,
            manifest,
            run_inputs,
            run_inputs.results_root
            / ".staging"
            / task.task_id
            / "__EXPECTED__",
        )
        assert task_report["invocation_config"] == dict(
            expected_invocation.config
        )
        assert runner.derive_config_hash(
            task,
            manifest,
            run_inputs,
            invocation_config=task_report["invocation_config"],
        ) == task_report["config_hash"]


@pytest.mark.parametrize(
    ("shard_name", "new_task_numbers"),
    [
        ("5080", (40, 41, 42)),
        ("5090", (43, 44, 45)),
    ],
)
@pytest.mark.parametrize("additional_skip_count", range(4))
def test_postflight_accepts_every_safe_restart_skip_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shard_name: str,
    new_task_numbers: tuple[int, ...],
    additional_skip_count: int,
):
    canonical = canonical_manifest()
    manifest = distributed.build_shard_manifests(canonical)[shard_name]
    run_inputs = postflight_inputs(tmp_path)
    additional_skipped_numbers = set(
        new_task_numbers[:additional_skip_count]
    )
    write_valid_shard_results(
        manifest,
        run_inputs,
        shard_name,
        additional_skipped_numbers=additional_skipped_numbers,
    )
    patch_postflight_environment(monkeypatch)

    report = postflight.build_postflight_report(
        shard_name=shard_name,
        canonical=canonical,
        manifest=manifest,
        repo_root=REPO_ROOT,
        run_inputs=run_inputs,
    )

    expected_inherited = set(
        distributed.SHARD_SPECS[shard_name].expected_skip_numbers
    )
    expected_skipped = expected_inherited | additional_skipped_numbers
    expected_skip_count = len(expected_skipped)
    assert report["expected_skipped_valid_tasks"] == len(
        expected_inherited
    )
    assert report["minimum_skipped_valid_tasks"] == len(
        expected_inherited
    )
    assert report["actual_skipped_valid_tasks"] == expected_skip_count
    assert report["task_state_counts"] == {
        "completed": len(manifest.tasks) - expected_skip_count,
        "skipped_valid": expected_skip_count,
    }
    assert {
        int(task_id.split("_", 1)[0])
        for task_id in report["actual_skipped_task_ids"]
    } == expected_skipped


def test_postflight_attestation_is_stable_for_an_unchanged_completed_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )

    first = postflight.build_postflight_report(
        shard_name="5080",
        canonical=canonical,
        manifest=manifest,
        repo_root=REPO_ROOT,
        run_inputs=run_inputs,
    )
    second = postflight.build_postflight_report(
        shard_name="5080",
        canonical=canonical,
        manifest=manifest,
        repo_root=REPO_ROOT,
        run_inputs=run_inputs,
    )

    assert first == second
    assert first["verified_at"] == first["queue_status"]["updated_at"]
    assert first["source_completed_at"] == first["verified_at"]


def test_postflight_rejects_restart_that_loses_inherited_skip_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )
    inherited = manifest.tasks[0]
    replacement = next(
        task
        for task in manifest.tasks
        if task_number(task)
        not in distributed.SHARD_SPECS["5080"].expected_skip_numbers
    )
    inherited_state_path = (
        run_inputs.results_root
        / "state/tasks"
        / f"{inherited.task_id}.json"
    )
    replacement_state_path = (
        run_inputs.results_root
        / "state/tasks"
        / f"{replacement.task_id}.json"
    )
    inherited_state = json.loads(
        inherited_state_path.read_text(encoding="utf-8")
    )
    replacement_state = json.loads(
        replacement_state_path.read_text(encoding="utf-8")
    )
    inherited_state["state"] = "completed"
    replacement_state["state"] = "skipped_valid"
    inherited_state_path.write_text(
        json.dumps(inherited_state),
        encoding="utf-8",
    )
    replacement_state_path.write_text(
        json.dumps(replacement_state),
        encoding="utf-8",
    )

    with pytest.raises(
        postflight.PostflightError,
        match="inherited skipped task",
    ):
        postflight.build_postflight_report(
            shard_name="5080",
            canonical=canonical,
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_inputs=run_inputs,
        )


def test_postflight_rejects_queue_skip_count_that_differs_from_task_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )
    queue_path = run_inputs.results_root / "queue_status.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["skipped_valid_tasks"] = 25
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    with pytest.raises(
        postflight.PostflightError,
        match="differs from task states",
    ):
        postflight.build_postflight_report(
            shard_name="5080",
            canonical=canonical,
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_inputs=run_inputs,
        )


def test_postflight_rejects_missing_assigned_task_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )
    missing = run_inputs.results_root / manifest.tasks[-1].output_dir
    for path in sorted(missing.rglob("*"), reverse=True):
        path.unlink()
    missing.rmdir()

    with pytest.raises(postflight.PostflightError, match="task directory"):
        postflight.build_postflight_report(
            shard_name="5080",
            canonical=canonical,
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_inputs=run_inputs,
        )


def test_postflight_rejects_wrong_completed_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )
    queue_path = run_inputs.results_root / "queue_status.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["completed_tasks"] = 26
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    with pytest.raises(postflight.PostflightError, match="completed_tasks"):
        postflight.build_postflight_report(
            shard_name="5080",
            canonical=canonical,
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_inputs=run_inputs,
        )


def test_postflight_rejects_summary_that_fails_runner_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )
    summary_path = (
        run_inputs.results_root
        / manifest.tasks[0].output_dir
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["region"] = "WrongRegion"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(runner.ResultValidationError, match="region"):
        postflight.build_postflight_report(
            shard_name="5080",
            canonical=canonical,
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_inputs=run_inputs,
        )


def test_postflight_hash_manifest_rejects_symbolic_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )
    first_dir = run_inputs.results_root / manifest.tasks[0].output_dir
    (first_dir / "linked-summary.json").symlink_to(
        first_dir / "summary.json"
    )

    with pytest.raises(postflight.PostflightError, match="symbolic link"):
        postflight.build_postflight_report(
            shard_name="5080",
            canonical=canonical,
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_inputs=run_inputs,
        )


def test_postflight_export_allowlist_excludes_staging_and_unassigned_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    canonical, manifest, run_inputs = valid_postflight_case(
        tmp_path,
        monkeypatch,
    )
    results_root = run_inputs.results_root
    unassigned_id = "005_event_staeformer_s0a_orange_seed42"
    stale_paths = (
        results_root
        / ".staging"
        / unassigned_id
        / "attempt_0001"
        / "partial.pt",
        results_root / "logs" / unassigned_id / "attempt_0001.log",
        results_root / "state/tasks" / f"{unassigned_id}.json",
    )
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("historical partial", encoding="utf-8")

    report = postflight.build_postflight_report(
        shard_name="5080",
        canonical=canonical,
        manifest=manifest,
        repo_root=REPO_ROOT,
        run_inputs=run_inputs,
    )
    exported = set(report["export_paths"])

    assert exported == {record["path"] for record in report["files"]}
    assert not any(path.startswith(".staging/") for path in exported)
    assert not any(unassigned_id in path for path in exported)
    assert "queue_status.json" in exported
