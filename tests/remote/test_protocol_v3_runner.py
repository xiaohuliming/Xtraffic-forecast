from __future__ import annotations

import gc
import io
import hashlib
import json
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

from scripts.remote.windows import protocol_v3_runner as runner_module
from scripts.remote.windows.protocol_v3_runner import (
    Invocation,
    ManifestError,
    ProtocolV3Inputs,
    ResultValidationError,
    build_production_invocation,
    derive_config_hash,
    load_manifest,
    _require_prediction_metric,
    run_queue,
    validate_result,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_MANIFEST = (
    REPO_ROOT / "scripts" / "remote" / "windows" / "protocol_v3_manifest.json"
)
PRODUCTION_LAUNCHER = (
    REPO_ROOT / "scripts" / "remote" / "windows" / "protocol_v3_launch.bat"
)
DUMMY_TRAINER = Path(__file__).with_name("dummy_trainer.py")
CODE_SHA = "a" * 40


def production_manifest():
    return load_manifest(PRODUCTION_MANIFEST, require_production_matrix=True)


def mini_manifest(
    tmp_path: Path,
    *,
    kind: str = "staeformer_deseason",
    model: str = "staeformer_deseason",
    sampling_protocol: str = "event",
):
    task_id = (
        f"001_{sampling_protocol}_{model}_s0a_alameda_seed42"
    )
    payload = {
        "manifest_version": 1,
        "protocol_version": "protocol_v3_noleak_dedup",
        "cache_schema_version": 3,
        "default_results_root": "outputs/protocol_v3_noleak_dedup/test_runner",
        "default_cache_dir": "outputs/dist_net/region_data_v3_noleak_dedup",
        "default_graph_dir": "outputs/region_graphs",
        "training": {"epochs": 30, "patience": 6, "device": "cuda:0"},
        "tasks": [
            {
                "task_id": task_id,
                "kind": kind,
                "region": "Alameda",
                "model": model,
                "variant": "s0a",
                "seed": 42,
                "sampling_protocol": sampling_protocol,
                "output_dir": f"tasks/{task_id}",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_manifest(path, require_production_matrix=False)


def inputs(tmp_path: Path) -> ProtocolV3Inputs:
    cache_dir = tmp_path / "cache"
    graph_dir = tmp_path / "graphs"
    cache_dir.mkdir()
    graph_dir.mkdir()
    graph_keys = {
        "Alameda": "alameda",
        "ContraCosta": "contra_costa",
        "Orange": "orange",
    }
    region_digests = {}
    regions = {}
    for region, graph_key in graph_keys.items():
        traffic_cache_id = f"{region.lower()}-traffic"
        samples_cache_id = f"{region.lower()}-samples"
        relation_cache_id = f"{region.lower()}-relation"
        paths = {
            "traffic": cache_dir / f"{region}_traffic.h5",
            "samples": cache_dir / f"{region}_samples.h5",
            "relation": cache_dir / f"{region}_event_rel_feat.npz",
            "graph": graph_dir / f"{graph_key}_sparse_adj.npz",
        }
        artifact_sha256 = {}
        for role, path in paths.items():
            path.write_bytes(f"{region}:{role}\n".encode())
            artifact_sha256[role] = hashlib.sha256(path.read_bytes()).hexdigest()
        region_digest = hashlib.sha256(
            json.dumps(
                {
                    "protocol_version": "protocol_v3_noleak_dedup",
                    "cache_schema_version": 3,
                    "bundle_id": "bundle-v3-test",
                    "traffic_cache_id": traffic_cache_id,
                    "samples_cache_id": samples_cache_id,
                    "relation_cache_id": relation_cache_id,
                    "artifact_sha256": artifact_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        region_digests[region] = region_digest
        regions[region] = {
            "cache_bundle_digest": region_digest,
            "cache_provenance": {
                "bundle_id": "bundle-v3-test",
                "traffic_cache_id": traffic_cache_id,
                "samples_cache_id": samples_cache_id,
                "relation_cache_id": relation_cache_id,
                "artifact_sha256": artifact_sha256,
            },
        }
    bundle_id = "bundle-v3-test"
    bundle_digest = hashlib.sha256(
        json.dumps(
            {
                "protocol_version": "protocol_v3_noleak_dedup",
                "cache_schema_version": 3,
                "bundle_id": bundle_id,
                "regions": {
                    region: region_digests[region]
                    for region in sorted(region_digests)
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    (cache_dir / "cache_bundle_manifest.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "protocol_version": "protocol_v3_noleak_dedup",
                "cache_schema_version": 3,
                "bundle_id": bundle_id,
                "cache_bundle_id": bundle_id,
                "cache_bundle_digest": bundle_digest,
                "regions": regions,
            }
        ),
        encoding="utf-8",
    )
    return ProtocolV3Inputs(
        results_root=tmp_path / "results",
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=bundle_id,
        cache_bundle_digest=bundle_digest,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
    )


def dummy_builder(
    mode: str = "success",
    *,
    required_metrics=("all", "affected", "unaffected"),
):
    def build(task, manifest, run_inputs, attempt_dir):
        result_relpath = Path("work") / "result"
        result_dir = attempt_dir / result_relpath
        artifacts = ("summary.json", "ckpt_best.pt", "test_predictions.npz")
        command = (
            run_inputs.python_executable,
            str(DUMMY_TRAINER),
            "--out",
            str(result_dir),
            "--mode",
            mode,
            "--artifacts",
            ",".join(artifacts[1:]),
        )
        return Invocation(
            command=command,
            result_relpath=result_relpath,
            required_artifacts=artifacts,
            required_metrics=required_metrics,
            config={"trainer": "dummy"},
        )

    return build


def prepare_recoverable_staging(manifest, run_inputs):
    task = manifest.tasks[0]
    assert run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder("wrong_prediction_metric"),
        verify_git=False,
        stdout=io.StringIO(),
    ) == 4
    attempt_dir = (
        run_inputs.results_root
        / ".staging"
        / task.task_id
        / "attempt_0001"
    )
    staged_result = attempt_dir / "work" / "result"
    summary_path = staged_result / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["all"] = 1.25
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state_path = (
        run_inputs.results_root
        / "state"
        / "tasks"
        / f"{task.task_id}.json"
    )
    return task, attempt_dir, staged_result, state_path


def test_production_manifest_is_exact_unique_45_task_matrix():
    manifest = production_manifest()
    tasks = manifest.tasks

    assert len(tasks) == 45
    assert len({task.task_id for task in tasks}) == 45
    assert len({task.output_dir.as_posix().casefold() for task in tasks}) == 45

    counts = {}
    for task in tasks:
        key = (task.kind, task.sampling_protocol)
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("staeformer_deseason", "event"): 30,
        ("staeformer_icsf", "event"): 6,
        ("graphwavenet", "full_window"): 3,
        ("fourier_dual_net", "full_window"): 3,
        ("staeformer", "full_window"): 3,
    }

    deseason = {
        (task.region, task.variant, task.seed)
        for task in tasks
        if task.kind == "staeformer_deseason"
    }
    expected = {
        (region, variant, seed)
        for region in ("Alameda", "ContraCosta", "Orange")
        for variant in ("s0a", "s0c")
        for seed in (42, 1, 2)
    }
    expected |= {
        (region, variant, seed)
        for region in ("Alameda", "ContraCosta")
        for variant in ("s0f", "s0g")
        for seed in (42, 1, 2)
    }
    assert deseason == expected

    icsf = {
        (task.region, task.variant, task.seed)
        for task in tasks
        if task.kind == "staeformer_icsf"
    }
    assert icsf == {
        (region, variant, 42)
        for region in ("Alameda", "ContraCosta", "Orange")
        for variant in ("base", "icsf")
    }


def test_config_hash_is_stable_and_unique_for_production_tasks(tmp_path):
    manifest = production_manifest()
    run_inputs = inputs(tmp_path)
    hashes_first = [
        derive_config_hash(task, manifest, run_inputs)
        for task in manifest.tasks
    ]
    hashes_second = [
        derive_config_hash(task, manifest, run_inputs)
        for task in manifest.tasks
    ]

    assert hashes_first == hashes_second
    assert len(set(hashes_first)) == 45
    assert all(len(value) == 64 for value in hashes_first)


def test_every_production_task_requires_predictions_and_model_artifacts(tmp_path):
    manifest = production_manifest()
    run_inputs = inputs(tmp_path)

    for task in manifest.tasks:
        invocation = build_production_invocation(
            task,
            manifest,
            run_inputs,
            tmp_path / "attempt",
        )
        required = set(invocation.required_artifacts)
        assert {"summary.json", "ckpt_best.pt", "test_predictions.npz"} <= required
        if task.kind in {"graphwavenet", "fourier_dual_net"}:
            assert "train.log" in required
        if task.kind == "fourier_dual_net":
            assert "decomp_mask.json" in required
        if task.kind == "staeformer_deseason" and task.variant in {"s0f", "s0g"}:
            assert "gate_diag.npz" in required


def test_float32_metric_contract_accepts_observed_task040_reduction_delta():
    _require_prediction_metric(
        {"all": 11.305081367492676},
        "all",
        11.305081367492676,
        audit_recomputed=11.305101714317631,
        valid_count=103270922,
        producer_dtype=np.dtype(np.float32),
        allow_producer_fallback=True,
    )


def test_float32_metric_contract_rejects_real_metric_mismatch():
    with pytest.raises(ResultValidationError, match="tolerance="):
        _require_prediction_metric(
            {"all": 11.305121367492676},
            "all",
            11.305081367492676,
            audit_recomputed=11.305101714317631,
            valid_count=103270922,
            producer_dtype=np.dtype(np.float32),
            allow_producer_fallback=True,
        )


def test_float32_producer_error_reuses_prediction_buffer_exactly():
    rng = np.random.default_rng(42)
    prediction = rng.normal(size=(64, 12, 31)).astype(np.float32)
    actual = rng.normal(size=prediction.shape).astype(np.float32)
    prediction[0, 0, 0] = np.nan
    expected = np.abs(prediction.copy() - actual)
    selected = rng.random(size=prediction.shape) > 0.2
    selected[0, 0, 0] = False

    result = runner_module._float32_absolute_error_in_place(
        prediction,
        actual,
    )

    assert result is prediction
    assert np.shares_memory(result, prediction)
    assert np.array_equal(result, expected, equal_nan=True)
    assert float(result[selected].mean()) == float(expected[selected].mean())


def test_low_precision_producer_metric_is_validated_through_runner(
    tmp_path,
    monkeypatch,
):
    manifest = mini_manifest(
        tmp_path,
        kind="graphwavenet",
        model="graphwavenet",
    )
    run_inputs = inputs(tmp_path)
    calls = 0
    actual_refs = []
    release_checks = []
    original = runner_module._float32_absolute_error_in_place
    original_require = runner_module._require_prediction_metric

    def tracked(prediction, actual):
        nonlocal calls
        calls += 1
        actual_refs.append(weakref.ref(actual))
        return original(prediction, actual)

    def require_after_actual_release(*args, **kwargs):
        gc.collect()
        release_checks.append(actual_refs[-1]() is None)
        return original_require(*args, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "_float32_absolute_error_in_place",
        tracked,
    )
    monkeypatch.setattr(
        runner_module,
        "_require_prediction_metric",
        require_after_actual_release,
    )

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder("low_precision_metric"),
        verify_git=False,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert calls == 2
    assert release_checks
    assert all(release_checks)


def test_full_window_validation_skips_event_group_reductions(
    tmp_path,
    monkeypatch,
):
    manifest = mini_manifest(
        tmp_path,
        kind="graphwavenet",
        model="graphwavenet",
        sampling_protocol="full_window",
    )
    run_inputs = inputs(tmp_path)

    def reject_broadcast(*args, **kwargs):
        raise AssertionError("full-window validation built event groups")

    monkeypatch.setattr(runner_module.np, "broadcast_to", reject_broadcast)

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(
            "low_precision_metric",
            required_metrics=("all",),
        ),
        verify_git=False,
        stdout=io.StringIO(),
    )

    assert rc == 0


def test_success_runs_in_staging_promotes_and_writes_completed_status(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    stream = io.StringIO()

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=stream,
    )

    task = manifest.tasks[0]
    final_dir = run_inputs.results_root / task.output_dir
    assert rc == 0
    assert (final_dir / "summary.json").is_file()
    assert (final_dir / "ckpt_best.pt").is_file()
    assert (final_dir / "test_predictions.npz").is_file()
    summary = json.loads((final_dir / "summary.json").read_text())
    assert Path(summary["seen_staging_root"]).name == "attempt_0001"
    assert json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )["state"] == "completed"
    assert "ALL_DONE" in stream.getvalue()
    logs = list((run_inputs.results_root / "logs" / task.task_id).glob("attempt_*.log"))
    assert [path.name for path in logs] == ["attempt_0001.log"]
    staging_attempts = list(
        (run_inputs.results_root / ".staging" / task.task_id).glob("attempt_*")
    )
    assert [path.name for path in staging_attempts] == ["attempt_0001"]


def test_zero_exit_staging_is_recovered_without_retraining(
    tmp_path,
    monkeypatch,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task = manifest.tasks[0]

    assert run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder("wrong_prediction_metric"),
        verify_git=False,
        stdout=io.StringIO(),
    ) == 4

    attempt_dir = (
        run_inputs.results_root
        / ".staging"
        / task.task_id
        / "attempt_0001"
    )
    staged_result = attempt_dir / "work" / "result"
    summary_path = staged_result / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["all"] = 1.25
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifact_names = (
        "summary.json",
        "ckpt_best.pt",
        "test_predictions.npz",
    )
    before_hashes = {
        name: hashlib.sha256((staged_result / name).read_bytes()).hexdigest()
        for name in artifact_names
    }
    failed_state_path = (
        run_inputs.results_root
        / "state"
        / "tasks"
        / f"{task.task_id}.json"
    )
    failed_state_sha256 = hashlib.sha256(
        failed_state_path.read_bytes()
    ).hexdigest()

    def forbid_subprocess(*args, **kwargs):
        raise AssertionError("staging recovery must not launch a trainer")

    monkeypatch.setattr(subprocess, "Popen", forbid_subprocess)
    stream = io.StringIO()
    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=stream,
        recover_staging_task=task.task_id,
    )

    final_dir = run_inputs.results_root / task.output_dir
    after_hashes = {
        name: hashlib.sha256((final_dir / name).read_bytes()).hexdigest()
        for name in artifact_names
    }
    task_state = json.loads(failed_state_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert before_hashes == after_hashes
    assert task_state["state"] == "completed"
    assert task_state["attempt"] == 1
    assert task_state["recovered_from_attempt"] == 1
    assert (
        task_state["recovered_failure_state_sha256"]
        == failed_state_sha256
    )
    assert task_state["recovered_artifact_sha256"] == before_hashes
    assert "RECOVER_STAGING" in stream.getvalue()
    assert "attempt_0002" not in stream.getvalue()
    assert not (
        run_inputs.results_root
        / ".staging"
        / task.task_id
        / "attempt_0002"
    ).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("returncode", False),
        ("returncode", 7),
        ("attempt", 2),
        ("state", "completed"),
        ("task_id", "999_wrong_task"),
        ("config_hash", "0" * 64),
    ],
)
def test_recovery_rejects_untrusted_failed_state(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task, _, _, state_path = prepare_recoverable_staging(
        manifest,
        run_inputs,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def forbid_subprocess(*args, **kwargs):
        raise AssertionError("failed recovery must not launch a trainer")

    monkeypatch.setattr(subprocess, "Popen", forbid_subprocess)
    stream = io.StringIO()
    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=stream,
        recover_staging_task=task.task_id,
    )

    assert rc == 4
    assert "RECOVERY_FAILED" in stream.getvalue()
    assert not (run_inputs.results_root / task.output_dir).exists()
    assert not (
        run_inputs.results_root
        / ".staging"
        / task.task_id
        / "attempt_0002"
    ).exists()


@pytest.mark.parametrize("field", ["staging_dir", "log_path"])
def test_recovery_rejects_state_path_mismatch(
    tmp_path,
    monkeypatch,
    field,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task, attempt_dir, _, state_path = prepare_recoverable_staging(
        manifest,
        run_inputs,
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = str(attempt_dir.parent / "wrong_path")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail(
            "failed recovery must not launch a trainer"
        ),
    )

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
        recover_staging_task=task.task_id,
    )

    assert rc == 4
    assert not (run_inputs.results_root / task.output_dir).exists()


def test_recovery_rejects_incomplete_staging_without_retraining(
    tmp_path,
    monkeypatch,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task, _, staged_result, _ = prepare_recoverable_staging(
        manifest,
        run_inputs,
    )
    (staged_result / "ckpt_best.pt").unlink()
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail(
            "failed recovery must not launch a trainer"
        ),
    )

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
        recover_staging_task=task.task_id,
    )

    assert rc == 4
    assert not (run_inputs.results_root / task.output_dir).exists()
    assert not (
        run_inputs.results_root
        / ".staging"
        / task.task_id
        / "attempt_0002"
    ).exists()


def test_restart_after_recovery_keeps_valid_final_without_overwrite(
    tmp_path,
    monkeypatch,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task, _, _, _ = prepare_recoverable_staging(manifest, run_inputs)
    assert run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
        recover_staging_task=task.task_id,
    ) == 0
    final_dir = run_inputs.results_root / task.output_dir
    task_state_path = (
        run_inputs.results_root
        / "state"
        / "tasks"
        / f"{task.task_id}.json"
    )
    task_state_before = task_state_path.read_bytes()
    before = {
        path.relative_to(final_dir).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in final_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail(
            "a valid recovered result must not relaunch"
        ),
    )
    stream = io.StringIO()

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder("fail"),
        verify_git=False,
        stdout=stream,
        recover_staging_task=task.task_id,
    )

    after = {
        path.relative_to(final_dir).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in final_dir.rglob("*")
        if path.is_file()
    }
    queue = json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )
    assert rc == 0
    assert before == after
    assert task_state_path.read_bytes() == task_state_before
    assert queue["skipped_valid_tasks"] == 0
    assert "KEEP_COMPLETED" in stream.getvalue()
    assert not (
        run_inputs.results_root
        / ".staging"
        / task.task_id
        / "attempt_0002"
    ).exists()


def test_valid_completed_result_is_kept_without_new_attempt(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    first = io.StringIO()
    second = io.StringIO()

    assert run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=first,
    ) == 0
    assert run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(mode="fail"),
        verify_git=False,
        stdout=second,
    ) == 0

    task = manifest.tasks[0]
    logs = list((run_inputs.results_root / "logs" / task.task_id).glob("attempt_*.log"))
    assert [path.name for path in logs] == ["attempt_0001.log"]
    assert "KEEP_COMPLETED" in second.getvalue()
    assert "ALL_DONE" in second.getvalue()


def test_valid_result_with_untrusted_completed_state_falls_back_to_skip(
    tmp_path,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    assert run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
    ) == 0
    task = manifest.tasks[0]
    state_path = (
        run_inputs.results_root
        / "state"
        / "tasks"
        / f"{task.task_id}.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["final_dir"] = str(run_inputs.results_root / "wrong")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    stream = io.StringIO()

    assert run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder("fail"),
        verify_git=False,
        stdout=stream,
    ) == 0

    rewritten = json.loads(state_path.read_text(encoding="utf-8"))
    queue = json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )
    assert rewritten["state"] == "skipped_valid"
    assert queue["skipped_valid_tasks"] == 1
    assert "SKIP_VALID" in stream.getvalue()


def test_nonempty_invalid_final_directory_is_rejected_before_launch(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task = manifest.tasks[0]
    final_dir = run_inputs.results_root / task.output_dir
    final_dir.mkdir(parents=True)
    marker = final_dir / "old.txt"
    marker.write_text("must remain", encoding="utf-8")
    stream = io.StringIO()

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=stream,
    )

    assert rc != 0
    assert marker.read_text(encoding="utf-8") == "must remain"
    assert not (run_inputs.results_root / "logs" / task.task_id).exists()
    assert "ALL_DONE" not in stream.getvalue()
    status = json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )
    assert status["state"] == "failed"


def test_results_root_rejects_unowned_content_without_deleting_it(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    run_inputs.results_root.mkdir(parents=True)
    foreign = run_inputs.results_root / "foreign_result.txt"
    foreign.write_text("preserve me", encoding="utf-8")

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
    )

    assert rc != 0
    assert foreign.read_text(encoding="utf-8") == "preserve me"
    assert not (run_inputs.results_root / "logs").exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("bad_identity", "region"),
        ("nan_metric", "finite"),
        ("zero_valid", "valid_count"),
        ("wrong_valid_count", "prediction valid_count"),
        ("wrong_prediction_metric", "prediction metric"),
        ("bad_prediction_schema", "prediction artifact"),
        ("nonfinite_masked_prediction", "non-finite prediction"),
        ("missing_artifact", "artifact"),
    ],
)
def test_invalid_subprocess_result_fails_closed(
    tmp_path,
    mode,
    message,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    stream = io.StringIO()

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(mode),
        verify_git=False,
        stdout=stream,
    )

    task = manifest.tasks[0]
    assert rc != 0
    assert not (run_inputs.results_root / task.output_dir).exists()
    assert "ALL_DONE" not in stream.getvalue()
    status = json.loads(
        (
            run_inputs.results_root
            / "state"
            / "tasks"
            / f"{task.task_id}.json"
        ).read_text()
    )
    assert status["state"] == "failed"
    assert message in status["error"]


def test_preflight_rejects_cache_manifest_identity_before_launch(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    manifest_path = run_inputs.cache_dir / "cache_bundle_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["cache_bundle_id"] = "wrong-bundle"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
    )

    assert rc == 2
    assert not (run_inputs.results_root / "logs").exists()
    queue = json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )
    assert queue["state"] == "failed"
    assert "cache_bundle_id" in queue["error"]


def test_preflight_rejects_missing_region_artifact_before_launch(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    (run_inputs.cache_dir / "Orange_samples.h5").unlink()

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
    )

    assert rc == 2
    assert not (run_inputs.results_root / "logs").exists()
    queue = json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )
    assert "Orange" in queue["error"]
    assert "samples" in queue["error"]


def test_preflight_rejects_region_provenance_digest_mismatch(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    manifest_path = run_inputs.cache_dir / "cache_bundle_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["regions"]["Alameda"]["cache_provenance"][
        "traffic_cache_id"
    ] = "tampered-id"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
    )

    assert rc == 2
    assert not (run_inputs.results_root / "logs").exists()
    queue = json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )
    assert "Alameda" in queue["error"]
    assert "digest" in queue["error"]


def test_failed_attempts_are_numbered_and_logs_are_never_overwritten(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)

    for _ in range(2):
        assert run_queue(
            manifest,
            run_inputs,
            repo_root=REPO_ROOT,
            command_builder=dummy_builder("fail"),
            verify_git=False,
            stdout=io.StringIO(),
        ) != 0

    task = manifest.tasks[0]
    log_dir = run_inputs.results_root / "logs" / task.task_id
    assert sorted(path.name for path in log_dir.glob("attempt_*.log")) == [
        "attempt_0001.log",
        "attempt_0002.log",
    ]
    assert "intentional failure" in (log_dir / "attempt_0001.log").read_text()
    assert "intentional failure" in (log_dir / "attempt_0002.log").read_text()


def test_keyboard_interrupt_marks_task_and_queue_interrupted(
    tmp_path,
    monkeypatch,
):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    calls = {"terminate": 0, "kill": 0}

    class InterruptingProcess:
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            return 0

        def terminate(self):
            calls["terminate"] += 1
            self.returncode = -15

        def kill(self):
            calls["kill"] += 1
            self.returncode = -9

    monkeypatch.setattr(subprocess, "Popen", InterruptingProcess)

    rc = run_queue(
        manifest,
        run_inputs,
        repo_root=REPO_ROOT,
        command_builder=dummy_builder(),
        verify_git=False,
        stdout=io.StringIO(),
    )

    task = manifest.tasks[0]
    task_status = json.loads(
        (
            run_inputs.results_root
            / "state"
            / "tasks"
            / f"{task.task_id}.json"
        ).read_text()
    )
    queue_status = json.loads(
        (run_inputs.results_root / "queue_status.json").read_text()
    )
    assert rc == 130
    assert task_status["state"] == "interrupted"
    assert queue_status["state"] == "interrupted"
    assert calls["terminate"] == 1


def test_validate_result_rejects_wrong_config_hash(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task = manifest.tasks[0]
    result = tmp_path / "result"
    result.mkdir()
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
        "config_hash": "0" * 64,
        "all": 1.0,
        "valid_count": 1,
    }
    (result / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ResultValidationError, match="config_hash"):
        validate_result(
            result,
            task,
            manifest,
            run_inputs,
            expected_config_hash=derive_config_hash(task, manifest, run_inputs),
            required_artifacts=("summary.json",),
            required_metrics=("all",),
        )


def test_validate_result_rejects_duplicate_summary_keys(tmp_path):
    manifest = mini_manifest(tmp_path)
    run_inputs = inputs(tmp_path)
    task = manifest.tasks[0]
    result = tmp_path / "result"
    result.mkdir()
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
        "config_hash": derive_config_hash(task, manifest, run_inputs),
        "all": 1.0,
        "valid_count": 1,
    }
    serialized = json.dumps(summary)
    serialized = serialized.replace(
        f'"task_id": "{task.task_id}"',
        (
            f'"task_id": "{task.task_id}", '
            f'"task_id": "{task.task_id}"'
        ),
        1,
    )
    (result / "summary.json").write_text(serialized, encoding="utf-8")

    with pytest.raises(ResultValidationError, match="duplicate JSON key"):
        validate_result(
            result,
            task,
            manifest,
            run_inputs,
            expected_config_hash=derive_config_hash(
                task,
                manifest,
                run_inputs,
            ),
            required_artifacts=("summary.json",),
            required_metrics=("all",),
        )


def test_manifest_rejects_duplicate_case_insensitive_output_paths(tmp_path):
    payload = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
    payload["tasks"][1]["output_dir"] = payload["tasks"][0]["output_dir"].upper()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="output"):
        load_manifest(path, require_production_matrix=True)


def test_windows_launcher_requires_runtime_identity_and_propagates_exit_code():
    text = PRODUCTION_LAUNCHER.read_text(encoding="utf-8")

    for variable in (
        "XTRAFFIC_CACHE_BUNDLE_ID",
        "XTRAFFIC_CACHE_BUNDLE_DIGEST",
        "XTRAFFIC_CODE_SHA",
    ):
        assert f"if not defined {variable}" in text
        assert f"%{variable}%" in text
    assert "protocol_v3_runner.py" in text
    assert 'cd /d "%~dp0\\..\\..\\.."' in text
    assert "exit /b %RUNNER_RC%" in text
    assert "> protocol_v3" not in text
