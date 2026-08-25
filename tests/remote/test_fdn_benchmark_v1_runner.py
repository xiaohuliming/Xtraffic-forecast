from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "config/fdn_benchmark_v1/go_no_go_pems04_pems08_v1.json"


def test_manifest_is_exact_blocked_24_task_matrix() -> None:
    from scripts.remote.windows.fdn_benchmark_v1_runner import load_manifest

    manifest = load_manifest(MANIFEST)

    assert len(manifest.tasks) == 24
    assert {task.dataset for task in manifest.tasks} == {"pems04", "pems08"}
    assert {task.seed for task in manifest.tasks} == {42, 1, 2}
    assert {task.variant for task in manifest.tasks} == {
        "gwn",
        "dual_gwn",
        "fdn_fixed_k3",
        "fdn_learnable_k3",
    }
    for dataset in ("pems04", "pems08"):
        for seed in (42, 1, 2):
            block = [
                task.variant
                for task in manifest.tasks
                if task.dataset == dataset and task.seed == seed
            ]
            assert len(block) == 4
            assert set(block) == {
                "gwn",
                "dual_gwn",
                "fdn_fixed_k3",
                "fdn_learnable_k3",
            }


def test_config_hash_binds_task_training_data_and_code() -> None:
    from scripts.remote.windows.fdn_benchmark_v1_runner import (
        derive_config_hash,
        load_manifest,
    )

    manifest = load_manifest(MANIFEST)
    task = manifest.tasks[0]
    first = derive_config_hash(task, manifest, "a" * 40)
    repeated = derive_config_hash(task, manifest, "a" * 40)
    other_code = derive_config_hash(task, manifest, "b" * 40)
    other_task = derive_config_hash(manifest.tasks[1], manifest, "a" * 40)

    assert first == repeated
    assert len(first) == 64
    assert first != other_code
    assert first != other_task


def test_four_gpu_shards_are_disjoint_complete_and_order_preserving() -> None:
    from scripts.remote.windows.fdn_benchmark_v1_runner import (
        load_manifest,
        select_shard_tasks,
    )

    manifest = load_manifest(MANIFEST)
    shards = [
        select_shard_tasks(manifest, shard_index=index, shard_count=4)
        for index in range(4)
    ]

    assert [len(shard) for shard in shards] == [6, 6, 6, 6]
    task_ids = [task.task_id for shard in shards for task in shard]
    assert len(task_ids) == len(set(task_ids)) == 24
    assert set(task_ids) == {task.task_id for task in manifest.tasks}
    for index, shard in enumerate(shards):
        assert shard == manifest.tasks[index::4]


def test_nested_metrics_slice_only_the_requested_horizon() -> None:
    from scripts.remote.windows.fdn_benchmark_v1_runner import (
        _nested_metrics_from_arrays,
    )

    actual = np.full((2, 3, 12), 10.0, dtype=np.float32)
    prediction = actual.copy()
    for horizon in range(12):
        prediction[:, :, horizon] += np.float32(horizon + 1)
    mask = np.ones_like(actual, dtype=np.bool_)

    metrics = _nested_metrics_from_arrays(prediction, actual, mask)

    assert metrics["h3"]["mae"] == 3.0
    assert metrics["h6"]["mae"] == 6.0
    assert metrics["h12"]["mae"] == 12.0
    assert metrics["h3"]["valid_count"] == 6


def test_result_validator_recomputes_metrics_and_binds_test_origins(
    tmp_path: Path,
) -> None:
    from scripts.fdn_benchmark_v1 import load_benchmark_data
    from scripts.remote.windows.fdn_benchmark_v1_runner import (
        _nested_metrics_from_arrays,
        derive_config_hash,
        load_manifest,
        validate_result,
    )

    manifest = load_manifest(MANIFEST)
    task = next(task for task in manifest.tasks if task.variant == "gwn")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    total_steps = 120
    node_count = 4
    values = np.ones((total_steps, node_count, 3), dtype=np.float64)
    values[..., 0] = 10 + np.arange(total_steps)[:, None]
    np.savez(data_dir / f"{task.dataset.upper()}.npz", data=values)
    with (data_dir / f"{task.dataset.upper()}_distance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["from", "to", "distance"])
        writer.writeheader()
        for node in range(node_count - 1):
            writer.writerow({"from": node, "to": node + 1, "distance": 1})
    dataset_identity = dict(manifest.datasets[task.dataset])
    import hashlib

    dataset_identity["data_sha256"] = hashlib.sha256(
        (data_dir / f"{task.dataset.upper()}.npz").read_bytes()
    ).hexdigest()
    dataset_identity["adjacency_sha256"] = hashlib.sha256(
        (data_dir / f"{task.dataset.upper()}_distance.csv").read_bytes()
    ).hexdigest()
    raw = dict(manifest.raw)
    datasets = dict(manifest.datasets)
    datasets[task.dataset] = dataset_identity
    raw["datasets"] = datasets
    manifest = manifest.__class__(
        path=manifest.path,
        raw=raw,
        matrix_id=manifest.matrix_id,
        datasets=datasets,
        training=manifest.training,
        tasks=manifest.tasks,
    )
    source = load_benchmark_data(task.dataset, data_dir)
    origins = np.asarray(source.anchors_by_split["test"], dtype=np.int64)
    target_offsets = np.arange(1, 13, dtype=np.int64)
    target_index = origins[:, None] + target_offsets[None, :]
    actual = source.flow_raw[target_index].transpose(0, 2, 1).astype(np.float32)
    mask = source.observed_mask[target_index].transpose(0, 2, 1).astype(np.bool_)
    prediction = actual + np.float32(1.0)
    metrics = _nested_metrics_from_arrays(prediction, actual, mask)
    code_sha = "a" * 40
    config_hash = derive_config_hash(task, manifest, code_sha)
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    summary = {
        "artifact_schema_version": 1,
        "protocol_version": "fdn_benchmark_v1",
        "status": "completed",
        "task_id": task.task_id,
        "dataset": task.dataset,
        "variant": task.variant,
        "seed": task.seed,
        "code_sha": code_sha,
        "config_hash": config_hash,
        "data_sha256": dataset_identity["data_sha256"],
        "adjacency_sha256": dataset_identity["adjacency_sha256"],
        "test_metrics": metrics,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    np.savez_compressed(
        result_dir / "test_predictions.npz",
        prediction=prediction,
        actual=actual,
        mask=mask,
        forecast_origin=origins,
    )
    (result_dir / "training_diagnostics.jsonl").write_text(
        "".join(json.dumps({"epoch": epoch}) + "\n" for epoch in range(1, 101)),
        encoding="utf-8",
    )
    for name in ("config.json", "ckpt_best.pt", "train.log"):
        (result_dir / name).write_text(name, encoding="utf-8")

    validated = validate_result(
        result_dir,
        task,
        manifest,
        code_sha=code_sha,
        config_hash=config_hash,
        data_dir=data_dir,
    )

    assert validated["summary"]["test_metrics"]["mae"] == 1.0
    assert len(validated["files_digest"]) == 64
