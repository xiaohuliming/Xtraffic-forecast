from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.fdn_benchmark_v1 import (
    MetricAccumulator,
    build_model,
    iter_batches,
    load_benchmark_data,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_dataset(root: Path, name: str = "PEMS04") -> Path:
    root.mkdir(parents=True)
    total_steps = 120
    node_count = 4
    data = np.arange(total_steps * node_count * 3, dtype=np.float64).reshape(
        total_steps, node_count, 3
    )
    data[..., 0] = 10.0 + np.arange(total_steps)[:, None] + np.arange(node_count)[None, :]
    data[5, 2, 0] = 0.0
    np.savez(root / f"{name}.npz", data=data)
    with (root / f"{name}_distance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["from", "to", "distance"])
        writer.writeheader()
        for node in range(node_count - 1):
            writer.writerow({"from": node, "to": node + 1, "distance": 1.0})
    return root


def test_loader_uses_train_only_stats_and_target_disjoint_splits(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path / "data")
    data = load_benchmark_data("pems04", data_dir)

    assert data.train_end == 72
    assert data.val_end == 96
    assert not data.observed_mask[5, 2]
    expected = data.flow_raw[: data.train_end][data.observed_mask[: data.train_end]]
    assert np.isclose(data.mean, expected.astype(np.float64).mean())
    assert np.isclose(data.std, expected.astype(np.float64).std())
    assert data.flow_normalized[5, 2] == 0.0

    target_sets = {}
    for split, anchors in data.anchors_by_split.items():
        target_sets[split] = {
            target
            for anchor in anchors.tolist()
            for target in range(anchor + 1, anchor + 13)
        }
    assert target_sets["train"].isdisjoint(target_sets["val"])
    assert target_sets["train"].isdisjoint(target_sets["test"])
    assert target_sets["val"].isdisjoint(target_sets["test"])


def test_batches_are_seeded_by_epoch_and_preserve_formal_test_order(tmp_path: Path) -> None:
    data = load_benchmark_data("pems04", _write_dataset(tmp_path / "data"))
    device = torch.device("cpu")
    first = list(
        iter_batches(data, "train", batch_size=8, seed=42, epoch=1, device=device)
    )
    repeated = list(
        iter_batches(data, "train", batch_size=8, seed=42, epoch=1, device=device)
    )
    changed = list(
        iter_batches(data, "train", batch_size=8, seed=42, epoch=2, device=device)
    )
    assert np.array_equal(first[0][3], repeated[0][3])
    assert not np.array_equal(first[0][3], changed[0][3])

    test_origins = np.concatenate(
        [batch[3] for batch in iter_batches(data, "test", batch_size=7, seed=42, epoch=0, device=device)]
    )
    assert np.array_equal(test_origins, data.anchors_by_split["test"])


def test_metric_accumulator_pools_cells_and_reports_horizons() -> None:
    prediction = torch.zeros((2, 3, 12), dtype=torch.float32)
    actual = torch.ones_like(prediction)
    mask = torch.ones_like(prediction, dtype=torch.bool)
    mask[0, 0, 0] = False
    accumulator = MetricAccumulator()
    accumulator.update(prediction[:1], actual[:1], mask[:1])
    accumulator.update(prediction[1:], actual[1:], mask[1:])

    result = accumulator.compute()

    assert result["mae"] == 1.0
    assert result["rmse"] == 1.0
    assert result["mape_percent"] == 100.0
    assert result["valid_count"] == 71
    assert result["h3"]["valid_count"] == 6


def test_direct_control_parameter_contract() -> None:
    supports = [torch.eye(5), torch.eye(5)]
    device = torch.device("cpu")
    models = {
        variant: build_model(
            variant,
            node_count=5,
            supports=supports,
            device=device,
            nhid=4,
        )
        for variant in ("gwn", "dual_gwn", "fdn_fixed_k3", "fdn_learnable_k3")
    }
    counts = {
        name: sum(value.numel() for value in model.parameters() if value.requires_grad)
        for name, model in models.items()
    }
    assert counts["dual_gwn"] == 2 * counts["gwn"]
    assert counts["fdn_fixed_k3"] == counts["dual_gwn"]
    assert counts["fdn_learnable_k3"] == counts["dual_gwn"] + 7

    x = torch.randn(2, 5, 12, 1)
    for model in models.values():
        assert model(x).shape == (2, 5, 12)


def test_formal_cpu_training_writes_complete_recomputable_artifacts(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path / "data")
    output_dir = tmp_path / "result"
    data_sha = hashlib.sha256((data_dir / "PEMS04.npz").read_bytes()).hexdigest()
    adjacency_sha = hashlib.sha256(
        (data_dir / "PEMS04_distance.csv").read_bytes()
    ).hexdigest()
    command = [
        sys.executable,
        "scripts/fdn_benchmark_v1.py",
        "--dataset",
        "pems04",
        "--variant",
        "gwn",
        "--seed",
        "42",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--batch-size",
        "16",
        "--nhid",
        "4",
        "--device",
        "cpu",
        "--task-id",
        "fdnb_test_cpu",
        "--code-sha",
        "a" * 40,
        "--config-hash",
        "b" * 64,
        "--expected-data-sha256",
        data_sha,
        "--expected-adjacency-sha256",
        adjacency_sha,
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert {path.name for path in output_dir.iterdir()} == {
        "config.json",
        "training_diagnostics.jsonl",
        "ckpt_best.pt",
        "test_predictions.npz",
        "summary.json",
    }
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["best_epoch"] == 1
    assert summary["test_metrics"]["valid_count"] > 0
    with np.load(output_dir / "test_predictions.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "prediction",
            "actual",
            "mask",
            "forecast_origin",
        }
        assert archive["prediction"].shape == archive["actual"].shape
        assert archive["mask"].dtype == np.bool_


def test_day_block_pairing_and_hierarchical_bootstrap_are_deterministic() -> None:
    from scripts.fdn_benchmark_v1_report import (
        _daily_window_differences,
        _hierarchical_bootstrap,
    )

    actual = np.full((8, 2, 12), 10.0, dtype=np.float32)
    mask = np.ones_like(actual, dtype=np.bool_)
    origins = np.asarray([100, 101, 102, 103, 400, 401, 402, 403], dtype=np.int64)
    control = {
        "prediction": actual + 2.0,
        "actual": actual,
        "mask": mask,
        "forecast_origin": origins,
    }
    learnable = {
        "prediction": actual + 1.0,
        "actual": actual,
        "mask": mask,
        "forecast_origin": origins,
    }

    day_differences = _daily_window_differences(learnable, control)
    first = _hierarchical_bootstrap(
        [day_differences, day_differences, day_differences],
        iterations=100,
        seed=7,
    )
    second = _hierarchical_bootstrap(
        [day_differences, day_differences, day_differences],
        iterations=100,
        seed=7,
    )

    assert np.array_equal(day_differences, np.asarray([-1.0, -1.0]))
    assert first == second == (-1.0, -1.0)
