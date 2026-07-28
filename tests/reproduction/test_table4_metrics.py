import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.reproduction.igstgnn_table4.metrics import (
    compute_official_table4_metrics,
)


def test_table4_metrics_selects_h3_h6_h12_and_averages_all_horizons():
    labels = np.broadcast_to(
        np.array([10.0, 20.0], dtype=np.float32),
        (1, 12, 2),
    ).copy()
    horizon_errors = np.arange(1, 13, dtype=np.float32).reshape(1, 12, 1)
    predictions = labels + horizon_errors

    metrics = compute_official_table4_metrics(predictions, labels)

    assert metrics["h3"]["mae"] == pytest.approx(3.0)
    assert metrics["h6"]["mae"] == pytest.approx(6.0)
    assert metrics["h12"]["mae"] == pytest.approx(12.0)
    assert metrics["average"]["mae"] == pytest.approx(6.5)
    assert metrics["h3"]["rmse"] == pytest.approx(3.0)
    assert metrics["average"]["rmse"] == pytest.approx(6.5)
    assert metrics["h3"]["mape_percent"] == pytest.approx(22.5)
    assert metrics["average"]["mape_percent"] == pytest.approx(48.75)


def test_official_table4_metrics_matches_upstream_null_value_semantics():
    labels = np.broadcast_to(
        np.array([0.0, 10.0], dtype=np.float32),
        (1, 12, 2),
    ).copy()
    predictions = np.broadcast_to(
        np.array([100.0, 12.0], dtype=np.float32),
        labels.shape,
    ).copy()

    metrics = compute_official_table4_metrics(predictions, labels)

    assert metrics["null_value"] == 0.0
    assert metrics["h3"]["mae"] == pytest.approx(2.0)
    assert metrics["h3"]["rmse"] == pytest.approx(2.0)
    assert metrics["h3"]["mape_percent"] == pytest.approx(20.0)


def test_official_table4_metrics_rejects_mismatched_shapes():
    predictions = np.zeros((1, 12, 2), dtype=np.float32)
    labels = np.zeros((1, 12, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        compute_official_table4_metrics(predictions, labels)


def test_official_table4_metrics_requires_exactly_12_horizons():
    predictions = np.zeros((1, 11, 2), dtype=np.float32)
    labels = np.zeros_like(predictions)

    with pytest.raises(ValueError, match="exactly 12"):
        compute_official_table4_metrics(predictions, labels)


def test_table4_metrics_match_the_pinned_upstream_implementation():
    metrics_path = (
        Path(__file__).resolve().parents[2]
        / ".third_party"
        / "igstgnn_table4"
        / "pristine"
        / "largest_b7c6ff5"
        / "src"
        / "utils"
        / "metrics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pinned_largest_metrics",
        metrics_path,
    )
    assert spec is not None and spec.loader is not None
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    labels = np.arange(1, 1 + 2 * 12 * 3, dtype=np.float32).reshape(
        2,
        12,
        3,
    )
    labels[0, 0, 0] = 0
    predictions = labels + np.linspace(
        -2,
        3,
        labels.size,
        dtype=np.float32,
    ).reshape(labels.shape)

    observed = compute_official_table4_metrics(predictions, labels)
    prediction_tensor = torch.as_tensor(predictions)
    label_tensor = torch.as_tensor(labels)
    null_value = label_tensor.min()

    for horizon in range(12):
        mae, mape, rmse = upstream.compute_all_metrics(
            prediction_tensor[:, horizon],
            label_tensor[:, horizon],
            null_value,
        )
        assert observed["per_horizon"][horizon]["mae"] == pytest.approx(mae)
        assert observed["per_horizon"][horizon]["rmse"] == pytest.approx(rmse)
        assert observed["per_horizon"][horizon]["mape_percent"] == pytest.approx(
            mape * 100.0
        )


def test_table4_metrics_reproduce_upstream_three_thread_reduction():
    generator = np.random.default_rng(2025)
    labels = generator.uniform(
        1,
        400,
        size=(97, 12, 521),
    ).astype(np.float32)
    predictions = labels + generator.normal(
        0,
        30,
        size=labels.shape,
    ).astype(np.float32)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        single_thread_caller = compute_official_table4_metrics(
            predictions,
            labels,
        )
        torch.set_num_threads(4)
        four_thread_caller = compute_official_table4_metrics(
            predictions,
            labels,
        )
    finally:
        torch.set_num_threads(original_threads)

    assert single_thread_caller == four_thread_caller
