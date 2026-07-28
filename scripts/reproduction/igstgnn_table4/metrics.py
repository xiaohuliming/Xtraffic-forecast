from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch


_UPSTREAM_METRIC_THREADS = 3
_METRIC_LOCK = threading.RLock()


def _official_null_value(labels: torch.Tensor) -> torch.Tensor:
    minimum = labels.min()
    if minimum < 1:
        return minimum
    return torch.tensor(0.0, dtype=labels.dtype, device=labels.device)


def _official_horizon_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    null_value: torch.Tensor,
) -> dict[str, float]:
    mask = (labels != null_value).to(dtype=torch.float32)
    mask = mask / mask.mean()
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    absolute_error = torch.abs(predictions - labels)
    squared_error = (predictions - labels) ** 2
    percentage_error = absolute_error / labels

    absolute_error = torch.where(
        torch.isnan(absolute_error * mask),
        torch.zeros_like(absolute_error),
        absolute_error * mask,
    )
    squared_error = torch.where(
        torch.isnan(squared_error * mask),
        torch.zeros_like(squared_error),
        squared_error * mask,
    )
    percentage_error = torch.where(
        torch.isnan(percentage_error * mask),
        torch.zeros_like(percentage_error),
        percentage_error * mask,
    )

    return {
        "mae": float(absolute_error.mean().item()),
        "rmse": float(torch.sqrt(squared_error.mean()).item()),
        "mape_percent": float(percentage_error.mean().item() * 100.0),
    }


def compute_official_table4_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    prediction_shape = np.shape(predictions)
    label_shape = np.shape(labels)
    if prediction_shape != label_shape:
        raise ValueError(
            "Predictions and labels must have the same shape, got "
            f"{prediction_shape} and {label_shape}"
        )
    if len(prediction_shape) < 2 or prediction_shape[1] != 12:
        raise ValueError(
            "Table 4 metrics require exactly 12 prediction horizons, got "
            f"shape {prediction_shape}"
        )
    with _METRIC_LOCK:
        previous_threads = torch.get_num_threads()
        if previous_threads != _UPSTREAM_METRIC_THREADS:
            torch.set_num_threads(_UPSTREAM_METRIC_THREADS)
        try:
            prediction_tensor = torch.as_tensor(
                predictions,
                dtype=torch.float32,
            )
            label_tensor = torch.as_tensor(labels, dtype=torch.float32)
            null_value = _official_null_value(label_tensor)
            per_horizon = [
                _official_horizon_metrics(
                    prediction_tensor[:, horizon_index, ...],
                    label_tensor[:, horizon_index, ...],
                    null_value,
                )
                for horizon_index in range(12)
            ]
        finally:
            if torch.get_num_threads() != previous_threads:
                torch.set_num_threads(previous_threads)
    average = {
        metric: float(np.mean([row[metric] for row in per_horizon]))
        for metric in ("mae", "rmse", "mape_percent")
    }

    return {
        "h3": per_horizon[2],
        "h6": per_horizon[5],
        "h12": per_horizon[11],
        "average": average,
        "per_horizon": per_horizon,
        "null_value": float(null_value.item()),
    }
