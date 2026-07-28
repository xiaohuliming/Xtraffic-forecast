from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .secure_io import open_verified_regular_file


_METHODS = frozenset(
    {
        "hl",
        "lstm",
        "dcrnn",
        "agcrn",
        "stgcn",
        "gwnet",
        "astgcn",
        "sttn",
        "dstagnn",
        "dgcrn",
        "d2stgnn",
        "bist",
        "igstgnn",
    }
)


def _validate_method(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(f"Unsupported Table 4 method: {method}")


def run_upstream_training(*, method: str, engine: Any) -> None:
    _validate_method(method)
    if method == "hl":
        engine.evaluate("test")
    else:
        engine.train()


def _squeezed_cpu_array(value: torch.Tensor, *, label: str) -> np.ndarray:
    if not torch.is_tensor(value):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.ndim < 3 or value.shape[1] != 12:
        raise ValueError(f"{label} must contain exactly 12 horizons")
    if value.ndim >= 4:
        if value.shape[-1] != 1:
            raise ValueError(f"{label} final feature dimension must be 1")
        value = value.squeeze(-1)
    result = value.detach().cpu().numpy()
    if result.dtype != np.dtype("float32"):
        raise ValueError(f"{label} must use float32")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.array(result, copy=True)


def capture_test_predictions(
    *,
    method: str,
    engine: Any,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_method(method)
    engine.model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in engine._dataloader["test_loader"].get_iterator():
            if method == "igstgnn":
                (
                    model_input,
                    label,
                    incident_data,
                    sensor_data,
                ) = engine._prepare_batch(batch)
                prediction = engine._predict(
                    model_input,
                    label,
                    incident_data,
                    sensor_data,
                )
            else:
                model_input, label = engine._to_device(
                    engine._to_tensor(
                        [
                            batch[0],
                            batch[1][:, : engine.model.horizon],
                        ]
                    )
                )
                prediction = engine.model(model_input, label)
            prediction, label = engine._inverse_transform([prediction, label])
            predictions.append(_squeezed_cpu_array(prediction, label="prediction"))
            labels.append(_squeezed_cpu_array(label, label="label"))
    if not predictions:
        raise ValueError("Test loader produced no batches")
    prediction_array = np.concatenate(predictions, axis=0)
    label_array = np.concatenate(labels, axis=0)
    if prediction_array.shape != label_array.shape:
        raise ValueError("Captured predictions and labels have different shapes")
    return prediction_array, label_array


def save_model_state_checkpoint(
    *,
    model: torch.nn.Module,
    target: Path | str,
) -> Path:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("xb") as handle:
        torch.save(model.state_dict(), handle)
        handle.flush()
        os.fsync(handle.fileno())
    return target_path


def load_model_state_checkpoint(
    *,
    model: torch.nn.Module,
    source: Path | str,
    map_location: torch.device | str = "cpu",
) -> Path:
    source_path = Path(source)
    with open_verified_regular_file(
        source_path,
        purpose="Model state checkpoint",
        require_nonempty=True,
    ) as opened:
        state = torch.load(
            opened.handle,
            map_location=map_location,
            weights_only=True,
        )
    if not isinstance(state, dict):
        raise ValueError("Model state checkpoint must contain a state dict")
    model.load_state_dict(state, strict=True)
    return source_path
