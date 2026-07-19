"""Fail-closed masked metrics shared by training entry points."""

from __future__ import annotations

import numpy as np
import torch


def masked_mae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return differentiable MAE and reject unusable or corrupted batches."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    valid = _validated_mask(mask, prediction.shape).to(prediction.device)
    if int(valid.sum().item()) == 0:
        raise ValueError("masked MAE batch has zero valid observations")
    if not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError("prediction contains non-finite valid observations")
    target = target.to(prediction.device)
    if not torch.isfinite(target[valid]).all():
        raise FloatingPointError("target contains non-finite valid observations")
    return torch.abs(prediction[valid] - target[valid]).mean()


class MaskedMAEAccumulator:
    """Accumulate absolute error over valid observations, not batches."""

    def __init__(self) -> None:
        self._absolute_error_sum = 0.0
        self._valid_count = 0

    def update(self, prediction, target, mask) -> None:
        prediction = _as_tensor(prediction)
        target = _as_tensor(target).to(prediction.device)
        mask = _validated_mask(mask, prediction.shape).to(prediction.device)
        if target.shape != prediction.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
            )

        valid_count = int(mask.sum().item())
        if valid_count == 0:
            raise ValueError("masked MAE batch has zero valid observations")
        if not torch.isfinite(prediction[mask]).all():
            raise FloatingPointError("prediction contains non-finite valid observations")
        if not torch.isfinite(target[mask]).all():
            raise FloatingPointError("target contains non-finite valid observations")

        absolute_error = torch.abs(prediction[mask] - target[mask])
        self._absolute_error_sum += float(absolute_error.double().sum().item())
        self._valid_count += valid_count

    def compute(self) -> float:
        if self._valid_count == 0:
            raise ValueError("masked MAE has zero valid observations")
        value = self._absolute_error_sum / self._valid_count
        if not np.isfinite(value):
            raise FloatingPointError("masked MAE is non-finite")
        return float(value)

    @property
    def valid_count(self) -> int:
        return self._valid_count


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def _validated_mask(mask, expected_shape: torch.Size) -> torch.Tensor:
    mask = _as_tensor(mask)
    if mask.shape != expected_shape:
        raise ValueError(f"mask shape differs from prediction: {mask.shape} != {expected_shape}")
    if mask.dtype == torch.bool:
        return mask
    if not torch.isfinite(mask).all():
        raise FloatingPointError("mask contains non-finite values")
    if not torch.logical_or(mask == 0, mask == 1).all():
        raise ValueError("mask must contain only zero and one")
    return mask.bool()
