from __future__ import annotations

import pytest

from scripts.remote.linux.protocol_v3_event_extension_gpu_probe import (
    estimate_training_seconds,
)


def test_calibration_includes_loader_and_optimizer_time() -> None:
    assert estimate_training_seconds(
        batch_build_seconds=1.25,
        optimization_seconds=2.75,
        train_batches=100,
        epochs=30,
    ) == pytest.approx(12_000.0)


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_build_seconds", -0.1),
        ("optimization_seconds", 0.0),
        ("train_batches", 0),
        ("epochs", 0),
    ],
)
def test_calibration_rejects_invalid_measurements(key: str, value: float) -> None:
    arguments = {
        "batch_build_seconds": 1.0,
        "optimization_seconds": 2.0,
        "train_batches": 10,
        "epochs": 30,
    }
    arguments[key] = value
    with pytest.raises(ValueError):
        estimate_training_seconds(**arguments)
