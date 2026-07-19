import numpy as np
import pytest
import torch

from dist_net.metrics import MaskedMAEAccumulator, masked_mae_loss


def test_masked_mae_weights_batches_by_valid_observation_count():
    first_pred = np.array([0.0, 10.0, 99.0], dtype=np.float32)
    first_true = np.array([0.0, 0.0, 99.0], dtype=np.float32)
    first_mask = np.array([1, 1, 0], dtype=bool)
    second_pred = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    second_true = np.zeros(4, dtype=np.float32)
    second_mask = np.ones(4, dtype=bool)

    metric = MaskedMAEAccumulator()
    metric.update(first_pred, first_true, first_mask)
    metric.update(second_pred, second_true, second_mask)

    assert metric.compute() == 5.0


def test_masked_mae_loss_rejects_a_batch_without_valid_observations():
    with pytest.raises(ValueError, match="zero valid observations"):
        masked_mae_loss(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([False, False]),
        )


def test_masked_mae_rejects_non_finite_valid_predictions():
    metric = MaskedMAEAccumulator()

    with pytest.raises(FloatingPointError, match="prediction"):
        metric.update(
            np.array([1.0, np.nan]),
            np.array([1.0, 2.0]),
            np.array([True, True]),
        )


def test_masked_mae_rejects_non_finite_predictions_even_when_target_is_masked():
    metric = MaskedMAEAccumulator()

    with pytest.raises(FloatingPointError, match="prediction"):
        metric.update(
            np.array([1.0, np.nan]),
            np.array([1.0, np.nan]),
            np.array([True, False]),
        )


def test_masked_mae_skips_an_empty_batch_but_not_an_empty_split():
    metric = MaskedMAEAccumulator()
    metric.update(
        np.array([1.0, 2.0]),
        np.array([np.nan, np.nan]),
        np.array([False, False]),
    )
    metric.update(
        np.array([3.0]),
        np.array([1.0]),
        np.array([True]),
    )

    assert metric.valid_count == 1
    assert metric.compute() == 2.0


def test_masked_mae_rejects_an_empty_evaluation():
    with pytest.raises(ValueError, match="zero valid observations"):
        MaskedMAEAccumulator().compute()
