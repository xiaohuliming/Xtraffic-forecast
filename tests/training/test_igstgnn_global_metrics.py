import numpy as np
import pytest

from scripts.evaluate_igstgnn_baseline import evaluate


def test_headline_metric_weights_every_observed_target_equally():
    preds = np.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=np.float32)
    labels = np.zeros_like(preds)
    observed = np.array([[[True, True], [True, False]]])

    result = evaluate(preds, labels, observed)

    assert result["aggregation"] == "all_observed_targets"
    assert result["valid_count"] == 3
    assert result["average"]["MAE"] == pytest.approx(10.0 / 3.0)
