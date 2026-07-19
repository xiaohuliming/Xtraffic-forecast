from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.summarize_ndr_horizon_metrics import expected_cache_counts, validate_summary
from scripts.train_staeformer_deseason import (
    finite_masked_absolute_error_totals,
    numpy_forecast_metrics,
)


def test_masked_error_totals_support_true_global_validation_mae():
    pred_a = torch.tensor([[[2.0, 99.0]]])
    target_a = torch.tensor([[[1.0, 0.0]]])
    mask_a = torch.tensor([[[True, False]]])
    pred_b = torch.tensor([[[2.0, 4.0], [6.0, 99.0]]])
    target_b = torch.zeros_like(pred_b)
    mask_b = torch.tensor([[[True, True], [True, False]]])

    sum_a, count_a = finite_masked_absolute_error_totals(pred_a, target_a, mask_a)
    sum_b, count_b = finite_masked_absolute_error_totals(pred_b, target_b, mask_b)

    assert (sum_a + sum_b).item() / (count_a + count_b).item() == pytest.approx(3.25)
    assert (count_a + count_b).item() == 4


def test_nonfinite_prediction_fails_even_at_masked_position():
    pred = torch.tensor([[1.0, float("nan")]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False]])
    with pytest.raises(FloatingPointError, match="NaN/Inf predictions"):
        finite_masked_absolute_error_totals(pred, target, mask)
    with pytest.raises(FloatingPointError, match="NaN/Inf predictions"):
        numpy_forecast_metrics(pred.numpy(), target.numpy(), mask.numpy())


def _formal_summary(seed: int = 42) -> dict:
    rows = [
        {
            "horizon": horizon,
            "mae": 1.0 + horizon,
            "rmse": 2.0 + horizon,
            "mape_pct": 3.0 + horizon,
            "n_valid": 20,
            "n_mape": 19,
        }
        for horizon in range(1, 13)
    ]
    return {
        "summary_schema_version": 2,
        "region": "Alameda",
        "protocol": "event",
        "variant": "s0g",
        "seed": seed,
        "all": 1.0,
        "best_val": 1.1,
        "per_horizon": rows,
        "selected_horizons": {str(h): dict(rows[h - 1]) for h in (3, 6, 12)},
        "training_config": {
            "region": "Alameda", "variant": "s0g", "seed": seed,
            "epochs": 30, "batch_size": 16, "lr": 1e-3,
            "weight_decay": 1e-4, "lambda_gate": 1.0,
            "lambda_spec": 0.3, "num_workers": 0, "device": "cuda",
            "strict_causal_cache": True,
        },
        "provenance": {
            "protocol": "event",
            "region": "Alameda",
            "strict_causal_cache": True,
            "cache": {
                "imputation_method": "causal_forward_fill",
                "baseline_train_frac": 0.70,
                "traffic_cache_id": "traffic-1",
                "samples_traffic_cache_id": "traffic-1",
                "traffic_attrs": {"builder_source_sha256": "a" * 64},
                "samples_attrs": {
                    "traffic_builder_source_sha256": "a" * 64,
                    "samples_builder_source_sha256": "a" * 64,
                },
            },
            "source_files": {
                "scripts/build_full_county_cache.py": {"sha256": "a" * 64},
            },
        },
    }


def test_formal_summary_requires_complete_consistent_horizons():
    summary = _formal_summary()
    counts = validate_summary(
        summary, path=Path("summary.json"), region="Alameda", seed=42
    )
    assert counts == ((20,) * 12, (19,) * 12)

    summary["per_horizon"][11]["mae"] = float("nan")
    with pytest.raises(ValueError, match="non-finite mae"):
        validate_summary(summary, path=Path("summary.json"), region="Alameda", seed=42)

    summary = _formal_summary()
    summary["training_config"]["batch_size"] = 8
    with pytest.raises(ValueError, match="unexpected formal config batch_size"):
        validate_summary(summary, path=Path("summary.json"), region="Alameda", seed=42)


def test_expected_cache_counts_and_duplicate_rejection(tmp_path: Path):
    samples_path = tmp_path / "Alameda_samples.h5"
    traffic_path = tmp_path / "Alameda_traffic.h5"
    with h5py.File(samples_path, "w") as f:
        f.create_dataset("sample_start", data=np.array([2, 4, 6], dtype=np.int64))
        f.create_dataset("split", data=np.array([0, 2, 2], dtype=np.int8))
        f.attrs["T_p"] = 12
    flow = np.ones((20, 2, 1), dtype=np.float32)
    mask = np.ones_like(flow, dtype=bool)
    mask[5, 0, 0] = False
    with h5py.File(traffic_path, "w") as f:
        f.create_dataset("flow_series_imputed", data=flow)
        f.create_dataset("flow_mask", data=mask)
    provenance = {
        "data_files": {
            "samples": {"path": str(samples_path)},
            "traffic": {"path": str(traffic_path)},
        }
    }
    n_valid, n_mape = expected_cache_counts(provenance)
    assert n_valid[0] == 3
    assert n_valid[1:] == (4,) * 11
    assert n_mape == n_valid

    with h5py.File(samples_path, "r+") as f:
        f["sample_start"][...] = np.array([2, 4, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate sample_start"):
        expected_cache_counts(provenance)
