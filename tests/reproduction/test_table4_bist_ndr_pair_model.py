from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.bist_ndr_pair_manifest import (
    load_bist_ndr_pair_profile,
)
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_model import (
    build_bist_ndr_pair_model,
)
from scripts.reproduction.igstgnn_table4.ndr_extension_runner import (
    NDRTrainingStatistics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "bist_ndr_pair_profile_v1.json"
)


def _profile():
    digest = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    return load_bist_ndr_pair_profile(
        PROFILE_PATH,
        expected_sha256=digest,
    )


def _statistics(node_count: int) -> NDRTrainingStatistics:
    return NDRTrainingStatistics(
        baseline=np.zeros((7, 288, node_count), dtype=np.float32),
        flow_mean=20.0,
        flow_std=5.0,
        residual_std=3.0,
        residual_std_by_node=np.ones(node_count, dtype=np.float32),
        valid_training_points=100,
        source_normalization_mean=10.0,
        source_normalization_std=2.0,
    )


def _inputs(node_count: int):
    flow_hist = torch.full((2, node_count, 12), 20.0)
    x_baseline = torch.full_like(flow_hist, 18.0)
    y_baseline = torch.full_like(flow_hist, 21.0)
    time_feat = torch.zeros((2, 12, 2))
    time_feat[..., 0] = 10 / 96
    time_feat[..., 1] = 2 / 7
    return flow_hist, x_baseline, y_baseline, time_feat


def test_bist_and_bist_ndr_expose_the_same_raw_flow_forecast_interface():
    profile = _profile()
    statistics = _statistics(node_count=4)
    inputs = _inputs(node_count=4)

    aligned = build_bist_ndr_pair_model(
        method="bist_aligned",
        node_count=4,
        effective_profile=profile["effective"],
        statistics=statistics,
        device="cpu",
    )
    hybrid = build_bist_ndr_pair_model(
        method="bist_ndr",
        node_count=4,
        effective_profile=profile["effective"],
        statistics=statistics,
        device="cpu",
    )

    aligned.model.eval()
    hybrid.model.eval()
    with torch.no_grad():
        aligned_prediction = aligned.model(*inputs)
        hybrid_prediction = hybrid.model(*inputs)

    assert aligned_prediction.shape == hybrid_prediction.shape == (2, 4, 12)
    assert torch.isfinite(aligned_prediction).all()
    assert torch.isfinite(hybrid_prediction).all()
    assert aligned.model.aux is None
    assert hybrid.model.oracle_sup is True
    assert hybrid.model.dual is True
    assert set(hybrid.model.aux) == {"g", "B", "P", "yn", "yd"}
    torch.testing.assert_close(
        hybrid_prediction,
        hybrid.model.aux["g"] * hybrid.model.aux["yn"]
        + (1.0 - hybrid.model.aux["g"]) * hybrid.model.aux["yd"],
    )
    assert aligned.provenance["upstream_commit"] == (
        "dd94adf7721fcbb9e3feb5d1b44040305199a4cc"
    )
    assert hybrid.provenance["head_initialization"] == (
        "identical_copy_from_bist_native_heads"
    )
