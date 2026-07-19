from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from baselines.DCRNN.graph import build_bidirectional_supports, random_walk_matrix
from baselines.DCRNN.model import DCRNN
from baselines.DCRNN.train_xtraffic import (
    ORIGINAL_DCRNN_CONFIG,
    build_encoder_input,
    compute_train_stats,
    original_learning_rate,
)


def small_supports(nodes: int = 5):
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for node in range(nodes - 1):
        adjacency[node, node + 1] = 1.0
        adjacency[node + 1, node] = 0.5
    return build_bidirectional_supports(adjacency)


def test_random_walk_rows_are_normalized():
    adjacency = np.array(
        [[0.0, 1.0, 3.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    walk = random_walk_matrix(adjacency)
    assert np.allclose(walk[:2].sum(axis=1), 1.0)
    assert np.allclose(walk[2], 0.0)


def test_support_orientation_matches_official_dcrnn():
    adjacency = np.array(
        [[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    forward, backward = build_bidirectional_supports(adjacency)
    assert torch.allclose(
        forward.to_dense(), torch.from_numpy(random_walk_matrix(adjacency).T)
    )
    assert torch.allclose(
        backward.to_dense(), torch.from_numpy(random_walk_matrix(adjacency.T).T)
    )


def test_forward_shape_finite_and_gradient():
    batch, history, horizon, nodes = 2, 4, 3, 5
    model = DCRNN(
        num_nodes=nodes,
        input_dim=1,
        output_dim=1,
        horizon=horizon,
        supports=small_supports(nodes),
        hidden_dim=8,
        num_layers=2,
        max_diffusion_step=2,
    )
    inputs = torch.randn(batch, history, nodes, 1)
    labels = torch.randn(batch, horizon, nodes, 1)
    mask = torch.ones_like(labels, dtype=torch.bool)
    output = model(inputs, labels=labels, labels_mask=mask, batches_seen=0)
    assert output.shape == (batch, horizon, nodes, 1)
    assert torch.isfinite(output).all()
    output.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(
        g is not None and torch.isfinite(g).all() for g in gradients
    )


def test_eval_has_no_teacher_forcing_or_future_leakage():
    model = DCRNN(
        num_nodes=5,
        input_dim=1,
        output_dim=1,
        horizon=3,
        supports=small_supports(),
        hidden_dim=4,
        num_layers=1,
    )
    model.eval()
    inputs = torch.randn(2, 4, 5, 1)
    labels_a = torch.zeros(2, 3, 5, 1)
    labels_b = torch.full_like(labels_a, 1000.0)
    with torch.no_grad():
        output_a = model(inputs, labels=labels_a, batches_seen=0)
        output_b = model(inputs, labels=labels_b, batches_seen=0)
    assert torch.equal(output_a, output_b)


def test_sampling_threshold_decreases():
    model = DCRNN(
        num_nodes=5,
        input_dim=1,
        output_dim=1,
        horizon=3,
        supports=small_supports(),
        hidden_dim=4,
        num_layers=1,
        cl_decay_steps=100,
    )
    assert model.sampling_threshold(0) > model.sampling_threshold(500)


def test_privileged_incident_fields_cannot_change_model_input():
    history = torch.randn(2, 5, 4, 3)
    time_feat = torch.rand(2, 4, 2)
    batch_a = {
        "x_hist": history,
        "time_feat": time_feat,
        "incident_feat": torch.zeros(2, 3, 13),
        "affected_mask": torch.zeros(2, 5, dtype=torch.bool),
    }
    batch_b = {
        **batch_a,
        "incident_feat": torch.randn(2, 3, 13) * 1000,
        "affected_mask": torch.ones(2, 5, dtype=torch.bool),
    }
    stats = {"mean": 10.0, "std": 2.0}
    encoded_a = build_encoder_input(batch_a, stats, torch.device("cpu"))
    encoded_b = build_encoder_input(batch_b, stats, torch.device("cpu"))
    assert encoded_a.shape == (2, 4, 5, 2)
    expected_tod = time_feat[..., 0].unsqueeze(2).expand(-1, -1, 5)
    assert torch.equal(encoded_a[..., 1], expected_tod)
    assert torch.equal(encoded_a, encoded_b)


def test_official_configuration_and_learning_rate_schedule_are_locked():
    assert ORIGINAL_DCRNN_CONFIG["history"] == 12
    assert ORIGINAL_DCRNN_CONFIG["horizon"] == 12
    assert ORIGINAL_DCRNN_CONFIG["input_dim"] == 2
    assert ORIGINAL_DCRNN_CONFIG["hidden_dim"] == 64
    assert ORIGINAL_DCRNN_CONFIG["num_layers"] == 2
    assert ORIGINAL_DCRNN_CONFIG["max_diffusion_step"] == 2
    assert ORIGINAL_DCRNN_CONFIG["batch_size"] == 64
    assert ORIGINAL_DCRNN_CONFIG["base_lr"] == 0.01
    assert ORIGINAL_DCRNN_CONFIG["adam_epsilon"] == 1.0e-3
    assert ORIGINAL_DCRNN_CONFIG["epochs"] == 100
    assert np.isclose(original_learning_rate(1), 0.01)
    assert np.isclose(original_learning_rate(20), 0.001)
    assert np.isclose(original_learning_rate(30), 0.0001)
    assert np.isclose(original_learning_rate(40), 0.00001)
    assert np.isclose(original_learning_rate(50), 2.0e-6)

    model = DCRNN(
        num_nodes=5,
        input_dim=ORIGINAL_DCRNN_CONFIG["input_dim"],
        output_dim=ORIGINAL_DCRNN_CONFIG["output_dim"],
        horizon=ORIGINAL_DCRNN_CONFIG["horizon"],
        supports=small_supports(),
        hidden_dim=ORIGINAL_DCRNN_CONFIG["hidden_dim"],
        num_layers=ORIGINAL_DCRNN_CONFIG["num_layers"],
        max_diffusion_step=ORIGINAL_DCRNN_CONFIG["max_diffusion_step"],
        cl_decay_steps=ORIGINAL_DCRNN_CONFIG["cl_decay_steps"],
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 372_352
    assert model.decoder.projection.bias is None


def test_training_scaler_matches_repeated_official_x_windows():
    flow = np.zeros((6, 1, 3), dtype=np.float32)
    flow[:3, 0, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    mask = np.ones_like(flow, dtype=bool)
    rdata = SimpleNamespace(
        sample_start=np.array([1, 2], dtype=np.int64),
        split=np.array([0, 0], dtype=np.int8),
        T=6,
        T_h=2,
        flow_series=flow,
        flow_mask=mask,
    )
    stats = compute_train_stats(rdata)
    # Official x_train contains [0,1] and [1,2], hence [0,1,1,2].
    assert np.isclose(stats["mean"], 1.0)
    assert np.isclose(stats["std"], np.sqrt(0.5))
