from __future__ import annotations

import inspect

import pytest
import torch

from ndr_research.shared_graph import SharedGraphNDR, canonical_history


def _inputs(nodes=5, batch=2):
    generator = torch.Generator().manual_seed(41)
    history = 60 + torch.randn(batch, nodes, 12, generator=generator) * 4
    baseline = torch.full_like(history, 55)
    future = torch.full_like(history, 58)
    time = torch.zeros(batch, 12, 2)
    time[..., 0] = (torch.arange(12) + 90) / 288
    time[..., 1] = 2 / 7
    observed = torch.ones_like(history, dtype=torch.bool)
    sources = torch.arange(max(nodes - 1, 0), dtype=torch.long)
    edges = torch.stack((sources, sources + 1))
    scales = torch.linspace(3, 6, nodes)
    return history, baseline, future, time, observed, edges, scales


def _model(**kwargs):
    return SharedGraphNDR(hidden_dim=16, num_heads=4, **kwargs)


def test_same_strictly_loaded_weights_forecast_different_node_sets():
    model = _model()
    configuration = model.get_config()
    parameter_shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    model.eval()
    for nodes in (1, 4, 9):
        restored = SharedGraphNDR(**configuration)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        with torch.no_grad():
            prediction = restored(*_inputs(nodes))
        assert prediction.shape == (2, nodes, 12)
        assert torch.isfinite(prediction).all()
        assert parameter_shapes == {name: tuple(value.shape) for name, value in restored.state_dict().items()}
        assert not list(restored.named_buffers())
    assert "node_count" not in configuration
    assert "region" not in configuration


def test_node_permutation_and_corresponding_graph_permutation_are_equivariant():
    model = _model()
    model.eval()
    inputs = _inputs(nodes=6)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    history, baseline, future, time, observed, edges, scale = inputs
    with torch.no_grad():
        expected = model(*inputs)
        actual = model(
            history[:, permutation], baseline[:, permutation], future[:, permutation],
            time, observed[:, permutation], inverse[edges], scale[permutation],
        )
    torch.testing.assert_close(actual, expected[:, permutation], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("missing_pattern", ["all", "prefix", "tail", "interior"])
def test_missing_sentinels_do_not_change_any_forecast(missing_pattern):
    history, baseline, future, time, observed, edges, scale = _inputs()
    if missing_pattern == "all":
        observed[:] = False
    elif missing_pattern == "prefix":
        observed[:, :, :5] = False
    elif missing_pattern == "tail":
        observed[:, :, -5:] = False
    else:
        observed[:, :, 3:7] = False
    nan_history = torch.where(observed, history, torch.full_like(history, float("nan")))
    poison_history = torch.where(observed, history, torch.full_like(history, -1e9))
    model = _model()
    with torch.no_grad():
        model.gate.context[-1].weight.normal_(std=0.1)
        model.gate.gap_coefficient.fill_(0.2)
    model.eval()
    with torch.no_grad():
        first = model(nan_history, baseline, future, time, observed, edges, scale)
        second = model(poison_history, baseline, future, time, observed, edges, scale)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(second, first, rtol=0, atol=0)


def test_window_fill_is_causal_and_real_zero_is_an_observation():
    history = torch.tensor([[[999.0, 0.0, 999.0, 12.0, 999.0, 999.0]]])
    baseline = torch.tensor([[[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]])
    observed = torch.tensor([[[False, True, False, True, False, False]]])
    state = canonical_history(history, baseline, observed)
    torch.testing.assert_close(state.filled, torch.tensor([[[5.0, 0.0, 0.0, 12.0, 12.0, 12.0]]]))
    changed = history.clone()
    changed[..., 3] = 1200
    changed_state = canonical_history(changed, baseline, observed)
    torch.testing.assert_close(changed_state.filled[..., :3], state.filled[..., :3])
    assert state.reference.item() == 12.0
    assert state.age.item() == 2


@pytest.mark.parametrize("edges", [torch.empty(2, 0, dtype=torch.long), torch.tensor([[0], [1]])])
def test_isolated_nodes_and_empty_roads_remain_finite_with_finite_parameter_gradients(edges):
    inputs = list(_inputs(nodes=4))
    inputs[5] = edges
    inputs[4][:, 0] = False
    inputs[0] = torch.where(inputs[4], inputs[0], torch.full_like(inputs[0], float("nan")))
    model = _model()
    prediction = model(*inputs)
    assert torch.isfinite(prediction).all()
    loss = (prediction - 62.0).abs().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert model.temporal.projections[0][0].weight.grad.abs().sum() > 0
    assert model.global_mixer.tokens.grad.abs().sum() > 0
    assert model.normal_head.weight.grad.abs().sum() > 0
    assert model.disturbance_head.weight.grad.abs().sum() > 0


def test_raw_flow_output_restores_runtime_scale_exactly_once():
    model = _model()
    model.eval()
    history, baseline, future, time, observed, edges, scale = _inputs()
    with torch.no_grad():
        original = model(history, baseline, future, time, observed, edges, scale)
        transformed = model(history * 7 + 100, baseline * 7 + 100, future * 7 + 100, time, observed, edges, scale * 7)
    assert model.output_units == "raw_flow"
    torch.testing.assert_close(transformed, original * 7 + 100, rtol=2e-5, atol=2e-5)
    aux = model.aux
    torch.testing.assert_close(transformed, aux["g"] * aux["yn"] + (1 - aux["g"]) * aux["yd"])


def test_batch_specific_scales_equal_broadcast_node_scales():
    model = _model()
    model.eval()
    inputs = list(_inputs())
    with torch.no_grad():
        expected = model(*inputs)
        inputs[-1] = inputs[-1].unsqueeze(0).expand(2, -1)
        actual = model(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_forward_rejects_future_targets_and_affected_labels():
    signature = inspect.signature(SharedGraphNDR.forward)
    forbidden = {"target", "y_true", "affected", "affected_mask", "target_mask", "region_code"}
    assert not forbidden.intersection(signature.parameters)
    model = _model()
    with pytest.raises(TypeError):
        model(*_inputs(), target=torch.ones(2, 5, 12))
    with pytest.raises(TypeError):
        model(*_inputs(), affected_mask=torch.zeros(2, 5, dtype=torch.bool))


def test_global_attention_uses_only_node_to_latent_and_latent_to_node_pairs():
    model = _model(latent_tokens=8)
    captured = []

    def record(module, args):
        query, key, value = args
        captured.append((query.shape[-2], key.shape[-2], value.shape[-2]))

    first = model.global_mixer.to_tokens.register_forward_pre_hook(record)
    second = model.global_mixer.to_nodes.register_forward_pre_hook(record)
    with torch.no_grad():
        model(*_inputs(nodes=19))
    first.remove()
    second.remove()
    assert captured == [(8, 19, 19), (19, 8, 8)]


@pytest.mark.parametrize("invalid", ["negative_scale", "nonfinite_observed", "invalid_edge", "nonboolean_mask"])
def test_invalid_runtime_inputs_fail_closed(invalid):
    model = _model()
    inputs = list(_inputs())
    if invalid == "negative_scale":
        inputs[-1][0] = -1
    elif invalid == "nonfinite_observed":
        inputs[0][0, 0, 0] = float("nan")
    elif invalid == "invalid_edge":
        inputs[5][0, 0] = 100
    else:
        inputs[4] = inputs[4].float()
    with pytest.raises((ValueError, FloatingPointError)):
        model(*inputs)
