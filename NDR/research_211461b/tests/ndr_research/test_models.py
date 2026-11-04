from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from ndr_research.models import (
    MaskSignedNDR,
    SanitizedNDR,
    build_research_loss_normalizers,
    compute_research_loss,
    forward_research,
    make_research_model,
    projected_oracle_teacher,
    research_factory_kwargs,
    research_persistence,
)
from ndr_research.routers import MaskSignedGate, persistence_anchor, reliable_history
from scripts.train_staeformer_deseason import STAEformerDeseason, VARIANTS, compute_loss


def _inputs(nodes=3, batch=2):
    torch.manual_seed(7)
    history = 50 + torch.randn(batch, nodes, 12)
    baseline = torch.full_like(history, 48)
    future = torch.full_like(history, 51)
    time = torch.zeros(batch, 12, 2)
    time[..., 0] = 80 / 288
    time[..., 1] = 2 / 7
    mask = torch.ones_like(history, dtype=torch.bool)
    return history, baseline, future, time, mask


def test_baseline_is_the_original_s0g_and_has_exact_forward_and_loss_parity():
    torch.manual_seed(11)
    original = STAEformerDeseason(3, 12, 12, **dict(VARIANTS["s0g"]))
    torch.manual_seed(11)
    baseline = make_research_model(3, variant="baseline")
    assert type(baseline) is STAEformerDeseason
    assert original.state_dict().keys() == baseline.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
    inputs = _inputs()
    original.eval()
    baseline.eval()
    with torch.no_grad():
        expected = original(*inputs[:4])
        actual = baseline(*inputs[:4])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = inputs[2] + 2
    original.train()
    baseline.train()
    expected_loss, _ = compute_loss(
        original, expected,
        {"y_true": target.unsqueeze(-1), "y_mask": inputs[4].unsqueeze(-1)},
        SimpleNamespace(lambda_gate=1.0, lambda_spec=0.3), "cpu",
    )
    actual_loss, _ = compute_research_loss(baseline, actual, target, inputs[4])
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)


def test_projected_variant_changes_only_the_loss_teacher():
    torch.manual_seed(13)
    baseline = make_research_model(3, variant="baseline")
    torch.manual_seed(13)
    projected = make_research_model(3, variant="projected_oracle")
    assert type(projected) is STAEformerDeseason
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, projected.state_dict()[name], rtol=0, atol=0)
    assert projected.research_spec["teacher"] == "projected_raw_anchor"
    inputs = _inputs()
    baseline.eval()
    projected.eval()
    with torch.no_grad():
        torch.testing.assert_close(baseline(*inputs[:4]), projected(*inputs[:4]), rtol=0, atol=0)


def test_zero_initialized_mask_router_matches_original_on_complete_history():
    inputs = _inputs()
    torch.manual_seed(17)
    baseline = make_research_model(3)
    torch.manual_seed(17)
    candidate = make_research_model(3, variant="mask_signed")
    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        expected = baseline(*inputs[:4])
        actual = candidate(*inputs[:4], history_mask=inputs[4])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sanitized_control_isolates_the_history_policy_from_the_router():
    history, baseline, future, time, mask = _inputs()
    mask[:, 0, :3] = False
    mask[:, 1, -3:] = False
    history = torch.where(mask, history, torch.full_like(history, float("nan")))
    torch.manual_seed(23)
    control = make_research_model(3, variant="sanitized_control")
    torch.manual_seed(23)
    candidate = make_research_model(3, variant="mask_signed")
    control.eval()
    candidate.eval()
    with torch.no_grad():
        expected = forward_research(control, history, baseline, future, time, mask)
        actual = forward_research(candidate, history, baseline, future, time, mask)
    assert type(control) is SanitizedNDR
    assert control.research_spec["history_policy"] == candidate.research_spec["history_policy"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["baseline", "projected_oracle", "sanitized_control", "mask_signed"])
def test_recipe_rebuild_and_anchor_helper_match_forward(variant):
    inputs = _inputs()
    model = make_research_model(
        3, variant=variant, residual_std=3.0, residual_std_by_node=[2.0, 3.0, 4.0],
        reliable_anchor=variant == "mask_signed", flow_mean=50.0, flow_std=2.0,
    )
    kwargs = research_factory_kwargs(model)
    reconstructed = make_research_model(**kwargs)
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert model.research_spec == reconstructed.research_spec
    model.eval()
    reconstructed.eval()
    with torch.no_grad():
        expected = forward_research(model, *inputs)
        actual = forward_research(reconstructed, *inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.aux["P"], research_persistence(model, inputs[0], inputs[1], inputs[4]))


def test_shared_gate_is_node_permutation_equivariant_after_nonzero_training_update():
    history, baseline, future, _, mask = _inputs(nodes=5)
    mask[:, 0, -3:] = False
    scales = torch.arange(1, 6).float()
    gate = MaskSignedGate(12)
    with torch.no_grad():
        gate.correction[-1].weight.normal_(std=0.2)
        gate.correction[-1].bias.fill_(0.1)
    state = reliable_history(history, baseline, mask)
    anchor = persistence_anchor(state, baseline)
    expected = gate(state, baseline, future, anchor, torch.tensor(3.0), scales)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    state_p = reliable_history(history[:, permutation], baseline[:, permutation], mask[:, permutation])
    actual = gate(
        state_p, baseline[:, permutation], future[:, permutation],
        persistence_anchor(state_p, baseline[:, permutation]), torch.tensor(3.0), scales[permutation],
    )
    torch.testing.assert_close(actual, expected[:, permutation], rtol=0, atol=0)


def test_one_gate_state_handles_different_node_counts_without_parameter_changes():
    gate = MaskSignedGate(12)
    identity = {name: tuple(value.shape) for name, value in gate.state_dict().items()}
    for nodes in (2, 7):
        history, baseline, future, _, mask = _inputs(nodes=nodes)
        state = reliable_history(history, baseline, mask)
        result = gate(
            state, baseline, future, persistence_anchor(state, baseline),
            torch.tensor(3.0), torch.ones(nodes),
        )
        assert result.shape == (2, nodes, 12)
        assert identity == {name: tuple(value.shape) for name, value in gate.state_dict().items()}


@pytest.mark.parametrize("reliable_anchor", [False, True])
def test_missing_sentinels_cannot_change_candidate_forecasts(reliable_anchor):
    history, baseline, future, time, mask = _inputs()
    mask[:, 0, :] = False
    mask[:, 1, :3] = False
    mask[:, 2, -3:] = False
    first = torch.where(mask, history, torch.full_like(history, float("nan")))
    second = torch.where(mask, history, torch.full_like(history, -1e9))
    model = make_research_model(3, variant="mask_signed", reliable_anchor=reliable_anchor)
    with torch.no_grad():
        model.gate_mod.correction[-1].weight.normal_(std=0.2)
        model.gate_mod.correction[-1].bias.fill_(0.1)
    model.eval()
    with torch.no_grad():
        expected = model(first, baseline, future, time, mask)
        actual = model(second, baseline, future, time, mask)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reliable_reference_is_age_limited_and_never_uses_future_values():
    history, baseline, _, _, mask = _inputs()
    mask[:, 0, -3:] = False
    mask[:, 1, -1:] = False
    mask[:, 2, :] = False
    state = reliable_history(history, baseline, mask)
    anchor = persistence_anchor(state, baseline, reliable=True, max_age=2)
    torch.testing.assert_close(anchor[:, 0, 0], baseline[:, 0, -1])
    torch.testing.assert_close(anchor[:, 1, 0], history[:, 1, -2])
    torch.testing.assert_close(anchor[:, 2, 0], baseline[:, 2, -1])
    torch.testing.assert_close(state.age, torch.tensor([[3, 1, 12], [3, 1, 12]]))


def test_recovery_feature_distinguishes_rising_and_falling_deviations():
    baseline = torch.full((1, 2, 12), 50.0)
    history = baseline.clone()
    history[:, 0] += torch.linspace(1, 12, 12)
    history[:, 1] += torch.linspace(12, 1, 12)
    observed = torch.ones_like(history, dtype=torch.bool)
    state = reliable_history(history, baseline, observed)
    gate = MaskSignedGate(12)
    features = gate.features(state, baseline, baseline, persistence_anchor(state, baseline), torch.ones(2))
    recovery_index = gate.feature_names.index("signed_recovery")
    assert features[0, 0, 0, recovery_index] < 0
    assert features[0, 1, 0, recovery_index] > 0


def test_forward_apis_do_not_accept_future_targets_or_analysis_labels():
    for cls in (STAEformerDeseason, SanitizedNDR, MaskSignedNDR):
        names = inspect.signature(cls.forward).parameters
        assert not {"target", "y_true", "affected", "affected_mask", "target_mask"}.intersection(names)
    model = make_research_model(3, variant="mask_signed")
    inputs = _inputs()
    with pytest.raises(TypeError):
        model(*inputs[:4], history_mask=inputs[4], target=inputs[2])
    with pytest.raises(TypeError):
        model(*inputs[:4], history_mask=inputs[4], affected_mask=inputs[4])


def test_projected_teacher_is_the_analytic_minimum_including_outside_anchor_interval():
    baseline = torch.tensor([[[100.0, 60.0, 100.0, 60.0]]])
    persistence = torch.tensor([[[60.0, 100.0, 60.0, 100.0]]])
    target = torch.tensor([[[120.0, 40.0, 80.0, 80.0]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    teacher = projected_oracle_teacher(target, baseline, persistence, mask, torch.ones(1))
    torch.testing.assert_close(teacher.target, torch.tensor([[[1.0, 1.0, 0.5, 0.5]]]))
    optimum_error = (teacher.target * baseline + (1 - teacher.target) * persistence - target).abs()
    grid = torch.linspace(0, 1, 1001).view(-1, 1, 1, 1)
    grid_error = (grid * baseline + (1 - grid) * persistence - target).abs().amin(0)
    torch.testing.assert_close(optimum_error, grid_error)


def test_ambiguous_anchors_have_zero_confidence_and_masked_target_sentinels_are_ignored():
    target = torch.tensor([[[80.0, float("nan"), 100.0]]])
    baseline = torch.tensor([[[50.0, 60.0, 50.0 + 1e-7]]])
    persistence = torch.tensor([[[50.0, 40.0, 50.0]]])
    mask = torch.tensor([[[True, False, True]]])
    teacher = projected_oracle_teacher(target, baseline, persistence, mask, torch.ones(1))
    torch.testing.assert_close(teacher.confidence, torch.zeros_like(target))
    torch.testing.assert_close(teacher.target, torch.full_like(target, 0.5))


@pytest.mark.parametrize("variant", ["baseline", "projected_oracle", "sanitized_control", "mask_signed"])
def test_loss_microbatch_gradients_match_effective_batch_with_unequal_masks(variant):
    model = make_research_model(2, variant=variant)
    model.train()
    target = torch.tensor([[[110.0, 75.0], [95.0, 80.0]], [[90.0, 55.0], [85.0, 90.0]]])
    mask = torch.tensor([[[True, True], [True, False]], [[False, True], [True, True]]])
    baseline = torch.full_like(target, 100)
    persistence = torch.full_like(target, 60)
    original = torch.tensor([0.2, 70.0, 50.0], requires_grad=True)

    def loss_for(parameters, batch_slice, normalizers=None):
        selected = target[batch_slice]
        gate = parameters[0].sigmoid().expand_as(selected)
        normal = parameters[1].expand_as(selected)
        disturbed = parameters[2].expand_as(selected)
        prediction = gate * normal + (1 - gate) * disturbed
        model.aux = {
            "g": gate, "B": baseline[batch_slice], "P": persistence[batch_slice],
            "yn": normal, "yd": disturbed,
        }
        return compute_research_loss(
            model, prediction, selected, mask[batch_slice], normalizers=normalizers,
        )[0]

    full_loss = loss_for(original, slice(None))
    full_loss.backward()
    micro_parameters = original.detach().clone().requires_grad_(True)
    normalizers = build_research_loss_normalizers(
        model, target, mask, baseline=baseline, persistence=persistence,
    )
    total = 0
    for row in range(2):
        local_loss = loss_for(micro_parameters, slice(row, row + 1), normalizers)
        total += local_loss.detach()
        local_loss.backward()
    torch.testing.assert_close(total, full_loss.detach(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(micro_parameters.grad, original.grad, rtol=1e-6, atol=1e-6)
