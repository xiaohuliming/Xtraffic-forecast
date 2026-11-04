from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ndr_research.data import ResearchData, ResearchEventLoader
from ndr_research.models import compute_research_loss, make_research_model, research_factory_kwargs, research_persistence
from ndr_research.multichannel import (
    AUXILIARY_CHANNELS,
    CACHE_CHANNEL_NAMES,
    MULTICHANNEL_AUXILIARY_CHANNELS,
    MultiChannelData,
    MultiChannelEventLoader,
    MultiChannelNDR,
    fit_channel_climatology,
    load_multichannel_data,
    make_multichannel_model,
    multichannel_factory_kwargs,
    prepare_multichannel_batch,
)
from scripts import train_ndr_multichannel as mc
from scripts import train_ndr_research as base_trainer
from scripts.protocol_v3_event_extension_data import WeeklyClimatology, fit_weekly_climatology

CHANNELS = MULTICHANNEL_AUXILIARY_CHANNELS
AUX_SCALES = [0.02, 5.0]
REPO = Path(__file__).resolve().parents[2]
CACHE = Path("/Users/xhlm/Desktop/Study/科研实习/outputs/dist_net/region_data_v3_noleak_dedup")
STAGE1 = Path("/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013/stage1_copied/tasks")


class TinyMultiChannelRegion:
    N = 2
    T_h = 12
    T_p = 12
    include_analysis_labels = False
    region_name = "Alameda"
    train_end = 50
    val_end = 100
    split = np.asarray([0] * 6 + [1] * 3 + [2], dtype=np.int8)
    sample_start = np.asarray([12, 14, 16, 18, 20, 22, 65, 67, 69, 115], dtype=np.int64)

    def __init__(self, auxiliary_observed: bool = True) -> None:
        self.auxiliary_observed = auxiliary_observed

    def get_sample(self, index: int) -> dict:
        if self.split[index] == 2:
            raise AssertionError("a test target was read")
        grid = np.arange(24, dtype=np.float32).reshape(2, 12)
        flow = grid + 30 + index
        occupancy = 0.05 + 0.002 * grid * (index % 3 + 1)
        speed = 62 - 0.5 * grid - index
        history = np.stack((flow, occupancy, speed), axis=-1)
        x_mask = np.ones((2, 12, 3), dtype=np.bool_)
        x_mask[0, -2:, 0] = False
        x_mask[0, 5, 1] = False
        x_mask[1, :3, 2] = False
        # The cache imputes unobserved speed with zeros; the candidate must ignore them.
        history[1, :3, 2] = 0.0
        if not self.auxiliary_observed:
            x_mask[..., 1:] = False
        target = flow + index + 1
        y_mask = np.ones_like(x_mask)
        y_mask[0, :index % 11, 0] = False
        return {
            "x_hist": history,
            "x_hist_mask": x_mask,
            "y_true": np.repeat(target[..., None], 3, axis=-1),
            "y_mask": y_mask,
            "time_feat": np.stack((np.arange(12, dtype=np.float32) / 288,
                                   np.zeros(12, dtype=np.float32)), axis=-1),
            "sample_start": self.sample_start[index],
        }


def flow_statistics() -> WeeklyClimatology:
    return WeeklyClimatology(
        baseline=np.full((7, 288, 2), 40.0, dtype=np.float32), flow_mean=40.0, flow_std=10.0,
        residual_std=5.0, residual_std_by_node=np.asarray([4.0, 6.0], dtype=np.float32),
        valid_training_points=100,
    )


def auxiliary_statistics() -> tuple[WeeklyClimatology, ...]:
    occupancy = np.full((7, 288, 2), 0.06, dtype=np.float32)
    occupancy[0, :12, 1] = 0.08
    speed = np.full((7, 288, 2), 60.0, dtype=np.float32)
    speed[0, 3:9, 0] = 55.0
    return (
        WeeklyClimatology(baseline=occupancy, flow_mean=0.05, flow_std=0.05, residual_std=AUX_SCALES[0],
                          residual_std_by_node=np.asarray([0.02, 0.03], dtype=np.float32),
                          valid_training_points=90),
        WeeklyClimatology(baseline=speed, flow_mean=60.0, flow_std=8.0, residual_std=AUX_SCALES[1],
                          residual_std_by_node=np.asarray([4.0, 6.0], dtype=np.float32),
                          valid_training_points=80),
    )


IDENTITY = {"region_cache_digest": "tiny_cache", "cache_bundle_digest": "tiny_bundle"}


def loader_kwargs(region, name, code, batch_size, seed):
    return dict(split=name, region_data=region, indices=np.flatnonzero(region.split == code),
                batch_size=batch_size, flow_mean=40.0, flow_std=10.0, sensor_support=None,
                include_incident=False, shuffle=name == "train", drop_last=False, seed=seed)


def tiny_data(batch_size=2, seed=42, auxiliary_observed=True) -> MultiChannelData:
    region = TinyMultiChannelRegion(auxiliary_observed)
    loaders = {name: MultiChannelEventLoader(auxiliary_channels=CHANNELS,
                                             **loader_kwargs(region, name, code, batch_size, seed))
               for name, code in (("train", 0), ("val", 1))}
    return MultiChannelData(region, flow_statistics(), loaders["train"], loaders["val"], dict(IDENTITY),
                            CHANNELS, auxiliary_statistics())


def tiny_flow_data(batch_size=2, seed=42, auxiliary_observed=True) -> ResearchData:
    region = TinyMultiChannelRegion(auxiliary_observed)
    loaders = {name: ResearchEventLoader(**loader_kwargs(region, name, code, batch_size, seed))
               for name, code in (("train", 0), ("val", 1))}
    return ResearchData(region, flow_statistics(), loaders["train"], loaders["val"], dict(IDENTITY))


def tiny_model() -> MultiChannelNDR:
    return make_multichannel_model(2, residual_std=5.0, residual_std_by_node=[4.0, 6.0], flow_mean=40.0,
                                   flow_std=10.0, auxiliary_residual_std=AUX_SCALES)


def tiny_baseline():
    return make_research_model(2, variant="baseline", residual_std=5.0, residual_std_by_node=[4.0, 6.0],
                               flow_mean=40.0, flow_std=10.0)


def config_for(data, *, epochs=2, microbatch=1):
    return {
        "epochs": epochs, "seed": 42, "batch_size": data.train.bs, "microbatch_size": microbatch,
        "learning_rate": 0.001, "weight_decay": 0.0001, "eta_min": 0.00001, "gradient_clip": 5.0,
        "lambda_gate": 1.0, "lambda_specialization": 0.3, "max_train_batches": None,
        "max_val_batches": None, "log_interval": 100, "claim_scope": "smoke_only",
        "source_snapshot": mc.core.source_snapshot(), "runtime": mc.core.runtime_identity(torch.device("cpu")),
    }


def model_inputs(nodes=3, batch=2):
    generator = torch.Generator().manual_seed(7)
    history = 50 + torch.randn(batch, nodes, 12, generator=generator)
    baseline = torch.full_like(history, 48)
    future = torch.full_like(history, 51)
    time = torch.zeros(batch, 12, 2)
    time[..., 0] = 80 / 288
    time[..., 1] = 2 / 7
    aux_hist = torch.stack((0.06 + 0.03 * torch.randn(batch, nodes, 12, generator=generator),
                            60 + 8 * torch.randn(batch, nodes, 12, generator=generator)), dim=-1)
    aux_baseline = torch.stack((torch.full_like(history, 0.05), torch.full_like(history, 62.0)), dim=-1)
    aux_mask = torch.ones_like(aux_hist, dtype=torch.bool)
    aux_mask[0, 0, :4, 1] = False
    aux_mask[1, 2, -3:, 0] = False
    return history, baseline, future, time, aux_hist, aux_baseline, aux_mask


def seeded_pair(nodes=3, seed=11):
    torch.manual_seed(seed)
    baseline = make_research_model(nodes, variant="baseline")
    torch.manual_seed(seed)
    candidate = make_multichannel_model(nodes, auxiliary_residual_std=AUX_SCALES)
    return baseline, candidate


def activate_auxiliary_projection(model: MultiChannelNDR) -> None:
    with torch.no_grad():
        model.aux_input_proj.copy_(0.3 * torch.randn(model.aux_input_proj.shape,
                                                     generator=torch.Generator().manual_seed(5)))


def disable_dropout(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not (STAGE1 / "alameda_baseline_seed42/config.json").exists(),
                    reason="stage-1 reference configuration unavailable")
def test_reference_snapshot_files_stay_byte_identical():
    for region in ("alameda", "contracosta"):
        snapshot = json.loads((STAGE1 / f"{region}_baseline_seed42/config.json").read_text())["source_snapshot"]
        changed = [relative for relative, value in snapshot["files"].items()
                   if sha256(REPO / relative) != value["sha256"]]
        assert changed == [], region


def test_core_is_an_independent_baseline_trainer_instance_with_four_bound_collaborators():
    assert mc.core is not base_trainer
    assert Path(mc.core.__file__).resolve() == Path(base_trainer.__file__).resolve()
    assert mc.core.batch_tensors is mc.batch_tensors and mc.core.model_forward is mc.model_forward
    assert mc.core.research_factory_kwargs is multichannel_factory_kwargs
    assert mc.core.make_research_model is make_multichannel_model
    assert base_trainer.batch_tensors is not mc.batch_tensors
    assert base_trainer.make_research_model is make_research_model
    for name in ("train_run", "evaluate", "backward_effective_batch", "optimize_batch", "verify_checkpoint",
                 "save_checkpoint", "restore_training", "load_checkpoint"):
        assert getattr(mc.core, name).__code__.co_code == getattr(base_trainer, name).__code__.co_code, name
    with pytest.raises(ValueError, match="exactly"):
        mc.load_baseline_trainer(batch_tensors=mc.batch_tensors)


def test_seeded_candidate_draws_the_original_weights_and_adds_only_a_zero_projection():
    baseline, candidate = seeded_pair()
    assert type(candidate) is MultiChannelNDR
    original, extended = baseline.state_dict(), candidate.state_dict()
    assert set(extended) - set(original) == {"aux_sd", "aux_input_proj"}
    for name, value in original.items():
        assert torch.equal(value, extended[name]), name
    assert torch.equal(candidate.aux_input_proj, torch.zeros(24, 2))
    torch.testing.assert_close(candidate.aux_sd, torch.tensor(AUX_SCALES))
    added = sum(value.numel() for value in candidate.parameters()) - sum(value.numel() for value in baseline.parameters())
    assert added == 48


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_zero_initialized_candidate_forward_is_bitwise_the_baseline(mode):
    baseline, candidate = seeded_pair()
    history, anchor, future, time, aux_hist, aux_baseline, aux_mask = model_inputs()
    for model in (baseline, candidate):
        model.train(mode == "train")
    with torch.no_grad():
        torch.manual_seed(3)
        expected = baseline(history, anchor, future, time)
        torch.manual_seed(3)
        actual = candidate(history, anchor, future, time, aux_hist, aux_baseline, aux_mask)
    assert torch.equal(actual, expected)
    for key in ("g", "B", "P", "yn", "yd"):
        assert torch.equal(candidate.aux[key], baseline.aux[key]), key


def test_zero_initialized_candidate_has_the_baseline_loss_and_shared_gradients():
    baseline, candidate = seeded_pair()
    history, anchor, future, time, aux_hist, aux_baseline, aux_mask = model_inputs()
    target = future + torch.linspace(-3, 3, 12)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 1, 4:] = False
    disable_dropout(baseline, candidate)
    baseline.train()
    candidate.train()
    expected_loss, expected_parts = compute_research_loss(baseline, baseline(history, anchor, future, time), target, mask)
    prediction = candidate(history, anchor, future, time, aux_hist, aux_baseline, aux_mask)
    actual_loss, actual_parts = compute_research_loss(candidate, prediction, target, mask)
    assert torch.equal(actual_loss, expected_loss)
    assert actual_parts == expected_parts
    expected_loss.backward()
    actual_loss.backward()
    shared = dict(baseline.named_parameters())
    for name, parameter in candidate.named_parameters():
        if name == "aux_input_proj":
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        elif shared[name].grad is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, shared[name].grad), name


def test_auxiliary_channels_change_the_forecast_once_the_projection_is_trained():
    _, candidate = seeded_pair()
    activate_auxiliary_projection(candidate)
    candidate.eval()
    history, anchor, future, time, aux_hist, aux_baseline, aux_mask = model_inputs()
    with torch.no_grad():
        first = candidate(history, anchor, future, time, aux_hist, aux_baseline, aux_mask)
        second = candidate(history, anchor, future, time, aux_hist + torch.tensor([0.05, -10.0]), aux_baseline, aux_mask)
    assert not torch.equal(first, second)
    torch.testing.assert_close(candidate.aux["P"], research_persistence(candidate, history, anchor))


@pytest.mark.parametrize("sentinel", [float("nan"), float("inf"), -1e9, 0.0])
def test_unobserved_auxiliary_points_cannot_change_or_poison_the_forecast(sentinel):
    _, candidate = seeded_pair()
    activate_auxiliary_projection(candidate)
    history, anchor, future, time, aux_hist, aux_baseline, aux_mask = model_inputs()
    aux_mask[:, 1] = False
    clean = torch.where(aux_mask, aux_hist, aux_baseline)
    poisoned = torch.where(aux_mask, aux_hist, torch.full_like(aux_hist, sentinel))
    candidate.eval()
    with torch.no_grad():
        expected = candidate(history, anchor, future, time, clean, aux_baseline, aux_mask)
        actual = candidate(history, anchor, future, time, poisoned, aux_baseline, aux_mask)
        signal = candidate.auxiliary_signal(poisoned, aux_baseline, aux_mask)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)
    assert torch.equal(signal[~aux_mask], torch.zeros_like(signal[~aux_mask]))
    candidate.train()
    target = future + 1
    loss, _ = compute_research_loss(candidate, candidate(history, anchor, future, time, poisoned, aux_baseline, aux_mask),
                                    target, torch.ones_like(target, dtype=torch.bool))
    loss.backward()
    assert all(torch.isfinite(value.grad).all() for value in candidate.parameters() if value.grad is not None)


def test_observed_non_finite_auxiliary_history_and_bad_shapes_are_rejected():
    _, candidate = seeded_pair()
    history, anchor, future, time, aux_hist, aux_baseline, aux_mask = model_inputs()
    aux_hist[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="observed auxiliary"):
        candidate(history, anchor, future, time, aux_hist, aux_baseline, aux_mask)
    with pytest.raises(ValueError, match="shape"):
        candidate(history, anchor, future, time, aux_hist[..., :1], aux_baseline, aux_mask)


def test_backbone_is_guarded_and_factory_rejects_other_recipes():
    _, candidate = seeded_pair()
    with pytest.raises(RuntimeError, match="must be called through"):
        candidate.backbone(torch.zeros(1, 12, 3, 3))
    with pytest.raises(ValueError, match="fitted auxiliary residual scales"):
        make_multichannel_model(3)
    with pytest.raises(ValueError, match="finite, positive"):
        make_multichannel_model(3, auxiliary_residual_std=[1.0, 0.0])
    with pytest.raises(ValueError, match="distinct names"):
        make_multichannel_model(3, auxiliary_channels=("speed", "speed"), auxiliary_residual_std=[1.0, 1.0])
    with pytest.raises(ValueError, match="only the multichannel"):
        make_multichannel_model(3, variant="baseline", auxiliary_residual_std=AUX_SCALES)
    with pytest.raises(ValueError, match="original causal anchor"):
        make_multichannel_model(3, reliable_anchor=True, auxiliary_residual_std=AUX_SCALES)
    with pytest.raises(TypeError, match="only MultiChannelNDR"):
        mc.model_forward(make_research_model(3), {})


def test_factory_recipe_rebuilds_the_candidate_and_records_the_input_scope():
    _, candidate = seeded_pair()
    activate_auxiliary_projection(candidate)
    kwargs = multichannel_factory_kwargs(candidate)
    assert {key: kwargs[key] for key in research_factory_kwargs(candidate)} == research_factory_kwargs(candidate)
    assert kwargs["auxiliary_channels"] == list(CHANNELS)
    assert kwargs["auxiliary_residual_std"] == pytest.approx(AUX_SCALES)
    rebuilt = make_multichannel_model(**json.loads(json.dumps(kwargs)))
    rebuilt.load_state_dict(candidate.state_dict(), strict=True)
    assert rebuilt.research_spec == candidate.research_spec
    spec = candidate.research_spec
    baseline_spec = make_research_model(3).research_spec
    assert {key: spec[key] for key in baseline_spec if key != "variant"} == {
        key: value for key, value in baseline_spec.items() if key != "variant"}
    assert spec["gate_features"] == "flow_only_unchanged"
    assert spec["auxiliary_missing_policy"] == "unobserved_points_enter_as_zero_residual"
    inputs = model_inputs()
    candidate.eval()
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(rebuilt(*inputs), candidate(*inputs))
    with pytest.raises(ValueError, match="require a MultiChannelNDR"):
        multichannel_factory_kwargs(make_research_model(3))


def test_deepcopy_keeps_an_independent_working_hook():
    _, candidate = seeded_pair()
    duplicate = copy.deepcopy(candidate)
    activate_auxiliary_projection(duplicate)
    inputs = model_inputs()
    candidate.eval()
    duplicate.eval()
    with torch.no_grad():
        original = candidate(*inputs)
        changed = duplicate(*inputs)
    assert torch.equal(candidate.aux_input_proj, torch.zeros(24, 2))
    assert not torch.equal(original, changed)


def test_loader_adds_aligned_auxiliary_history_without_touching_flow_fields():
    with_aux, without = tiny_data(), tiny_flow_data()
    batch, plain = with_aux.val.next_batch(), without.val.next_batch()
    assert set(batch) - set(plain) == {"x_raw_aux", "x_aux_mask"}
    for key, value in plain.items():
        np.testing.assert_array_equal(batch[key], value)
    region = with_aux.region
    for row, start in enumerate(batch["sample_start"]):
        sample = region.get_sample(int(np.flatnonzero(region.sample_start == start)[0]))
        np.testing.assert_array_equal(batch["x_raw_aux"][row], sample["x_hist"][..., [1, 2]])
        np.testing.assert_array_equal(batch["x_aux_mask"][row], sample["x_hist_mask"][..., [1, 2]])
    prepared = prepare_multichannel_batch(batch, with_aux.statistics, with_aux.auxiliary_statistics)
    flow_prepared = base_trainer.prepare_research_batch(plain, without.statistics)
    for key, value in flow_prepared.items():
        np.testing.assert_array_equal(prepared[key], value)
    assert set(prepared) - set(flow_prepared) == {"aux_hist", "aux_baseline", "aux_mask"}
    assert prepared["aux_hist"].shape == prepared["aux_baseline"].shape == prepared["aux_mask"].shape == (2, 2, 12, 2)
    np.testing.assert_array_equal(prepared["aux_baseline"][:, 1, :, 0], np.full((2, 12), 0.08, dtype=np.float32))
    expected_speed = np.full((2, 12), 60.0, dtype=np.float32)
    expected_speed[:, 3:9] = 55.0
    np.testing.assert_array_equal(prepared["aux_baseline"][:, 0, :, 1], expected_speed)
    assert not prepared["aux_mask"][:, 1, :3, 1].any()
    with pytest.raises(ValueError, match="require auxiliary history"):
        prepare_multichannel_batch(plain, without.statistics, with_aux.auxiliary_statistics)
    with pytest.raises(ValueError, match="matching its statistics"):
        prepare_multichannel_batch(batch, with_aux.statistics, ())


def test_loader_resume_state_is_the_research_loader_state():
    first, restored = tiny_data(), tiny_data()
    first.train.start_epoch()
    first.train.next_batch()
    flow = tiny_flow_data()
    flow.train.start_epoch()
    flow.train.next_batch()
    state = first.train.state_dict()
    assert state.keys() == flow.train.state_dict().keys()
    np.testing.assert_array_equal(state["order"], flow.train.state_dict()["order"])
    restored.train.load_state_dict(state)
    np.testing.assert_array_equal(first.train.next_batch()["x_raw_aux"], restored.train.next_batch()["x_raw_aux"])


def test_loader_rejects_misaligned_auxiliary_history_and_unknown_channels():
    data = tiny_data()
    original = data.region.get_sample
    calls = {"count": 0}

    def shifted(index):
        sample = original(index)
        calls["count"] += 1
        if calls["count"] > 2:
            sample["x_hist"] = sample["x_hist"] + 1
        return sample

    data.region.get_sample = shifted
    with pytest.raises(RuntimeError, match="not aligned"):
        data.val.next_batch()
    region = TinyMultiChannelRegion()
    for channels in (("flow",), ()):
        with pytest.raises(ValueError, match="distinct names"):
            MultiChannelEventLoader(auxiliary_channels=channels, **loader_kwargs(region, "val", 1, 2, 42))


def channel_region(steps=900):
    rng = np.random.default_rng(0)
    series = np.stack((rng.uniform(0, 400, (steps, 3)), rng.uniform(0, 0.3, (steps, 3)),
                       rng.uniform(20, 70, (steps, 3))), axis=-1).astype(np.float32)
    mask = rng.uniform(size=series.shape) > 0.2
    mask[:, 2, 2] = False
    positions = np.arange(steps)
    return SimpleNamespace(train_end=700, N=3, flow_series=series, flow_mask=mask,
                           dow=(positions // 288) % 7, tod=positions % 288)


def test_channel_climatology_reproduces_the_flow_fit_and_uses_only_its_own_training_channel():
    region = channel_region()
    expected, actual = fit_weekly_climatology(region), fit_channel_climatology(region, 0)
    np.testing.assert_array_equal(actual.baseline, expected.baseline)
    np.testing.assert_array_equal(actual.residual_std_by_node, expected.residual_std_by_node)
    assert (actual.flow_mean, actual.flow_std, actual.residual_std, actual.valid_training_points) == (
        expected.flow_mean, expected.flow_std, expected.residual_std, expected.valid_training_points)
    speed = fit_channel_climatology(region, 2)
    assert speed.valid_training_points == int(region.flow_mask[:700, :, 2].sum())
    assert np.all(speed.baseline[..., 2] == 0) and speed.residual_std_by_node[2] == 1.0
    changed = copy.deepcopy(region)
    changed.flow_series[700:, :, 2] += 1000
    changed.flow_series[:, :, 0] += 1000
    np.testing.assert_array_equal(fit_channel_climatology(changed, 2).baseline, speed.baseline)
    with pytest.raises(ValueError, match="outside flow"):
        fit_channel_climatology(region, 3)


@pytest.mark.parametrize("microbatch", [1, 2])
def test_microbatch_gradient_matches_effective_full_batch(microbatch):
    mc.core.configure_runtime("cpu", 42, 1)
    data = tiny_data(batch_size=4)
    tensors = mc.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    tensors["target_mask"][0] = False
    tensors["target"][0] = 0
    full = tiny_model()
    activate_auxiliary_projection(full)
    disable_dropout(full)
    micro = copy.deepcopy(full)
    full.train()
    micro.train()
    expected = mc.core.backward_effective_batch(full, tensors, microbatch_size=4)
    actual = mc.core.backward_effective_batch(micro, tensors, microbatch_size=microbatch)
    assert actual["valid_count"] == expected["valid_count"]
    assert actual["objective"] == pytest.approx(expected["objective"], rel=2e-6)
    for (name, left), right in zip(full.named_parameters(), micro.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-6, msg=name)
    assert full.aux_input_proj.grad.abs().sum() > 0


def train_both_trainers(tmp_path, gradient_clip):
    device = base_trainer.configure_runtime("cpu", 42, 1)
    flow_data = tiny_flow_data(auxiliary_observed=False)
    baseline = tiny_baseline()
    config = config_for(flow_data) | {"gradient_clip": gradient_clip}
    (tmp_path / "baseline").mkdir()
    expected = base_trainer.train_run(model=baseline, data=flow_data, device=device, config=config,
                                      output=tmp_path / "baseline")
    mc.core.configure_runtime("cpu", 42, 1)
    candidate = tiny_model()
    (tmp_path / "candidate").mkdir()
    actual = mc.core.train_run(model=candidate, data=tiny_data(auxiliary_observed=False), device=device,
                               config=config, output=tmp_path / "candidate")
    rows = {}
    for name in ("baseline", "candidate"):
        lines = (tmp_path / name / "training_diagnostics.jsonl").read_text().splitlines()
        rows[name] = [{key: value for key, value in json.loads(line).items()
                       if key not in ("train_seconds", "validation_seconds")} for line in lines]
    return baseline, candidate, expected, actual, rows


def test_uninformative_auxiliary_history_reproduces_the_baseline_trainer_bitwise(tmp_path):
    # With clipping inactive the reused loop must reproduce the baseline trainer exactly.
    baseline, candidate, expected, actual, rows = train_both_trainers(tmp_path, 1e9)
    assert actual["best_val_mae"] == expected["best_val_mae"]
    assert actual["best_validation"] == expected["best_validation"]
    assert rows["candidate"] == rows["baseline"]
    for name, value in baseline.state_dict().items():
        assert torch.equal(candidate.state_dict()[name], value), name
    assert torch.equal(candidate.aux_input_proj, torch.zeros(24, 2))


def test_uninformative_auxiliary_history_matches_the_baseline_within_clip_norm_rounding(tmp_path):
    # The clip norm reduces over one more, all-zero gradient, which can move its last bit.
    baseline, candidate, expected, actual, rows = train_both_trainers(tmp_path, 5.0)
    assert actual["best_val_mae"] == pytest.approx(expected["best_val_mae"], rel=1e-6)
    assert actual["best_epoch"] == expected["best_epoch"]
    assert rows["candidate"][0]["train_mae"] == rows["baseline"][0]["train_mae"]
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=1e-5, atol=1e-6, msg=name)
    assert torch.equal(candidate.aux_input_proj, torch.zeros(24, 2))


def test_interruption_resumes_identically_and_independent_verification_rebuilds_the_candidate(tmp_path, monkeypatch):
    device = mc.core.configure_runtime("cpu", 42, 1)
    continuous_data, continuous_model = tiny_data(), tiny_model()
    config = config_for(continuous_data)
    (tmp_path / "full").mkdir()
    mc.core.train_run(model=continuous_model, data=continuous_data, device=device, config=config,
                      output=tmp_path / "full")
    assert continuous_model.aux_input_proj.abs().sum() > 0

    mc.core.configure_runtime("cpu", 42, 1)
    interrupted_data, interrupted_model = tiny_data(), tiny_model()
    output = tmp_path / "resumed"
    output.mkdir()
    clock = iter(range(100))
    with monkeypatch.context() as limited:
        limited.setattr(mc.core.time, "monotonic", lambda: float(next(clock)))
        stopped = mc.core.train_run(model=interrupted_model, data=interrupted_data, device=device,
                                    config=config, output=output, max_wallclock_seconds=6)
    assert stopped["status"] == "stopped_resumable" and stopped["next_batch"] == 2
    checkpoint = mc.core.load_checkpoint(output / "last.pt")
    assert checkpoint["model_factory"]["variant"] == "multichannel"
    assert checkpoint["model_factory"]["auxiliary_channels"] == list(CHANNELS)
    restored_data, restored_model = tiny_data(), tiny_model()
    resumed = mc.core.train_run(model=restored_model, data=restored_data, device=device, config=config,
                                output=output, checkpoint=checkpoint)
    assert resumed["completed_epochs"] == 2
    for name, expected in continuous_model.state_dict().items():
        assert torch.equal(restored_model.state_dict()[name], expected), name
    verified = mc.core.verify_checkpoint(checkpoint_path=output / "best.pt", data=tiny_data(), device=device,
                                         output=output, current_runtime=mc.core.runtime_identity(device))
    assert verified["status"] == "verified_smoke_checkpoint_complete_validation"
    assert verified["absolute_mae_difference"] == 0


def test_cli_records_the_multichannel_recipe_and_verifies_in_a_new_call(tmp_path, monkeypatch):
    requested = []

    def fake_loader(**kwargs):
        requested.append(kwargs["region_name"])
        return tiny_data(batch_size=kwargs["batch_size"], seed=kwargs["seed"])

    monkeypatch.setattr(mc, "load_multichannel_data", fake_loader)
    output = tmp_path / "run"
    common = ["--device", "cpu", "--threads", "1"]
    result = mc.main(["--mode", "train", "--variant", "multichannel", "--region", "ContraCosta",
                      "--epochs", "1", "--batch-size", "2", "--microbatch-size", "1",
                      "--output-dir", str(output), *common])
    assert result["status"] == "training_completed_validation_pending_independent_verification"
    config = json.loads((output / "config.json").read_text())
    recipe = config["multichannel_recipe"]
    assert config["variant"] == "multichannel" and config["claim_scope"] == "validation_development_only"
    assert (config["batch_size"], config["microbatch_size"], config["learning_rate"]) == (2, 1, 0.001)
    assert (config["reliable_anchor"], config["max_reference_age"], config["hidden_dim"]) == (False, 2, 16)
    assert recipe["auxiliary_channels"] == list(CHANNELS)
    assert recipe["auxiliary_residual_std"] == AUX_SCALES
    assert recipe["gate_features"] == "flow only, unchanged"
    assert "not a flow-only comparison" in recipe["data_scope"]
    snapshot = config["source_snapshot"]["files"]
    assert {"scripts/train_ndr_multichannel.py", "scripts/train_ndr_research.py",
            "ndr_research/multichannel.py"} <= set(snapshot)
    checkpoint = mc.core.load_checkpoint(output / "best.pt")
    assert checkpoint["model_factory"]["auxiliary_residual_std"] == pytest.approx(AUX_SCALES)
    verified = mc.main(["--mode", "verify", "--checkpoint", str(output / "best.pt"), *common])
    assert verified["status"] == "verified_checkpoint_complete_validation"
    assert requested == ["ContraCosta", "ContraCosta"]


def test_cli_keeps_the_baseline_defaults_and_rejects_other_variants(tmp_path):
    parser, reference = mc.argument_parser(), base_trainer.argument_parser()
    defaults = vars(parser.parse_args([]))
    assert defaults.pop("variant") == "multichannel"
    expected = vars(reference.parse_args([]))
    expected.pop("variant")
    assert defaults == expected
    with pytest.raises(SystemExit):
        parser.parse_args(["--variant", "baseline"])
    with pytest.raises(ValueError, match="original anchor"):
        mc.main(["--mode", "train", "--reliable-anchor", "--device", "cpu", "--output-dir", str(tmp_path / "x")])


def test_cpu_calibration_reports_p90_budget_fields_without_scores(tmp_path):
    device = mc.core.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    result = mc.calibrate(tiny_model(), data, device, config_for(data, epochs=30), warmup=1, timed=3,
                          output=tmp_path)
    assert result["cuda_budget_eligible"] is False and result["device_type"] == "cpu"
    assert result["budget_epochs"] == 30 and result["status"] == "calibrated_engineering_only"
    assert result["p90_estimated_train_val_seconds"] > 0
    assert len(result["timed_train_batch_seconds"]) == 3
    assert result["p90_train_batch_seconds"] == pytest.approx(np.percentile(result["timed_train_batch_seconds"], 90))
    assert not any("mae" in key.lower() for key in result)
    assert json.loads((tmp_path / "calibration.json").read_text()) == result


@pytest.mark.skipif(not (CACHE / "cache_bundle_manifest.json").exists(), reason="formal local cache is unavailable")
def test_real_cache_candidate_batches_are_finite_and_match_the_baseline_at_initialization():
    data = load_multichannel_data(region_name="ContraCosta", data_dir=CACHE,
                                  graph_dir=CACHE.parent.parent / "region_graphs", batch_size=16)
    assert data.identity["cache_channel_names"] == list(CACHE_CHANNEL_NAMES)
    entries = data.identity["auxiliary_channels"]
    assert [entry["name"] for entry in entries] == list(CHANNELS)
    train_end = data.region.train_end
    for entry, statistics in zip(entries, data.auxiliary_statistics):
        channel = AUXILIARY_CHANNELS[entry["name"]]
        assert entry["valid_training_points"] == int(data.region.flow_mask[:train_end, :, channel].sum())
        assert np.isfinite(statistics.baseline).all() and statistics.residual_std > 0
    flow = fit_channel_climatology(data.region, 0)
    np.testing.assert_array_equal(flow.baseline, data.statistics.baseline)
    assert flow.residual_std == data.statistics.residual_std
    data.train.start_epoch()
    batch = data.train.next_batch()
    tensors = mc.batch_tensors(batch, data, torch.device("cpu"))
    flow_tensors = base_trainer.batch_tensors(batch, data, torch.device("cpu"))
    assert set(tensors) - set(flow_tensors) == {"aux_hist", "aux_baseline", "aux_mask"}
    for key, value in flow_tensors.items():
        assert torch.equal(tensors[key], value), key
    assert torch.isfinite(tensors["aux_hist"]).all() and torch.isfinite(tensors["aux_baseline"]).all()
    assert not tensors["aux_mask"].all()
    scales = [value.residual_std for value in data.auxiliary_statistics]
    torch.manual_seed(42)
    baseline = make_research_model(data.region.N, variant="baseline", residual_std=data.statistics.residual_std)
    torch.manual_seed(42)
    candidate = make_multichannel_model(data.region.N, residual_std=data.statistics.residual_std,
                                        auxiliary_residual_std=scales)
    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        signal = candidate.auxiliary_signal(tensors["aux_hist"], tensors["aux_baseline"], tensors["aux_mask"])
        expected = base_trainer.model_forward(baseline, flow_tensors)
        actual = mc.model_forward(candidate, tensors)
    assert torch.isfinite(signal).all()
    assert torch.equal(signal[~tensors["aux_mask"]], torch.zeros_like(signal[~tensors["aux_mask"]]))
    assert torch.equal(actual, expected)
