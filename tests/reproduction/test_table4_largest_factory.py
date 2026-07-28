from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.reproduction.igstgnn_table4.execution import (
    capture_test_predictions,
    load_model_state_checkpoint,
    save_model_state_checkpoint,
)
from scripts.reproduction.igstgnn_table4.largest_factory import (
    LargeSTFactoryResult,
    build_largest_factory,
)
from scripts.reproduction.igstgnn_table4.profiles import load_profile_set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "profiles.public_material_v2.json"
)


class _IdentityScaler:
    def inverse_transform(self, value):
        return value


class _Loader:
    num_batch = 2


class _LifecycleLoader:
    def __init__(self, *, input_dim, seed):
        generator = np.random.default_rng(seed)
        self.x = generator.uniform(
            1.0,
            20.0,
            size=(2, 12, 4, input_dim),
        ).astype(np.float32)
        if input_dim == 3:
            self.x[..., 1] = np.arange(12, dtype=np.float32)[None, :, None] / 288
            self.x[..., 2] = np.float32(2 / 7)
        self.y = generator.uniform(
            1.0,
            20.0,
            size=(2, 12, 4, 1),
        ).astype(np.float32)
        self.size = 2
        self.num_batch = 1
        self.current_ind = 0
        self.shuffle_calls = 0

    def shuffle(self):
        self.shuffle_calls += 1

    def get_iterator(self):
        self.current_ind = 1
        return iter([(self.x.copy(), self.y.copy())])


def _dataloader():
    return {
        "train_loader": _Loader(),
        "val_loader": _Loader(),
        "test_loader": _Loader(),
    }


def _lifecycle_dataloader(method):
    input_dim = 1 if method in {"hl", "dstagnn"} else 3
    return {
        "train_loader": _LifecycleLoader(input_dim=input_dim, seed=1),
        "val_loader": _LifecycleLoader(input_dim=input_dim, seed=2),
        "test_loader": _LifecycleLoader(input_dim=input_dim, seed=3),
    }


def _adjacency(node_num=4):
    # A four-node directed ring is the shared non-degenerate graph fixture.
    # The fixed 12-step profile also satisfies STGCN's two temporal blocks
    # and DSTAGNN's widest seven-step gated temporal convolution.
    adjacency = np.eye(node_num, dtype=np.float32)
    for node in range(node_num):
        adjacency[node, (node + 1) % node_num] = 1.0
    return adjacency


def _traffic_inputs(node_num=4):
    # DGCRN and D2STGNN index learned 288-slot and seven-day embeddings, so
    # their synthetic calendar channels must stay in the documented ranges.
    x = torch.randn(2, 12, node_num, 3)
    x[..., 1] = torch.arange(12, dtype=torch.float32).view(1, 12, 1) / 288
    x[..., 2] = 2 / 7
    y = torch.randn(2, 12, node_num, 1)
    return x, y


def test_hl_factory_builds_pinned_upstream_engine_and_forecasts_12_steps(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["hl"]["effective"]
    logger = logging.getLogger("test_table4_largest_factory.hl")

    result = build_largest_factory(
        method="hl",
        profile_effective=effective,
        node_num=4,
        device=torch.device("cpu"),
        adjacency=None,
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logger,
        output_dir=tmp_path / "hl",
    )

    assert isinstance(result, LargeSTFactoryResult)
    assert result.engine.model is result.model
    assert result.optimizer is None
    assert result.scheduler is None
    assert result.effective_config == effective
    assert result.provenance["source_id"] == "largest_b7c6ff5"
    assert result.provenance["commit"] == ("b7c6ff5f7f0c10051ed10a319b2cfe58021312f9")

    x = torch.randn(2, 12, 4, 1)
    with torch.no_grad():
        prediction = result.model(x)
    assert prediction.shape == (2, 12, 4, 1)
    np.testing.assert_allclose(
        prediction.numpy(),
        x[:, -1:, :, :].expand(-1, 12, -1, -1).numpy(),
    )


def test_lstm_factory_maps_profile_optimizer_and_forward_shape(tmp_path):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["lstm"]["effective"]
    logger = logging.getLogger("test_table4_largest_factory.lstm")

    result = build_largest_factory(
        method="lstm",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=None,
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logger,
        output_dir=tmp_path / "lstm",
    )

    assert result.optimizer.param_groups[0]["lr"] == 0.001
    assert result.optimizer.param_groups[0]["weight_decay"] == 0.0001
    assert result.scheduler is None
    assert result.provenance["engine_class"] == "src.base.engine.BaseEngine"
    x = torch.randn(2, 12, 4, 3)
    y = torch.randn(2, 12, 4, 1)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x, y)
    assert prediction.shape == (2, 12, 4, 1)


def test_learned_factory_applies_profile_seed_before_model_construction(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["lstm"]["effective"]
    common = {
        "method": "lstm",
        "profile_effective": effective,
        "node_num": 4,
        "device": "cpu",
        "adjacency": None,
        "dataloader": _dataloader(),
        "scaler": _IdentityScaler(),
        "logger": logging.getLogger("test_table4_largest_factory.seed"),
    }

    torch.manual_seed(7)
    first = build_largest_factory(**common, output_dir=tmp_path / "first")
    torch.manual_seed(999)
    second = build_largest_factory(**common, output_dir=tmp_path / "second")

    first_parameter = next(first.model.parameters()).detach()
    second_parameter = next(second.model.parameters()).detach()
    torch.testing.assert_close(first_parameter, second_parameter)
    assert first.provenance["seed_applied"] == 2023
    assert torch.get_num_threads() == 3
    assert first.provenance["runtime_contract"]["torch_num_threads"] == 3
    assert first.factory_config == second.factory_config
    assert (
        first.provenance["factory_config_hash"]
        == second.provenance["factory_config_hash"]
    )
    assert len(first.provenance["factory_config_hash"]) == 64


def test_dcrnn_factory_applies_only_scheduler_verbose_compatibility(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["dcrnn"]["effective"]
    result = build_largest_factory(
        method="dcrnn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=_adjacency(),
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.dcrnn"),
        output_dir=tmp_path / "dcrnn",
    )

    assert isinstance(result.scheduler, torch.optim.lr_scheduler.MultiStepLR)
    assert result.scheduler.milestones == {10: 1, 50: 1, 90: 1}
    assert result.provenance["compatibility_patches_applied_by_factory"] == [
        "dcrnn_scheduler_verbose_compatibility_v1"
    ]
    x = torch.randn(2, 12, 4, 3)
    y = torch.randn(2, 12, 4, 1)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x, y)
    assert prediction.shape == (2, 12, 4, 1)


def test_agcrn_factory_uses_original_initializing_engine_and_forward_shape(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["agcrn"]["effective"]
    result = build_largest_factory(
        method="agcrn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=None,
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.agcrn"),
        output_dir=tmp_path / "agcrn",
    )

    assert result.provenance["engine_class"] == (
        "src.engines.agcrn_engine.AGCRN_Engine"
    )
    assert result.scheduler is None
    x = torch.randn(2, 12, 4, 3)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x)
    assert prediction.shape == (2, 12, 4, 1)


def test_stgcn_factory_builds_profile_scheduler_without_mutating_adjacency(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["stgcn"]["effective"]
    adjacency = _adjacency()
    original = adjacency.copy()
    result = build_largest_factory(
        method="stgcn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=adjacency,
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.stgcn"),
        output_dir=tmp_path / "stgcn",
    )

    np.testing.assert_array_equal(adjacency, original)
    assert isinstance(result.scheduler, torch.optim.lr_scheduler.StepLR)
    assert result.scheduler.step_size == 10
    x = torch.randn(2, 12, 4, 3)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x)
    assert prediction.shape == (2, 12, 4, 1)


def test_gwnet_factory_builds_double_transition_plus_adaptive_graph(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["gwnet"]["effective"]
    result = build_largest_factory(
        method="gwnet",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=_adjacency(),
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.gwnet"),
        output_dir=tmp_path / "gwnet",
    )

    assert len(result.model.supports) == 2
    assert result.model.adp_adj is True
    assert result.scheduler is None
    x = torch.randn(2, 12, 4, 3)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x)
    assert prediction.shape == (2, 12, 4, 1)


def test_astgcn_factory_builds_chebyshev_graph_and_cosine_scheduler(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["astgcn"]["effective"]
    result = build_largest_factory(
        method="astgcn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=_adjacency(),
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.astgcn"),
        output_dir=tmp_path / "astgcn",
    )

    assert isinstance(result.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert result.provenance["engine_class"] == (
        "src.engines.astgcn_engine.ASTGCN_Engine"
    )
    x = torch.randn(2, 12, 4, 3)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x)
    assert prediction.shape == (2, 12, 4, 1)


def test_sttn_factory_builds_original_transformer_and_forward_shape(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["sttn"]["effective"]
    result = build_largest_factory(
        method="sttn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=_adjacency(),
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.sttn"),
        output_dir=tmp_path / "sttn",
    )

    assert len(result.model.supports) == 2
    assert len(result.model.t_modules) == 2
    x = torch.randn(2, 12, 4, 3)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x)
    assert prediction.shape == (2, 12, 4, 1)


def test_dstagnn_factory_uses_flow_only_contract_and_original_engine(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["dstagnn"]["effective"]
    result = build_largest_factory(
        method="dstagnn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=_adjacency(),
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.dstagnn"),
        output_dir=tmp_path / "dstagnn",
    )

    assert result.model.input_dim == 1
    assert result.provenance["engine_class"] == (
        "src.engines.dstagnn_engine.DSTAGNN_Engine"
    )
    x = torch.randn(2, 12, 4, 1)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x)
    assert prediction.shape == (2, 12, 4, 1)


def test_dgcrn_factory_maps_288_slots_and_curriculum_engine(tmp_path):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["dgcrn"]["effective"]
    result = build_largest_factory(
        method="dgcrn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=_adjacency(),
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.dgcrn"),
        output_dir=tmp_path / "dgcrn",
    )

    assert result.model.tpd == 288
    assert result.engine._step_size == 2500
    assert result.engine._horizon == 12
    assert result.provenance["runtime_defaults"] == {
        "middle_dim": 2,
        "list_weight": [0.05, 0.95, 0.95],
        "cl_decay_step": 2000,
        "dropout": 0.3,
    }
    x, y = _traffic_inputs()
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x, y)
    assert prediction.shape == (2, 12, 4, 1)


def test_d2stgnn_factory_derives_curriculum_steps_from_original_loader(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["d2stgnn"]["effective"]
    result = build_largest_factory(
        method="d2stgnn",
        profile_effective=effective,
        node_num=4,
        device="cpu",
        adjacency=_adjacency(),
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_largest_factory.d2stgnn"),
        output_dir=tmp_path / "d2stgnn",
    )

    assert result.engine._cl_step == 6
    assert result.engine._warm_step == 60
    assert isinstance(result.scheduler, torch.optim.lr_scheduler.MultiStepLR)
    assert result.provenance["runtime_defaults"] == {"dropout": 0.1}
    x, y = _traffic_inputs()
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x, y)
    assert prediction.shape == (2, 12, 4, 1)


@pytest.mark.parametrize(
    "method",
    (
        "hl",
        "lstm",
        "dcrnn",
        "agcrn",
        "stgcn",
        "gwnet",
        "astgcn",
        "sttn",
        "dstagnn",
        "dgcrn",
        "d2stgnn",
    ),
)
def test_largest_method_completes_single_batch_artifact_lifecycle(
    method,
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    dataloader = _lifecycle_dataloader(method)
    result = build_largest_factory(
        method=method,
        profile_effective=profile_set["profiles"][method]["effective"],
        node_num=4,
        device="cpu",
        adjacency=(_adjacency() if method not in {"hl", "lstm", "agcrn"} else None),
        dataloader=dataloader,
        scaler=_IdentityScaler(),
        logger=logging.getLogger(f"test_table4_lifecycle.{method}"),
        output_dir=tmp_path / method,
    )

    if method != "hl":
        metrics = result.engine.train_batch()
        assert np.isfinite(metrics).all()
        assert dataloader["train_loader"].shuffle_calls == 1
        assert result.optimizer.state

    predictions, labels = capture_test_predictions(
        method=method,
        engine=result.engine,
    )
    assert predictions.shape == labels.shape == (2, 12, 4)
    assert predictions.dtype == labels.dtype == np.float32
    assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(
        labels,
        dataloader["test_loader"].y[..., 0],
    )

    checkpoint = save_model_state_checkpoint(
        model=result.model,
        target=tmp_path / f"{method}.pt",
    )
    float_states = [
        (name, value)
        for name, value in result.model.state_dict().items()
        if torch.is_floating_point(value) and value.numel()
    ]
    if float_states:
        state_name, state_value = float_states[0]
        expected = state_value.detach().clone()
        with torch.no_grad():
            state_value.add_(1)
        load_model_state_checkpoint(
            model=result.model,
            source=checkpoint,
        )
        torch.testing.assert_close(
            result.model.state_dict()[state_name],
            expected,
        )
    else:
        load_model_state_checkpoint(
            model=result.model,
            source=checkpoint,
        )
