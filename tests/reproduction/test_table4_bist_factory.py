from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.bist_factory import (
    build_bist_factory,
)
from scripts.reproduction.igstgnn_table4.execution import (
    capture_test_predictions,
    load_model_state_checkpoint,
    save_model_state_checkpoint,
)
from scripts.reproduction.igstgnn_table4.profiles import load_profile_set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FROZEN_SEED = 424242
PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "profiles.public_material_v2.json"
)


class _IdentityScaler:
    @staticmethod
    def inverse_transform(value):
        return value


class _TupleLoader:
    def __init__(self, *, seed: int) -> None:
        generator = np.random.default_rng(seed)
        self.x = generator.normal(size=(2, 12, 4, 3)).astype(np.float32)
        self.x[..., 1] = np.float32(10 / 288)
        self.x[..., 2] = np.float32(2 / 7)
        self.y = generator.uniform(1, 10, size=(2, 12, 4, 1)).astype(np.float32)
        self.size = 2
        self.num_batch = 1
        self.current_ind = 0

    def shuffle(self):
        return None

    def get_iterator(self):
        self.current_ind = 1
        return iter([(self.x.copy(), self.y.copy())])


def _dataloader():
    return {
        "train_loader": _TupleLoader(seed=1),
        "val_loader": _TupleLoader(seed=2),
        "test_loader": _TupleLoader(seed=3),
    }


def test_bist_factory_maps_locked_public_profile_and_forward_shape(tmp_path):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["bist"]["effective"]
    result = build_bist_factory(
        profile_effective=effective,
        manifest_frozen_seed=MANIFEST_FROZEN_SEED,
        node_num=4,
        device="cpu",
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_bist_factory"),
        output_dir=tmp_path / "bist",
    )

    assert result.effective_config == effective
    assert result.provenance["source_id"] == "bist_dd94adf"
    assert result.provenance["commit"] == ("dd94adf7721fcbb9e3feb5d1b44040305199a4cc")
    assert result.provenance["license_spdx"] is None
    assert result.provenance["engine_class"] == "src.base.engine.BaseEngine"
    assert result.provenance["seed_policy"] == (
        "upstream_runtime_random_draw_frozen_in_manifest"
    )
    assert result.provenance["manifest_frozen_seed"] == MANIFEST_FROZEN_SEED
    assert result.model.stmodel.time_of_day_size == 96
    assert result.optimizer.param_groups[0]["lr"] == 0.002
    assert result.optimizer.param_groups[0]["weight_decay"] == 1e-5
    assert result.optimizer.param_groups[0]["eps"] == 1e-8
    assert result.scheduler.milestones == {30: 1, 60: 1, 80: 1, 100: 1}

    x = torch.randn(2, 12, 4, 3)
    x[..., 1] = 10 / 288
    x[..., 2] = 2 / 7
    y = torch.randn(2, 12, 4, 1)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(x, y)
    assert prediction.shape == (2, 12, 4, 1)


def test_bist_factory_native100_override_changes_only_epoch_cap(tmp_path):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["bist"]["effective"]
    result = build_bist_factory(
        profile_effective=effective,
        manifest_frozen_seed=MANIFEST_FROZEN_SEED,
        node_num=4,
        device="cpu",
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_bist_factory.native100"),
        output_dir=tmp_path / "bist_native100",
        max_epochs_override=100,
    )

    assert effective["max_epochs"] == 500
    assert result.effective_config["max_epochs"] == 100
    assert result.effective_config["patience"] == effective["patience"] == 20
    assert result.engine._max_epochs == 100
    assert result.engine._patience == 20
    assert result.provenance["effective_overrides"] == {
        "max_epochs": {"from": 500, "to": 100}
    }


def test_bist_factory_manifest_frozen_runtime_seed_is_reproducible(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["bist"]["effective"]
    first = build_bist_factory(
        profile_effective=effective,
        manifest_frozen_seed=MANIFEST_FROZEN_SEED,
        node_num=3,
        device="cpu",
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_bist_factory.seed.first"),
        output_dir=tmp_path / "first",
    )
    torch.rand(100)
    second = build_bist_factory(
        profile_effective=effective,
        manifest_frozen_seed=MANIFEST_FROZEN_SEED,
        node_num=3,
        device="cpu",
        dataloader=_dataloader(),
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_bist_factory.seed.second"),
        output_dir=tmp_path / "second",
    )

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name])


def test_bist_factory_completes_single_batch_artifact_lifecycle(tmp_path):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["bist"]["effective"]
    dataloader = _dataloader()
    result = build_bist_factory(
        profile_effective=effective,
        manifest_frozen_seed=MANIFEST_FROZEN_SEED,
        node_num=4,
        device="cpu",
        dataloader=dataloader,
        scaler=_IdentityScaler(),
        logger=logging.getLogger("test_table4_bist_factory.train"),
        output_dir=tmp_path / "train",
    )

    loss, mape, rmse = result.engine.train_batch()

    assert np.isfinite([loss, mape, rmse]).all()

    predictions, labels = capture_test_predictions(
        method="bist",
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
        target=tmp_path / "bist.pt",
    )
    expected = result.model.decoder[-1].weight.detach().clone()
    with torch.no_grad():
        result.model.decoder[-1].weight.add_(1)
    load_model_state_checkpoint(
        model=result.model,
        source=checkpoint,
    )
    torch.testing.assert_close(result.model.decoder[-1].weight, expected)


def test_bist_factory_rejects_missing_or_out_of_range_manifest_seed(tmp_path):
    profile_set = load_profile_set(PROFILE_PATH)
    effective = profile_set["profiles"]["bist"]["effective"]

    for invalid in (None, -1, 999999, True):
        try:
            build_bist_factory(
                profile_effective=effective,
                manifest_frozen_seed=invalid,
                node_num=3,
                device="cpu",
                dataloader=_dataloader(),
                scaler=_IdentityScaler(),
                logger=logging.getLogger("test_table4_bist_factory.invalid_seed"),
                output_dir=tmp_path / f"invalid_{invalid}",
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Invalid manifest seed was accepted: {invalid}")
