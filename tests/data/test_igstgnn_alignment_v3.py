import json
import sys
from types import SimpleNamespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from dist_net.protocol import PROTOCOL_VERSION
from scripts import evaluate_igstgnn_baseline
from scripts.build_igstgnn_data_from_h5 import build_split

IGSTGNN_ROOT = Path(__file__).resolve().parents[2] / "baselines" / "IGSTGNN"
sys.path.insert(0, str(IGSTGNN_ROOT))

from src.utils.dataloader import (
    DataLoader as IGSTGNNDataLoader,
    IncidentDataLoader,
    load_dataset,
)
from src.base.engine import BaseEngine
from src.engines.igstgnn_engine import IGSTGNN_Engine
from src.utils.metrics import masked_mae, masked_mape, masked_rmse


def _write_minimal_source_bundle(root):
    region = "Alameda"
    total_steps = 32
    n_nodes = 2
    history = 3
    horizon = 2
    anchor = 10

    flow = np.arange(total_steps * n_nodes * 3, dtype=np.float32).reshape(
        total_steps, n_nodes, 3
    )
    flow_mask = np.ones_like(flow, dtype=bool)
    flow_mask[8, 0, 0] = False
    flow_mask[11, 1, 0] = False
    flow_mask[12, 0, 0] = False

    with h5py.File(root / f"{region}_traffic.h5", "w") as h5:
        h5.attrs["protocol_version"] = PROTOCOL_VERSION
        h5.create_dataset("flow_series_imputed", data=flow)
        h5.create_dataset("flow_mask", data=flow_mask)
        h5.create_dataset("time_enc", data=np.zeros((total_steps, 5), dtype=np.float32))
        times = pd.date_range("2023-01-01", periods=total_steps, freq="5min")
        h5.create_dataset("times_ns", data=times.asi8)

    incident_feat = np.zeros((1, 1, 12), dtype=np.float32)
    incident_feat[0, 0, 0] = 1.0
    with h5py.File(root / f"{region}_samples.h5", "w") as h5:
        h5.attrs["protocol_version"] = PROTOCOL_VERSION
        h5.attrs["T_h"] = history
        h5.attrs["T_p"] = horizon
        h5.attrs["N"] = n_nodes
        h5.attrs["M_max"] = 1
        h5.attrs["train_end"] = 20
        h5.create_dataset("split", data=np.array([0], dtype=np.int8))
        h5.create_dataset("sample_start", data=np.array([anchor], dtype=np.int64))
        h5.create_dataset("incident_feat", data=incident_feat)
        h5.create_dataset(
            "n_active_incidents", data=np.array([1], dtype=np.int16)
        )
        h5.create_dataset(
            "active_event_idx", data=np.array([[0]], dtype=np.int32)
        )
        h5.create_dataset(
            "primary_incident_slot", data=np.array([0], dtype=np.int16)
        )
        h5.create_dataset("event_idx", data=np.array([0], dtype=np.int32))
        labels = h5.create_group("analysis_labels")
        labels.create_dataset(
            "affected_mask", data=np.zeros((1, n_nodes), dtype=bool)
        )

    rel_feat = np.ones((1, n_nodes, 4), dtype=np.float32)
    np.savez(root / f"{region}_event_rel_feat.npz", rel_feat=rel_feat)
    return region, flow_mask, anchor, history, horizon


def test_conversion_preserves_source_observation_masks_elementwise(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "converted"
    source_dir.mkdir()
    region, flow_mask, anchor, history, horizon = _write_minimal_source_bundle(
        source_dir
    )

    build_split(
        region,
        source_dir,
        output_dir,
        sigma_euc_log=1.5,
        sigma_road_log=1.5,
        cutoff_euc_log=4.0,
        cutoff_road_log=5.0,
        normalization={
            "mean_per_ch": np.zeros(3, dtype=np.float32),
            "std_per_ch": np.ones(3, dtype=np.float32),
            "flow_mean": 0.0,
            "flow_std": 1.0,
        },
        validate_source=False,
    )

    converted = np.load(
        output_dir / region / "incident_data_train.npy", allow_pickle=True
    )
    sample = converted[0]
    np.testing.assert_array_equal(
        sample["x_mask"],
        flow_mask[anchor - history + 1 : anchor + 1, :, 0:1],
    )
    np.testing.assert_array_equal(
        sample["y_mask"],
        flow_mask[anchor + 1 : anchor + 1 + horizon, :, 0:1],
    )

    metadata = json.loads(
        (output_dir / region / "conversion_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["observation_mask_semantics"] == {
        "required": True,
        "source": "flow_mask",
        "x_mask": "x_data[..., 0:1]",
        "y_mask": "y_data",
        "meaning": "True iff the raw source flow value was observed",
    }


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


def test_igstgnn_loader_rejects_samples_without_observation_masks(tmp_path):
    sample = {
        "x_data": np.zeros((3, 2, 3), dtype=np.float32),
        "y_data": np.zeros((2, 2, 1), dtype=np.float32),
    }
    for split in ("train", "val", "test"):
        np.save(
            tmp_path / f"incident_data_{split}.npy",
            np.array([sample], dtype=object),
            allow_pickle=True,
        )
    np.savez(tmp_path / "incident_data_stats.npz", mean=0.0, std=1.0)

    args = SimpleNamespace(
        use_sensor_info=False,
        incident=True,
        bs=1,
        input_dim=3,
    )
    with pytest.raises(RuntimeError, match=r"requires x_mask and y_mask"):
        load_dataset(str(tmp_path), args, _Logger())


def test_igstgnn_loader_emits_exact_sample_masks(tmp_path):
    x_mask = np.array(
        [
            [[True], [False]],
            [[True], [True]],
            [[False], [True]],
        ],
        dtype=bool,
    )
    y_mask = np.array(
        [
            [[False], [True]],
            [[True], [False]],
        ],
        dtype=bool,
    )
    sample = {
        "x_data": np.zeros((3, 2, 3), dtype=np.float32),
        "x_mask": x_mask,
        "y_data": np.zeros((2, 2, 1), dtype=np.float32),
        "y_mask": y_mask,
        "event_features": {},
        "event_position": 0,
        "event_distances": np.zeros((2, 3), dtype=np.float32),
        "durations": 0.0,
    }
    for split in ("train", "val", "test"):
        np.save(
            tmp_path / f"incident_data_{split}.npy",
            np.array([sample], dtype=object),
            allow_pickle=True,
        )
    np.savez(tmp_path / "incident_data_stats.npz", mean=0.0, std=1.0)

    args = SimpleNamespace(
        use_sensor_info=False,
        incident=True,
        bs=1,
        input_dim=3,
    )
    dataloaders, _ = load_dataset(str(tmp_path), args, _Logger())
    batch = next(dataloaders["train_loader"].get_iterator())

    np.testing.assert_array_equal(batch["x_mask"][0], x_mask)
    np.testing.assert_array_equal(batch["y_mask"][0], y_mask)


def test_igstgnn_loader_keeps_masks_when_incident_features_are_disabled(tmp_path):
    x_mask = np.ones((3, 2, 1), dtype=bool)
    y_mask = np.array(
        [
            [[True], [False]],
            [[False], [True]],
        ],
        dtype=bool,
    )
    sample = {
        "x_data": np.zeros((3, 2, 3), dtype=np.float32),
        "x_mask": x_mask,
        "y_data": np.zeros((2, 2, 1), dtype=np.float32),
        "y_mask": y_mask,
    }
    for split in ("train", "val", "test"):
        np.save(
            tmp_path / f"incident_data_{split}.npy",
            np.array([sample], dtype=object),
            allow_pickle=True,
        )
    np.savez(tmp_path / "incident_data_stats.npz", mean=0.0, std=1.0)

    args = SimpleNamespace(
        use_sensor_info=False,
        incident=False,
        bs=1,
        input_dim=3,
    )
    dataloaders, _ = load_dataset(str(tmp_path), args, _Logger())
    _, _, batch_x_mask, batch_y_mask = next(
        dataloaders["train_loader"].get_iterator()
    )

    np.testing.assert_array_equal(batch_x_mask[0], x_mask)
    np.testing.assert_array_equal(batch_y_mask[0], y_mask)


def test_igstgnn_loader_emits_trailing_partial_batch():
    samples = []
    for sample_index in range(5):
        samples.append(
            {
                "x_data": np.full(
                    (3, 2, 3), sample_index, dtype=np.float32
                ),
                "x_mask": np.ones((3, 2, 1), dtype=bool),
                "y_data": np.full(
                    (2, 2, 1), sample_index, dtype=np.float32
                ),
                "y_mask": np.ones((2, 2, 1), dtype=bool),
            }
        )

    loader = IncidentDataLoader(
        samples,
        event=False,
        bs=2,
        logger=_Logger(),
        input_dim=3,
    )
    batches = list(loader.get_iterator())
    seen = [
        int(value)
        for batch_x, _batch_y, _x_mask, _y_mask in batches
        for value in batch_x[:, 0, 0, 0]
    ]

    assert loader.num_batch == 3
    assert [batch[0].shape[0] for batch in batches] == [2, 2, 1]
    assert seen == [0, 1, 2, 3, 4]


def test_igstgnn_standard_loader_emits_trailing_partial_batch():
    data = np.zeros((20, 1, 3), dtype=np.float32)
    data[:, 0, 0] = np.arange(20, dtype=np.float32)
    loader = IGSTGNNDataLoader(
        data=data,
        idx=np.arange(3, 8, dtype=np.int64),
        seq_len=2,
        horizon=1,
        bs=2,
        logger=_Logger(),
    )
    batches = list(loader.get_iterator())
    seen = [
        int(value)
        for batch_x, _batch_y in batches
        for value in batch_x[:, -1, 0, 0]
    ]

    assert loader.num_batch == 3
    assert [batch[0].shape[0] for batch in batches] == [2, 2, 1]
    assert seen == [3, 4, 5, 6, 7]


def test_igstgnn_loader_rejects_misaligned_observation_masks(tmp_path):
    sample = {
        "x_data": np.zeros((3, 2, 3), dtype=np.float32),
        "x_mask": np.ones((3, 2, 1), dtype=bool),
        "y_data": np.zeros((2, 2, 1), dtype=np.float32),
        "y_mask": np.ones((2, 1, 1), dtype=bool),
    }
    for split in ("train", "val", "test"):
        np.save(
            tmp_path / f"incident_data_{split}.npy",
            np.array([sample], dtype=object),
            allow_pickle=True,
        )
    np.savez(tmp_path / "incident_data_stats.npz", mean=0.0, std=1.0)

    args = SimpleNamespace(
        use_sensor_info=False,
        incident=True,
        bs=1,
        input_dim=3,
    )
    with pytest.raises(RuntimeError, match=r"y_mask shape"):
        load_dataset(str(tmp_path), args, _Logger())


def test_igstgnn_metrics_use_explicit_observation_mask():
    pred = torch.tensor([[0.0, 10.0, 1000.0]])
    label = torch.tensor([[0.0, 12.0, -5.0]])
    observed = torch.tensor([[True, True, False]])

    torch.testing.assert_close(masked_mae(pred, label, observed), torch.tensor(1.0))
    torch.testing.assert_close(
        masked_rmse(pred, label, observed),
        torch.tensor(np.sqrt(2.0), dtype=pred.dtype),
    )
    torch.testing.assert_close(
        masked_mape(pred, label, observed), torch.tensor(1.0 / 12.0)
    )


class _IdentityScaler:
    def inverse_transform(self, tensor):
        return tensor


class _SingleBatchLoader:
    def __init__(self, batch):
        self.batch = batch

    def get_iterator(self):
        return iter([self.batch])


class _SingleTrainBatchLoader(_SingleBatchLoader):
    num_batch = 1

    def shuffle(self):
        pass


class _ZeroForecast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def param_num(self):
        return 1

    def forward(
        self,
        x,
        label,
        incident_data=None,
        sensor_data=None,
    ):
        return torch.zeros_like(label) + self.bias


def test_igstgnn_validation_metrics_ignore_only_unobserved_targets(tmp_path):
    y_data = np.array([[[[2.0], [1000.0]]]], dtype=np.float32)
    y_mask = np.array([[[[True], [False]]]], dtype=bool)
    batch = {
        "x_data": np.zeros((1, 1, 2, 3), dtype=np.float32),
        "x_mask": np.ones((1, 1, 2, 1), dtype=bool),
        "y_data": y_data,
        "y_mask": y_mask,
        "incident_features": np.zeros((1, 4), dtype=np.float32),
        "incident_position": np.zeros(1, dtype=np.int64),
        "incident_distances": np.zeros((1, 2, 3), dtype=np.float32),
        "durations": np.zeros(1, dtype=np.float32),
    }
    model = _ZeroForecast()
    engine = IGSTGNN_Engine(
        device=torch.device("cpu"),
        model=model,
        dataloader={"val_loader": _SingleBatchLoader(batch)},
        scaler=_IdentityScaler(),
        sampler=None,
        loss_fn=masked_mae,
        lrate=1e-3,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        scheduler=None,
        clip_grad_value=0,
        max_epochs=1,
        patience=1,
        log_dir=str(tmp_path),
        logger=_Logger(),
        seed=42,
        cl_step=1,
        warm_step=1,
        horizon=1,
        incident=True,
    )

    mae, mape, rmse = engine.evaluate("val")

    assert mae == pytest.approx(2.0)
    assert mape == pytest.approx(1.0)
    assert rmse == pytest.approx(2.0)


def test_igstgnn_training_loss_uses_explicit_y_mask(tmp_path):
    batch = {
        "x_data": np.zeros((1, 1, 2, 3), dtype=np.float32),
        "x_mask": np.ones((1, 1, 2, 1), dtype=bool),
        "y_data": np.array([[[[2.0], [1000.0]]]], dtype=np.float32),
        "y_mask": np.array([[[[True], [False]]]], dtype=bool),
        "incident_features": np.zeros((1, 4), dtype=np.float32),
        "incident_position": np.zeros(1, dtype=np.int64),
        "incident_distances": np.zeros((1, 2, 3), dtype=np.float32),
        "durations": np.zeros(1, dtype=np.float32),
    }
    model = _ZeroForecast()
    engine = IGSTGNN_Engine(
        device=torch.device("cpu"),
        model=model,
        dataloader={"train_loader": _SingleTrainBatchLoader(batch)},
        scaler=_IdentityScaler(),
        sampler=None,
        loss_fn=masked_mae,
        lrate=0.0,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        clip_grad_value=0,
        max_epochs=1,
        patience=1,
        log_dir=str(tmp_path),
        logger=_Logger(),
        seed=42,
        cl_step=1,
        warm_step=1,
        horizon=1,
        incident=True,
    )

    mae, mape, rmse = engine.train_batch()

    assert mae == pytest.approx(2.0)
    assert mape == pytest.approx(1.0)
    assert rmse == pytest.approx(2.0)


def test_igstgnn_test_predictions_persist_exact_y_mask(tmp_path):
    y_mask = np.array([[[[True], [False]]]], dtype=bool)
    batch = {
        "x_data": np.zeros((1, 1, 2, 3), dtype=np.float32),
        "x_mask": np.ones((1, 1, 2, 1), dtype=bool),
        "y_data": np.array([[[[2.0], [1000.0]]]], dtype=np.float32),
        "y_mask": y_mask,
        "incident_features": np.zeros((1, 4), dtype=np.float32),
        "incident_position": np.zeros(1, dtype=np.int64),
        "incident_distances": np.zeros((1, 2, 3), dtype=np.float32),
        "durations": np.zeros(1, dtype=np.float32),
    }
    model = _ZeroForecast()
    log_dir = f"{tmp_path}/"
    engine = IGSTGNN_Engine(
        device=torch.device("cpu"),
        model=model,
        dataloader={"test_loader": _SingleBatchLoader(batch)},
        scaler=_IdentityScaler(),
        sampler=None,
        loss_fn=masked_mae,
        lrate=1e-3,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        scheduler=None,
        clip_grad_value=0,
        max_epochs=1,
        patience=1,
        log_dir=log_dir,
        logger=_Logger(),
        seed=42,
        cl_step=1,
        warm_step=1,
        horizon=1,
        incident=True,
    )
    engine.save_model(log_dir)

    engine.evaluate("test")

    with np.load(tmp_path / "test_predictions.npz") as saved:
        np.testing.assert_array_equal(saved["y_mask"], y_mask[..., 0])


def test_fair_evaluation_rejects_predictions_without_y_mask(tmp_path):
    prediction_path = tmp_path / "test_predictions.npz"
    np.savez(
        prediction_path,
        preds=np.zeros((1, 2, 3), dtype=np.float32),
        labels=np.zeros((1, 2, 3), dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match=r"missing required y_mask"):
        evaluate_igstgnn_baseline.load_prediction_bundle(prediction_path)


def test_base_engine_rejects_unmasked_evaluation_batches(tmp_path):
    batch = (
        np.zeros((1, 1, 2, 3), dtype=np.float32),
        np.zeros((1, 1, 2, 1), dtype=np.float32),
    )
    model = _ZeroForecast()
    engine = BaseEngine(
        device=torch.device("cpu"),
        model=model,
        dataloader={"val_loader": _SingleBatchLoader(batch)},
        scaler=_IdentityScaler(),
        sampler=None,
        loss_fn=masked_mae,
        lrate=1e-3,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        scheduler=None,
        clip_grad_value=0,
        max_epochs=1,
        patience=1,
        log_dir=str(tmp_path),
        logger=_Logger(),
        seed=42,
    )

    with pytest.raises(RuntimeError, match=r"requires an explicit y_mask"):
        engine.evaluate("val")
