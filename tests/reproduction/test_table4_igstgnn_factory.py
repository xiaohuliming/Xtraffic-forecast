from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.execution import (
    capture_test_predictions,
    load_model_state_checkpoint,
    save_model_state_checkpoint,
)
from scripts.reproduction.igstgnn_table4.igstgnn_factory import (
    IGSTGNNFactoryResult,
    build_igstgnn_factory,
)
from scripts.reproduction.igstgnn_table4.profiles import load_profile_set
from scripts.reproduction.igstgnn_table4.upstream_lock import (
    verify_pristine_upstream_checkout,
)


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


class _SingleBatchLoader:
    def __init__(self, batch):
        self._batch = batch
        self.num_batch = 1
        self.shuffle_calls = 0

    def shuffle(self):
        self.shuffle_calls += 1

    def get_iterator(self):
        yield self._batch


def _write_mapping_files(path):
    path.mkdir()
    (path / "desc_mapping.json").write_text(
        '{"collision": 0, "closure": 1}\n',
        encoding="utf-8",
    )
    (path / "type_mapping.json").write_text(
        '{"accident": 0, "hazard": 1}\n',
        encoding="utf-8",
    )


def _synthetic_batch(*, batch_size=2, node_num=3):
    x_data = np.ones(
        (batch_size, 12, node_num, 3),
        dtype=np.float32,
    )
    x_data[..., 0] = np.linspace(
        10.0,
        20.0,
        num=batch_size * 12 * node_num,
        dtype=np.float32,
    ).reshape(batch_size, 12, node_num)
    x_data[..., 1] = np.arange(12, dtype=np.float32)[None, :, None] / 288
    x_data[..., 2] = 3 / 7
    y_data = np.linspace(
        12.0,
        24.0,
        num=batch_size * 12 * node_num,
        dtype=np.float32,
    ).reshape(batch_size, 12, node_num, 1)
    return {
        "x_data": x_data,
        "y_data": y_data,
        "incident_features": np.asarray(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "incident_position": np.asarray([11, 10], dtype=np.int64),
        "incident_distances": np.full(
            (batch_size, node_num, 3),
            0.5,
            dtype=np.float32,
        ),
        "sensor_data": {
            "sensor_type": np.asarray(
                [[0, 1, 0], [0, 1, 0]],
                dtype=np.int64,
            ),
            "surface": np.asarray(
                [[0, 1, 1], [0, 1, 1]],
                dtype=np.int64,
            ),
            "roadway_use": np.asarray(
                [[0, 0, 1], [0, 0, 1]],
                dtype=np.int64,
            ),
            "road_width": np.full(
                (batch_size, node_num),
                80.0,
                dtype=np.float32,
            ),
            "speed_limit": np.full(
                (batch_size, node_num),
                65.0,
                dtype=np.float32,
            ),
        },
    }


def _factory_arguments(tmp_path):
    data_path = tmp_path / "Alameda"
    _write_mapping_files(data_path)
    batch = _synthetic_batch()
    train_loader = _SingleBatchLoader(batch)
    dataloader = {
        "train_loader": train_loader,
        "val_loader": _SingleBatchLoader(batch),
        "test_loader": _SingleBatchLoader(batch),
    }
    adjacency = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    profile_set = load_profile_set(PROFILE_PATH)
    return {
        "profile_effective": profile_set["profiles"]["igstgnn"]["effective"],
        "node_num": 3,
        "device": "cpu",
        "adjacency": adjacency,
        "dataloader": dataloader,
        "scaler": _IdentityScaler(),
        "sensor_dimensions": SimpleNamespace(
            sensor_type_size=2,
            surface_size=2,
            roadway_use_size=2,
        ),
        "logger": logging.getLogger("test_table4_igstgnn_factory"),
        "output_dir": tmp_path / "run",
        "dataset": "Alameda",
        "data_path": data_path,
    }, batch


def test_igstgnn_factory_builds_pinned_incident_sensor_pipeline_and_forecasts(
    tmp_path,
):
    arguments, batch = _factory_arguments(tmp_path)
    original_adjacency = arguments["adjacency"].copy()

    result = build_igstgnn_factory(**arguments)

    assert isinstance(result, IGSTGNNFactoryResult)
    assert result.engine.model is result.model
    assert result.engine._incident is True
    assert result.model.icsf_module.use_sensor_info is True
    assert result.effective_config == arguments["profile_effective"]
    assert result.optimizer.param_groups[0]["lr"] == 0.002
    assert result.optimizer.param_groups[0]["weight_decay"] == 0.00001
    assert result.optimizer.param_groups[0]["eps"] == 1e-8
    assert sorted(result.scheduler.milestones.elements()) == [
        1,
        38,
        46,
        54,
        62,
        70,
        80,
    ]
    assert result.scheduler.gamma == 0.5
    assert result.engine._cl_step == 3
    assert result.engine._warm_step == 30
    assert result.provenance["source_id"] == "igstgnn_64c6e24"
    assert result.provenance["commit"] == ("64c6e243466f791f1b75e50421d63e39577ec9fa")
    assert result.provenance["model_class"] == ("src.models.igstgnn.IGSTGNN")
    assert result.provenance["engine_class"] == (
        "src.engines.igstgnn_engine.IGSTGNN_Engine"
    )
    source_status = verify_pristine_upstream_checkout(
        source_id=result.provenance["source_id"],
        checkout=Path(result.provenance["checkout"]),
    )
    assert source_status["git_dirty"] is False
    np.testing.assert_array_equal(
        arguments["adjacency"],
        original_adjacency,
    )

    X, label, incident_data, sensor_data = result.engine._prepare_batch(batch)
    result.model.eval()
    with torch.no_grad():
        prediction = result.model(
            X,
            label,
            incident_data=incident_data,
            sensor_data=sensor_data,
        )
    assert prediction.shape == (2, 12, 3, 1)
    assert torch.isfinite(prediction).all()


def test_igstgnn_factory_completes_single_batch_artifact_lifecycle(
    tmp_path,
):
    arguments, batch = _factory_arguments(tmp_path)
    result = build_igstgnn_factory(**arguments)
    train_loader = arguments["dataloader"]["train_loader"]
    before = result.model.out_fc_2.weight.detach().clone()

    metrics = result.engine.train_batch()

    assert len(metrics) == 3
    assert np.isfinite(metrics).all()
    assert train_loader.shuffle_calls == 1
    assert result.engine._iter_cnt == 1
    assert result.engine._cl_len == 12
    assert not torch.equal(
        before,
        result.model.out_fc_2.weight.detach(),
    )
    assert result.optimizer.state

    predictions, labels = capture_test_predictions(
        method="igstgnn",
        engine=result.engine,
    )
    assert predictions.shape == labels.shape == (2, 12, 3)
    assert predictions.dtype == labels.dtype == np.float32
    assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(labels, batch["y_data"][..., 0])

    checkpoint = save_model_state_checkpoint(
        model=result.model,
        target=tmp_path / "igstgnn.pt",
    )
    expected = result.model.out_fc_2.weight.detach().clone()
    with torch.no_grad():
        result.model.out_fc_2.weight.add_(1)
    load_model_state_checkpoint(
        model=result.model,
        source=checkpoint,
    )
    torch.testing.assert_close(result.model.out_fc_2.weight, expected)


def test_igstgnn_factory_applies_official_seed_before_model_initialization(
    tmp_path,
):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    torch.set_num_threads(5)
    first_arguments, _ = _factory_arguments(tmp_path / "first")
    first = build_igstgnn_factory(**first_arguments)
    torch.rand(100)
    second_arguments, _ = _factory_arguments(tmp_path / "second")
    second = build_igstgnn_factory(**second_arguments)

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()
    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(first_state[name], second_state[name])
    assert torch.get_num_threads() == 3
    assert first.provenance["runtime_contract"]["torch_num_threads"] == 3
