from __future__ import annotations

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.execution import (
    capture_test_predictions,
    load_model_state_checkpoint,
    run_upstream_training,
    save_model_state_checkpoint,
)


class _IdentityScaler:
    @staticmethod
    def inverse_transform(value):
        return value


class _TupleLoader:
    def __init__(self):
        self.x = np.arange(2 * 12 * 3 * 3, dtype=np.float32).reshape(
            2,
            12,
            3,
            3,
        )
        self.y = np.arange(2 * 12 * 3, dtype=np.float32).reshape(
            2,
            12,
            3,
            1,
        )

    def get_iterator(self):
        return iter([(self.x.copy(), self.y.copy())])


class _RepeatLast(torch.nn.Module):
    horizon = 12

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, label=None):
        return x[:, -1:, :, :1].expand(-1, 12, -1, -1) * self.weight


class _StandardEngine:
    def __init__(self):
        self.model = _RepeatLast()
        self._device = torch.device("cpu")
        self._scaler = _IdentityScaler()
        self._dataloader = {"test_loader": _TupleLoader()}
        self.train_calls = 0
        self.evaluate_calls = []

    @staticmethod
    def _to_tensor(values):
        return [torch.as_tensor(value, dtype=torch.float32) for value in values]

    def _to_device(self, values):
        return [value.to(self._device) for value in values]

    def _inverse_transform(self, values):
        return [self._scaler.inverse_transform(value) for value in values]

    def train(self):
        self.train_calls += 1

    def evaluate(self, mode):
        self.evaluate_calls.append(mode)


class _IncidentEngine:
    def __init__(self):
        self.model = _RepeatLast()
        self._scaler = _IdentityScaler()
        self._dataloader = {
            "test_loader": _IncidentLoader(),
        }

    @staticmethod
    def _prepare_batch(batch):
        return (
            torch.as_tensor(batch["x_data"]),
            torch.as_tensor(batch["y_data"]),
            {"incident_features": torch.ones(2, 4)},
            {"sensor_type": torch.zeros(2, 3)},
        )

    def _predict(self, x, label, incident_data, sensor_data):
        assert incident_data is not None
        assert sensor_data is not None
        return self.model(x, label)

    def _inverse_transform(self, values):
        return values


class _IncidentLoader:
    def get_iterator(self):
        source = _TupleLoader()
        return iter(
            [
                {
                    "x_data": source.x.copy(),
                    "y_data": source.y.copy(),
                }
            ]
        )


def test_run_upstream_training_uses_hl_evaluation_and_learned_train():
    hl_engine = _StandardEngine()
    learned_engine = _StandardEngine()

    run_upstream_training(method="hl", engine=hl_engine)
    run_upstream_training(method="lstm", engine=learned_engine)

    assert hl_engine.evaluate_calls == ["test"]
    assert hl_engine.train_calls == 0
    assert learned_engine.train_calls == 1
    assert learned_engine.evaluate_calls == []


def test_capture_test_predictions_handles_standard_and_incident_engines():
    standard = _StandardEngine()
    incident = _IncidentEngine()

    standard_predictions, standard_labels = capture_test_predictions(
        method="lstm",
        engine=standard,
    )
    incident_predictions, incident_labels = capture_test_predictions(
        method="igstgnn",
        engine=incident,
    )

    assert standard_predictions.dtype == np.float32
    assert standard_labels.dtype == np.float32
    np.testing.assert_array_equal(
        standard_predictions,
        incident_predictions,
    )
    np.testing.assert_array_equal(standard_labels, incident_labels)
    assert standard_predictions.shape == (2, 12, 3)


def test_save_model_state_checkpoint_round_trips_exact_state(tmp_path):
    model = _RepeatLast()
    target = tmp_path / "checkpoint.pt"

    save_model_state_checkpoint(model=model, target=target)

    expected = model.weight.detach().clone()
    model.weight.data.fill_(99.0)
    loaded = load_model_state_checkpoint(model=model, source=target)

    assert loaded == target
    torch.testing.assert_close(model.weight, expected)
    assert target.is_file()
