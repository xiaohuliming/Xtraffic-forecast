from types import SimpleNamespace

import numpy as np
import torch

from scripts.train_baseline_zoo import evaluate as evaluate_baseline_zoo
from scripts.train_dist_net import evaluate as evaluate_dist_net
from scripts.train_rad import evaluate as evaluate_rad
from scripts.train_rfn import evaluate as evaluate_rfn
from scripts.train_stid import evaluate as evaluate_stid


class _STIDEchoModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, x_hist, _tod, _dow):
        return x_hist[..., 0]


def _stid_batch(prediction, target, mask):
    prediction = torch.tensor(prediction, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    batch_size, num_nodes = prediction.shape
    return {
        "x_hist": prediction.reshape(batch_size, num_nodes, 1, 1),
        "y_true": target.reshape(batch_size, num_nodes, 1, 1),
        "y_mask": mask.reshape(batch_size, num_nodes, 1, 1),
        "time_feat": torch.zeros(batch_size, 1, 2),
    }


def test_stid_validation_uses_all_valid_observations():
    loader = [
        _stid_batch(
            [[0, 10, 99, 99]],
            [[0, 0, 99, 99]],
            [[1, 1, 0, 0]],
        ),
        _stid_batch(
            [[2, 2, 2, 2], [2, 2, 2, 2]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
        ),
    ]

    result = evaluate_stid(_STIDEchoModel(), loader, torch.device("cpu"))

    assert result == 2.6


class _BaselineZooEchoModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, model_input):
        return model_input[..., :1]


def _baseline_zoo_batch(prediction, target, mask):
    prediction = torch.tensor(prediction, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    batch_size, num_nodes = prediction.shape
    x_hist = torch.zeros(batch_size, num_nodes, 1, 3)
    x_hist[..., 0] = prediction.reshape(batch_size, num_nodes, 1)
    return {
        "x_hist": x_hist,
        "y_true": target.reshape(batch_size, num_nodes, 1, 1),
        "y_mask": mask.reshape(batch_size, num_nodes, 1, 1),
        "time_feat": torch.zeros(batch_size, 1, 2),
    }


def test_baseline_zoo_validation_uses_all_valid_observations():
    loader = [
        _baseline_zoo_batch(
            [[0, 10, 99, 99]],
            [[0, 0, 99, 99]],
            [[1, 1, 0, 0]],
        ),
        _baseline_zoo_batch(
            [[2, 2, 2, 2], [2, 2, 2, 2]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
        ),
    ]

    result = evaluate_baseline_zoo(
        _BaselineZooEchoModel(),
        "lstm",
        loader,
        torch.device("cpu"),
        0.0,
        1.0,
        1,
        4,
    )

    assert result == 2.6


class _ModeOnlyModel:
    def eval(self):
        return self

    def train(self):
        return self


def _direct_prediction_batch(prediction, target, mask):
    prediction = torch.tensor(prediction, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    batch_size, num_nodes = prediction.shape
    return {
        "prediction": prediction.reshape(batch_size, num_nodes, 1),
        "y_true": target.reshape(batch_size, num_nodes, 1, 1),
        "y_mask": mask.reshape(batch_size, num_nodes, 1, 1),
    }


def test_rad_validation_uses_all_valid_observations():
    loader = [
        _direct_prediction_batch(
            [[0, 10, 99, 99]],
            [[0, 0, 99, 99]],
            [[1, 1, 0, 0]],
        ),
        _direct_prediction_batch(
            [[2, 2, 2, 2], [2, 2, 2, 2]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
        ),
    ]

    result = evaluate_rad(
        _ModeOnlyModel(),
        loader,
        torch.device("cpu"),
        lambda _model, batch: batch["prediction"],
    )

    assert result == 2.6


def test_rfn_validation_uses_all_valid_observations():
    loader = [
        _direct_prediction_batch(
            [[0, 10, 99, 99]],
            [[0, 0, 99, 99]],
            [[1, 1, 0, 0]],
        ),
        _direct_prediction_batch(
            [[2, 2, 2, 2], [2, 2, 2, 2]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
        ),
    ]

    result = evaluate_rfn(
        _ModeOnlyModel(),
        loader,
        torch.device("cpu"),
        lambda _model, batch: batch["prediction"],
    )

    assert result == 2.6


class _DISTEchoModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, batch, edge_index):
        prediction = batch["prediction"]
        return {
            "pred": prediction,
            "pred_normal_final": prediction,
        }


class _DISTValidationDataset:
    def __init__(self, samples):
        self.samples = samples
        self.index_table = [("Alameda", index) for index in range(len(samples))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _dist_sample(prediction, target, mask):
    prediction = np.asarray(prediction, dtype=np.float32).reshape(-1, 1, 1)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 1, 1)
    mask = np.asarray(mask, dtype=bool).reshape(-1, 1, 1)
    num_nodes = prediction.shape[0]
    return {
        "prediction": prediction,
        "x_hist": np.zeros((num_nodes, 1, 1), dtype=np.float32),
        "region_code": np.int64(0),
        "incident_feat": np.zeros((1, 1), dtype=np.float32),
        "incident_mask": np.zeros(1, dtype=bool),
        "y_true": target,
        "y_baseline": np.zeros_like(target),
        "y_mask": mask,
    }


def test_dist_net_validation_main_mae_uses_all_valid_observations():
    dataset = _DISTValidationDataset(
        [
            _dist_sample(
                [0, 10, 99, 99],
                [0, 0, 99, 99],
                [1, 1, 0, 0],
            ),
            _dist_sample(
                [99, 99, 99, 99],
                [99, 99, 99, 99],
                [0, 0, 0, 0],
            ),
            _dist_sample(
                [2, 2, 2, 2],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ),
        ]
    )
    args = SimpleNamespace(
        batch_size=2,
        seed=42,
        blind=True,
        single_branch=False,
        alpha=1.0,
        beta=0.3,
        gamma=0.5,
        lam_aff=0.3,
    )

    result = evaluate_dist_net(
        _DISTEchoModel(),
        dataset,
        {"Alameda": None},
        args,
        torch.device("cpu"),
    )

    assert result["L_main"] == 3.0
