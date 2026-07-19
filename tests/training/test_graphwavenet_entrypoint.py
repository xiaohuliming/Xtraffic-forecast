import torch

from scripts.train_graphwavenet import evaluate


class _FlowEchoModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, graphwavenet_input):
        values = graphwavenet_input[:, 0, :, 0]
        return values[:, None, :, None]


def _batch(prediction, target, mask):
    prediction = torch.tensor(prediction, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    batch_size, num_nodes = prediction.shape
    return {
        "x_hist": prediction.reshape(batch_size, num_nodes, 1, 1),
        "y_true": target.reshape(batch_size, num_nodes, 1, 1),
        "y_mask": mask.reshape(batch_size, num_nodes, 1, 1),
    }


def test_graphwavenet_validation_uses_all_valid_observations():
    loader = [
        _batch([[0, 10, 99, 99]], [[0, 0, 99, 99]], [[1, 1, 0, 0]]),
        _batch([[2, 2, 2, 2], [2, 2, 2, 2]], [[0] * 4, [0] * 4], [[1] * 4, [1] * 4]),
    ]

    result = evaluate(_FlowEchoModel(), loader, torch.device("cpu"), supports=[])

    assert result["L_main"] == 2.6
