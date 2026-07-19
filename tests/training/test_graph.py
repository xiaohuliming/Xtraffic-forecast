import numpy as np
import torch

from dist_net.graph import bidirectional_random_walk_supports


def test_bidirectional_supports_normalize_each_direction_independently():
    edge_index = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64)

    forward, backward = bidirectional_random_walk_supports(
        edge_index,
        num_nodes=3,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(forward.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(backward.sum(dim=1), torch.ones(3))
