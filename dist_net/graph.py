"""Graph construction shared by graph-based training entry points."""

from __future__ import annotations

import numpy as np
import torch


def bidirectional_random_walk_supports(
    edge_index: np.ndarray,
    num_nodes: int,
    device: torch.device | str,
) -> list[torch.Tensor]:
    """Build independently row-normalized forward and reverse supports."""
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if edge_index.size and (
        edge_index.min() < 0 or edge_index.max() >= num_nodes
    ):
        raise ValueError("edge_index contains a node outside [0, num_nodes)")

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(adjacency, 1.0)

    forward = _row_normalize(adjacency)
    backward = _row_normalize(adjacency.T)
    return [
        torch.from_numpy(forward).to(device),
        torch.from_numpy(backward).to(device),
    ]


def _row_normalize(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1, keepdims=True)
    if np.any(degree <= 0):
        raise ValueError("support contains a row without outgoing mass")
    return (adjacency / degree).astype(np.float32, copy=False)
