"""Graph preparation for the XTraffic DCRNN baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def slice_official_adjacency(
    adjacency_path: str | Path, region_idx: np.ndarray
) -> np.ndarray:
    """Slice XTraffic's weighted global adjacency in cache node order."""
    adjacency_path = Path(adjacency_path)
    if not adjacency_path.exists():
        raise FileNotFoundError(f"XTraffic adjacency not found: {adjacency_path}")
    full = np.load(adjacency_path, mmap_mode="r")
    if full.ndim != 2 or full.shape[0] != full.shape[1]:
        raise ValueError(f"expected a square adjacency matrix, got {full.shape}")
    region_idx = np.asarray(region_idx, dtype=np.int64)
    if region_idx.ndim != 1 or region_idx.size == 0:
        raise ValueError("region_idx must be a non-empty 1-D array")
    if region_idx.min() < 0 or region_idx.max() >= full.shape[0]:
        raise IndexError("region_idx falls outside the global adjacency")
    adjacency = np.asarray(full[np.ix_(region_idx, region_idx)], dtype=np.float32)
    adjacency[~np.isfinite(adjacency)] = 0.0
    adjacency[adjacency < 0.0] = 0.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def adjacency_from_edge_index(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Build a binary adjacency from the graph shared by existing project baselines."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2,E), got {edge_index.shape}")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def random_walk_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Return row-normalized ``D^-1 A`` with isolated rows left at zero."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inverse = np.divide(
        1.0,
        degree,
        out=np.zeros_like(degree, dtype=np.float32),
        where=degree > 0,
    )
    return inverse[:, None] * adjacency


def numpy_to_sparse_tensor(matrix: np.ndarray) -> torch.Tensor:
    # The matrix is already materialized while slicing the regional graph. Using
    # Tensor.to_sparse_coo performs PyTorch's own invariant-safe conversion once;
    # all recurrent graph operations remain sparse afterwards.
    dense = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
    return dense.to_sparse_coo().coalesce()


def build_bidirectional_supports(adjacency: np.ndarray) -> list[torch.Tensor]:
    """Forward and backward supports used by canonical DCRNN.

    The official TensorFlow cell stores ``random_walk(A).T`` and
    ``random_walk(A.T).T`` before left-multiplying node features. Keeping this
    transpose matters for XTraffic because its official adjacency is directed.
    """
    forward = random_walk_matrix(adjacency).T
    backward = random_walk_matrix(adjacency.T).T
    return [numpy_to_sparse_tensor(forward), numpy_to_sparse_tensor(backward)]


def graph_diagnostics(adjacency: np.ndarray) -> dict[str, float | int]:
    nonzero = adjacency > 0
    n = int(adjacency.shape[0])
    return {
        "num_nodes": n,
        "num_edges": int(nonzero.sum()),
        "density": float(nonzero.mean()),
        "asymmetric_fraction": float(np.mean(adjacency != adjacency.T)),
        "isolated_out": int(np.sum(adjacency.sum(axis=1) == 0)),
        "isolated_in": int(np.sum(adjacency.sum(axis=0) == 0)),
        "weight_min": float(adjacency[nonzero].min()) if nonzero.any() else 0.0,
        "weight_max": float(adjacency[nonzero].max()) if nonzero.any() else 0.0,
    }
