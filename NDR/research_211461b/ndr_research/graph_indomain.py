"""Single-region event-data bridge for the real node-shared NDR model."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from ndr_research.data import ResearchData, array_digest, prepare_research_batch
from scripts.train_ndr_transfer import graph_batch_tensors


HIDDEN_DIMS = (64, 240)


def model_config(hidden_dim: int) -> dict[str, Any]:
    """The two predeclared widths retain the same graph and temporal recipe."""
    if type(hidden_dim) is not int or hidden_dim not in HIDDEN_DIMS:
        raise ValueError("single-region graph training requires fixed hidden width 64 or 240")
    return {"history_steps": 12, "horizon": 12, "hidden_dim": hidden_dim,
            "road_layers": 2, "latent_tokens": 8, "num_heads": 4, "dropout": 0.0}


def graph_identity(data: ResearchData) -> dict[str, Any]:
    edges = np.asarray(data.region.edge_index)
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu"
            or edges.size and (edges.min() < 0 or edges.max() >= data.region.N)):
        raise ValueError("single-region static roads must be integer (2,E) edges within node bounds")
    return {"region": data.identity["region"], "nodes": data.region.N,
            "edge_index_sha256": array_digest(edges.astype(np.int64)),
            "edges": int(edges.shape[1]), "graph_role": "audited_static_roads"}


def normalizer_identity(data: ResearchData) -> dict[str, Any]:
    keys = ("protocol_version", "cache_schema_version", "cache_bundle_digest", "region",
            "region_cache_digest", "fit_range", "fit_positions", "weekly_climatology_sha256",
            "node_residual_std_sha256", "valid_training_points", "training_analysis_labels")
    identity = {key: data.identity[key] for key in keys}
    if (identity["fit_range"] != [0, data.region.train_end]
            or identity["training_analysis_labels"] is not False):
        raise ValueError("single-region graph normalization must fit only raw observed train values")
    return identity | {"mode": "regional_train_only_runtime_arrays",
                       "model_registered_node_statistics": False,
                       "climatology_usage": "history_fallback_and_future_normal_anchor",
                       "scale_usage": "per_node_train_residual_scale_at_runtime"}


def batch_tensors(batch: Mapping[str, Any], data: ResearchData,
                  device: torch.device) -> dict[str, torch.Tensor]:
    """Keep the original event adapter and append only roads and train scales."""
    if data.region.include_analysis_labels:
        raise ValueError("analysis labels must remain disabled for graph training")
    prepared = prepare_research_batch(batch, data.statistics)
    prepared["edge_index"] = np.array(data.region.edge_index, dtype=np.int64, copy=True)
    prepared["node_scale"] = np.array(data.statistics.residual_std_by_node, dtype=np.float32, copy=True)
    return graph_batch_tensors(prepared, device)
