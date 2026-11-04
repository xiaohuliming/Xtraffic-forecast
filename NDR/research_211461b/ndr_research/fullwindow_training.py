"""Independent full-window data, model and objective adapters for NDR training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ndr_research.data import canonical_digest
from ndr_research.fullwindow_data import (
    FullWindowBatchLoader, FullWindowMaterialContract, FullWindowRegionReader,
    FullWindowStatistics, fit_fullwindow_statistics, legacy_array_digest,
)
from ndr_research.graph_indomain import model_config as graph_model_config
from ndr_research.models import forward_research, make_research_model, research_factory_kwargs
from ndr_research.shared_graph import SharedGraphNDR
from scripts.train_ndr_research import optimize_batch as optimize_native_ndr
from scripts.train_ndr_transfer import FORWARD_FIELDS, graph_batch_tensors, optimize_graph_batch

METHODS = ("baseline", "mask_signed", "sanitized_control", "projected_oracle", "shared_graph")
NATIVE_FIELDS = ("flow_hist", "x_baseline", "y_baseline", "time_feat", "history_mask", "target", "target_mask")


@dataclass(frozen=True)
class FullWindowTrainingData:
    reader: FullWindowRegionReader
    statistics: FullWindowStatistics
    train: FullWindowBatchLoader
    val: FullWindowBatchLoader
    identity: Mapping[str, Any]
    loading: Mapping[str, Any]


def graph_identity(reader: FullWindowRegionReader) -> dict[str, Any]:
    return {"region": reader.region, "nodes": reader.N, "edges": int(reader.edge_index.shape[1]),
            "edge_index_sha256": legacy_array_digest(reader.edge_index.astype(np.int64)),
            "authority": "raw_material_static_roads"}


def make_training_data(reader: FullWindowRegionReader, statistics: FullWindowStatistics, *,
                       batch_size: int, seed: int, preload: bool = True) -> FullWindowTrainingData:
    if reader.role != "source" or reader.history_policy != "canonical_causal_imputation":
        raise PermissionError("full-window training requires a source reader with native causal clean history")
    statistics.validate_reader(reader)
    loading = (reader.preload_source_train_val() if preload else
               {"loading_policy": "fullwindow_source_lazy_raw_windows",
                "reader_identity_digest": reader.identity["identity_digest"], "channels": [0]})
    train = FullWindowBatchLoader(reader, statistics, split="train", batch_size=batch_size, seed=seed)
    val = FullWindowBatchLoader(reader, statistics, split="val", batch_size=batch_size, seed=seed, shuffle=False)
    identity = {"schema_version": "fullwindow_training_data_v1", "sampling_protocol": "full_window",
                "protocol_version": reader.identity["protocol_version"], "region": reader.region,
                "reader_identity": reader.identity, "sampling_identity": reader.sampling_identity,
                "statistics_identity": dict(statistics.identity), "history_policy": reader.history_policy,
                "graph_identity": graph_identity(reader), "loading_policy": loading["loading_policy"],
                "test_numeric_api_available": False}
    identity["identity_digest"] = canonical_digest(identity)
    return FullWindowTrainingData(reader, statistics, train, val, identity, loading)


def load_training_data(*, region: str, data_dir: Path | str, graph_dir: Path | str,
                       batch_size: int = 16, seed: int = 42, preload: bool = True,
                       contract: FullWindowMaterialContract | None = None) -> FullWindowTrainingData:
    reader = FullWindowRegionReader(region, data_dir, graph_dir, role="source", contract=contract,
                                    history_policy="canonical_causal_imputation")
    statistics = fit_fullwindow_statistics(reader)
    return make_training_data(reader, statistics, batch_size=batch_size, seed=seed, preload=preload)


def validation_identity(data: FullWindowTrainingData, max_batches: int | None = None) -> dict[str, Any]:
    count = data.val.size if max_batches is None else min(data.val.size, max_batches * data.val.batch_size)
    indices = data.val.indices[:count]
    return {"split": "val", "sampling_protocol": "full_window",
            "sampling_digest": data.reader.sampling_identity["sampling_digest"],
            "reader_identity_digest": data.reader.identity["identity_digest"],
            "statistics_identity_digest": data.statistics.identity["identity_digest"],
            "samples": int(count), "sample_indices_sha256": legacy_array_digest(indices),
            "forecast_origins_sha256": legacy_array_digest(data.reader.sample_start[indices]),
            "all_validation_samples": count == data.val.size, "drop_last": False,
            "target_disjoint_between_splits": True, "adjacent_origins_share_targets_within_split": True}


def factory_from_data(method: str, data: FullWindowTrainingData, *, graph_hidden_dim: int = 64) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError("unsupported full-window NDR method")
    if method == "shared_graph":
        return graph_model_config(graph_hidden_dim)
    stats = data.statistics
    # Match the original model's float32 registered statistics exactly.
    return {"node_count": data.reader.N, "variant": method,
            "residual_std": float(np.float32(stats.residual_std)),
            "residual_std_by_node": stats.residual_std_by_node.tolist(),
            "flow_mean": float(np.float32(stats.flow_mean)), "flow_std": float(np.float32(stats.flow_std)),
            "reliable_anchor": False, "max_reference_age": 2, "hidden_dim": 16}


def rebuild_model(method: str, factory: Mapping[str, Any], device: torch.device | str = "cpu") -> torch.nn.Module:
    if method not in METHODS:
        raise ValueError("unsupported full-window NDR method")
    if method == "shared_graph":
        if dict(factory) != graph_model_config(factory["hidden_dim"]):
            raise ValueError("full-window graph architecture differs from the two fixed recipes")
        return SharedGraphNDR(**dict(factory)).to(device)
    if factory["variant"] != method or factory["hidden_dim"] != 16 or factory["reliable_anchor"] is not False:
        raise ValueError("full-window native NDR does not use its fixed original variant recipe")
    return make_research_model(**dict(factory), device=device)


def saved_factory(model: torch.nn.Module, method: str) -> dict[str, Any]:
    return model.get_config() if method == "shared_graph" else research_factory_kwargs(model)


def batch_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return graph_batch_tensors(batch, device)


def model_forward(model: Any, tensors: Mapping[str, torch.Tensor], method: str) -> torch.Tensor:
    if method == "shared_graph":
        prediction = model(**{field: tensors[field] for field in FORWARD_FIELDS})
    else:
        prediction = forward_research(model, tensors["flow_hist"], tensors["x_baseline"],
                                      tensors["y_baseline"], tensors["time_feat"], tensors["history_mask"])
    if prediction.shape != tensors["target"].shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("full-window forecast must be finite everywhere and match B,N,H")
    return prediction


def optimize_batch(model: Any, optimizer: Any, scheduler: Any, tensors: Mapping[str, torch.Tensor],
                   config: Mapping[str, Any]) -> dict[str, Any]:
    if config["method"] == "shared_graph":
        return optimize_graph_batch(model, optimizer, scheduler, tensors, config)
    # Reuse the original objective without exposing full-window roads/scales to
    # the original helper's batch-axis slicing. No event data adapter is used.
    step = optimize_native_ndr(model, optimizer, scheduler,
                               {field: tensors[field] for field in NATIVE_FIELDS}, config)
    return step | {"absolute_error_sum": step["error_sum"], "parts": step["loss_parts"]}


def objective_recipe(method: str, teacher: str) -> dict[str, Any]:
    if method not in METHODS or teacher not in ("original", "projected"):
        raise ValueError("full-window objective has an unsupported method or teacher")
    expected = "projected" if method == "projected_oracle" else "original"
    if method != "shared_graph" and teacher != expected:
        raise ValueError("native NDR teacher is determined by its actual variant")
    return {"loss": "raw_flow_masked_mae_plus_oracle_gate_plus_specialization", "teacher": teacher,
            "normalizer_scope": "complete_effective_batch_valid_count_gate_confidence_and_mean_teacher",
            "per_node_sigma": "runtime_train_node_residual_scale" if method == "shared_graph" else "original_model_registered_train_node_residual_scale",
            "lambda_gate": 1.0, "lambda_specialization": 0.3}
