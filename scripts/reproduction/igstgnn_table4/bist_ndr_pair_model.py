from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from scripts.train_staeformer_deseason import HorizonGate

from .bist_factory import BIST_CHECKOUT, BIST_SOURCE_ID, _load_upstream
from .upstream_lock import verify_pristine_upstream_checkout


@dataclass(frozen=True)
class BiSTNDRPairModelResult:
    model: nn.Module
    provenance: dict[str, Any]


def _make_bist_input(
    signal: torch.Tensor,
    time_feat: torch.Tensor,
) -> torch.Tensor:
    if signal.ndim != 3:
        raise ValueError("BiST paired signal must have shape batch, node, time")
    batch_size, node_count, history_steps = signal.shape
    if time_feat.shape != (batch_size, history_steps, 2):
        raise ValueError("BiST paired time features do not match the signal")
    tod = time_feat[..., 0].unsqueeze(-1).expand(
        batch_size,
        history_steps,
        node_count,
    )
    dow = time_feat[..., 1].unsqueeze(-1).expand_as(tod)
    return torch.stack(
        [signal.permute(0, 2, 1), tod, dow],
        dim=-1,
    )


def _build_native_bist(
    *,
    upstream: Any,
    node_count: int,
    config: Mapping[str, Any],
) -> nn.Module:
    base = upstream.MLP(
        node_num=node_count,
        input_dim=3,
        output_dim=1,
        num_layer=config["layers"],
        model_dim=config["model_dim"],
        prompt_dim=config["prompt_dim"],
        tod_size=config["time_slots_per_day"],
        kernel_size=config["kernel_size"],
    )
    hidden_embedding_dim = config["model_dim"] + 3 * config["prompt_dim"]
    model_args = {
        "extra_type": config["extra_type"],
        "same": config["same"],
        "hid_dim": config["hidden_dim"],
        "horizon": 12,
        "predefined_adj": int(config["predefined_adjacency"]),
        "adjs": [],
        "rp_layer": config["residual_propagation_layers"],
        "datadriven_adj": int(config["data_driven_adjacency"]),
        "datadriven_adj_dim": 32,
        "datadriven_adj_head": 0,
        "adaptive_adj": int(config["adaptive_adjacency"]),
        "adaptive_adj_dim": config["adaptive_adjacency_dim"],
        "use_global_opt": False,
        "mrf": 1,
    }
    return upstream.BiST(
        node_num=node_count,
        input_dim=3,
        output_dim=1,
        model_args=model_args,
        stmodel=base,
        dim=[hidden_embedding_dim, hidden_embedding_dim],
        core=config["core_dim"],
    )


class AlignedBiST(nn.Module):
    def __init__(
        self,
        *,
        native: nn.Module,
        source_mean: float,
        source_std: float,
    ) -> None:
        super().__init__()
        self.native = native
        self.register_buffer(
            "source_mean",
            torch.tensor(float(source_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "source_std",
            torch.tensor(float(source_std), dtype=torch.float32),
        )
        self.oracle_sup = False
        self.dual = False
        self.abank = False
        self.aux = None

    def forward(
        self,
        flow_hist: torch.Tensor,
        x_baseline: torch.Tensor,
        y_baseline: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> torch.Tensor:
        del x_baseline, y_baseline
        normalized = (flow_hist - self.source_mean) / self.source_std
        inputs = _make_bist_input(normalized, time_feat)
        prediction = self.native(inputs)
        self.aux = None
        return (
            prediction.squeeze(-1).permute(0, 2, 1) * self.source_std
            + self.source_mean
        )


class BiSTNDR(nn.Module):
    def __init__(
        self,
        *,
        native: nn.Module,
        residual_std: float,
        residual_std_by_node: np.ndarray,
    ) -> None:
        super().__init__()
        self.native = native
        self.disturbance_regression = copy.deepcopy(
            native.stmodel.regression_layer
        )
        self.disturbance_decoder = copy.deepcopy(native.decoder)
        self.gate_mod = HorizonGate(12)
        self.register_buffer(
            "sd_res",
            torch.tensor(float(residual_std), dtype=torch.float32),
        )
        self.register_buffer(
            "sd_node",
            torch.from_numpy(
                np.array(residual_std_by_node, dtype=np.float32, copy=True)
            ),
        )
        self.oracle_sup = True
        self.dual = True
        self.abank = False
        self.aux: dict[str, torch.Tensor] | None = None

    def _residual_heads(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shallow, encoded, normal_base = self.native.stmodel(inputs)
        backcast = self.native.backcast(encoded)
        if self.native.same:
            residual_encoded = self.native.stmodel.module(
                shallow - backcast,
            )
        else:
            residual_encoded = self.native.stmodel_detach.module(
                shallow - backcast,
            )
        residual_encoded = self.native.residual_information_propagation(
            residual_encoded
        )
        normal = normal_base + self.native.decoder(
            residual_encoded
        ).transpose(1, -1)
        disturbance = self.disturbance_regression(
            encoded.transpose(1, -1)
        ) + self.disturbance_decoder(residual_encoded).transpose(1, -1)
        return (
            normal.squeeze(-1).permute(0, 2, 1),
            disturbance.squeeze(-1).permute(0, 2, 1),
        )

    def forward(
        self,
        flow_hist: torch.Tensor,
        x_baseline: torch.Tensor,
        y_baseline: torch.Tensor,
        time_feat: torch.Tensor,
    ) -> torch.Tensor:
        residual = (flow_hist - x_baseline) / self.sd_res
        inputs = _make_bist_input(residual, time_feat)
        normal_residual, disturbance_residual = self._residual_heads(inputs)
        signed_residual = residual
        deviation = signed_residual.abs().mean(dim=2)
        slope = (
            signed_residual[:, :, -1] - signed_residual[:, :, -2]
        ).abs()
        gate = self.gate_mod(deviation, slope)
        persistence = flow_hist[:, :, -1:]
        normal = y_baseline + normal_residual * self.sd_res
        disturbance = persistence + disturbance_residual * self.sd_res
        self.aux = {
            "g": gate,
            "B": y_baseline,
            "P": persistence,
            "yn": normal,
            "yd": disturbance,
        }
        return gate * normal + (1.0 - gate) * disturbance


def build_bist_ndr_pair_model(
    *,
    method: str,
    node_count: int,
    effective_profile: Mapping[str, Any],
    statistics: Any,
    device: torch.device | str,
) -> BiSTNDRPairModelResult:
    if method not in {"bist_aligned", "bist_ndr"}:
        raise ValueError(f"Unsupported BiST-NDR paired method: {method}")
    if type(node_count) is not int or node_count <= 0:
        raise ValueError("node_count must be a positive integer")
    backbone = effective_profile.get("backbone")
    if not isinstance(backbone, Mapping):
        raise ValueError("BiST-NDR paired backbone config is missing")
    upstream_provenance = verify_pristine_upstream_checkout(
        source_id=BIST_SOURCE_ID,
        checkout=BIST_CHECKOUT,
    )
    if (
        backbone.get("source_id") != BIST_SOURCE_ID
        or backbone.get("commit") != upstream_provenance["commit"]
    ):
        raise ValueError("BiST-NDR paired backbone binding is invalid")
    upstream = _load_upstream()
    native = _build_native_bist(
        upstream=upstream,
        node_count=node_count,
        config=backbone,
    )
    if method == "bist_aligned":
        model: nn.Module = AlignedBiST(
            native=native,
            source_mean=statistics.source_normalization_mean,
            source_std=statistics.source_normalization_std,
        )
        head_initialization = "native_single_head"
    else:
        model = BiSTNDR(
            native=native,
            residual_std=statistics.residual_std,
            residual_std_by_node=statistics.residual_std_by_node,
        )
        head_initialization = "identical_copy_from_bist_native_heads"
    model = model.to(torch.device(device))
    return BiSTNDRPairModelResult(
        model=model,
        provenance={
            "adapter_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "upstream_source_id": upstream_provenance["source_id"],
            "upstream_commit": upstream_provenance["commit"],
            "upstream_tree": upstream_provenance["tree"],
            "license_spdx": upstream_provenance["license_spdx"],
            "head_initialization": head_initialization,
            "upstream_source_execution": "in_place_external_adapter",
        },
    )
