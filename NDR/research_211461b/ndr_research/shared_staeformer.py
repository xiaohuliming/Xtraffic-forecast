"""Fixed node-inductive adaptation of the tracked STAEformer flow-only model."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from baselines.STAEformer.STAEformer import STAEformer
from ndr_research.shared_graph import canonical_history


SHARED_STAEFORMER_CONFIG = {
    "in_steps": 12, "out_steps": 12, "steps_per_day": 288,
    "input_dim": 1, "output_dim": 1, "input_embedding_dim": 24,
    "tod_embedding_dim": 24, "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0, "adaptive_embedding_dim": 80,
    "feed_forward_dim": 256, "num_heads": 4, "num_layers": 3,
    "dropout": 0.1, "use_mixed_proj": True,
}


class _RuntimeSTAEformer(STAEformer):
    """Native shared modules with output layouts based on the input node count."""

    def __init__(self) -> None:
        super().__init__(num_nodes=1, **SHARED_STAEFORMER_CONFIG)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, channels = x.shape
        if steps != self.in_steps or channels != 3 or nodes <= 0:
            raise ValueError("native transfer input must match (B,12,N,3)")
        # Keep the tracked native embedding, temporal/spatial attention and head.
        features = [self.input_proj(x[..., :self.input_dim])]
        features.append(self.tod_embedding((x[..., 1] * self.steps_per_day).long()))
        features.append(self.dow_embedding(x[..., 2].long()))
        features.append(self.adaptive_embedding.expand(batch, steps, nodes, -1))
        encoded = torch.cat(features, dim=-1)
        for attention in self.attn_layers_t:
            encoded = attention(encoded, dim=1)
        for attention in self.attn_layers_s:
            encoded = attention(encoded, dim=2)
        output = encoded.transpose(1, 2).reshape(batch, nodes, self.in_steps * self.model_dim)
        output = self.output_proj(output).view(batch, nodes, self.out_steps, self.output_dim)
        return output.transpose(1, 2)


class SharedSTAEformer(nn.Module):
    """Predict raw flow without target data or node-specific learned parameters."""

    history_steps = horizon = 12

    def __init__(self, *, flow_center: float, flow_scale: float) -> None:
        super().__init__()
        statistics = torch.tensor([flow_center, flow_scale], dtype=torch.float64)
        if not torch.isfinite(statistics).all() or flow_scale <= 0:
            raise ValueError("source normalization requires a finite center and positive scale")
        self.register_buffer("flow_center", torch.tensor(float(flow_center), dtype=torch.float32))
        self.register_buffer("flow_scale", torch.tensor(float(flow_scale), dtype=torch.float32))
        self.native = _RuntimeSTAEformer()
        self.constructor = {"flow_center": float(flow_center), "flow_scale": float(flow_scale)}
        self.research_spec = {
            "method": "shared_staeformer", "output_units": "raw_flow",
            "architecture": copy.deepcopy(SHARED_STAEFORMER_CONFIG),
            "normalization": "source_template_week_mean_center_and_pooled_residual_scale",
            "history_fill": "causal_in_window_last_observation_with_source_weekly_fallback",
            "calendar": "tod_normalized_dow_raw_0_to_6",
            "node_identity_parameters": False, "region_identity_parameters": False,
            "node_count_in_state": False, "runtime_node_layout": True,
            "spatial_attention": "dense_native_attention_quadratic_in_nodes",
            "raw_published_width": 152, "adapted_width": 152,
            "capacity_change": "tie_adaptive_embedding_across_nodes_as_shared_temporal_embedding",
            "adaptive_parameter_shape": [12, 1, 80],
            "target_input": False, "incident_input": False,
            "loss": "pure_raw_flow_masked_mae",
        }

    def get_config(self) -> dict[str, Any]:
        return copy.deepcopy(self.constructor)

    def build_inputs(
        self, flow_hist: torch.Tensor, history_mask: torch.Tensor,
        x_baseline: torch.Tensor, time_feat: torch.Tensor,
    ) -> torch.Tensor:
        if flow_hist.ndim != 3 or flow_hist.shape[-1] != self.history_steps:
            raise ValueError("shared STAEformer history must match (B,N,12)")
        batch, nodes, _ = flow_hist.shape
        if time_feat.shape != (batch, 12, 2) or time_feat.device != flow_hist.device:
            raise ValueError("shared STAEformer calendar must match (B,12,2) on the history device")
        if not torch.isfinite(time_feat).all() or not ((time_feat >= 0) & (time_feat < 1)).all():
            raise FloatingPointError("historical calendar must be finite and normalized")
        state = canonical_history(flow_hist, x_baseline, history_mask)
        signal = ((state.filled - self.flow_center) / self.flow_scale).permute(0, 2, 1)
        tod = time_feat[..., 0].unsqueeze(-1).expand(batch, 12, nodes)
        dow = (time_feat[..., 1] * 7).round().clamp(0, 6).unsqueeze(-1).expand_as(tod)
        return torch.stack((signal, tod, dow), dim=-1)

    def forward(
        self, flow_hist: torch.Tensor, x_baseline: torch.Tensor,
        y_baseline: torch.Tensor, time_feat: torch.Tensor,
        history_mask: torch.Tensor, edge_index: torch.Tensor,
        node_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Shared evaluation interface; only the four historical fields are used.
        inputs = self.build_inputs(flow_hist, history_mask, x_baseline, time_feat)
        output = self.native(inputs).squeeze(-1).permute(0, 2, 1)
        return output * self.flow_scale + self.flow_center
