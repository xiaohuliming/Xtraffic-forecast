#!/usr/bin/env python3
"""Train a dual-branch gate with an ST-TIS-style incident branch.

The normal branch and gated residual formulation are kept unchanged from the
current best dual-branch model. The incident branch is replaced with a
lightweight spatio-temporal Transformer module: temporal self-attention encodes
each candidate node history, and top-k graph-biased spatial attention fuses
incident-centered candidate nodes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from train_dual_branch_gate_baseline import DualBranchGateBaseline, cap_indices, infer_cache_shapes
from train_full_candidate_stgnn_heatmap_model import (
    CHANNELS,
    compute_loss,
    compute_stats,
    evaluate_loader,
    forecast_metrics_for_loader,
    make_loader,
    region_codes,
    split_indices,
)
from train_impact_residual_model import choose_device, json_safe_args


class TemporalFusionEncoder(nn.Module):
    """Per-node temporal attention encoder with last/mean fusion."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        max_input_steps: int = 64,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_input_steps, hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=heads,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, hist_residual: torch.Tensor) -> torch.Tensor:
        batch_size, input_steps, nodes, hist_channels = hist_residual.shape
        x = hist_residual.permute(0, 2, 1, 3).reshape(batch_size * nodes, input_steps, hist_channels)
        x = self.input_proj(x)
        if input_steps <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, :input_steps]
        else:
            x = x + self.pos_embedding[:, -1:].expand(1, input_steps, -1)
        for layer in self.layers:
            x = layer(x)
        h_last = x[:, -1]
        h_mean = x.mean(dim=1)
        h = self.fusion(torch.cat([h_last, h_mean], dim=-1))
        return h.reshape(batch_size, nodes, -1)


class GraphBiasedSpatialAttention(nn.Module):
    """Top-k graph-biased spatial attention over candidate nodes."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        dropout: float,
        spatial_topk: int,
        adj_bias_scale: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.spatial_topk = spatial_topk
        self.adj_bias_scale = adj_bias_scale
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def make_topk_mask(self, adj_all: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch_size, nodes, _ = adj_all.shape
        valid_pair = valid[:, :, None].bool() & valid[:, None, :].bool()
        if self.spatial_topk <= 0 or self.spatial_topk >= nodes:
            return valid_pair
        k = max(1, min(self.spatial_topk, nodes))
        masked_adj = adj_all.masked_fill(~valid_pair, -1.0)
        top_idx = torch.topk(masked_adj, k=k, dim=-1, largest=True, sorted=False).indices
        top_mask = torch.zeros(batch_size, nodes, nodes, dtype=torch.bool, device=adj_all.device)
        top_mask.scatter_(-1, top_idx, True)
        eye = torch.eye(nodes, dtype=torch.bool, device=adj_all.device).unsqueeze(0)
        return (top_mask | eye) & valid_pair

    def forward(self, h: torch.Tensor, adj_all: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch_size, nodes, _ = h.shape
        q = self.q_proj(h).reshape(batch_size, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(batch_size, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(batch_size, nodes, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.adj_bias_scale != 0.0:
            scores = scores + self.adj_bias_scale * torch.log(adj_all.clamp_min(1e-6)).unsqueeze(1)
        mask = self.make_topk_mask(adj_all, valid).unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, nodes, self.hidden_dim)
        out = self.out_proj(out)
        h = self.attn_norm(h + out)
        h = self.ffn_norm(h + self.ffn(h))
        return h * valid.unsqueeze(-1)


class DualBranchSTTISGate(DualBranchGateBaseline):
    """Dual-branch gated residual model with ST-TIS-style incident branch."""

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
        )
        self.sttis_heads = sttis_heads
        self.sttis_temporal_layers = sttis_temporal_layers
        self.sttis_spatial_topk = sttis_spatial_topk
        self.sttis_adj_bias = sttis_adj_bias
        self.incident_temporal_encoder = TemporalFusionEncoder(
            input_channels=hist_input_channels,
            hidden_dim=hidden_dim,
            layers=sttis_temporal_layers,
            heads=sttis_heads,
            dropout=dropout,
        )
        self.incident_spatial_layers = nn.ModuleList(
            [
                GraphBiasedSpatialAttention(
                    hidden_dim=hidden_dim,
                    heads=sttis_heads,
                    dropout=dropout,
                    spatial_topk=sttis_spatial_topk,
                    adj_bias_scale=sttis_adj_bias,
                )
                for _ in range(graph_layers)
            ]
        )

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        gate_input = torch.cat([h_normal, h_incident, *delta_features], dim=-1) if delta_features else torch.cat([h_normal, h_incident], dim=-1)

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        gate = torch.sigmoid(self.gate_head(gate_input)).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISProposalGate(DualBranchSTTISGate):
    """ST-TIS dual-branch model whose gate sees the branch residual proposals.

    The original gate only receives latent branch embeddings and normal-delta
    features. This version first decodes the normal and incident residual
    proposals, then lets the gate inspect their signed values, magnitudes, and
    disagreement before choosing the fusion weight. No branch-error labels or
    oracle gate targets are used.
    """

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
        )
        self.proposal_feature_count = proposal_feature_count
        self.proposal_feature_dim = proposal_feature_count * horizon_steps * channels
        base_gate_dim = int(self.gate_head[0].in_features)
        self.proposal_norm = nn.LayerNorm(self.proposal_feature_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(base_gate_dim + self.proposal_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps * channels),
        )

    def proposal_features(self, normal_residual: torch.Tensor, incident_residual: torch.Tensor) -> torch.Tensor:
        diff = incident_residual - normal_residual
        features = [
            normal_residual,
            incident_residual,
            diff,
            diff.abs(),
            incident_residual.abs() - normal_residual.abs(),
        ]
        if self.proposal_feature_count != len(features):
            raise ValueError(f"proposal_feature_count={self.proposal_feature_count} is not supported")
        proposal = torch.cat([item.flatten(start_dim=2) for item in features], dim=-1)
        return self.proposal_norm(proposal)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_input)).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISReliabilityGate(DualBranchSTTISProposalGate):
    """Proposal-aware gate with a small branch-reliability adapter.

    The base proposal-aware gate remains intact. A zero-initialized reliability
    head predicts an additive gate-logit correction from proposal-aware gate
    features and the base gate logits. Positive corrections increase reliance
    on the incident branch; negative corrections move the fusion toward the
    normal-style residual branch.
    """

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        reliability_scale: float = 1.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.reliability_scale = reliability_scale
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        self.base_gate_logit_norm = nn.LayerNorm(logit_dim)
        self.reliability_head = nn.Sequential(
            nn.Linear(gate_input_dim + logit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        final = self.reliability_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        reliability_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat)], dim=-1)
        reliability_logits_flat = self.reliability_head(reliability_input)
        gate_logits = base_gate_logits_flat + self.reliability_scale * reliability_logits_flat
        gate = torch.sigmoid(gate_logits).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": torch.sigmoid(base_gate_logits_flat)
                .reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "reliability_logits": reliability_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISVetoGate(DualBranchSTTISProposalGate):
    """Proposal-aware gate with a conservative incident-branch veto.

    This adapter can only subtract from the incident-branch gate logit. It is
    intended for hard negatives where the incident residual branch is locally
    much worse than the normal-style residual branch.
    """

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        veto_scale: float = 1.0,
        veto_max: float = 2.0,
        veto_init_bias: float = -6.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.veto_scale = veto_scale
        self.veto_max = veto_max
        self.veto_init_bias = veto_init_bias
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        self.base_gate_logit_norm = nn.LayerNorm(logit_dim)
        self.veto_head = nn.Sequential(
            nn.Linear(gate_input_dim + logit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        final = self.veto_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, veto_init_bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        veto_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat)], dim=-1)
        veto_logits_flat = self.veto_head(veto_input)
        veto_amount_flat = self.veto_max * torch.sigmoid(veto_logits_flat)
        gate_logits = base_gate_logits_flat - self.veto_scale * veto_amount_flat
        gate = torch.sigmoid(gate_logits).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": torch.sigmoid(base_gate_logits_flat)
                .reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "veto_logits": veto_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "veto_amount": veto_amount_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISDeltaGate(DualBranchSTTISProposalGate):
    """Proposal-aware gate with a bounded bidirectional logit adapter.

    Unlike the veto gate, this adapter can move the incident-branch weight in
    either direction. It is useful when the base gate sometimes underuses the
    incident branch and sometimes overuses it.
    """

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        delta_scale: float = 1.0,
        delta_max: float = 2.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.delta_scale = delta_scale
        self.delta_max = delta_max
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        self.base_gate_logit_norm = nn.LayerNorm(logit_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(gate_input_dim + logit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        final = self.delta_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        delta_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat)], dim=-1)
        delta_logits_flat = self.delta_head(delta_input)
        gate_delta_flat = self.delta_max * torch.tanh(delta_logits_flat)
        gate_logits_flat = base_gate_logits_flat + self.delta_scale * gate_delta_flat
        gate = torch.sigmoid(gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": torch.sigmoid(base_gate_logits_flat)
                .reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "gate_logits": gate_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "delta_logits": delta_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "gate_delta": gate_delta_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISBranchConfidenceGate(DualBranchSTTISProposalGate):
    """Proposal-aware gate corrected by branch-specific confidence scores.

    The base proposal gate remains unchanged. Two zero-initialized confidence
    heads predict horizon-level confidence logits from the normal and incident
    branch representations. Their difference shifts the gate logit: if the
    incident branch is more confident, the model leans toward the incident
    residual proposal; if the normal branch is more confident, it leans back
    toward the normal-style residual proposal.
    """

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        confidence_scale: float = 1.0,
        confidence_max: float = 2.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.confidence_scale = confidence_scale
        self.confidence_max = confidence_max
        branch_input_dim = hidden_dim + horizon_steps * channels * int(use_normal_delta) + horizon_steps * channels * int(use_normal_delta_abs)
        logit_dim = horizon_steps * channels
        self.normal_confidence_head = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        self.incident_confidence_head = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        for head in [self.normal_confidence_head, self.incident_confidence_head]:
            final = head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)

        normal_conf_flat = self.normal_confidence_head(normal_input)
        incident_conf_flat = self.incident_confidence_head(incident_input)
        confidence_delta_flat = self.confidence_max * torch.tanh(incident_conf_flat - normal_conf_flat)
        gate_logits_flat = base_gate_logits_flat + self.confidence_scale * confidence_delta_flat
        gate = torch.sigmoid(gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": torch.sigmoid(base_gate_logits_flat)
                .reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "gate_logits": gate_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "normal_confidence_logits": normal_conf_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "incident_confidence_logits": incident_conf_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "confidence_delta": confidence_delta_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISUncertaintyGate(DualBranchSTTISProposalGate):
    """Proposal-aware gate corrected by branch error-risk predictions.

    Two branch heads predict log error magnitude for the normal and incident
    residual proposals. The gate logit is shifted by the predicted risk
    difference: high incident risk moves the model toward the normal branch,
    while high normal risk moves it toward the incident branch.
    """

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        uncertainty_scale: float = 1.0,
        uncertainty_max: float = 2.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.uncertainty_scale = uncertainty_scale
        self.uncertainty_max = uncertainty_max
        branch_input_dim = hidden_dim + horizon_steps * channels * int(use_normal_delta) + horizon_steps * channels * int(use_normal_delta_abs)
        logit_dim = horizon_steps * channels
        self.normal_uncertainty_head = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        self.incident_uncertainty_head = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        for head in [self.normal_uncertainty_head, self.incident_uncertainty_head]:
            final = head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)

        normal_risk_flat = self.normal_uncertainty_head(normal_input)
        incident_risk_flat = self.incident_uncertainty_head(incident_input)
        risk_delta_flat = self.uncertainty_max * torch.tanh(normal_risk_flat - incident_risk_flat)
        gate_logits_flat = base_gate_logits_flat + self.uncertainty_scale * risk_delta_flat
        gate = torch.sigmoid(gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": torch.sigmoid(base_gate_logits_flat)
                .reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "gate_logits": gate_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "normal_risk": normal_risk_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "incident_risk": incident_risk_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "risk_delta": risk_delta_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISProposalUncertaintyGate(DualBranchSTTISProposalGate):
    """Proposal-aware uncertainty gate whose risk heads inspect gate features."""

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        uncertainty_scale: float = 1.0,
        uncertainty_max: float = 2.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.uncertainty_scale = uncertainty_scale
        self.uncertainty_max = uncertainty_max
        risk_input_dim = int(self.gate_head[0].in_features) + horizon_steps * channels
        logit_dim = horizon_steps * channels
        self.base_gate_logit_norm = nn.LayerNorm(logit_dim)
        self.normal_uncertainty_head = nn.Sequential(
            nn.Linear(risk_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        self.incident_uncertainty_head = nn.Sequential(
            nn.Linear(risk_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        for head in [self.normal_uncertainty_head, self.incident_uncertainty_head]:
            final = head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)

        risk_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat)], dim=-1)
        normal_risk_flat = self.normal_uncertainty_head(risk_input)
        incident_risk_flat = self.incident_uncertainty_head(risk_input)
        risk_delta_flat = self.uncertainty_max * torch.tanh(normal_risk_flat - incident_risk_flat)
        gate_logits_flat = base_gate_logits_flat + self.uncertainty_scale * risk_delta_flat
        gate = torch.sigmoid(gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = (1.0 - gate) * normal_residual + gate * incident_residual
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": torch.sigmoid(base_gate_logits_flat)
                .reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "gate_logits": gate_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "normal_risk": normal_risk_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "incident_risk": incident_risk_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "risk_delta": risk_delta_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISLocalSelectorGate(DualBranchSTTISProposalGate):
    """Local mixture selector over normal, incident, and base fused proposals."""

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        selector_temperature: float = 1.0,
        selector_init_base_bias: float = 2.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.selector_temperature = selector_temperature
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        self.base_gate_logit_norm = nn.LayerNorm(logit_dim)
        self.selector_head = nn.Sequential(
            nn.Linear(gate_input_dim + logit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim * 3),
        )
        final = self.selector_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                final.bias.reshape(logit_dim, 3)[:, 2].fill_(selector_init_base_bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        base_gate = torch.sigmoid(base_gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        base_fused_residual = (1.0 - base_gate) * normal_residual + base_gate * incident_residual

        selector_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat)], dim=-1)
        selector_logits_flat = self.selector_head(selector_input)
        selector_logits = selector_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels, 3)
        temperature = max(float(self.selector_temperature), 1e-6)
        selector_weights = torch.softmax(selector_logits / temperature, dim=-1)
        proposal_stack = torch.stack([normal_residual, incident_residual, base_fused_residual], dim=-1)
        residual = (selector_weights * proposal_stack).sum(dim=-1)

        effective_gate = selector_weights[..., 1] + selector_weights[..., 2] * base_gate
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = effective_gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "base_fused_residual": base_fused_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": effective_gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": base_gate.permute(0, 2, 1, 3).contiguous(),
                "selector_logits": selector_logits.permute(0, 2, 1, 3, 4).contiguous(),
                "selector_weights": selector_weights.permute(0, 2, 1, 3, 4).contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISNormalVetoGate(DualBranchSTTISProposalGate):
    """Two-stage gate that only vetoes the base fused proposal toward normal."""

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        normal_veto_scale: float = 1.0,
        normal_veto_temperature: float = 1.0,
        normal_veto_init_bias: float = -4.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
        )
        self.normal_veto_scale = normal_veto_scale
        self.normal_veto_temperature = normal_veto_temperature
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        self.base_gate_logit_norm = nn.LayerNorm(logit_dim)
        self.normal_veto_head = nn.Sequential(
            nn.Linear(gate_input_dim + logit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        final = self.normal_veto_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, normal_veto_init_bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        base_gate = torch.sigmoid(base_gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        base_fused_residual = (1.0 - base_gate) * normal_residual + base_gate * incident_residual

        veto_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat)], dim=-1)
        normal_veto_logits_flat = self.normal_veto_head(veto_input)
        temperature = max(float(self.normal_veto_temperature), 1e-6)
        normal_veto = torch.sigmoid(normal_veto_logits_flat / temperature).reshape(
            batch_size, nodes, self.horizon_steps, self.channels
        )
        normal_veto_amount = (float(self.normal_veto_scale) * normal_veto).clamp(0.0, 1.0)
        residual = (1.0 - normal_veto_amount) * base_fused_residual + normal_veto_amount * normal_residual
        effective_gate = (1.0 - normal_veto_amount) * base_gate
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = effective_gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "base_fused_residual": base_fused_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": effective_gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": base_gate.permute(0, 2, 1, 3).contiguous(),
                "normal_veto": normal_veto.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_amount": normal_veto_amount.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_logits": normal_veto_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


class DualBranchSTTISImpactConditionedNormalVetoGate(DualBranchSTTISNormalVetoGate):
    """Normal-veto gate whose detector sees predicted incident-impact cues."""

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        normal_veto_scale: float = 1.0,
        normal_veto_temperature: float = 1.0,
        normal_veto_init_bias: float = -4.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
            normal_veto_scale=normal_veto_scale,
            normal_veto_temperature=normal_veto_temperature,
            normal_veto_init_bias=normal_veto_init_bias,
        )
        self.impact_feature_norm = nn.LayerNorm(horizon_steps)
        self.event_aux_feature_norm = nn.LayerNorm(3)
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        aux_feature_dim = horizon_steps + 1 + 3
        self.normal_veto_head = nn.Sequential(
            nn.Linear(gate_input_dim + logit_dim + aux_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, logit_dim),
        )
        final = self.normal_veto_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, normal_veto_init_bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        base_gate = torch.sigmoid(base_gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        base_fused_residual = (1.0 - base_gate) * normal_residual + base_gate * incident_residual

        impact_node = self.impact_head(h_incident)
        impact_feature = self.impact_feature_norm(impact_node)
        event_pool = (h_incident * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(event_pool)
        event_feature = self.event_aux_feature_norm(event_aux).unsqueeze(1).expand(-1, nodes, -1)
        node_logits = self.node_aux_head(h_incident).squeeze(-1)
        node_feature = torch.tanh(node_logits).unsqueeze(-1)
        aux_feature = torch.cat([impact_feature, node_feature, event_feature], dim=-1)

        veto_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat), aux_feature], dim=-1)
        normal_veto_logits_flat = self.normal_veto_head(veto_input)
        temperature = max(float(self.normal_veto_temperature), 1e-6)
        normal_veto = torch.sigmoid(normal_veto_logits_flat / temperature).reshape(
            batch_size, nodes, self.horizon_steps, self.channels
        )
        normal_veto_amount = (float(self.normal_veto_scale) * normal_veto).clamp(0.0, 1.0)
        residual = (1.0 - normal_veto_amount) * base_fused_residual + normal_veto_amount * normal_residual
        effective_gate = (1.0 - normal_veto_amount) * base_gate
        residual = residual.permute(0, 2, 1, 3).contiguous()

        if return_details:
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "base_fused_residual": base_fused_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": effective_gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": base_gate.permute(0, 2, 1, 3).contiguous(),
                "normal_veto": normal_veto.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_amount": normal_veto_amount.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_logits": normal_veto_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "impact_condition_features": aux_feature,
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact_node.permute(0, 2, 1).contiguous(), event_aux, node_logits, details
        return residual, impact_node.permute(0, 2, 1).contiguous(), event_aux, node_logits


class DualBranchSTTISHierarchicalImpactNormalVetoGate(DualBranchSTTISImpactConditionedNormalVetoGate):
    """Impact-conditioned normal-veto with a node-event prior and element refinement."""

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        normal_veto_scale: float = 1.0,
        normal_veto_temperature: float = 1.0,
        normal_veto_init_bias: float = -4.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
            normal_veto_scale=normal_veto_scale,
            normal_veto_temperature=normal_veto_temperature,
            normal_veto_init_bias=normal_veto_init_bias,
        )
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        aux_feature_dim = horizon_steps + 1 + 3
        detector_dim = gate_input_dim + logit_dim + aux_feature_dim
        self.node_event_veto_head = nn.Sequential(
            nn.Linear(detector_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        node_final = self.node_event_veto_head[-1]
        if isinstance(node_final, nn.Linear):
            nn.init.zeros_(node_final.weight)
            nn.init.constant_(node_final.bias, normal_veto_init_bias)
        element_final = self.normal_veto_head[-1]
        if isinstance(element_final, nn.Linear):
            nn.init.zeros_(element_final.weight)
            nn.init.zeros_(element_final.bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        base_gate = torch.sigmoid(base_gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        base_fused_residual = (1.0 - base_gate) * normal_residual + base_gate * incident_residual

        impact_node = self.impact_head(h_incident)
        impact_feature = self.impact_feature_norm(impact_node)
        event_pool = (h_incident * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(event_pool)
        event_feature = self.event_aux_feature_norm(event_aux).unsqueeze(1).expand(-1, nodes, -1)
        node_logits = self.node_aux_head(h_incident).squeeze(-1)
        node_feature = torch.tanh(node_logits).unsqueeze(-1)
        aux_feature = torch.cat([impact_feature, node_feature, event_feature], dim=-1)

        detector_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat), aux_feature], dim=-1)
        element_logits_flat = self.normal_veto_head(detector_input)
        node_event_logits = self.node_event_veto_head(detector_input).reshape(batch_size, nodes, 1, 1)
        normal_veto_logits = element_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels) + node_event_logits
        temperature = max(float(self.normal_veto_temperature), 1e-6)
        normal_veto = torch.sigmoid(normal_veto_logits / temperature)
        normal_veto_amount = (float(self.normal_veto_scale) * normal_veto).clamp(0.0, 1.0)
        residual = (1.0 - normal_veto_amount) * base_fused_residual + normal_veto_amount * normal_residual
        effective_gate = (1.0 - normal_veto_amount) * base_gate
        residual = residual.permute(0, 2, 1, 3).contiguous()

        if return_details:
            node_event_veto = torch.sigmoid(node_event_logits / temperature).squeeze(-1).squeeze(-1)
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "base_fused_residual": base_fused_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": effective_gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": base_gate.permute(0, 2, 1, 3).contiguous(),
                "normal_veto": normal_veto.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_amount": normal_veto_amount.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_logits": normal_veto_logits.permute(0, 2, 1, 3).contiguous(),
                "element_normal_veto_logits": element_logits_flat.reshape(batch_size, nodes, self.horizon_steps, self.channels)
                .permute(0, 2, 1, 3)
                .contiguous(),
                "node_event_normal_veto": node_event_veto,
                "node_event_normal_veto_logits": node_event_logits.squeeze(-1).squeeze(-1),
                "impact_condition_features": aux_feature,
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact_node.permute(0, 2, 1).contiguous(), event_aux, node_logits, details
        return residual, impact_node.permute(0, 2, 1).contiguous(), event_aux, node_logits


class DualBranchSTTISNodeEventNormalVetoGate(DualBranchSTTISNormalVetoGate):
    """Normal-veto gate with one node-event veto score broadcast to all horizons/channels."""

    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        sttis_heads: int,
        sttis_temporal_layers: int,
        sttis_spatial_topk: int,
        sttis_adj_bias: float,
        proposal_feature_count: int = 5,
        normal_veto_scale: float = 1.0,
        normal_veto_temperature: float = 1.0,
        normal_veto_init_bias: float = -4.0,
    ) -> None:
        super().__init__(
            channels=channels,
            hist_input_channels=hist_input_channels,
            node_context_dim=node_context_dim,
            global_context_dim=global_context_dim,
            horizon_steps=horizon_steps,
            hidden_dim=hidden_dim,
            graph_layers=graph_layers,
            dropout=dropout,
            graph_sigma=graph_sigma,
            graph_mode=graph_mode,
            use_normal_delta=use_normal_delta,
            use_normal_delta_abs=use_normal_delta_abs,
            sttis_heads=sttis_heads,
            sttis_temporal_layers=sttis_temporal_layers,
            sttis_spatial_topk=sttis_spatial_topk,
            sttis_adj_bias=sttis_adj_bias,
            proposal_feature_count=proposal_feature_count,
            normal_veto_scale=normal_veto_scale,
            normal_veto_temperature=normal_veto_temperature,
            normal_veto_init_bias=normal_veto_init_bias,
        )
        gate_input_dim = int(self.gate_head[0].in_features)
        logit_dim = horizon_steps * channels
        self.normal_veto_head = nn.Sequential(
            nn.Linear(gate_input_dim + logit_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final = self.normal_veto_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, normal_veto_init_bias)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        batch_size, _input_steps, nodes, _hist_channels = hist_residual.shape
        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx_input = torch.cat([node_context, global_rep], dim=-1)

        h_normal = self.encode_temporal(self.normal_encoder, hist_residual)
        h_normal = self.normal_input_norm(h_normal + self.normal_context_proj(ctx_input))

        h_incident = self.incident_temporal_encoder(hist_residual)
        h_incident = self.incident_input_norm(h_incident + self.incident_context_proj(ctx_input))
        adj_all, _adj_left, _adj_right, valid = self.build_adjacency(node_context)
        h_normal = h_normal * valid.unsqueeze(-1)
        h_incident = h_incident * valid.unsqueeze(-1)
        for layer in self.incident_spatial_layers:
            h_incident = layer(h_incident, adj_all, valid)

        delta_features = self.normal_delta_features(
            normal_delta=normal_delta,
            batch_size=batch_size,
            nodes=nodes,
            dtype=hist_residual.dtype,
            device=hist_residual.device,
        )
        normal_input = torch.cat([h_normal, *delta_features], dim=-1) if delta_features else h_normal
        incident_input = torch.cat([h_incident, *delta_features], dim=-1) if delta_features else h_incident
        base_gate_input = (
            torch.cat([h_normal, h_incident, *delta_features], dim=-1)
            if delta_features
            else torch.cat([h_normal, h_incident], dim=-1)
        )

        normal_residual = self.normal_decoder(normal_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        incident_residual = self.incident_decoder(incident_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        proposal_input = self.proposal_features(normal_residual, incident_residual)
        gate_input = torch.cat([base_gate_input, proposal_input], dim=-1)
        base_gate_logits_flat = self.gate_head(gate_input)
        base_gate = torch.sigmoid(base_gate_logits_flat).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        base_fused_residual = (1.0 - base_gate) * normal_residual + base_gate * incident_residual

        veto_input = torch.cat([gate_input, self.base_gate_logit_norm(base_gate_logits_flat)], dim=-1)
        node_veto_logits = self.normal_veto_head(veto_input)
        temperature = max(float(self.normal_veto_temperature), 1e-6)
        node_veto = torch.sigmoid(node_veto_logits / temperature).reshape(batch_size, nodes, 1, 1)
        normal_veto = node_veto.expand(-1, -1, self.horizon_steps, self.channels)
        normal_veto_amount = (float(self.normal_veto_scale) * normal_veto).clamp(0.0, 1.0)
        residual = (1.0 - normal_veto_amount) * base_fused_residual + normal_veto_amount * normal_residual
        effective_gate = (1.0 - normal_veto_amount) * base_gate
        residual = residual.permute(0, 2, 1, 3).contiguous()

        gate_node = effective_gate.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        fused = (1.0 - gate_node) * h_normal + gate_node * h_incident
        impact = self.impact_head(h_incident).permute(0, 2, 1).contiguous()
        pooled = (fused * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(fused).squeeze(-1)
        if return_details:
            normal_veto_logits = node_veto_logits.reshape(batch_size, nodes, 1, 1).expand(
                -1, -1, self.horizon_steps, self.channels
            )
            details = {
                "normal_residual": normal_residual.permute(0, 2, 1, 3).contiguous(),
                "incident_residual": incident_residual.permute(0, 2, 1, 3).contiguous(),
                "base_fused_residual": base_fused_residual.permute(0, 2, 1, 3).contiguous(),
                "gate": effective_gate.permute(0, 2, 1, 3).contiguous(),
                "base_gate": base_gate.permute(0, 2, 1, 3).contiguous(),
                "normal_veto": normal_veto.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_amount": normal_veto_amount.permute(0, 2, 1, 3).contiguous(),
                "normal_veto_logits": normal_veto_logits.permute(0, 2, 1, 3).contiguous(),
                "node_event_normal_veto": node_veto.squeeze(-1).squeeze(-1),
                "node_event_normal_veto_logits": node_veto_logits.squeeze(-1),
                "h_normal": h_normal,
                "h_incident": h_incident,
                "valid": valid,
            }
            return residual, impact, event_aux, node_logits, details
        return residual, impact, event_aux, node_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_gate_no_aux"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-mode", choices=["directional", "undirected"], default="undirected")
    parser.add_argument("--graph-sigma", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heatmap-aux-weight", type=float, default=0.0)
    parser.add_argument("--event-aux-weight", type=float, default=0.0)
    parser.add_argument("--node-aux-weight", type=float, default=0.0)
    parser.add_argument("--sttis-heads", type=int, default=4)
    parser.add_argument("--sttis-temporal-layers", type=int, default=1)
    parser.add_argument("--sttis-spatial-topk", type=int, default=8)
    parser.add_argument("--sttis-adj-bias", type=float, default=0.25)
    parser.add_argument("--use-normal-delta", action="store_true")
    parser.add_argument("--use-normal-delta-abs", action="store_true")
    parser.add_argument("--use-dual-hist-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Optional per-split cap for quick smoke evaluation; 0 evaluates each full split.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Dual-branch ST-TIS gate training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    residual_beta: float,
    log_df: pd.DataFrame,
) -> None:
    test = metrics["test"]
    lines = [
        "# Dual-Branch ST-TIS Gate",
        "",
        "This variant keeps the normal branch and residual gate unchanged, but replaces the incident branch with temporal self-attention plus top-k graph-biased spatial attention.",
        "",
        "## Test Result",
        "",
        f"- all candidates robust MAE: `{test['all_candidates_baseline_robust_mae']:.4f} -> {test['all_candidates_model_robust_mae']:.4f}` ({test['all_candidates_improvement_pct']:.2f}%)",
        f"- affected candidates robust MAE: `{test['affected_candidates_baseline_robust_mae']:.4f} -> {test['affected_candidates_model_robust_mae']:.4f}` ({test['affected_candidates_improvement_pct']:.2f}%)",
        "",
        "## ST-TIS Settings",
        "",
        f"- sttis_heads: {args.sttis_heads}",
        f"- sttis_temporal_layers: {args.sttis_temporal_layers}",
        f"- sttis_spatial_topk: {args.sttis_spatial_topk}",
        f"- sttis_adj_bias: {args.sttis_adj_bias}",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Data Settings",
        "",
        f"- cache_path: `{args.cache_path}`",
        f"- epochs: {args.epochs}",
        f"- hidden_dim: {args.hidden_dim}",
        f"- graph_layers: {args.graph_layers}",
        f"- graph_mode: {args.graph_mode}",
        f"- graph_sigma: {args.graph_sigma}",
        f"- use_normal_delta: {args.use_normal_delta}",
        f"- use_normal_delta_abs: {args.use_normal_delta_abs}",
        f"- use_dual_hist_residual: {args.use_dual_hist_residual}",
        "",
        "## Split Counts",
        "",
        pd.DataFrame([{"split": key, "samples": value} for key, value in split_counts.items()]).to_markdown(index=False),
        "",
        "## Metrics",
        "",
        pd.DataFrame([{"split": split, **values} for split, values in metrics.items()]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Training",
        "",
        f"- best_epoch: {int(log_df.loc[log_df['val_loss'].idxmin(), 'epoch']) if not log_df.empty else 'n/a'}",
        f"- best_val_loss: {float(log_df['val_loss'].min()) if not log_df.empty else float('nan'):.4f}",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path.resolve()
    device = choose_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)

    stats = compute_stats(cache_path)
    indices = split_indices(cache_path)
    eval_indices = {
        split: cap_indices(idx, args.max_eval_samples, args.seed + offset)
        for offset, (split, idx) in enumerate(indices.items())
    }
    train_indices_full = indices["train"]
    if args.max_train_samples > 0 and train_indices_full.size > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_indices = np.sort(rng.choice(train_indices_full, size=args.max_train_samples, replace=False))
    else:
        train_indices = train_indices_full

    train_loader = make_loader(cache_path, train_indices, stats, args.batch_size, shuffle=True)
    val_loader = make_loader(cache_path, eval_indices["val"], stats, args.batch_size, shuffle=False)
    shapes = infer_cache_shapes(cache_path)
    model = DualBranchSTTISGate(
        channels=shapes["channels"],
        hist_input_channels=len(CHANNELS) * (2 if args.use_dual_hist_residual else 1),
        node_context_dim=shapes["node_context_dim"],
        global_context_dim=shapes["global_context_dim"],
        horizon_steps=shapes["horizon_steps"],
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        graph_sigma=args.graph_sigma,
        graph_mode=args.graph_mode,
        use_normal_delta=args.use_normal_delta,
        use_normal_delta_abs=args.use_normal_delta_abs,
        sttis_heads=args.sttis_heads,
        sttis_temporal_layers=args.sttis_temporal_layers,
        sttis_spatial_topk=args.sttis_spatial_topk,
        sttis_adj_bias=args.sttis_adj_bias,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            loss, _parts = compute_loss(model, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loader(model, val_loader, args, device)
        log_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"}}
        )
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_metrics['loss']:.4f}", flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    beta_candidates = [round(x, 2) for x in np.arange(0.0, 1.51, 0.05)]
    val_metrics_by_beta = forecast_metrics_for_loader(model, val_loader, beta_candidates, device)
    beta_df = pd.DataFrame([{"residual_beta": beta, **values} for beta, values in val_metrics_by_beta.items()])
    beta_df.to_csv(output_dir / "residual_beta_sweep.csv", index=False)
    residual_beta = float(beta_df.loc[beta_df["all_candidates_model_robust_mae"].idxmin(), "residual_beta"])

    metrics: dict[str, dict[str, float]] = {}
    for split, idx in eval_indices.items():
        loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
        metrics[split] = forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]

    region_code_arr = region_codes(cache_path)
    region_metrics = []
    for code in sorted(np.unique(region_code_arr[eval_indices["test"]]).tolist()):
        mask_idx = indices["test"][region_code_arr[indices["test"]] == code]
        mask_idx = cap_indices(mask_idx, args.max_eval_samples, args.seed + 100 + int(code))
        loader = make_loader(cache_path, mask_idx, stats, args.batch_size, shuffle=False)
        row = {"region_code": int(code), "samples": int(mask_idx.size)}
        row.update(forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta])
        region_metrics.append(row)

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    ckpt_args = json_safe_args(args)
    ckpt_args["model_class"] = "DualBranchSTTISGate"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": ckpt_args,
            "residual_beta": residual_beta,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "region_metrics": region_metrics,
                "samples": split_counts,
                "eval_samples": eval_counts,
                "residual_beta": residual_beta,
                "cache_path": str(cache_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["model_class"] = "DualBranchSTTISGate"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, metrics, split_counts, residual_beta, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
