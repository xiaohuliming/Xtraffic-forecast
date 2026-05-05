#!/usr/bin/env python3
"""Train a calibrated dual-branch gate with an ST-TIS-style incident branch.

This keeps the ST-TIS-style incident branch architecture unchanged and adds a
small training-only gate calibration regularizer. The regularizer nudges the
gate toward the locally better branch only when the two branch errors differ
enough to be informative, avoiding the stronger confidence-head intervention
that previously hurt final fusion quality.
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
    compute_stats,
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
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_calibrated_gate_no_aux"),
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
    parser.add_argument("--branch-loss-weight", type=float, default=0.02)
    parser.add_argument("--gate-alignment-weight", type=float, default=0.01)
    parser.add_argument("--gate-target-temperature", type=float, default=0.15)
    parser.add_argument("--gate-gap-margin", type=float, default=0.05)
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


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_calibrated_loss(
    model: DualBranchSTTISGate,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        hist,
        hist_normal,
        node,
        global_context,
        normal_delta,
        y,
        y_mask,
        impact,
        impact_mask,
        event_aux,
        node_affected,
        node_valid,
    ) = [item.to(device) for item in batch]
    if args.use_dual_hist_residual:
        hist = torch.cat([hist, hist_normal], dim=-1)

    pred_y, pred_impact, pred_event_aux, pred_node_logits, details = model(
        hist,
        node,
        global_context,
        normal_delta,
        return_details=True,
    )
    residual_loss_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = masked_mean(residual_loss_raw, y_mask)

    impact_loss_raw = nn.functional.smooth_l1_loss(pred_impact, impact, reduction="none")
    impact_loss = masked_mean(impact_loss_raw, impact_mask)
    event_aux_loss = nn.functional.smooth_l1_loss(pred_event_aux, event_aux)
    node_bce = nn.functional.binary_cross_entropy_with_logits(pred_node_logits, node_affected, reduction="none")
    node_aux_loss = masked_mean(node_bce, node_valid)

    normal_branch_error = nn.functional.smooth_l1_loss(details["normal_residual"], y, reduction="none")
    incident_branch_error = nn.functional.smooth_l1_loss(details["incident_residual"], y, reduction="none")
    branch_loss = 0.5 * (masked_mean(normal_branch_error, y_mask) + masked_mean(incident_branch_error, y_mask))

    error_gap = normal_branch_error.detach() - incident_branch_error.detach()
    target_gate = torch.sigmoid(error_gap / max(args.gate_target_temperature, 1e-6))
    gap_weight = (error_gap.abs() / max(args.gate_gap_margin, 1e-6)).clamp(max=1.0)
    gate_alignment_raw = nn.functional.smooth_l1_loss(details["gate"], target_gate, reduction="none")
    gate_alignment_loss = masked_mean(gate_alignment_raw * gap_weight, y_mask)

    loss = (
        residual_loss
        + args.heatmap_aux_weight * impact_loss
        + args.event_aux_weight * event_aux_loss
        + args.node_aux_weight * node_aux_loss
        + args.branch_loss_weight * branch_loss
        + args.gate_alignment_weight * gate_alignment_loss
    )
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "impact_loss": float(impact_loss.detach().cpu()),
        "event_aux_loss": float(event_aux_loss.detach().cpu()),
        "node_aux_loss": float(node_aux_loss.detach().cpu()),
        "branch_loss": float(branch_loss.detach().cpu()),
        "gate_alignment_loss": float(gate_alignment_loss.detach().cpu()),
    }


def evaluate_loader(model: DualBranchSTTISGate, loader: torch.utils.data.DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "residual_loss": 0.0,
        "impact_loss": 0.0,
        "event_aux_loss": 0.0,
        "node_aux_loss": 0.0,
        "branch_loss": 0.0,
        "gate_alignment_loss": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            loss, parts = compute_calibrated_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


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
        "# Dual-Branch ST-TIS Calibrated Gate",
        "",
        "This variant keeps the ST-TIS-style incident branch architecture unchanged and adds a small training-only gate calibration regularizer.",
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
        f"- branch_loss_weight: {args.branch_loss_weight}",
        f"- gate_alignment_weight: {args.gate_alignment_weight}",
        f"- gate_target_temperature: {args.gate_target_temperature}",
        f"- gate_gap_margin: {args.gate_gap_margin}",
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
            loss, _parts = compute_calibrated_loss(model, batch, args, device)
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
    ckpt_args["training_variant"] = "sttis_calibrated_gate"
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
    config["training_variant"] = "sttis_calibrated_gate"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, metrics, split_counts, residual_beta, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
