"""IncidentBranch — event-aware encoder.

Implementation status: v3 — full real design (doc §6).
  ✓ §6.3 Multi-scale temporal patching (short-focus weighted)
  ✓ §6.4 Encoder = L layers of (sparse spatial GAT + dense incident→sensor cross-attention)
  ✓ §6.2 Learned D tensor (B, M, N, d_D) + attn_bias = Linear(d_D, 1)(D)
  ✓ §6.5 Three parallel decay heads with per-incident learned σ/amp
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patching import MultiScalePatching
from .sparse_attn import SparseAttention, batched_edge_index


class _IncidentToSensorAttn(nn.Module):
    """Dense cross-attention: sensors query, incidents provide K/V.

    For each batched sample b, sensor n attends across all M_max active
    incidents (masked by incident_mask). M is small (≤32) so dense is cheap.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.d = hidden_dim
        self.h = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        # residual-init: zero output at start
        nn.init.zeros_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)

    def forward(self, h_sensor: torch.Tensor, inc_emb: torch.Tensor,
                incident_mask: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """h_sensor : (B, N, d), inc_emb : (B, M, d),
        incident_mask : (B, M) bool, attn_bias : (B, M, N) or None.
        Returns delta sensor representation (B, N, d)."""
        B, N, d = h_sensor.shape
        _, M, _ = inc_emb.shape
        H = self.h
        hd = self.head_dim

        Q = self.W_Q(h_sensor).view(B, N, H, hd)                     # (B, N, H, hd)
        K = self.W_K(inc_emb).view(B, M, H, hd)
        V = self.W_V(inc_emb).view(B, M, H, hd)

        # scores: (B, H, N, M)
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) * self.scale
        if attn_bias is not None:
            # attn_bias: (B, M, N) -> broadcast to (B, 1, N, M) by transpose
            scores = scores + attn_bias.transpose(-1, -2).unsqueeze(1)

        # mask: hide padded incidents per batch
        mask = incident_mask.view(B, 1, 1, M)                        # bool
        scores = scores.masked_fill(~mask, float("-inf"))

        # if a sensor has NO valid incidents (all masked), softmax would be NaN.
        # handle by zero-replacing those rows after softmax.
        any_valid = incident_mask.any(dim=-1)                        # (B,)
        alpha = torch.softmax(scores, dim=-1)                        # (B, H, N, M)
        alpha = torch.nan_to_num(alpha, nan=0.0)                     # NaN-safe

        # weighted sum: (B, N, H, hd)
        out = torch.einsum("bhnm,bmhd->bnhd", alpha, V).reshape(B, N, d)
        out = self.W_O(out)
        # zero out batch items that had no valid incidents
        out = out * any_valid.view(B, 1, 1).float()
        return out


class _IncidentEncoderLayer(nn.Module):
    """One encoder layer: pre-LN spatial GAT + pre-LN incident cross-attn + FFN."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        # spatial
        self.norm_spatial = nn.LayerNorm(hidden_dim)
        self.spatial_attn = SparseAttention(hidden_dim, n_heads=n_heads,
                                            zero_init_output=False)
        # incident cross-attn
        self.norm_inc = nn.LayerNorm(hidden_dim)
        self.inc_attn = _IncidentToSensorAttn(hidden_dim, n_heads=n_heads)
        # FFN
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        d_ff = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index_batched: torch.Tensor,
                inc_emb: torch.Tensor, incident_mask: torch.Tensor,
                attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """h : (B, N, d) — sensor representation
        inc_emb : (B, M, d) — incident features projected
        edge_index_batched : (2, B*E) flat indexing into (B*N, d).
        """
        B, N, d = h.shape
        # spatial step (flatten to (B*N, d) for sparse op)
        h_n = self.norm_spatial(h)
        h_flat = h.reshape(B * N, d)
        h_n_flat = h_n.reshape(B * N, d)
        spatial_msg = self.spatial_attn(h_n_flat, h_n_flat, edge_index_batched)
        h_flat = h_flat + spatial_msg
        h = h_flat.reshape(B, N, d)

        # incident cross-attn step
        h_n2 = self.norm_inc(h)
        inc_msg = self.inc_attn(h_n2, inc_emb, incident_mask, attn_bias=attn_bias)
        h = h + inc_msg

        # FFN
        h = h + self.ffn(self.norm_ffn(h))
        return h


class _DecayHead(nn.Module):
    """One decay head per design doc §6.5.

    Learns σ + amplitude per (B, M) incident from `incident_feat`. Builds a
    Gaussian time envelope ω_τ = amp · exp(-τ²/(2σ²)) over the prediction
    horizon. Aggregates (incident → node) via D-derived impact weights, then
    bounds the aggregation via tanh.

    Output: delta_k of shape (B, N, T_p, C_x).
    """

    def __init__(self, hidden_dim: int, d_D: int, c_e: int,
                 t_p: int, c_x: int, sigma_init: float):
        super().__init__()
        self.t_p = t_p
        self.c_x = c_x
        self.sigma_init = sigma_init   # in 5-min step units (short=2, mid=6, long=12)
        self.decay_params = nn.Linear(c_e, 2)   # → (σ_raw, amp_raw) per incident
        self.impact_proj = nn.Linear(d_D, 1)
        self.pred_base = nn.Linear(hidden_dim, t_p * c_x)
        # Zero-init pred_base so delta_pred starts at 0
        nn.init.zeros_(self.pred_base.weight)
        nn.init.zeros_(self.pred_base.bias)

    def forward(self, incident_feat: torch.Tensor, incident_mask: torch.Tensor,
                D: torch.Tensor, z_incident: torch.Tensor) -> torch.Tensor:
        """incident_feat : (B, M, C_e)
        incident_mask : (B, M) bool
        D             : (B, M, N, d_D) learned tensor
        z_incident    : (B, N, hidden_dim)
        Returns delta_k : (B, N, T_p, C_x)
        """
        B, M, _ = incident_feat.shape
        N = z_incident.size(1)
        device = incident_feat.device

        # Decay params per incident
        params = self.decay_params(incident_feat)                       # (B, M, 2)
        sigma_raw, amp_raw = params.unbind(-1)
        sigma = F.softplus(sigma_raw) * self.sigma_init                 # step units
        amp = torch.sigmoid(amp_raw)                                    # [0, 1]

        # Gaussian envelope over future horizon
        tau = torch.arange(1, self.t_p + 1, device=device, dtype=sigma.dtype)
        # broadcast: (B, M, T_p)
        envelope = amp.unsqueeze(-1) * torch.exp(
            -tau.view(1, 1, -1).pow(2) / (2.0 * sigma.unsqueeze(-1).pow(2) + 1e-8)
        )
        # zero-out padded incidents
        envelope = envelope * incident_mask.unsqueeze(-1).float()

        # Per-(incident, node) impact in [0, 1]
        impact = torch.sigmoid(self.impact_proj(D).squeeze(-1))         # (B, M, N)
        impact = impact * incident_mask.unsqueeze(-1).float()

        # Aggregate over incidents: (B, N, T_p), bounded via tanh (§6.5d)
        raw_sum = torch.einsum("bmn,bmt->bnt", impact, envelope)
        node_decay = torch.tanh(raw_sum)

        # Per-channel raw-flow base from z_incident
        pred_base = self.pred_base(z_incident).view(B, N, self.t_p, self.c_x)
        delta_k = node_decay.unsqueeze(-1) * pred_base                  # (B, N, T_p, C_x)
        return delta_k


class IncidentBranch(nn.Module):
    def __init__(self, c_x: int, c_meta: int, c_e: int, n_regions: int,
                 hidden_dim: int, t_h: int, t_p: int,
                 n_heads: int = 4, n_enc_layers: int = 2,
                 d_D: int = 32):
        super().__init__()
        self.c_x = c_x
        self.t_p = t_p
        self.hidden_dim = hidden_dim
        self.d_D = d_D

        self.patching = MultiScalePatching(
            c_x=c_x, d_t=5, hidden_dim=hidden_dim, t_h=t_h,
            init_weights=[0.2, 0.3, 0.5],  # short-focus
        )
        self.static_proj = nn.Linear(c_meta, hidden_dim)
        self.region_emb = nn.Embedding(n_regions, hidden_dim)
        self.inc_proj = nn.Linear(c_e, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Learned D tensor (B, M, N, d_D). We use d_D=32 (< hidden_dim=64) to
        # keep memory bounded: (B=32, M=32, N=990, d_D=32) ≈ 130 MB fp32.
        # Three input streams summed (all projected to d_D):
        #   incident features (B, M, C_e)         — type / severity / duration
        #   sensor static     (B, N, C_meta)      — sensor properties
        #   rel_feat          (B, M, N, 4)        — incident↔sensor geometry:
        #                                           [log_euclid, log_road,
        #                                            up/down, same_freeway]
        #     (the spatial inductive bias IGSTGNN gets for free via its preset
        #      D tensor; here we let the MLP learn how to use it.)
        self.D_inc_proj = nn.Linear(c_e, d_D)
        self.D_sensor_proj = nn.Linear(c_meta, d_D)
        self.D_rel_proj = nn.Linear(4, d_D)
        self.D_mlp = nn.Sequential(
            nn.Linear(d_D, 2 * d_D),
            nn.GELU(),
            nn.Linear(2 * d_D, d_D),
        )
        self.D_to_bias = nn.Linear(d_D, 1)
        # zero-init the bias projection so D contributes nothing at start;
        # the model learns to use it gradually.
        nn.init.zeros_(self.D_to_bias.weight)
        nn.init.zeros_(self.D_to_bias.bias)

        self.encoder_layers = nn.ModuleList([
            _IncidentEncoderLayer(hidden_dim, n_heads=n_heads, ffn_mult=4)
            for _ in range(n_enc_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # v3: three parallel decay heads (short / mid / long) per §6.5
        # σ_init in 5-min step units (10 min / 30 min / 60 min)
        self.decay_heads = nn.ModuleList([
            _DecayHead(hidden_dim=hidden_dim, d_D=d_D, c_e=c_e,
                       t_p=t_p, c_x=c_x, sigma_init=s_init)
            for s_init in (2.0, 6.0, 12.0)
        ])

    def forward(self, x_hist: torch.Tensor, x_hist_mask: torch.Tensor,
                incident_feat: torch.Tensor, incident_mask: torch.Tensor,
                static_meta: torch.Tensor, region_code: torch.Tensor,
                time_enc: torch.Tensor,
                edge_index: torch.Tensor,
                rel_feat: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, T_h, C_x = x_hist.shape

        x_in = torch.cat([x_hist, x_hist_mask.float()], dim=-1)
        e_temporal = self.patching(x_in, time_enc)                          # (B, N, d)

        s = self.static_proj(static_meta)
        r = self.region_emb(region_code).unsqueeze(1).expand(-1, N, -1)
        h = self.input_norm(e_temporal + s + r)                             # (B, N, d)

        inc_emb = self.inc_proj(incident_feat)                              # (B, M, d)

        # Build learned D tensor (B, M, N, d_D) from up to three input streams.
        inc_D    = self.D_inc_proj(incident_feat)                           # (B, M, d_D)
        sensor_D = self.D_sensor_proj(static_meta)                          # (B, N, d_D)
        D_in = inc_D.unsqueeze(2) + sensor_D.unsqueeze(1)                   # (B, M, N, d_D)
        if rel_feat is not None:
            rel_D = self.D_rel_proj(rel_feat)                               # (B, M, N, d_D)
            D_in = D_in + rel_D
        D = self.D_mlp(D_in)                                                # (B, M, N, d_D)
        attn_bias = self.D_to_bias(D).squeeze(-1)                           # (B, M, N)
        # also mask attn_bias for padded incidents (extra defense)
        attn_bias = attn_bias.masked_fill(~incident_mask.unsqueeze(-1), 0.0)
        self._D_last = D  # exposed for v3 decay heads

        big_edges = batched_edge_index(edge_index.to(h.device), B, N)

        for layer in self.encoder_layers:
            h = layer(h, big_edges, inc_emb, incident_mask, attn_bias=attn_bias)
        h = self.encoder_norm(h)
        z_incident = h                                                      # (B, N, d)

        # Three decay heads aggregating over active incidents
        delta_pred = torch.zeros(B, N, self.t_p, self.c_x, device=h.device, dtype=h.dtype)
        for head in self.decay_heads:
            delta_pred = delta_pred + head(incident_feat, incident_mask, D, z_incident)
        return z_incident, delta_pred
