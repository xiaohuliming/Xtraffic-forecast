"""Reusable sparse multi-head attention layer (pure PyTorch).

Used by:
  • dist_net/cross_attn.py for branch-to-branch cross-attention
  • dist_net/normal_branch.py + incident_branch.py as GAT-style self-attention

Semantics:
  • Given edge_index (2, E), for each edge (src, tgt) the message goes src → tgt
  • Multi-head dot-product attention with per-head softmax over edges
    going to the same target node
  • Output projection W_O applied after concatenating heads

Convention:
  edge_index[0] = source nodes (key/value providers)
  edge_index[1] = target nodes (query owners)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _segment_softmax(scores: torch.Tensor, target_idx: torch.Tensor,
                     num_targets: int) -> torch.Tensor:
    """Softmax of scores grouped by target_idx (per-head).

    scores      : (E, H)  per-edge per-head pre-softmax
    target_idx  : (E,)
    num_targets : int

    Returns alpha (E, H).
    """
    H = scores.size(-1)
    # max per (target, head) for numerical stability
    max_per_t = torch.full((num_targets, H), float("-inf"),
                           device=scores.device, dtype=scores.dtype)
    # scatter_reduce on dim 0 with multi-channel
    expanded_idx = target_idx.unsqueeze(-1).expand(-1, H)               # (E, H)
    max_per_t = max_per_t.scatter_reduce(0, expanded_idx, scores,
                                         reduce="amax", include_self=True)
    max_per_t = torch.where(torch.isfinite(max_per_t), max_per_t,
                            torch.zeros_like(max_per_t))
    exp = (scores - max_per_t[target_idx]).exp()                        # (E, H)
    sum_per_t = torch.zeros((num_targets, H), device=scores.device, dtype=scores.dtype)
    sum_per_t.index_add_(0, target_idx, exp)
    return exp / sum_per_t[target_idx].clamp(min=1e-12)


class SparseAttention(nn.Module):
    """Generic sparse multi-head attention.

    If z_query is z_kv (passed as the same tensor), this is sparse self-attention
    (a "GAT-like" layer). Otherwise it's cross-attention.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4,
                 zero_init_output: bool = True):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.d = hidden_dim
        self.h = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        if zero_init_output:
            nn.init.zeros_(self.W_O.weight)
            nn.init.zeros_(self.W_O.bias)

    def forward(self, z_query: torch.Tensor, z_kv: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Inputs flat-shaped: (M, d) where M = batch * nodes.

        Returns output of shape (M, d).
        """
        M = z_query.size(0)
        src = edge_index[0]
        tgt = edge_index[1]

        Q = self.W_Q(z_query).view(M, self.h, self.head_dim)              # (M, H, hd)
        K = self.W_K(z_kv).view(M, self.h, self.head_dim)
        V = self.W_V(z_kv).view(M, self.h, self.head_dim)

        scores = (Q[tgt] * K[src]).sum(dim=-1) * self.scale               # (E, H)
        alpha = _segment_softmax(scores, tgt, num_targets=M)              # (E, H)
        # weighted value per (edge, head, hd)
        msg = alpha.unsqueeze(-1) * V[src]                                # (E, H, hd)

        out = torch.zeros((M, self.h, self.head_dim),
                          device=z_query.device, dtype=z_query.dtype)
        out.index_add_(0, tgt, msg)                                       # (M, H, hd)
        out = out.reshape(M, self.d)                                      # concat heads
        return self.W_O(out)


def batched_edge_index(edge_index: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """Replicate (2, E) edge_index across B disjoint copies offset by N.

    Returns (2, B*E) suitable for use with flattened (B*N, d) node features.
    """
    E = edge_index.size(1)
    device = edge_index.device
    rep = edge_index.repeat(1, B)
    offsets = torch.arange(B, device=device).repeat_interleave(E) * N
    return rep + offsets.unsqueeze(0)


class SparseTransformerLayer(nn.Module):
    """Transformer-style layer: sparse self-attention + FFN, both with residual.

    Used by the spatial-temporal encoder in NormalBranch / IncidentBranch.
    Input/output shape: (M, d) where M = B*N (flattened batched graph).
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SparseAttention(hidden_dim, n_heads=n_heads, zero_init_output=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        d_ff = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_dim),
        )

    def forward(self, h_flat: torch.Tensor, edge_index_batched: torch.Tensor) -> torch.Tensor:
        # Pre-LN style for stability
        h_n = self.norm1(h_flat)
        h_flat = h_flat + self.attn(h_n, h_n, edge_index_batched)
        h_flat = h_flat + self.ffn(self.norm2(h_flat))
        return h_flat
