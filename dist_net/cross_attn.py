"""BidirectionalCrossAttention — sparse cross-attention between branches.

Real implementation (v1) per design doc §7, using pure PyTorch (no PyG dependency).
Mathematically equivalent to a single-head PyG MessagePassing + segment_softmax.

For each (i,j) edge in the region-sampled graph:
  • target i's query attends to source j's key/value
  • attention scores are softmax-normalized PER TARGET (all edges into i)
  • only edges in the sparse graph contribute → O(E·d) compute, no N×N matrix

Bidirectional fusion:
  Step 1: z_normal queries from z_incident       → z_normal_updated
  Step 2: z_incident queries from z_normal_u     → z_incident_updated

Init: W_O initialized to zero so attention starts as residual identity
(z_branch passes through unchanged). Lets the model learn to mix gradually.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _segment_softmax(scores: torch.Tensor, target_idx: torch.Tensor,
                     num_targets: int) -> torch.Tensor:
    """Softmax of `scores` grouped by `target_idx`.

    scores      : (E,)         pre-softmax logits per edge
    target_idx  : (E,)         which group each edge belongs to (= target node)
    num_targets : int          total number of groups (= B*N)

    Returns alpha (E,) where sum of alphas for each target == 1.
    """
    # numerical stability: subtract max per target
    max_per_target = torch.full((num_targets,), float("-inf"),
                                device=scores.device, dtype=scores.dtype)
    max_per_target = max_per_target.scatter_reduce(
        0, target_idx, scores, reduce="amax", include_self=True,
    )
    # replace -inf (target with no edges, shouldn't happen here) with 0
    max_per_target = torch.where(
        torch.isfinite(max_per_target), max_per_target,
        torch.zeros_like(max_per_target),
    )
    exp = (scores - max_per_target[target_idx]).exp()
    sum_per_target = torch.zeros(num_targets, device=scores.device, dtype=scores.dtype)
    sum_per_target.index_add_(0, target_idx, exp)
    return exp / sum_per_target[target_idx].clamp(min=1e-12)


class SparseCrossAttn(nn.Module):
    """One-direction sparse cross-attention. z_query asks z_kv via edge_index.

    edge_index[0] = source (key/value side), edge_index[1] = target (query side).
    For our symmetrized region graph, each undirected edge appears in both
    directions so attention is naturally bidirectional within a single call.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.d = hidden_dim
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        # residual init: cross-attn contributes 0 at start, lets z passthrough
        nn.init.zeros_(self.W_O.weight)
        nn.init.zeros_(self.W_O.bias)

    def forward(self, z_query: torch.Tensor, z_kv: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        # z_query, z_kv: (M, d) where M = B*N (flattened super-graph)
        # edge_index: (2, E_total) batched supergraph edges
        M = z_query.size(0)
        src = edge_index[0]
        tgt = edge_index[1]

        Q = self.W_Q(z_query)
        K = self.W_K(z_kv)
        V = self.W_V(z_kv)

        scores = (Q[tgt] * K[src]).sum(dim=-1) / (self.d ** 0.5)   # (E,)
        alpha = _segment_softmax(scores, tgt, num_targets=M)        # (E,)
        msg = alpha.unsqueeze(-1) * V[src]                          # (E, d)

        out = torch.zeros_like(z_query)
        out.index_add_(0, tgt, msg)
        return self.W_O(out)


class BidirectionalCrossAttention(nn.Module):
    """Sparse bidirectional cross-attention over the region-sampled graph.

    Inputs:
      z_normal, z_incident : (B, N, d)
      edge_index           : (2, E) -- per-region symmetrized adjacency

    Outputs:
      z_normal_updated, z_incident_updated : (B, N, d)

    Internally we replicate edge_index across the batch (one disjoint graph per
    sample) so message passing never crosses batch boundaries.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cross_n = SparseCrossAttn(hidden_dim)   # normal queries from incident
        self.cross_i = SparseCrossAttn(hidden_dim)   # incident queries from normal_u
        self.norm_n = nn.LayerNorm(hidden_dim)
        self.norm_i = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _batched_edge_index(edge_index: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Replicate edge_index B times with offset N per copy.

        edge_index: (2, E) on host or matching device.
        Returns (2, B*E) suitable for use with flattened (B*N, d) features.
        """
        E = edge_index.size(1)
        device = edge_index.device
        rep = edge_index.repeat(1, B)                                   # (2, B*E)
        offsets = (torch.arange(B, device=device).repeat_interleave(E) * N)
        rep = rep + offsets.unsqueeze(0)                                 # broadcast over 2 rows
        return rep

    def forward(self, z_normal: torch.Tensor, z_incident: torch.Tensor,
                edge_index: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None:
            raise ValueError("BidirectionalCrossAttention requires edge_index "
                             "(stub fallback removed in v1).")

        B, N, d = z_normal.shape
        big_edges = self._batched_edge_index(
            edge_index.to(z_normal.device), B, N,
        )

        zn_flat = z_normal.reshape(B * N, d)
        zi_flat = z_incident.reshape(B * N, d)

        # Step 1: normal queries from incident
        update_n = self.cross_n(zn_flat, zi_flat, big_edges)
        zn_u_flat = self.norm_n(zn_flat + update_n)

        # Step 2: incident queries from normal_updated
        update_i = self.cross_i(zi_flat, zn_u_flat, big_edges)
        zi_u_flat = self.norm_i(zi_flat + update_i)

        return zn_u_flat.reshape(B, N, d), zi_u_flat.reshape(B, N, d)
