#!/usr/bin/env python3
"""Diagnose over-smoothing in DIST-Net's spatial encoder.

For each region, we measure the *mean pairwise cosine similarity* between
node representations at four checkpoints:

  e_input      — multi-scale patching output (before any GAT layer)
  encoder_L1   — after layer 1 of spatial encoder
  encoder_L2   — after layer 2 (final z_normal)
  cross_attn   — after bidirectional cross-attention

We split node pairs into three groups to localize where smoothing happens:
  • core ↔ core   — cores are the high-degree hubs (32 of them in Orange)
  • leaf ↔ leaf
  • hub  ↔ random — the top-1 super-hub (degree 717 in Orange) vs random leaves

Healthy ranges (rule of thumb):
  cos < 0.6   ok
  0.6–0.85   watch
  > 0.85     pathological (over-smoothed)

We run this on FRESHLY-INITIALIZED model (no training). At init, attention
weights are nearly uniform → if there's a structural over-smoothing tendency,
it shows up immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from dist_net.data import MultiRegionDataset, make_loader, REGION_NAME_TO_CODE
from dist_net.model import DISTNet, DISTNetConfig
from dist_net.sparse_attn import batched_edge_index


@torch.no_grad()
def cosine_pair_stats(z: torch.Tensor, idx_a: np.ndarray, idx_b: np.ndarray
                      ) -> tuple[float, float]:
    """z: (N, d). idx_a, idx_b: 1-D arrays of equal length.
    Returns (mean, std) of cos sim across the requested pairs."""
    zn = F.normalize(z, dim=-1)
    cos = (zn[idx_a] * zn[idx_b]).sum(dim=-1)
    return float(cos.mean()), float(cos.std())


def sample_disjoint_pairs(pool: np.ndarray, n_pairs: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.choice(pool, size=n_pairs, replace=True)
    b = rng.choice(pool, size=n_pairs, replace=True)
    mask = a != b
    return a[mask], b[mask]


def probe_region(region_name: str, model: DISTNet, ds, batch_size: int = 8,
                 n_pairs: int = 2000, seed: int = 0) -> None:
    rdata = ds.regions[region_name]
    region_code = REGION_NAME_TO_CODE[region_name]
    N = rdata.N
    core_idx_np = np.array(rdata.core_idx, dtype=np.int64)
    leaf_idx_np = np.setdiff1d(np.arange(N), core_idx_np)
    # super-hub = the highest-degree node
    edge_index = torch.from_numpy(rdata.edge_index)
    degree = torch.bincount(edge_index[0], minlength=N).numpy()
    hub = int(np.argmax(degree))
    is_hub_core = hub in core_idx_np

    print(f"\n=== {region_name}  N={N}  cores={len(core_idx_np)}  "
          f"super-hub=node{hub} (deg={int(degree[hub])}{', core' if is_hub_core else ''}) ===",
          flush=True)

    # Pick a fixed batch_size of training samples from this region
    gis = [gi for gi, (rn, _) in enumerate(ds.index_table) if rn == region_name]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(gis, size=batch_size, replace=False)
    batch_items = [ds[int(g)] for g in chosen]
    from dist_net.data import collate
    batch = collate(batch_items)

    # Capture intermediate activations via hooks
    captured: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def fn(_module, _inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            captured[name] = t.detach()
        return fn

    # Hooks on normal branch encoder
    hooks = []
    hooks.append(model.normal.encoder_layers[0].register_forward_hook(hook("normal_L1")))
    hooks.append(model.normal.encoder_layers[1].register_forward_hook(hook("normal_L2")))
    hooks.append(model.incident.encoder_layers[0].register_forward_hook(hook("incident_L1")))
    hooks.append(model.incident.encoder_layers[1].register_forward_hook(hook("incident_L2")))
    hooks.append(model.cross_attn.register_forward_hook(hook("cross_attn")))

    edge_t = torch.from_numpy(rdata.edge_index)
    _ = model(batch, edge_index=edge_t)

    for h in hooks:
        h.remove()

    # Reshape captured to (B, N, d) (some are flat (B*N, d))
    def to_BN(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3:
            return t
        if t.dim() == 2:
            return t.reshape(batch_size, N, t.size(-1))
        raise ValueError(f"unexpected dim: {t.shape}")

    # Cross-attn returns a tuple, captured["cross_attn"] is the first output (z_normal_u)
    stages = {
        "normal_L1": to_BN(captured["normal_L1"]),
        "normal_L2": to_BN(captured["normal_L2"]),
        "incident_L1": to_BN(captured["incident_L1"]),
        "incident_L2": to_BN(captured["incident_L2"]),
        "cross_normal_updated": to_BN(captured["cross_attn"]),
    }

    # Build pair samples once
    cc_a, cc_b = sample_disjoint_pairs(core_idx_np, n_pairs, seed=seed)
    ll_a, ll_b = sample_disjoint_pairs(leaf_idx_np, n_pairs, seed=seed + 1)
    hr_a = np.full(n_pairs, hub, dtype=np.int64)
    hr_b = np.random.default_rng(seed + 2).choice(leaf_idx_np, size=n_pairs)

    header = f"{'stage':<22} {'core↔core':>20} {'leaf↔leaf':>20} {'hub→leaves':>20}"
    print(header, flush=True)
    print("-" * len(header))
    for stage, z in stages.items():
        z0 = z[0]  # first batch sample
        cc = cosine_pair_stats(z0, cc_a, cc_b)
        ll = cosine_pair_stats(z0, ll_a, ll_b)
        hr = cosine_pair_stats(z0, hr_a, hr_b)
        print(f"{stage:<22} {cc[0]:8.3f}±{cc[1]:.3f}  "
              f"{ll[0]:8.3f}±{ll[1]:.3f}  "
              f"{hr[0]:8.3f}±{hr[1]:.3f}",
              flush=True)


def main() -> None:
    torch.manual_seed(0)
    ds = MultiRegionDataset(
        region_names=["Alameda", "ContraCosta", "Orange"],
        data_dir="outputs/dist_net/region_data",
        graph_dir="outputs/region_graphs",
        split="train", lazy=False,
    )
    cfg = DISTNetConfig(
        c_x=3, c_meta=ds.regions["Alameda"].C_meta, c_e=13, d_t=5,
        n_regions=3, hidden_dim=64, t_h=12, t_p=12,
    )
    model = DISTNet(cfg).eval()
    print("Probing fresh (untrained) DISTNet for over-smoothing patterns.\n"
          "Healthy: cos<0.6 | Watch: 0.6-0.85 | Bad: >0.85", flush=True)

    for r in ["Alameda", "ContraCosta", "Orange"]:
        probe_region(r, model, ds, batch_size=8, n_pairs=2000, seed=0)


if __name__ == "__main__":
    main()
