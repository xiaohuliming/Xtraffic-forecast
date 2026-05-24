#!/usr/bin/env python3
"""Build region-sampled sparse graph per DIST-Net design doc §4.

Per region (Alameda / Contra Costa / Orange):
  Step 1 - physical k-NN base layer (k=8 nearest road-network neighbors).
           BIDIRECTIONAL — physical neighbors exchange info both ways.
  Step 2 - STTIS-style shortcut:
           • ⌊√N⌋ core nodes fully connected to each other (BIDIRECTIONAL)
           • Every leaf links to its 3 nearest cores. DIRECTED: hub → leaf
             only by default (--directed-shortcuts). Semantic: the hub
             broadcasts regional context to leaves; it does NOT aggregate
             from 717 distant leaves (over-smoothing defense).
  Step 3 - assemble directed edge_index, verify n-hop reachability.

Output: outputs/region_graphs/{region}_sparse_adj.npz with
  edge_index   (2, E_total)  directed COO: edge_index[0]=source, [1]=target
  region_idx   (N,)          index into the global 16972-node space
  core_idx     (⌊√N⌋,)       positions of core nodes inside the region
  N, county, edge counts, in/out-degree stats, n-hop unreachable counts
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = {
    "alameda":      "Alameda",
    "contra_costa": "Contra Costa",
    "orange":       "Orange",
}


def load_region_meta(meta_path: Path, node_order_path: Path, county: str):
    """Match scripts/export_to_igstgnn.py:load_region_meta exactly so that
    region_idx ordering is identical to IGSTGNN's per-region adjacency."""
    node_order = np.load(node_order_path)
    meta = pd.read_csv(meta_path, sep="\t")
    meta_by_id = meta.set_index("station_id")
    aligned = meta_by_id.loc[node_order].reset_index(drop=False).rename(
        columns={"index": "station_id"}
    )
    is_region = (aligned["County"] == county) & (aligned["Type"] == "Mainline")
    region_idx = np.where(is_region.values)[0]
    region_meta = aligned.iloc[region_idx].reset_index(drop=True)
    return region_idx, region_meta


def load_dis_region(dis_matrix_path: Path, region_idx: np.ndarray) -> np.ndarray:
    """Slice the 16972x16972 global dis matrix to (N,N), convert m->km,
    mark invalid (-1) and self-loops as inf."""
    dis_full = np.load(dis_matrix_path, mmap_mode="r")
    dis_region = np.array(
        dis_full[region_idx][:, region_idx], dtype=np.float32
    )
    dis_region = np.where(dis_region < 0, np.inf, dis_region / 1000.0)
    np.fill_diagonal(dis_region, np.inf)
    return dis_region


def build_knn_base(dis_region: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Per-node k-nearest neighbors by road distance. Symmetrized via set
    of (min, max) pairs."""
    N = dis_region.shape[0]
    edges: set[tuple[int, int]] = set()
    for i in range(N):
        row = dis_region[i]
        finite_idx = np.where(np.isfinite(row))[0]
        if finite_idx.size == 0:
            continue
        order = finite_idx[np.argsort(row[finite_idx])][:k]
        for j in order:
            a, b = int(i), int(j)
            if a == b:
                continue
            edges.add((min(a, b), max(a, b)))
    return edges


def select_cores(base_edges: set[tuple[int, int]], N: int, n_cores: int) -> np.ndarray:
    """Select the n_cores nodes with highest degree in the base k-NN graph.
    Tie-break by lower node index for reproducibility."""
    degree = np.zeros(N, dtype=np.int32)
    for a, b in base_edges:
        degree[a] += 1
        degree[b] += 1
    # argsort: ascending; reverse via -degree so ties break by smaller idx
    order = np.lexsort((np.arange(N), -degree))
    return np.sort(order[:n_cores])


def build_core_clique_pairs(core_idx: np.ndarray) -> list[tuple[int, int]]:
    """All-pairs of core nodes, returned as undirected pairs (a < b)."""
    pairs: list[tuple[int, int]] = []
    n_cores = len(core_idx)
    for i in range(n_cores):
        for j in range(i + 1, n_cores):
            a, b = int(core_idx[i]), int(core_idx[j])
            pairs.append((min(a, b), max(a, b)))
    return pairs


def build_hub_leaf_assignments(dis_region: np.ndarray, core_idx: np.ndarray,
                               leaf_to_n_cores: int) -> list[tuple[int, int]]:
    """Each leaf is assigned to its leaf_to_n_cores nearest cores.

    Returned as (hub, leaf) tuples — the convention used downstream when
    edges are directed: hub broadcasts to leaf."""
    N = dis_region.shape[0]
    is_core = np.zeros(N, dtype=bool)
    is_core[core_idx] = True
    out: list[tuple[int, int]] = []
    for leaf in range(N):
        if is_core[leaf]:
            continue
        dist_to_cores = dis_region[leaf, core_idx]
        if not np.any(np.isfinite(dist_to_cores)):
            chosen = core_idx[:leaf_to_n_cores]
        else:
            order = np.argsort(dist_to_cores)[:leaf_to_n_cores]
            chosen = core_idx[order]
        for c in chosen:
            out.append((int(c), int(leaf)))
    return out


def assemble_edge_index(base_pairs: set[tuple[int, int]],
                        core_clique_pairs: list[tuple[int, int]],
                        hub_to_leaf: list[tuple[int, int]],
                        directed_shortcuts: bool) -> np.ndarray:
    """Combine the three edge groups into a directed edge_index.

    base_pairs      : undirected — emitted as both (a,b) and (b,a)
    core_clique_pairs: undirected — emitted as both directions
    hub_to_leaf     : (hub, leaf) tuples.
                      if directed_shortcuts=True (default), emitted only as
                      (hub → leaf). Otherwise, both directions (legacy behavior).
    """
    edge_set: set[tuple[int, int]] = set()
    for a, b in base_pairs:
        edge_set.add((a, b))
        edge_set.add((b, a))
    for a, b in core_clique_pairs:
        edge_set.add((a, b))
        edge_set.add((b, a))
    for h, l in hub_to_leaf:
        edge_set.add((h, l))
        if not directed_shortcuts:
            edge_set.add((l, h))
    if not edge_set:
        return np.empty((2, 0), dtype=np.int64)
    src, dst = zip(*sorted(edge_set))
    return np.array([list(src), list(dst)], dtype=np.int64)


def degree_stats(edge_index: np.ndarray, N: int) -> dict:
    out_deg = np.bincount(edge_index[0], minlength=N)
    in_deg  = np.bincount(edge_index[1], minlength=N)
    return {
        "in_mean":     float(in_deg.mean()),
        "in_median":   int(np.median(in_deg)),
        "in_min":      int(in_deg.min()),
        "in_max":      int(in_deg.max()),
        "in_p75":      int(np.percentile(in_deg, 75)),
        "out_mean":    float(out_deg.mean()),
        "out_median":  int(np.median(out_deg)),
        "out_min":     int(out_deg.min()),
        "out_max":     int(out_deg.max()),
        "out_p75":     int(np.percentile(out_deg, 75)),
        "isolated_in": int((in_deg == 0).sum()),
        "isolated_out": int((out_deg == 0).sum()),
    }


def verify_n_hop_reachability(edge_index: np.ndarray, N: int, n_hops: int) -> int:
    """Returns number of ordered (src, tgt) pairs NOT reachable from src to tgt
    in ≤ n_hops via the DIRECTED edges. Self-pairs are always reachable."""
    adj = np.zeros((N, N), dtype=bool)
    adj[edge_index[0], edge_index[1]] = True   # adj[src, tgt] = True
    np.fill_diagonal(adj, True)
    reach = adj.copy()
    cur = adj.copy()
    for _ in range(n_hops - 1):
        cur = cur @ adj
        reach = reach | cur
    return int((N * N) - int(reach.sum()))


def top_in_degree_nodes(edge_index: np.ndarray, N: int, k: int = 5) -> list[tuple[int, int]]:
    """Top-k nodes by IN-degree (how many sources broadcast to them).
    These are the worst over-smoothing candidates."""
    in_deg = np.bincount(edge_index[1], minlength=N)
    order = np.argsort(in_deg)[::-1][:k]
    return [(int(i), int(in_deg[i])) for i in order]


def top_out_degree_nodes(edge_index: np.ndarray, N: int, k: int = 5) -> list[tuple[int, int]]:
    """Top-k nodes by OUT-degree (how many targets they broadcast to).
    These are the hubs broadcasting to many leaves."""
    out_deg = np.bincount(edge_index[0], minlength=N)
    order = np.argsort(out_deg)[::-1][:k]
    return [(int(i), int(out_deg[i])) for i in order]


def build_for_region(region_key: str, args: argparse.Namespace) -> None:
    county = REGIONS[region_key]
    print(f"\n=== {region_key.upper()} ({county}) ===", flush=True)

    region_idx, region_meta = load_region_meta(args.meta, args.node_order, county)
    N = len(region_idx)
    print(f"  nodes              : {N}")
    print(f"  region_idx range   : [{int(region_idx.min())}, {int(region_idx.max())}]")

    print(f"  loading dis_matrix slice ({N}x{N}) ...", flush=True)
    dis_region = load_dis_region(args.dis, region_idx)
    finite_frac = float(np.isfinite(dis_region).sum()) / (N * N)
    print(f"    finite-distance fraction: {finite_frac:.3%}")
    if finite_frac < 0.05:
        print("    WARNING: very sparse connectivity; check dis_matrix unit/region selection")

    print(f"  Step 1: physical k-NN (k={args.k_nn}, bidirectional) ...", flush=True)
    base_pairs = build_knn_base(dis_region, k=args.k_nn)
    print(f"    base undirected pairs: {len(base_pairs)}")

    n_cores = int(np.floor(np.sqrt(N)))
    mode_str = "DIRECTED hub→leaf" if args.directed_shortcuts else "undirected"
    print(f"  Step 2: STTIS shortcut (n_cores={n_cores}, leaf_to_n_cores={args.leaf_to_n_cores}, "
          f"mode={mode_str}) ...", flush=True)
    core_idx = select_cores(base_pairs, N, n_cores)
    core_clique_pairs = build_core_clique_pairs(core_idx)
    hub_leaf = build_hub_leaf_assignments(dis_region, core_idx, args.leaf_to_n_cores)
    print(f"    core-core undirected pairs: {len(core_clique_pairs)}")
    print(f"    hub→leaf assignments     : {len(hub_leaf)}")

    edge_index = assemble_edge_index(
        base_pairs=base_pairs,
        core_clique_pairs=core_clique_pairs,
        hub_to_leaf=hub_leaf,
        directed_shortcuts=args.directed_shortcuts,
    )
    print(f"  Step 3: assembled edge_index (directed): {edge_index.shape}")

    deg = degree_stats(edge_index, N)
    print(f"  IN-degree : mean={deg['in_mean']:.1f}  med={deg['in_median']}  "
          f"min={deg['in_min']}  max={deg['in_max']}  p75={deg['in_p75']}  "
          f"isolated_in={deg['isolated_in']}")
    print(f"  OUT-degree: mean={deg['out_mean']:.1f}  med={deg['out_median']}  "
          f"min={deg['out_min']}  max={deg['out_max']}  p75={deg['out_p75']}  "
          f"isolated_out={deg['isolated_out']}")

    print(f"  verifying multi-hop reachability ...", flush=True)
    n_unreach_2 = verify_n_hop_reachability(edge_index, N, n_hops=2)
    pct_2 = n_unreach_2 / float(N * N)
    n_unreach_3 = verify_n_hop_reachability(edge_index, N, n_hops=3)
    pct_3 = n_unreach_3 / float(N * N)
    n_unreach_4 = verify_n_hop_reachability(edge_index, N, n_hops=4)
    pct_4 = n_unreach_4 / float(N * N)
    print(f"    unreachable pairs (≤2 hops): {n_unreach_2} ({pct_2:.3%})")
    print(f"    unreachable pairs (≤3 hops): {n_unreach_3} ({pct_3:.3%})")
    print(f"    unreachable pairs (≤4 hops): {n_unreach_4} ({pct_4:.3%})")

    top_in = top_in_degree_nodes(edge_index, N, k=5)
    print(f"  top-5 IN-degree (highest receivers — over-smoothing candidates):")
    for idx, d in top_in:
        is_core_str = " [CORE]" if idx in set(int(c) for c in core_idx) else ""
        print(f"    node {idx}: in-degree {d}{is_core_str}")
    top_out = top_out_degree_nodes(edge_index, N, k=5)
    print(f"  top-5 OUT-degree (highest broadcasters — hubs):")
    for idx, d in top_out:
        is_core_str = " [CORE]" if idx in set(int(c) for c in core_idx) else ""
        print(f"    node {idx}: out-degree {d}{is_core_str}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{region_key}_sparse_adj.npz"
    np.savez_compressed(
        out_path,
        edge_index=edge_index,
        region_idx=region_idx.astype(np.int64),
        core_idx=core_idx.astype(np.int64),
        N=np.int32(N),
        county=county,
        n_base_pairs=np.int32(len(base_pairs)),
        n_core_clique_pairs=np.int32(len(core_clique_pairs)),
        n_hub_leaf_assignments=np.int32(len(hub_leaf)),
        n_edges_directed=np.int32(edge_index.shape[1]),
        in_degree_mean=np.float32(deg["in_mean"]),
        in_degree_max=np.int32(deg["in_max"]),
        in_degree_median=np.int32(deg["in_median"]),
        out_degree_mean=np.float32(deg["out_mean"]),
        out_degree_max=np.int32(deg["out_max"]),
        out_degree_median=np.int32(deg["out_median"]),
        n_isolated_in=np.int32(deg["isolated_in"]),
        n_isolated_out=np.int32(deg["isolated_out"]),
        n_unreachable_2hop=np.int32(n_unreach_2),
        n_unreachable_3hop=np.int32(n_unreach_3),
        n_unreachable_4hop=np.int32(n_unreach_4),
        directed_shortcuts=np.bool_(args.directed_shortcuts),
        k_nn=np.int32(args.k_nn),
        leaf_to_n_cores=np.int32(args.leaf_to_n_cores),
        finite_distance_fraction=np.float32(finite_frac),
    )
    print(f"  saved -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("archive"))
    p.add_argument("--meta", type=Path, default=None,
                   help="Defaults to <archive>/sensor_meta_feature.csv")
    p.add_argument("--node-order", type=Path, default=None,
                   help="Defaults to <archive>/node_order.npy")
    p.add_argument("--dis", type=Path, default=None,
                   help="Defaults to <archive>/dis_matrix.npy")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--k-nn", type=int, default=8)
    p.add_argument("--leaf-to-n-cores", type=int, default=3)
    p.add_argument("--directed-shortcuts", action="store_true", default=True,
                   help="Hub→leaf shortcuts are directed only (no leaf→hub). "
                        "Default ON for the over-smoothing defense (v2 design).")
    p.add_argument("--undirected-shortcuts", dest="directed_shortcuts",
                   action="store_false",
                   help="Legacy: shortcut edges are bidirectional (v1 behavior).")
    p.add_argument("--regions", default="alameda,contra_costa,orange",
                   help="Comma-separated region keys to build")
    args = p.parse_args()

    if args.meta is None:
        args.meta = args.archive / "sensor_meta_feature.csv"
    if args.node_order is None:
        args.node_order = args.archive / "node_order.npy"
    if args.dis is None:
        args.dis = args.archive / "dis_matrix.npy"

    for region_key in args.regions.split(","):
        region_key = region_key.strip()
        if region_key not in REGIONS:
            raise ValueError(f"Unknown region '{region_key}'; choices: {list(REGIONS)}")
        build_for_region(region_key, args)


if __name__ == "__main__":
    main()
