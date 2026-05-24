#!/usr/bin/env python3
"""Visualize per-region sparse sampling graphs over sensor lat/lon.

Sanity check: do cores cluster at freeway interchanges / network hubs?
Reads outputs/region_graphs/{region}_sparse_adj.npz, plots to PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

REGIONS = {
    "alameda":      "Alameda",
    "contra_costa": "Contra Costa",
    "orange":       "Orange",
}


def load_region_lat_lon(meta: pd.DataFrame, node_order: np.ndarray,
                        region_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta_by_id = meta.set_index("station_id")
    aligned = meta_by_id.loc[node_order].reset_index(drop=False).rename(
        columns={"index": "station_id"}
    )
    region_meta = aligned.iloc[region_idx].reset_index(drop=True)
    return region_meta["Lat"].values.astype(float), region_meta["Lng"].values.astype(float)


def plot_single_region(ax, npz, lat: np.ndarray, lon: np.ndarray,
                       county: str, top_k_annotate: int = 5) -> None:
    edge_index = npz["edge_index"]
    core_idx   = npz["core_idx"]
    N          = int(npz["N"])

    is_core = np.zeros(N, dtype=bool)
    is_core[core_idx] = True
    degree = np.bincount(edge_index[0], minlength=N)

    src, dst = edge_index
    keep = src < dst
    src, dst = src[keep], dst[keep]
    segments = np.stack([
        np.stack([lon[src], lat[src]], axis=-1),
        np.stack([lon[dst], lat[dst]], axis=-1),
    ], axis=1)

    lc = LineCollection(segments, colors="lightgray", linewidths=0.35,
                        alpha=0.55, zorder=1)
    ax.add_collection(lc)

    ax.scatter(lon[~is_core], lat[~is_core], s=8, c="steelblue", alpha=0.65,
               zorder=2, label=f"leaves ({(~is_core).sum()})", edgecolors="none")

    core_degree = degree[is_core]
    core_size = 25 + 0.4 * core_degree
    ax.scatter(lon[is_core], lat[is_core], s=core_size, c="crimson",
               edgecolors="black", linewidths=0.5, zorder=3,
               label=f"cores ({is_core.sum()})")

    top_idx = np.argsort(degree)[::-1][:top_k_annotate]
    for rank, i in enumerate(top_idx):
        ax.annotate(
            f"#{rank + 1} d={int(degree[i])}",
            (lon[int(i)], lat[int(i)]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black",
                      alpha=0.7, linewidth=0.4),
        )

    ax.set_title(
        f"{county}\nN={N}  edges={len(segments)}  cores={len(core_idx)}  "
        f"mean_deg={float(npz['degree_mean']):.1f}  max_deg={int(npz['degree_max'])}",
        fontsize=10,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, default=Path("archive"))
    p.add_argument("--graph-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--combined", action="store_true",
                   help="Also produce a single 3-panel combined figure")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.archive / "sensor_meta_feature.csv", sep="\t")
    node_order = np.load(args.archive / "node_order.npy")

    npz_per_region: dict[str, dict] = {}
    for region_key, county in REGIONS.items():
        print(f"\n=== {region_key.upper()} ({county}) ===", flush=True)
        graph_path = args.graph_dir / f"{region_key}_sparse_adj.npz"
        npz = np.load(graph_path)
        lat, lon = load_region_lat_lon(meta, node_order, npz["region_idx"])

        fig, ax = plt.subplots(figsize=(9, 8))
        plot_single_region(ax, npz, lat, lon, county, top_k_annotate=5)
        out_path = args.out_dir / f"{region_key}_visualization.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out_path}")

        npz_per_region[region_key] = {"npz": npz, "lat": lat, "lon": lon, "county": county}

    if args.combined:
        print("\n=== combined 3-panel figure ===", flush=True)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        for ax, (region_key, payload) in zip(axes, npz_per_region.items()):
            plot_single_region(ax, payload["npz"], payload["lat"], payload["lon"],
                               payload["county"], top_k_annotate=3)
        out_path = args.out_dir / "combined_visualization.png"
        fig.suptitle("DIST-Net region sampling graphs (3 California counties)",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
