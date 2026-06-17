"""DIST-Net data loading utilities.

Loads cache produced by scripts/build_full_county_cache.py + region graph
produced by scripts/build_region_sampling_graph.py.

Per Q4 decision: joint training across 3 regions with region-bucketed batches
(every batch contains samples from a single region so N stays constant).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

REGION_NAME_TO_KEY = {
    "Alameda": "alameda",
    "ContraCosta": "contra_costa",
    "Orange": "orange",
}
REGION_NAME_TO_CODE = {"Alameda": 0, "ContraCosta": 1, "Orange": 2}
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


class RegionData:
    """Holds all per-region tensors in memory.

    Memory budget per region:
      flow_series_imputed (T,N,3) float32 ≈ 4 * 105120 * N * 3 bytes
        Alameda (N=521): 657 MB
        ContraCosta (N=496): 626 MB
        Orange (N=990): 1.25 GB
      flow_mask same shape bool: 1/4 of above
      Total across 3 regions: ~3.2 GB (manageable on 16 GB+ machines)

    For memory-tight runs, pass lazy=True (lazy h5 access, slower but ~0 RAM).
    """

    def __init__(self, region_name: str, data_dir: str | Path,
                 graph_dir: str | Path, lazy: bool = False):
        self.region_name = region_name
        self.region_code = REGION_NAME_TO_CODE[region_name]
        self.lazy = lazy

        # ---- samples file (always loaded fully — small) ----
        samples_path = Path(data_dir) / f"{region_name}_samples.h5"
        with h5py.File(samples_path, "r") as f:
            self.sample_start = f["sample_start"][...]
            self.split = f["split"][...]
            self.event_idx = f["event_idx"][...]
            self.anchor_region_idx = f["anchor_region_idx"][...]
            self.sample_offset = f["sample_offset"][...]
            self.incident_feat = f["incident_feat"][...]
            self.incident_mask = f["incident_mask"][...]
            self.affected_mask = f["affected_mask"][...]
            self.n_active_incidents = f["n_active_incidents"][...]
            # active_event_idx is new (v0.3 cache). Optional for back-compat.
            if "active_event_idx" in f:
                self.active_event_idx = f["active_event_idx"][...]
            else:
                self.active_event_idx = None
            self.M_max = int(f.attrs["M_max"])
            self.C_e = int(f.attrs["C_e"])
            self.N = int(f.attrs["N"])
            self.T_h = int(f.attrs["T_h"])
            self.T_p = int(f.attrs["T_p"])

        # ---- event rel_feat (n_events, N, 4) — incident↔sensor geometry ----
        rel_feat_path = Path(data_dir) / f"{region_name}_event_rel_feat.npz"
        if rel_feat_path.exists() and self.active_event_idx is not None:
            rel = np.load(rel_feat_path)
            self.event_rel_feat = rel["rel_feat"].astype(np.float32)   # (n_events, N, 4)
            self.has_rel_feat = True
        else:
            self.event_rel_feat = None
            self.has_rel_feat = False

        # ---- traffic file ----
        self.traffic_path = Path(data_dir) / f"{region_name}_traffic.h5"
        with h5py.File(self.traffic_path, "r") as f:
            # small stuff: load eagerly always
            self.time_enc = f["time_enc"][...]          # (T, 5) float32
            self.static_meta = f["static_meta"][...]    # (N, C_meta) float32
            self.baseline_median = f["baseline_median"][...]  # (2, 288, N, 3)
            self.baseline_scale = f["baseline_scale"][...]    # (2, 288, N, 3)
            self.region_idx = f["region_idx"][...]      # (N,) int64
            self.times_ns = f["times_ns"][...]          # (T,) int64
            self.T = int(f.attrs["T"])
            self.C_meta = int(self.static_meta.shape[1])

            # large: load eager unless lazy
            if not lazy:
                self.flow_series = f["flow_series_imputed"][...]  # (T, N, 3)
                self.flow_mask = f["flow_mask"][...]              # (T, N, 3)
            else:
                self.flow_series = None
                self.flow_mask = None

        # Pre-compute day_kind / tod arrays for fast baseline lookup
        times = pd.to_datetime(self.times_ns)
        self.day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
        self.tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
        # Day-of-week 0..6 (raw), for full week-cycle time encoding (used by FDN++ Main branch)
        self.dow = times.dayofweek.to_numpy().astype(np.int8)

        # ---- region graph ----
        graph_key = REGION_NAME_TO_KEY[region_name]
        graph = np.load(Path(graph_dir) / f"{graph_key}_sparse_adj.npz")
        self.edge_index = graph["edge_index"].astype(np.int64)  # (2, E)
        self.core_idx = graph["core_idx"].astype(np.int64)
        if not np.array_equal(self.region_idx, graph["region_idx"]):
            raise RuntimeError(
                f"region_idx mismatch: {region_name} traffic.h5 vs sparse_adj.npz"
            )

        self._lazy_traffic_f: h5py.File | None = None

    def _traffic_handle(self) -> h5py.File:
        if self._lazy_traffic_f is None:
            self._lazy_traffic_f = h5py.File(self.traffic_path, "r")
        return self._lazy_traffic_f

    def _slice_traffic(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        if self.flow_series is not None:
            return self.flow_series[start:stop], self.flow_mask[start:stop]
        f = self._traffic_handle()
        return f["flow_series_imputed"][start:stop], f["flow_mask"][start:stop]

    def get_sample(self, idx: int) -> dict[str, np.ndarray]:
        """Return one sample as numpy arrays. Stacking is done by collate()."""
        t = int(self.sample_start[idx])
        T_h = self.T_h
        T_p = self.T_p

        hist_lo, hist_hi = t - T_h + 1, t + 1            # [t-T_h+1, t]
        fut_lo, fut_hi = t + 1, t + 1 + T_p              # [t+1, t+T_p]

        x_hist_t, x_hist_mask_t = self._slice_traffic(hist_lo, hist_hi)  # (T_h, N, 3)
        y_true_t,  y_mask_t      = self._slice_traffic(fut_lo, fut_hi)   # (T_p, N, 3)

        # Reorder time<->node so node is first axis (matches model expectations)
        x_hist      = np.transpose(x_hist_t,      (1, 0, 2)).astype(np.float32)
        x_hist_mask = np.transpose(x_hist_mask_t, (1, 0, 2)).astype(np.bool_)
        y_true      = np.transpose(y_true_t,      (1, 0, 2)).astype(np.float32)
        y_mask      = np.transpose(y_mask_t,      (1, 0, 2)).astype(np.bool_)

        # y_baseline lookup via (day_kind, tod) for each future step
        fut_idx = np.arange(fut_lo, fut_hi)
        dk = self.day_kind[fut_idx]
        td = self.tod[fut_idx]
        y_baseline_t = self.baseline_median[dk, td]      # (T_p, N, 3)
        y_baseline = np.transpose(y_baseline_t, (1, 0, 2)).astype(np.float32)

        # x_baseline lookup via (day_kind, tod) for each history step (mirror y_baseline)
        hist_idx = np.arange(hist_lo, hist_hi)
        x_baseline_t = self.baseline_median[self.day_kind[hist_idx], self.tod[hist_idx]]  # (T_h,N,3)
        x_baseline = np.transpose(x_baseline_t, (1, 0, 2)).astype(np.float32)             # (N,T_h,3)

        time_enc = self.time_enc[hist_lo:hist_hi].astype(np.float32)  # (T_h, 5)

        # ToD / DoW normalized to [0,1] — used by FDN++ Main branch as time embedding input
        tod_window = self.tod[hist_lo:hist_hi].astype(np.float32) / 288.0  # (T_h,)
        dow_window = self.dow[hist_lo:hist_hi].astype(np.float32) / 7.0    # (T_h,)
        time_feat = np.stack([tod_window, dow_window], axis=-1)            # (T_h, 2)

        # rel_feat: (M_max, N, 4) gathered from event_rel_feat by active_event_idx.
        # Padded events (active_event_idx == -1) get zeros.
        if self.has_rel_feat:
            active = self.active_event_idx[idx]                           # (M_max,) int32
            rel_feat = np.zeros((self.M_max, self.N, 4), dtype=np.float32)
            valid = active >= 0
            if valid.any():
                rel_feat[valid] = self.event_rel_feat[active[valid]]      # (k, N, 4)
        else:
            rel_feat = np.zeros((self.M_max, self.N, 4), dtype=np.float32)

        return {
            "x_hist":         x_hist,                            # (N, T_h, 3)
            "x_hist_mask":    x_hist_mask,                       # (N, T_h, 3) bool
            "y_true":         y_true,                            # (N, T_p, 3)
            "y_mask":         y_mask,                            # (N, T_p, 3) bool
            "y_baseline":     y_baseline,                        # (N, T_p, 3)
            "x_baseline":     x_baseline,                        # (N, T_h, 3)
            "time_enc":       time_enc,                          # (T_h, 5)
            "time_feat":      time_feat.astype(np.float32),     # (T_h, 2) tod/288, dow/7
            "static_meta":    self.static_meta.astype(np.float32),  # (N, C_meta)
            "region_code":    np.int64(self.region_code),
            "incident_feat":  self.incident_feat[idx].astype(np.float32),  # (M_max, C_e)
            "incident_mask":  self.incident_mask[idx].astype(np.bool_),    # (M_max,)
            "affected_mask":  self.affected_mask[idx].astype(np.bool_),    # (N,)
            "rel_feat":       rel_feat,                           # (M_max, N, 4)
            "n_active_incidents": np.int32(self.n_active_incidents[idx]),
            "sample_start":   np.int64(t),
            "sample_idx":     np.int64(idx),
        }


class MultiRegionDataset(Dataset):
    """Combined dataset across regions with a split filter.

    IMPORTANT: assumes paired with RegionBucketedSampler. __getitem__ across
    regions returns tensors with different N — they cannot be stacked into one
    batch without bucketing.
    """

    def __init__(self, region_names: list[str], data_dir: str | Path,
                 graph_dir: str | Path, split: str = "train", lazy: bool = False):
        self.split = split
        split_code = SPLIT_TO_CODE[split]
        self.regions: dict[str, RegionData] = {}
        self.index_table: list[tuple[str, int]] = []  # global_idx -> (region_name, local_idx)

        for region_name in region_names:
            rdata = RegionData(region_name, data_dir, graph_dir, lazy=lazy)
            self.regions[region_name] = rdata
            local_indices = np.where(rdata.split == split_code)[0]
            for li in local_indices:
                self.index_table.append((region_name, int(li)))

    def __len__(self) -> int:
        return len(self.index_table)

    def __getitem__(self, global_idx: int) -> dict[str, np.ndarray]:
        region_name, local_idx = self.index_table[global_idx]
        return self.regions[region_name].get_sample(local_idx)

    def region_of(self, global_idx: int) -> str:
        return self.index_table[global_idx][0]


class RegionBucketedSampler(Sampler[list[int]]):
    """Batch sampler: every batch contains indices from a single region.

    Shuffle order across regions + within region (seed-deterministic if seeded).
    """

    def __init__(self, dataset: MultiRegionDataset, batch_size: int,
                 shuffle: bool = True, seed: int | None = None,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.bucket: dict[str, np.ndarray] = {}
        for gi, (region_name, _li) in enumerate(dataset.index_table):
            self.bucket.setdefault(region_name, []).append(gi)
        self.bucket = {k: np.asarray(v, dtype=np.int64) for k, v in self.bucket.items()}

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        batches: list[list[int]] = []
        for region_name, gis in self.bucket.items():
            order = gis.copy()
            if self.shuffle:
                rng.shuffle(order)
            for s in range(0, len(order), self.batch_size):
                chunk = order[s:s + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        total = 0
        for region_name, gis in self.bucket.items():
            n = len(gis)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack same-region samples (already region-bucketed)."""
    out: dict[str, torch.Tensor] = {}
    for k in batch[0]:
        items = [b[k] for b in batch]
        if isinstance(items[0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(items, axis=0))
        else:
            # 0-d numpy scalars
            out[k] = torch.from_numpy(np.array(items))
    return out


def make_loader(dataset: MultiRegionDataset, batch_size: int = 32,
                shuffle: bool = True, seed: int | None = None,
                num_workers: int = 0, drop_last: bool = False) -> torch.utils.data.DataLoader:
    """Convenience builder. num_workers=0 by default to avoid h5py fork issues."""
    sampler = RegionBucketedSampler(
        dataset, batch_size=batch_size, shuffle=shuffle, seed=seed, drop_last=drop_last,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate,
        num_workers=num_workers, pin_memory=False,
    )
