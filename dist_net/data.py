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
CAUSAL_IMPUTATION_METHOD = "causal_forward_fill"


def _decode_h5_attr(value):
    """Return a JSON-friendly scalar for the cache metadata we consume."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


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
                 graph_dir: str | Path, lazy: bool = False,
                 strict_causal_cache: bool = False,
                 expected_baseline_train_frac: float = 0.70,
                 expected_t_h: int | None = 12,
                 expected_t_p: int | None = 12,
                 expected_split_train_end: float | None = 0.70,
                 expected_split_val_end: float | None = 0.85,
                 expected_sample_offsets: tuple[int, ...] | None = (0, 6, 12),
                 expected_active_window: int | None = 24,
                 expected_m_max: int | None = 32,
                 expected_smoke_events: int | None = 0,
                 expected_candidate_pm_radius: float | None = 5.0,
                 expected_anchor_pm_radius: float | None = 2.0,
                 expected_baseline_mask_extra_steps: int | None = 12,
                 expected_min_baseline_count: int | None = 8):
        """Load one region's event cache and graph.

        ``strict_causal_cache=True`` is intended for formal NDR runs.  It
        fails closed unless traffic and samples are a matched, causal cache
        pair with the requested 70/15/15-style protocol.  The default remains
        non-strict so existing legacy callers keep their loading behaviour;
        formal NDR callers must explicitly opt in.
        """
        self.region_name = region_name
        self.region_code = REGION_NAME_TO_CODE[region_name]
        self.lazy = lazy
        self.strict_causal_cache = bool(strict_causal_cache)

        # ---- samples file (always loaded fully — small) ----
        self.samples_path = Path(data_dir) / f"{region_name}_samples.h5"
        with h5py.File(self.samples_path, "r") as f:
            samples_attrs = {k: _decode_h5_attr(v) for k, v in f.attrs.items()}
            samples_shapes = {k: tuple(v.shape) for k, v in f.items()}
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
            traffic_attrs = {k: _decode_h5_attr(v) for k, v in f.attrs.items()}
            traffic_shapes = {k: tuple(v.shape) for k, v in f.items()}
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

        self.cache_provenance = {
            "strict_causal_cache": self.strict_causal_cache,
            "cache_schema_version": traffic_attrs.get("cache_schema_version"),
            "traffic_cache_id": traffic_attrs.get("traffic_cache_id"),
            "traffic_build_id": traffic_attrs.get("build_id"),
            "samples_cache_id": samples_attrs.get("cache_id"),
            "samples_build_id": samples_attrs.get("build_id"),
            "imputation_method": traffic_attrs.get("imputation_method"),
            "baseline_train_frac": traffic_attrs.get("baseline_train_frac"),
            "baseline_train_cutoff": traffic_attrs.get("baseline_train_cutoff"),
            "split_train_end": samples_attrs.get("split_train_end"),
            "split_val_end": samples_attrs.get("split_val_end"),
            "region_name": region_name,
            "N": self.N,
            "T": self.T,
            "T_h": self.T_h,
            "T_p": self.T_p,
            "traffic_path": str(self.traffic_path.resolve()),
            "samples_path": str(self.samples_path.resolve()),
            "sample_start_unique": (
                int(np.unique(self.sample_start).size) == int(self.sample_start.size)
            ),
            "n_samples": int(self.sample_start.size),
        }

        if self.strict_causal_cache:
            self._validate_strict_causal_cache(
                samples_attrs=samples_attrs,
                traffic_attrs=traffic_attrs,
                samples_shapes=samples_shapes,
                traffic_shapes=traffic_shapes,
                expected_baseline_train_frac=expected_baseline_train_frac,
                expected_t_h=expected_t_h,
                expected_t_p=expected_t_p,
                expected_split_train_end=expected_split_train_end,
                expected_split_val_end=expected_split_val_end,
                expected_sample_offsets=expected_sample_offsets,
                expected_active_window=expected_active_window,
                expected_m_max=expected_m_max,
                expected_smoke_events=expected_smoke_events,
                expected_candidate_pm_radius=expected_candidate_pm_radius,
                expected_anchor_pm_radius=expected_anchor_pm_radius,
                expected_baseline_mask_extra_steps=expected_baseline_mask_extra_steps,
                expected_min_baseline_count=expected_min_baseline_count,
            )

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

    def _validate_strict_causal_cache(
        self,
        *,
        samples_attrs: dict,
        traffic_attrs: dict,
        samples_shapes: dict[str, tuple[int, ...]],
        traffic_shapes: dict[str, tuple[int, ...]],
        expected_baseline_train_frac: float,
        expected_t_h: int | None,
        expected_t_p: int | None,
        expected_split_train_end: float | None,
        expected_split_val_end: float | None,
        expected_sample_offsets: tuple[int, ...] | None,
        expected_active_window: int | None,
        expected_m_max: int | None,
        expected_smoke_events: int | None,
        expected_candidate_pm_radius: float | None,
        expected_anchor_pm_radius: float | None,
        expected_baseline_mask_extra_steps: int | None,
        expected_min_baseline_count: int | None,
    ) -> None:
        """Fail closed when a formal run points at stale or mismatched H5s."""
        errors: list[str] = []

        def required(attrs: dict, key: str, label: str):
            if key not in attrs:
                errors.append(f"{label} missing required attr {key!r}")
                return None
            value = attrs[key]
            if isinstance(value, str) and not value.strip():
                errors.append(f"{label} attr {key!r} is empty")
            return value

        # These fields make cache provenance auditable and prevent a traffic
        # file from being silently paired with independently-created samples.
        for label, attrs in (("traffic.h5", traffic_attrs),
                             ("samples.h5", samples_attrs)):
            for key in ("cache_schema_version", "cache_id", "build_id",
                        "traffic_cache_id", "imputation_method",
                        "baseline_train_frac", "baseline_train_cutoff",
                        "region_name", "N", "T"):
                required(attrs, key, label)
        for key in ("T_h", "T_p", "split_train_end", "split_val_end",
                    "n_duplicate_sample_start_dropped", "sample_offsets",
                    "smoke_events", "traffic_builder_source_sha256",
                    "samples_builder_source_sha256"):
            required(samples_attrs, key, "samples.h5")
        for key in ("candidate_pm_radius", "anchor_pm_radius",
                    "baseline_mask_extra_steps", "min_baseline_count",
                    "builder_source_sha256"):
            required(traffic_attrs, key, "traffic.h5")

        traffic_schema = traffic_attrs.get("cache_schema_version")
        samples_schema = samples_attrs.get("cache_schema_version")
        try:
            if int(traffic_schema) != 2 or int(samples_schema) != 2:
                errors.append(
                    "formal causal caches require cache_schema_version=2 in both "
                    f"files, got traffic={traffic_schema!r}, samples={samples_schema!r}"
                )
        except (TypeError, ValueError):
            errors.append(
                "cache_schema_version must be integer 2 in both traffic and samples"
            )

        traffic_method = traffic_attrs.get("imputation_method")
        samples_method = samples_attrs.get("imputation_method")
        if traffic_method != CAUSAL_IMPUTATION_METHOD:
            errors.append(
                "traffic.h5 imputation_method must be "
                f"{CAUSAL_IMPUTATION_METHOD!r}, got {traffic_method!r}"
            )
        if samples_method != traffic_method:
            errors.append(
                "samples.h5 imputation_method does not match traffic.h5: "
                f"{samples_method!r} != {traffic_method!r}"
            )

        traffic_pair_id = traffic_attrs.get("traffic_cache_id")
        samples_pair_id = samples_attrs.get("traffic_cache_id")
        if traffic_pair_id != samples_pair_id:
            errors.append(
                "traffic/samples pairing ID mismatch: "
                f"{traffic_pair_id!r} != {samples_pair_id!r}"
            )
        traffic_cache_id = traffic_attrs.get("cache_id")
        if traffic_pair_id != traffic_cache_id:
            errors.append(
                "traffic.h5 traffic_cache_id must equal its cache_id: "
                f"{traffic_pair_id!r} != {traffic_cache_id!r}"
            )
        traffic_build_id = traffic_attrs.get("build_id")
        samples_build_id = samples_attrs.get("build_id")
        if traffic_build_id != samples_build_id:
            errors.append(
                "formal causal traffic/samples must come from one complete build: "
                f"build_id {traffic_build_id!r} != {samples_build_id!r}"
            )
        traffic_builder_hash = traffic_attrs.get("builder_source_sha256")
        samples_traffic_builder_hash = samples_attrs.get(
            "traffic_builder_source_sha256"
        )
        samples_builder_hash = samples_attrs.get("samples_builder_source_sha256")
        if samples_traffic_builder_hash != traffic_builder_hash:
            errors.append(
                "samples.h5 traffic builder hash does not match traffic.h5: "
                f"{samples_traffic_builder_hash!r} != {traffic_builder_hash!r}"
            )
        for label, value in (
            ("traffic builder hash", traffic_builder_hash),
            ("samples builder hash", samples_builder_hash),
        ):
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"{label} must be a 64-character SHA256 value")

        # Validate the actual baseline fitting boundary, not merely a config
        # file written later by a samples-only invocation.
        try:
            baseline_frac = float(traffic_attrs["baseline_train_frac"])
            samples_baseline_frac = float(samples_attrs["baseline_train_frac"])
            if not np.isclose(baseline_frac, expected_baseline_train_frac,
                              rtol=0.0, atol=1e-12):
                errors.append(
                    "traffic.h5 baseline_train_frac mismatch: "
                    f"expected {expected_baseline_train_frac}, got {baseline_frac}"
                )
            if not np.isclose(samples_baseline_frac, baseline_frac,
                              rtol=0.0, atol=1e-12):
                errors.append(
                    "samples.h5 baseline_train_frac does not match traffic.h5: "
                    f"{samples_baseline_frac} != {baseline_frac}"
                )
            expected_cutoff = int(self.T * baseline_frac)
            traffic_cutoff = int(traffic_attrs["baseline_train_cutoff"])
            samples_cutoff = int(samples_attrs["baseline_train_cutoff"])
            if traffic_cutoff != expected_cutoff:
                errors.append(
                    "traffic.h5 baseline_train_cutoff mismatch: "
                    f"expected exclusive index {expected_cutoff}, got {traffic_cutoff}"
                )
            if samples_cutoff != traffic_cutoff:
                errors.append(
                    "samples.h5 baseline_train_cutoff does not match traffic.h5: "
                    f"{samples_cutoff} != {traffic_cutoff}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid baseline metadata: {exc}")
            baseline_frac = None

        for label, attrs in (("traffic.h5", traffic_attrs),
                             ("samples.h5", samples_attrs)):
            if attrs.get("region_name") != self.region_name:
                errors.append(
                    f"{label} region_name mismatch: expected {self.region_name!r}, "
                    f"got {attrs.get('region_name')!r}"
                )
            try:
                if int(attrs["N"]) != self.N:
                    errors.append(
                        f"{label} N mismatch: attr={attrs['N']}, loaded={self.N}"
                    )
                if int(attrs["T"]) != self.T:
                    errors.append(
                        f"{label} T mismatch: attr={attrs['T']}, loaded={self.T}"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{label} has invalid N/T metadata: {exc}")

        try:
            split_train_end = float(samples_attrs["split_train_end"])
            split_val_end = float(samples_attrs["split_val_end"])
            if not 0.0 < split_train_end < split_val_end < 1.0:
                errors.append(
                    "invalid split fractions: expected 0 < train < val < 1, got "
                    f"{split_train_end}, {split_val_end}"
                )
            if baseline_frac is not None and baseline_frac > split_train_end + 1e-12:
                errors.append(
                    "baseline fitting enters validation time: "
                    f"{baseline_frac} > split_train_end {split_train_end}"
                )
            if (expected_split_train_end is not None
                    and not np.isclose(split_train_end, expected_split_train_end,
                                       rtol=0.0, atol=1e-12)):
                errors.append(
                    "split_train_end mismatch: "
                    f"expected {expected_split_train_end}, got {split_train_end}"
                )
            if (expected_split_val_end is not None
                    and not np.isclose(split_val_end, expected_split_val_end,
                                       rtol=0.0, atol=1e-12)):
                errors.append(
                    "split_val_end mismatch: "
                    f"expected {expected_split_val_end}, got {split_val_end}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid split metadata: {exc}")
            split_train_end = split_val_end = None

        if expected_t_h is not None and self.T_h != int(expected_t_h):
            errors.append(f"T_h mismatch: expected {expected_t_h}, got {self.T_h}")
        if expected_t_p is not None and self.T_p != int(expected_t_p):
            errors.append(f"T_p mismatch: expected {expected_t_p}, got {self.T_p}")
        if self.T_h <= 0 or self.T_p <= 0:
            errors.append(f"T_h and T_p must be positive, got {self.T_h}/{self.T_p}")

        def check_int(label: str, stored, expected) -> None:
            if expected is None:
                return
            try:
                if int(stored) != int(expected):
                    errors.append(f"{label} mismatch: expected {expected}, got {stored}")
            except (TypeError, ValueError):
                errors.append(f"{label} must be an integer, got {stored!r}")

        def check_float(label: str, stored, expected) -> None:
            if expected is None:
                return
            try:
                if not np.isclose(float(stored), float(expected), rtol=0.0, atol=1e-12):
                    errors.append(f"{label} mismatch: expected {expected}, got {stored}")
            except (TypeError, ValueError):
                errors.append(f"{label} must be numeric, got {stored!r}")

        if expected_sample_offsets is not None:
            try:
                stored_offsets = tuple(int(v) for v in np.asarray(
                    samples_attrs.get("sample_offsets")
                ).reshape(-1))
                if stored_offsets != tuple(expected_sample_offsets):
                    errors.append(
                        "sample_offsets mismatch: expected "
                        f"{tuple(expected_sample_offsets)}, got {stored_offsets}"
                    )
            except (TypeError, ValueError):
                errors.append("sample_offsets metadata is malformed")
        check_int("active_window", samples_attrs.get("active_window"),
                  expected_active_window)
        check_int("M_max", samples_attrs.get("M_max"), expected_m_max)
        check_int("smoke_events", samples_attrs.get("smoke_events"),
                  expected_smoke_events)
        check_float("candidate_pm_radius", traffic_attrs.get("candidate_pm_radius"),
                    expected_candidate_pm_radius)
        check_float("anchor_pm_radius", traffic_attrs.get("anchor_pm_radius"),
                    expected_anchor_pm_radius)
        check_int("baseline_mask_extra_steps",
                  traffic_attrs.get("baseline_mask_extra_steps"),
                  expected_baseline_mask_extra_steps)
        check_int("min_baseline_count", traffic_attrs.get("min_baseline_count"),
                  expected_min_baseline_count)

        # Verify dataset dimensions against the metadata without forcing lazy
        # traffic arrays into RAM.
        expected_traffic_shapes = {
            "flow_series_imputed": (self.T, self.N, 3),
            "flow_mask": (self.T, self.N, 3),
            "time_enc": (self.T, 5),
            "region_idx": (self.N,),
            "times_ns": (self.T,),
        }
        for name, expected in expected_traffic_shapes.items():
            got = traffic_shapes.get(name)
            if got != expected:
                errors.append(
                    f"traffic.h5 dataset {name!r} shape mismatch: "
                    f"expected {expected}, got {got}"
                )
        static_meta_shape = traffic_shapes.get("static_meta")
        if not static_meta_shape or static_meta_shape[0] != self.N:
            errors.append("traffic.h5 static_meta first dimension does not equal N")
        for name in ("baseline_median", "baseline_scale"):
            shape = traffic_shapes.get(name)
            expected = (2, 288, self.N, 3)
            if shape != expected:
                errors.append(
                    f"traffic.h5 dataset {name!r} must have shape {expected}, "
                    f"got {shape}"
                )

        n_samples = int(self.sample_start.size)
        if n_samples == 0:
            errors.append("samples.h5 contains no samples")
        if self.sample_start.ndim != 1:
            errors.append(f"sample_start must be 1-D, got {self.sample_start.shape}")
        unique_count = int(np.unique(self.sample_start).size)
        if unique_count != n_samples:
            errors.append(
                f"sample_start contains {n_samples - unique_count} duplicate rows "
                f"({unique_count}/{n_samples} unique)"
            )
        for name, shape in samples_shapes.items():
            if name == "sample_start" or not shape:
                continue
            if shape[0] != n_samples:
                errors.append(
                    f"samples.h5 dataset {name!r} first dimension mismatch: "
                    f"expected {n_samples}, got {shape[0]}"
                )
        affected_shape = samples_shapes.get("affected_mask")
        if affected_shape != (n_samples, self.N):
            errors.append(
                "samples.h5 affected_mask shape mismatch: "
                f"expected {(n_samples, self.N)}, got {affected_shape}"
            )

        sample_vectors_aligned = (
            self.sample_start.ndim == 1
            and self.split.ndim == 1
            and self.split.shape == self.sample_start.shape
        )
        if not sample_vectors_aligned:
            errors.append(
                "sample_start and split must be aligned 1-D vectors, got "
                f"{self.sample_start.shape} and {self.split.shape}"
            )

        if n_samples and self.sample_start.ndim == 1:
            min_start = self.T_h - 1
            max_start = self.T - self.T_p - 1
            bad_bounds = (self.sample_start < min_start) | (self.sample_start > max_start)
            if np.any(bad_bounds):
                errors.append(
                    f"{int(bad_bounds.sum())} sample_start values lack a complete "
                    "history/future window"
                )
            if sample_vectors_aligned and not np.isin(self.split, (0, 1, 2)).all():
                errors.append("split contains values outside {0, 1, 2}")

            # Recompute split membership from complete prediction windows.
            if (sample_vectors_aligned and split_train_end is not None
                    and split_val_end is not None):
                train_cutoff = int(self.T * split_train_end)
                val_cutoff = int(self.T * split_val_end)
                target_lo = self.sample_start + 1
                target_hi = self.sample_start + self.T_p
                valid_split = (
                    ((self.split == 0) & (target_hi < train_cutoff))
                    | ((self.split == 1) & (target_lo >= train_cutoff)
                       & (target_hi < val_cutoff))
                    | ((self.split == 2) & (target_lo >= val_cutoff)
                       & (target_hi < self.T))
                )
                if not np.all(valid_split):
                    errors.append(
                        f"{int((~valid_split).sum())} samples cross or disagree "
                        "with train/val/test boundaries"
                    )

        if errors:
            details = "\n  - ".join(errors)
            raise RuntimeError(
                f"Strict causal cache validation failed for {self.region_name}:"
                f"\n  - {details}\n"
                "Use a fully rebuilt causal cache pair, or explicitly set "
                "strict_causal_cache=False only for a legacy/non-formal protocol."
            )

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


class FullWindowRegionData(RegionData):
    """Standard full-sliding-window protocol (NOT event-anchored).

    Replaces the event-anchored sample_start/split from samples.h5 with an
    exhaustive sliding window over the whole series, chronologically split into
    train/val/test by fraction. Event-dependent per-sample fields are zeroed so
    the same batch schema (and label-free training scripts) work unchanged.
    The headline metric under this protocol is `all` MAE, directly comparable to
    standard benchmarks (METR-LA/PEMS) and to IGSTGNN's Table-4-style numbers.
    """

    def __init__(self, region_name: str, data_dir: str | Path,
                 graph_dir: str | Path, lazy: bool = False,
                 stride: int = 1, train_frac: float = 0.7, val_frac: float = 0.1):
        super().__init__(region_name, data_dir, graph_dir, lazy=lazy)
        T_h, T_p, T = self.T_h, self.T_p, self.T
        # anchor t needs history [t-T_h+1, t] and future [t+1, t+T_p]
        starts = np.arange(T_h - 1, T - T_p, max(1, stride), dtype=np.int64)
        n = len(starts)
        i_tr = int(n * train_frac)
        i_va = int(n * (train_frac + val_frac))
        split = np.empty(n, dtype=np.int64)
        split[:i_tr] = 0
        split[i_tr:i_va] = 1
        split[i_va:] = 2

        self.sample_start = starts
        self.split = split
        self.affected_mask = np.zeros((n, self.N), dtype=np.bool_)
        self.n_active_incidents = np.zeros(n, dtype=np.int32)
        self.incident_feat = np.zeros((n, self.M_max, self.C_e), dtype=np.float32)
        self.incident_mask = np.zeros((n, self.M_max), dtype=np.bool_)
        self.active_event_idx = np.full((n, self.M_max), -1, dtype=np.int32)
        self.has_rel_feat = False  # get_sample returns zero rel_feat


class MultiRegionDataset(Dataset):
    """Combined dataset across regions with a split filter.

    IMPORTANT: assumes paired with RegionBucketedSampler. __getitem__ across
    regions returns tensors with different N — they cannot be stacked into one
    batch without bucketing.
    """

    def __init__(self, region_names: list[str], data_dir: str | Path,
                 graph_dir: str | Path, split: str = "train", lazy: bool = False,
                 region_data_cls: type = RegionData, region_data_kwargs: dict | None = None):
        self.split = split
        split_code = SPLIT_TO_CODE[split]
        self.regions: dict[str, RegionData] = {}
        self.index_table: list[tuple[str, int]] = []  # global_idx -> (region_name, local_idx)

        for region_name in region_names:
            rdata = region_data_cls(region_name, data_dir, graph_dir, lazy=lazy,
                                    **(region_data_kwargs or {}))
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
        self.epoch = 0

        self.bucket: dict[str, np.ndarray] = {}
        for gi, (region_name, _li) in enumerate(dataset.index_table):
            self.bucket.setdefault(region_name, []).append(gi)
        self.bucket = {k: np.asarray(v, dtype=np.int64) for k, v in self.bucket.items()}

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic epoch order (mirrors distributed samplers)."""
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self.epoch = epoch

    def __iter__(self):
        current_epoch = self.epoch
        self.epoch += 1
        if self.seed is None:
            rng = np.random.default_rng()
        else:
            # Keeping seed and epoch as separate entropy components avoids
            # repeating the exact same order on every DataLoader epoch while
            # remaining reproducible across independent runs.
            seed_sequence = np.random.SeedSequence([int(self.seed), current_epoch])
            rng = np.random.default_rng(seed_sequence)
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
