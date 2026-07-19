"""DIST-Net data loading utilities.

Loads cache produced by scripts/build_full_county_cache.py + region graph
produced by scripts/build_region_sampling_graph.py.

Per Q4 decision: joint training across 3 regions with region-bucketed batches
(every batch contains samples from a single region so N stays constant).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from dist_net.protocol import (
    CACHE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    build_full_window_index,
    classify_forecast_window,
    forecast_window_bounds,
    git_worktree_provenance,
)

REGION_NAME_TO_KEY = {
    "Alameda": "alameda",
    "ContraCosta": "contra_costa",
    "Orange": "orange",
}
REGION_NAME_TO_CODE = {"Alameda": 0, "ContraCosta": 1, "Orange": 2}
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}
TRAINING_GIT = git_worktree_provenance(
    Path(__file__).resolve().parent.parent
)


@lru_cache(maxsize=64)
def _sha256_file_cached(
    path_text: str, size: int, modified_ns: int
) -> str:
    digest = hashlib.sha256()
    path = Path(path_text)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_cached(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns
    )


def _require_h5_members(
    h5: h5py.File, names: tuple[str, ...], artifact: Path
) -> None:
    missing = [name for name in names if name not in h5]
    if missing:
        raise RuntimeError(
            f"{artifact} is missing required v3 fields: {', '.join(missing)}"
        )


def _require_attrs(
    attrs: h5py.AttributeManager,
    names: tuple[str, ...],
    artifact: Path,
) -> None:
    missing = [name for name in names if name not in attrs]
    if missing:
        raise RuntimeError(
            f"{artifact} is missing required v3 attributes: "
            + ", ".join(missing)
        )


def gather_primary_incident(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather incident features and geometry using the stored anchor-event slot."""
    incident_feature = batch["incident_feat"].to(device)
    incident_mask = batch["incident_mask"].to(device)
    relation_feature = batch["rel_feat"].to(device)
    slots = batch["primary_incident_slot"].to(device).long()
    batch_size, max_events = incident_mask.shape
    if bool(((slots < 0) | (slots >= max_events)).any()):
        raise RuntimeError("primary_incident_slot is outside the incident tensor")

    rows = torch.arange(batch_size, device=device)
    selected_is_active = incident_mask[rows, slots]
    any_active = incident_mask.any(dim=1)
    if bool((any_active & ~selected_is_active).any()):
        raise RuntimeError("primary_incident_slot does not point to an active event")

    return (
        incident_feature[rows, slots],
        relation_feature[rows, slots],
        selected_is_active.float(),
    )


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

    def __init__(
        self,
        region_name: str,
        data_dir: str | Path,
        graph_dir: str | Path,
        lazy: bool = False,
        required_protocol: str | None = PROTOCOL_VERSION,
        include_analysis_labels: bool = False,
        require_rel_feat: bool = False,
        verify_artifact_hashes: bool = True,
    ):
        self.region_name = region_name
        self.region_code = REGION_NAME_TO_CODE[region_name]
        self.lazy = lazy
        self.sampling_protocol = "event"
        self.include_analysis_labels = bool(include_analysis_labels)
        self.require_rel_feat = bool(require_rel_feat)
        self.verify_artifact_hashes = bool(verify_artifact_hashes)
        strict_v3 = required_protocol == PROTOCOL_VERSION

        # ---- samples file (always loaded fully — small) ----
        samples_path = Path(data_dir) / f"{region_name}_samples.h5"
        self.samples_path = samples_path
        with h5py.File(samples_path, "r") as f:
            self.cache_schema_version = int(
                f.attrs.get("cache_schema_version", 1)
            )
            self.protocol_version = str(
                f.attrs.get("protocol_version", "legacy")
            )
            if (
                required_protocol is not None
                and self.protocol_version != required_protocol
            ):
                raise RuntimeError(
                    f"{samples_path} uses protocol {self.protocol_version!r}; "
                    f"required {required_protocol!r}"
                )
            if (
                strict_v3
                and self.cache_schema_version != CACHE_SCHEMA_VERSION
            ):
                raise RuntimeError(
                    f"{samples_path} uses cache schema "
                    f"{self.cache_schema_version}; required "
                    f"{CACHE_SCHEMA_VERSION}"
                )
            if strict_v3:
                _require_h5_members(
                    f,
                    (
                        "sample_start",
                        "split",
                        "event_idx",
                        "event_start_idx",
                        "event_id",
                        "anchor_region_idx",
                        "sample_offset",
                        "incident_feat",
                        "incident_mask",
                        "n_active_incidents",
                        "n_packed_incidents",
                        "active_event_idx",
                        "primary_incident_slot",
                        "anchor_event_mask",
                        "n_origin_events",
                        "analysis_labels/affected_mask",
                    ),
                    samples_path,
                )
                _require_attrs(
                    f.attrs,
                    (
                        "bundle_id",
                        "traffic_cache_id",
                        "samples_cache_id",
                        "artifact_id",
                        "cache_role",
                        "cache_mode",
                        "git_sha",
                        "git_branch",
                        "git_dirty",
                        "N",
                        "T",
                        "T_h",
                        "T_p",
                        "train_end",
                        "val_end",
                        "baseline_fit_start",
                        "baseline_fit_end",
                    ),
                    samples_path,
                )
                if "affected_mask" in f:
                    raise RuntimeError(
                        f"{samples_path} must keep affected_mask under "
                        "analysis_labels only"
                    )
                if str(f.attrs["cache_role"]) != "samples":
                    raise RuntimeError(
                        f"{samples_path} cache_role must be 'samples'"
                    )
                if str(f.attrs["cache_mode"]) != "formal":
                    raise RuntimeError(
                        f"{samples_path} is not a formal cache"
                    )
                if str(f.attrs["artifact_id"]) != str(
                    f.attrs["samples_cache_id"]
                ):
                    raise RuntimeError(
                        f"{samples_path} samples artifact_id does not match "
                        "samples_cache_id"
                    )
                if bool(f.attrs["git_dirty"]):
                    raise RuntimeError(
                        f"{samples_path} is a dirty formal cache"
                    )
            self.sample_start = f["sample_start"][...]
            self.split = f["split"][...]
            self.event_idx = f["event_idx"][...]
            self.event_start_idx = f["event_start_idx"][...]
            self.event_id = (
                f["event_id"].asstr()[...] if "event_id" in f else None
            )
            self.anchor_region_idx = f["anchor_region_idx"][...]
            self.sample_offset = f["sample_offset"][...]
            self.incident_feat = f["incident_feat"][...]
            self.incident_mask = f["incident_mask"][...]
            if "analysis_labels/affected_mask" in f:
                self._analysis_affected_mask = f[
                    "analysis_labels/affected_mask"
                ][...]
            else:
                self._analysis_affected_mask = f["affected_mask"][...]
            self.n_active_incidents = f["n_active_incidents"][...]
            self.n_packed_incidents = (
                f["n_packed_incidents"][...]
                if "n_packed_incidents" in f
                else self.incident_mask.sum(axis=1).astype(np.int32)
            )
            if "active_event_idx" in f:
                self.active_event_idx = f["active_event_idx"][...]
            else:
                self.active_event_idx = None
            if "primary_incident_slot" in f:
                self.primary_incident_slot = f["primary_incident_slot"][...]
            elif self.active_event_idx is not None:
                matches = self.active_event_idx == self.event_idx[:, None]
                self.primary_incident_slot = np.where(
                    matches.any(axis=1), matches.argmax(axis=1), 0
                ).astype(np.int16)
            else:
                self.primary_incident_slot = np.zeros(
                    len(self.sample_start), dtype=np.int16
                )
            self.anchor_event_mask = (
                f["anchor_event_mask"][...]
                if "anchor_event_mask" in f
                else np.zeros_like(self.incident_mask)
            )
            self.n_origin_events = (
                f["n_origin_events"][...]
                if "n_origin_events" in f
                else np.ones(len(self.sample_start), dtype=np.int16)
            )
            self.M_max = int(f.attrs["M_max"])
            self.C_e = int(f.attrs["C_e"])
            self.N = int(f.attrs["N"])
            self.sample_T = int(f.attrs.get("T", -1))
            self.T_h = int(f.attrs["T_h"])
            self.T_p = int(f.attrs["T_p"])
            self.train_end = int(f.attrs.get("train_end", -1))
            self.val_end = int(f.attrs.get("val_end", -1))
            self.bundle_id = str(f.attrs.get("bundle_id", ""))
            self.traffic_cache_id = str(
                f.attrs.get("traffic_cache_id", "")
            )
            self.samples_cache_id = str(
                f.attrs.get("samples_cache_id", "")
            )
            self.samples_cache_mode = str(
                f.attrs.get("cache_mode", "legacy")
            )
            self.samples_baseline_fit_start = int(
                f.attrs.get("baseline_fit_start", 0)
            )
            self.samples_baseline_fit_end = int(
                f.attrs.get("baseline_fit_end", self.train_end)
            )
            sample_manifest_json = str(
                f.attrs.get("source_manifest_json", "[]")
            )
            try:
                self.sample_source_manifest = json.loads(
                    sample_manifest_json
                )
            except json.JSONDecodeError:
                self.sample_source_manifest = []
            self.sample_cache_git = {
                "git_sha": str(f.attrs.get("git_sha", "unknown")),
                "git_branch": str(f.attrs.get("git_branch", "unknown")),
                "git_dirty": bool(f.attrs.get("git_dirty", True)),
            }

        if strict_v3:
            self._validate_sample_contract(samples_path)

        # ---- event rel_feat (n_events, N, 4) — incident↔sensor geometry ----
        rel_feat_path = Path(data_dir) / f"{region_name}_event_rel_feat.npz"
        self.rel_feat_path = rel_feat_path
        self.relation_cache_id = ""
        self.relation_cache_git: dict[str, object] = {}
        if rel_feat_path.exists() and self.active_event_idx is not None:
            with np.load(rel_feat_path, allow_pickle=False) as rel:
                if strict_v3:
                    required_rel = {
                        "rel_feat",
                        "event_start_idx",
                        "event_id",
                        "protocol_version",
                        "cache_schema_version",
                        "bundle_id",
                        "traffic_cache_id",
                        "samples_cache_id",
                        "relation_cache_id",
                        "artifact_id",
                        "cache_role",
                        "cache_mode",
                        "git_sha",
                        "git_branch",
                        "git_dirty",
                    }
                    missing_rel = sorted(required_rel.difference(rel.files))
                    if missing_rel:
                        raise RuntimeError(
                            f"{rel_feat_path} is missing required v3 fields: "
                            + ", ".join(missing_rel)
                        )
                rel_protocol = (
                    str(rel["protocol_version"].item())
                    if "protocol_version" in rel
                    else "legacy"
                )
                rel_schema = int(
                    rel["cache_schema_version"].item()
                    if "cache_schema_version" in rel
                    else -1
                )
                if (
                    required_protocol is not None
                    and rel_protocol != required_protocol
                ):
                    raise RuntimeError(
                        f"{rel_feat_path} uses protocol {rel_protocol!r}; "
                        f"required {required_protocol!r}"
                    )
                if strict_v3 and rel_schema != CACHE_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"{rel_feat_path} uses cache schema {rel_schema}; "
                        f"required {CACHE_SCHEMA_VERSION}"
                    )
                if not np.array_equal(
                    self.event_start_idx, rel["event_start_idx"]
                ):
                    raise RuntimeError(
                        f"event ordering mismatch in {rel_feat_path}"
                    )
                if self.event_id is not None and (
                    "event_id" not in rel
                    or not np.array_equal(
                        self.event_id.astype(str),
                        rel["event_id"].astype(str),
                    )
                ):
                    raise RuntimeError(
                        f"event identity mismatch in {rel_feat_path}"
                    )
                if strict_v3:
                    relation_ids = {
                        "bundle_id": self.bundle_id,
                        "traffic_cache_id": self.traffic_cache_id,
                        "samples_cache_id": self.samples_cache_id,
                    }
                    for key, expected in relation_ids.items():
                        actual = str(rel[key].item())
                        if actual != expected:
                            raise RuntimeError(
                                f"{key} mismatch in {rel_feat_path}: "
                                f"{actual!r} != {expected!r}"
                            )
                    if str(rel["cache_role"].item()) != "relation":
                        raise RuntimeError(
                            f"{rel_feat_path} cache_role must be 'relation'"
                        )
                    if str(rel["cache_mode"].item()) != "formal":
                        raise RuntimeError(
                            f"{rel_feat_path} is not a formal relation cache"
                        )
                    if str(rel["artifact_id"].item()) != str(
                        rel["relation_cache_id"].item()
                    ):
                        raise RuntimeError(
                            f"{rel_feat_path} relation artifact_id does not "
                            "match relation_cache_id"
                        )
                    if bool(rel["git_dirty"].item()):
                        raise RuntimeError(
                            f"{rel_feat_path} is a dirty formal relation cache"
                        )
                self.event_rel_feat = rel["rel_feat"].astype(np.float32)
                self.relation_cache_id = str(
                    rel["relation_cache_id"].item()
                ) if "relation_cache_id" in rel else ""
                self.relation_cache_git = {
                    "git_sha": str(rel["git_sha"].item())
                    if "git_sha" in rel
                    else "unknown",
                    "git_branch": str(rel["git_branch"].item())
                    if "git_branch" in rel
                    else "unknown",
                    "git_dirty": bool(rel["git_dirty"].item())
                    if "git_dirty" in rel
                    else True,
                }
            if self.event_rel_feat.shape != (
                len(self.event_start_idx),
                self.N,
                4,
            ):
                raise RuntimeError(
                    f"{rel_feat_path} rel_feat shape "
                    f"{self.event_rel_feat.shape} does not match "
                    f"({len(self.event_start_idx)}, {self.N}, 4)"
                )
            if not np.isfinite(self.event_rel_feat).all():
                raise RuntimeError(
                    f"{rel_feat_path} contains non-finite relation features"
                )
            self.has_rel_feat = True
        else:
            if require_rel_feat:
                raise RuntimeError(
                    f"required relation cache is missing: {rel_feat_path}"
                )
            self.event_rel_feat = None
            self.has_rel_feat = False

        # ---- traffic file ----
        self.traffic_path = Path(data_dir) / f"{region_name}_traffic.h5"
        with h5py.File(self.traffic_path, "r") as f:
            traffic_protocol = str(f.attrs.get("protocol_version", "legacy"))
            traffic_schema = int(f.attrs.get("cache_schema_version", 1))
            if traffic_protocol != self.protocol_version:
                raise RuntimeError(
                    f"protocol mismatch between {samples_path.name} "
                    f"({self.protocol_version}) and {self.traffic_path.name} "
                    f"({traffic_protocol})"
                )
            if traffic_schema != self.cache_schema_version:
                raise RuntimeError(
                    f"cache schema mismatch between {samples_path.name} "
                    f"({self.cache_schema_version}) and "
                    f"{self.traffic_path.name} ({traffic_schema})"
                )
            if strict_v3:
                _require_h5_members(
                    f,
                    (
                        "flow_series_imputed",
                        "flow_mask",
                        "time_enc",
                        "static_meta",
                        "baseline_median",
                        "baseline_scale",
                        "region_idx",
                        "times_ns",
                    ),
                    self.traffic_path,
                )
                _require_attrs(
                    f.attrs,
                    (
                        "bundle_id",
                        "traffic_cache_id",
                        "artifact_id",
                        "cache_role",
                        "cache_mode",
                        "git_sha",
                        "git_branch",
                        "git_dirty",
                        "N",
                        "T",
                        "train_end",
                        "val_end",
                        "baseline_fit_start",
                        "baseline_fit_end",
                        "imputation_method",
                        "baseline_method",
                    ),
                    self.traffic_path,
                )
                paired_attrs = {
                    "bundle_id": self.bundle_id,
                    "traffic_cache_id": self.traffic_cache_id,
                }
                for key, expected in paired_attrs.items():
                    actual = str(f.attrs[key])
                    if actual != expected:
                        raise RuntimeError(
                            f"{key} mismatch between {samples_path.name} "
                            f"and {self.traffic_path.name}: "
                            f"{expected!r} != {actual!r}"
                        )
                if str(f.attrs["cache_role"]) != "traffic":
                    raise RuntimeError(
                        f"{self.traffic_path} cache_role must be 'traffic'"
                    )
                if str(f.attrs["cache_mode"]) != "formal":
                    raise RuntimeError(
                        f"{self.traffic_path} is not a formal cache"
                    )
                if str(f.attrs["artifact_id"]) != str(
                    f.attrs["traffic_cache_id"]
                ):
                    raise RuntimeError(
                        f"{self.traffic_path} traffic artifact_id does not "
                        "match traffic_cache_id"
                    )
                if bool(f.attrs["git_dirty"]):
                    raise RuntimeError(
                        f"{self.traffic_path} is a dirty formal cache"
                    )
            # small stuff: load eagerly always
            self.time_enc = f["time_enc"][...]          # (T, 5) float32
            self.static_meta = f["static_meta"][...]    # (N, C_meta) float32
            self.baseline_median = f["baseline_median"][...]  # (2, 288, N, 3)
            self.baseline_scale = f["baseline_scale"][...]    # (2, 288, N, 3)
            self.region_idx = f["region_idx"][...]      # (N,) int64
            self.times_ns = f["times_ns"][...]          # (T,) int64
            self.T = int(f.attrs["T"])
            traffic_n = int(f.attrs.get("N", self.static_meta.shape[0]))
            traffic_train_end = int(f.attrs.get("train_end", -1))
            traffic_val_end = int(f.attrs.get("val_end", -1))
            flow_shape = f["flow_series_imputed"].shape
            flow_mask_shape = f["flow_mask"].shape
            self.C_meta = int(self.static_meta.shape[1])
            self.meta_feature_names = np.asarray(
                f.attrs.get("meta_feature_names", []), dtype=object
            )
            self.baseline_fit_start = int(
                f.attrs.get("baseline_fit_start", 0)
            )
            self.baseline_fit_end = int(
                f.attrs.get("baseline_fit_end", self.train_end)
            )
            self.imputation_method = str(
                f.attrs.get("imputation_method", "legacy_unknown")
            )
            self.baseline_method = str(
                f.attrs.get("baseline_method", "legacy_unknown")
            )
            manifest_json = str(
                f.attrs.get("source_manifest_json", "[]")
            )
            try:
                self.traffic_source_manifest = json.loads(manifest_json)
            except json.JSONDecodeError:
                self.traffic_source_manifest = []
            self.traffic_cache_git = {
                "git_sha": str(f.attrs.get("git_sha", "unknown")),
                "git_branch": str(f.attrs.get("git_branch", "unknown")),
                "git_dirty": bool(f.attrs.get("git_dirty", True)),
            }

            # large: load eager unless lazy
            if not lazy:
                self.flow_series = f["flow_series_imputed"][...]  # (T, N, 3)
                self.flow_mask = f["flow_mask"][...]              # (T, N, 3)
            else:
                self.flow_series = None
                self.flow_mask = None

        if strict_v3:
            paired_numeric = {
                "N": (self.N, traffic_n),
                "T": (self.sample_T, self.T),
                "train_end": (self.train_end, traffic_train_end),
                "val_end": (self.val_end, traffic_val_end),
                "baseline_fit_start": (
                    self.samples_baseline_fit_start,
                    self.baseline_fit_start,
                ),
                "baseline_fit_end": (
                    self.samples_baseline_fit_end,
                    self.baseline_fit_end,
                ),
            }
            for key, (sample_value, traffic_value) in paired_numeric.items():
                if sample_value != traffic_value:
                    raise RuntimeError(
                        f"{key} mismatch between {samples_path.name} and "
                        f"{self.traffic_path.name}: "
                        f"{sample_value} != {traffic_value}"
                    )
            expected_traffic_shapes = {
                "time_enc": (self.T, 5),
                "static_meta_nodes": self.N,
                "region_idx": (self.N,),
                "times_ns": (self.T,),
            }
            if self.time_enc.shape != expected_traffic_shapes["time_enc"]:
                raise RuntimeError(
                    f"{self.traffic_path} time_enc shape is invalid"
                )
            if self.static_meta.shape[0] != self.N:
                raise RuntimeError(
                    f"{self.traffic_path} static_meta node count is invalid"
                )
            if self.region_idx.shape != expected_traffic_shapes["region_idx"]:
                raise RuntimeError(
                    f"{self.traffic_path} region_idx shape is invalid"
                )
            if self.times_ns.shape != expected_traffic_shapes["times_ns"]:
                raise RuntimeError(
                    f"{self.traffic_path} times_ns shape is invalid"
                )
            if self.baseline_fit_end > self.train_end:
                raise RuntimeError(
                    "baseline_fit_end exceeds the cached training boundary"
                )
            expected_flow_shape = (self.T, self.N, 3)
            if (
                flow_shape != expected_flow_shape
                or flow_mask_shape != expected_flow_shape
            ):
                raise RuntimeError(
                    f"{self.traffic_path} flow tensors must have shape "
                    f"{expected_flow_shape}"
                )

        if self.train_end < 0:
            self.train_end = int(self.T * 0.70)
        if self.val_end < 0:
            self.val_end = int(self.T * 0.85)
        if strict_v3:
            self._validate_forecast_windows(samples_path)

        # Pre-compute day_kind / tod arrays for fast baseline lookup
        times = pd.to_datetime(self.times_ns)
        self.day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
        self.tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
        # Day-of-week 0..6 (raw), for full week-cycle time encoding (used by FDN++ Main branch)
        self.dow = times.dayofweek.to_numpy().astype(np.int8)

        # ---- region graph ----
        graph_key = REGION_NAME_TO_KEY[region_name]
        self.graph_path = (
            Path(graph_dir) / f"{graph_key}_sparse_adj.npz"
        )
        with np.load(self.graph_path, allow_pickle=False) as graph:
            for field in ("edge_index", "core_idx", "region_idx"):
                if field not in graph:
                    raise RuntimeError(
                        f"{self.graph_path} is missing required field {field}"
                    )
            self.edge_index = graph["edge_index"].astype(np.int64)
            self.core_idx = graph["core_idx"].astype(np.int64)
            graph_region_idx = graph["region_idx"]
        if not np.array_equal(self.region_idx, graph_region_idx):
            raise RuntimeError(
                f"region_idx mismatch: {region_name} traffic.h5 vs sparse_adj.npz"
            )
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise RuntimeError(
                f"{self.graph_path} edge_index must have shape (2, E)"
            )
        if self.edge_index.size and (
            self.edge_index.min() < 0 or self.edge_index.max() >= self.N
        ):
            raise RuntimeError(
                f"{self.graph_path} edge_index is outside [0, {self.N})"
            )

        artifact_paths = {
            "traffic": self.traffic_path,
            "samples": self.samples_path,
            "graph": self.graph_path,
        }
        if self.rel_feat_path.exists():
            artifact_paths["relation"] = self.rel_feat_path
        self.artifact_sha256 = (
            {
                role: _sha256_file(path)
                for role, path in artifact_paths.items()
            }
            if self.verify_artifact_hashes
            else {}
        )
        identity_payload = {
            "protocol_version": self.protocol_version,
            "cache_schema_version": self.cache_schema_version,
            "bundle_id": self.bundle_id,
            "traffic_cache_id": self.traffic_cache_id,
            "samples_cache_id": self.samples_cache_id,
            "relation_cache_id": self.relation_cache_id,
            "artifact_sha256": self.artifact_sha256,
        }
        self.cache_bundle_digest = hashlib.sha256(
            json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        self._lazy_traffic_f: h5py.File | None = None
        self.training_git = dict(TRAINING_GIT)

    def _validate_sample_contract(self, samples_path: Path) -> None:
        sample_count = len(self.sample_start)
        if self.sample_start.ndim != 1 or sample_count == 0:
            raise RuntimeError(
                f"{samples_path} sample_start must be a non-empty vector"
            )
        if np.unique(self.sample_start).size != sample_count:
            raise RuntimeError(
                f"{samples_path} sample_start must be unique"
            )
        if np.any(np.diff(self.sample_start) <= 0):
            raise RuntimeError(
                f"{samples_path} sample_start must be strictly increasing"
            )
        if not np.isin(self.split, (0, 1, 2)).all():
            raise RuntimeError(f"{samples_path} contains an invalid split code")

        vector_fields = {
            "split": self.split,
            "event_idx": self.event_idx,
            "anchor_region_idx": self.anchor_region_idx,
            "sample_offset": self.sample_offset,
            "n_active_incidents": self.n_active_incidents,
            "n_packed_incidents": self.n_packed_incidents,
            "primary_incident_slot": self.primary_incident_slot,
            "n_origin_events": self.n_origin_events,
        }
        for name, values in vector_fields.items():
            if values.shape != (sample_count,):
                raise RuntimeError(
                    f"{samples_path} {name} shape must be ({sample_count},)"
                )

        expected_matrix_shapes = {
            "incident_feat": (sample_count, self.M_max, self.C_e),
            "incident_mask": (sample_count, self.M_max),
            "active_event_idx": (sample_count, self.M_max),
            "anchor_event_mask": (sample_count, self.M_max),
            "affected_mask": (sample_count, self.N),
        }
        actual_matrix_shapes = {
            "incident_feat": self.incident_feat.shape,
            "incident_mask": self.incident_mask.shape,
            "active_event_idx": self.active_event_idx.shape,
            "anchor_event_mask": self.anchor_event_mask.shape,
            "affected_mask": self._analysis_affected_mask.shape,
        }
        for name, expected in expected_matrix_shapes.items():
            if actual_matrix_shapes[name] != expected:
                raise RuntimeError(
                    f"{samples_path} {name} shape "
                    f"{actual_matrix_shapes[name]} != {expected}"
                )

        event_count = len(self.event_start_idx)
        if (
            self.event_id is None
            or self.event_id.shape != (event_count,)
            or np.unique(self.event_id).size != event_count
        ):
            raise RuntimeError(
                f"{samples_path} event_id must be unique and aligned"
            )
        if np.any(np.diff(self.event_start_idx) < 0):
            raise RuntimeError(
                f"{samples_path} event_start_idx must be sorted"
            )
        if (
            self.event_idx.min(initial=0) < 0
            or self.event_idx.max(initial=-1) >= event_count
        ):
            raise RuntimeError(
                f"{samples_path} event_idx is outside the event table"
            )

        valid_active = self.active_event_idx >= 0
        if not np.array_equal(valid_active, self.incident_mask.astype(bool)):
            raise RuntimeError(
                f"{samples_path} active_event_idx padding does not match "
                "incident_mask"
            )
        if valid_active.any() and (
            self.active_event_idx[valid_active].max() >= event_count
        ):
            raise RuntimeError(
                f"{samples_path} active_event_idx is outside the event table"
            )
        packed = valid_active.sum(axis=1)
        if not np.array_equal(packed, self.n_packed_incidents):
            raise RuntimeError(
                f"{samples_path} n_packed_incidents is inconsistent"
            )
        if np.any(self.n_active_incidents < self.n_packed_incidents):
            raise RuntimeError(
                f"{samples_path} n_active_incidents is smaller than packed"
            )

        slots = self.primary_incident_slot.astype(np.int64)
        if np.any((slots < 0) | (slots >= self.M_max)):
            raise RuntimeError(
                f"{samples_path} primary_incident_slot is out of range"
            )
        rows = np.arange(sample_count)
        if not np.array_equal(
            self.active_event_idx[rows, slots], self.event_idx
        ):
            raise RuntimeError(
                f"{samples_path} primary_incident_slot is misaligned"
            )
        if np.any(self.anchor_event_mask & ~self.incident_mask):
            raise RuntimeError(
                f"{samples_path} anchor_event_mask includes padding"
            )
        if not np.array_equal(
            self.anchor_event_mask.sum(axis=1), self.n_origin_events
        ):
            raise RuntimeError(
                f"{samples_path} n_origin_events is inconsistent"
            )
        if not self.anchor_event_mask[rows, slots].all():
            raise RuntimeError(
                f"{samples_path} canonical event is not an origin event"
            )

        split_event_sets = []
        for code in (0, 1, 2):
            active = self.active_event_idx[self.split == code]
            split_event_sets.append(set(active[active >= 0].tolist()))
        if (
            not split_event_sets[0].isdisjoint(split_event_sets[1])
            or not split_event_sets[0].isdisjoint(split_event_sets[2])
            or not split_event_sets[1].isdisjoint(split_event_sets[2])
        ):
            raise RuntimeError(
                f"{samples_path} active event identity crosses splits"
            )

    def _validate_forecast_windows(self, samples_path: Path) -> None:
        expected = np.asarray(
            [
                classify_forecast_window(
                    int(anchor),
                    self.T_p,
                    self.train_end,
                    self.val_end,
                    self.T,
                )
                for anchor in self.sample_start
            ],
            dtype=object,
        )
        if any(value is None for value in expected) or not np.array_equal(
            expected.astype(np.int8), self.split
        ):
            raise RuntimeError(
                f"{samples_path} has a forecast target crossing split bounds"
            )

    def analysis_labels(self, idx: int) -> dict[str, np.ndarray]:
        """Return post-event labels through an explicit analysis-only API."""
        return {
            "affected_mask": self._analysis_affected_mask[idx].astype(
                np.bool_
            )
        }

    def result_metadata(self) -> dict[str, object]:
        """JSON-safe protocol and provenance fields for every new summary."""
        return {
            "protocol_version": self.protocol_version,
            "cache_schema_version": self.cache_schema_version,
            "sampling_protocol": self.sampling_protocol,
            "split": {
                "train_end": self.train_end,
                "val_end": self.val_end,
            },
            "preprocessing": {
                "imputation_method": self.imputation_method,
                "baseline_method": self.baseline_method,
                "baseline_fit_start": self.baseline_fit_start,
                "baseline_fit_end": self.baseline_fit_end,
            },
            "cache_provenance": {
                "bundle_id": self.bundle_id,
                "traffic_cache_id": self.traffic_cache_id,
                "samples_cache_id": self.samples_cache_id,
                "relation_cache_id": self.relation_cache_id,
                "cache_bundle_digest": self.cache_bundle_digest,
                "artifact_sha256": dict(self.artifact_sha256),
                "traffic_source_manifest": self.traffic_source_manifest,
                "sample_source_manifest": self.sample_source_manifest,
                "traffic_builder_git": self.traffic_cache_git,
                "sample_builder_git": self.sample_cache_git,
                "relation_builder_git": self.relation_cache_git,
            },
            "training_git": self.training_git,
        }

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

        hist_lo, hist_hi, fut_lo, fut_hi = forecast_window_bounds(
            t, T_h, T_p
        )

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

        # Day-lag anchor: observed flow at the same clock time yesterday for each future step.
        # Block read keeps h5 fancy-indexing legal; only first-day (train) samples clamp at 0.
        lag_idx = np.clip(fut_idx - 288, 0, None)
        blk_lo, blk_hi = int(lag_idx[0]), int(lag_idx[-1]) + 1
        if self.flow_series is not None:
            blk = self.flow_series[blk_lo:blk_hi, :, 0]
        else:
            blk = self._traffic_handle()["flow_series_imputed"][blk_lo:blk_hi, :, 0]
        y_daylag = np.transpose(blk[lag_idx - blk_lo], (1, 0)).astype(np.float32)         # (N,T_p)

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

        sample = {
            "x_hist":         x_hist,                            # (N, T_h, 3)
            "x_hist_mask":    x_hist_mask,                       # (N, T_h, 3) bool
            "y_true":         y_true,                            # (N, T_p, 3)
            "y_mask":         y_mask,                            # (N, T_p, 3) bool
            "y_baseline":     y_baseline,                        # (N, T_p, 3)
            "x_baseline":     x_baseline,                        # (N, T_h, 3)
            "y_daylag":       y_daylag,                          # (N, T_p) flow same time yesterday
            "time_enc":       time_enc,                          # (T_h, 5)
            "time_feat":      time_feat.astype(np.float32),     # (T_h, 2) tod/288, dow/7
            "static_meta":    self.static_meta.astype(np.float32),  # (N, C_meta)
            "region_code":    np.int64(self.region_code),
            "incident_feat":  self.incident_feat[idx].astype(np.float32),  # (M_max, C_e)
            "incident_mask":  self.incident_mask[idx].astype(np.bool_),    # (M_max,)
            "rel_feat":       rel_feat,                           # (M_max, N, 4)
            "n_active_incidents": np.int32(self.n_active_incidents[idx]),
            "primary_incident_slot": np.int16(
                self.primary_incident_slot[idx]
            ),
            "sample_start":   np.int64(t),
            "sample_idx":     np.int64(idx),
        }
        if self.include_analysis_labels:
            sample["affected_mask"] = self._analysis_affected_mask[
                idx
            ].astype(np.bool_)
        return sample


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
                 stride: int = 1, train_frac: float = 0.7, val_frac: float = 0.1,
                 required_protocol: str | None = PROTOCOL_VERSION):
        super().__init__(
            region_name,
            data_dir,
            graph_dir,
            lazy=lazy,
            required_protocol=required_protocol,
        )
        T_h, T_p, T = self.T_h, self.T_p, self.T
        if not 0 < train_frac < train_frac + val_frac < 1:
            raise ValueError("full-window fractions must satisfy 0 < train < train+val < 1")
        self.train_end = int(T * train_frac)
        self.val_end = int(T * (train_frac + val_frac))
        if self.baseline_fit_end > self.train_end:
            raise RuntimeError(
                "baseline_fit_end exceeds the full-window training boundary; "
                "rebuild baseline statistics for this split"
            )
        self.sampling_protocol = "full_window"
        self.full_window_config = {
            "stride": int(stride),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
        }
        starts, split = build_full_window_index(
            total_steps=T,
            history=T_h,
            horizon=T_p,
            stride=stride,
            train_end=self.train_end,
            val_end=self.val_end,
        )
        n = len(starts)

        self.sample_start = starts
        self.split = split
        self.event_idx = np.full(n, -1, dtype=np.int32)
        self.anchor_region_idx = np.full(n, -1, dtype=np.int32)
        self.sample_offset = np.full(n, -1, dtype=np.int8)
        self._analysis_affected_mask = np.zeros(
            (n, self.N), dtype=np.bool_
        )
        # This field is generated as an all-false structural placeholder for
        # full-window evaluation; it is not a post-event label.
        self.include_analysis_labels = True
        self.n_active_incidents = np.zeros(n, dtype=np.int32)
        self.incident_feat = np.zeros((n, self.M_max, self.C_e), dtype=np.float32)
        self.incident_mask = np.zeros((n, self.M_max), dtype=np.bool_)
        self.active_event_idx = np.full((n, self.M_max), -1, dtype=np.int32)
        self.primary_incident_slot = np.zeros(n, dtype=np.int16)
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
        self._seed_entropy = (
            int(seed) if seed is not None else int(np.random.SeedSequence().entropy)
        )

        self.bucket: dict[str, np.ndarray] = {}
        for gi, (region_name, _li) in enumerate(dataset.index_table):
            self.bucket.setdefault(region_name, []).append(gi)
        self.bucket = {k: np.asarray(v, dtype=np.int64) for k, v in self.bucket.items()}

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch deterministically for reproducible resumed runs."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed_entropy, epoch])
        )
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
