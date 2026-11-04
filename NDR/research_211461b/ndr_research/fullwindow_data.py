"""Independent full-window raw-material admission and train/validation adapters.

This module never opens event samples, incident features, or affected labels.
The verified v3 bundle supplies raw traffic and static graph authority only.
Full-window origins have a separate immutable sampling identity.

Runner integration:
    reader = FullWindowRegionReader(region, data_dir, graph_dir)
    statistics = fit_fullwindow_statistics(reader)
    loader = FullWindowBatchLoader(reader, statistics, split="train",
                                   batch_size=16, seed=42)
    # NDR: flow_hist, x_baseline, y_baseline, time_feat and frozen statistics.
    # SharedGraphNDR: the same fields plus history_mask, node_scale, edge_index.
    # BiST: flow_hist/time_feat and statistics.flow_mean/flow_std.

Native clean history preserves the cache's causal forward-fill imputations.
Observed-zero history is a separately identified policy for models that sanitize
masked inputs themselves. Neither policy changes the original observation mask.
Regional and pooled source-only preprocessing are deliberately distinct. Test
numeric access and target-role numeric access have no API in this adapter. A
future frozen full-window evaluator must implement its own admission gate.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import warnings
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np

from dist_net.protocol import build_full_window_index, classify_forecast_window
from ndr_research.transfer_data import (
    DAY_NS,
    FIVE_MINUTES_NS,
    FORMAL_BUNDLE_DIGEST,
    FORMAL_BUNDLE_ID,
    GRAPH_KEYS,
    PROTOCOL,
    REGIONS,
    TransferPreprocessing,
    _file_hash,
    _fingerprint,
    _json_hash,
    fit_source_only_pool,
)

FULLWINDOW_COUNTS = {"train": 73561, "val": 15757, "test": 15757, "total": 105075}
HISTORY_POLICIES = ("canonical_causal_imputation", "observed_zero")
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def legacy_array_digest(values: np.ndarray) -> str:
    """Match the verified27 analysis: shape, dtype.str, contiguous bytes."""
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _immutable(values: np.ndarray) -> np.ndarray:
    """Bytes-backed arrays cannot be made writable by setflags."""
    values = np.ascontiguousarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


@dataclass(frozen=True)
class FullWindowMaterialContract:
    """Raw-material authority; contains no event counts or sample identity."""

    bundle_id: str = FORMAL_BUNDLE_ID
    bundle_digest: str = FORMAL_BUNDLE_DIGEST
    total_steps: int = 105120
    regions: tuple[str, ...] = REGIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        if type(self.total_steps) is not int or self.total_steps < 80:
            raise ValueError("material timeline must support all three twelve-step splits")
        if not self.regions or len(set(self.regions)) != len(self.regions) or not set(self.regions) <= set(REGIONS):
            raise ValueError("material authority has invalid regions")
        if self.bundle_id == FORMAL_BUNDLE_ID or self.bundle_digest == FORMAL_BUNDLE_DIGEST:
            if not self.formal:
                raise ValueError("formal material authority cannot override timeline or region scope")

    @property
    def formal(self) -> bool:
        return (self.bundle_id, self.bundle_digest, self.total_steps, self.regions) == (
            FORMAL_BUNDLE_ID, FORMAL_BUNDLE_DIGEST, 105120, REGIONS,
        )


class FullWindowRegionReader:
    """Only source-role train/validation numerical access is currently allowed."""

    allowed_datasets = frozenset(("times_ns", "region_idx", "flow_series_imputed", "flow_mask"))

    def __init__(
        self, region: str, data_dir: Path | str, graph_dir: Path | str, *,
        role: str = "source", bundle_manifest: Path | str | None = None,
        contract: FullWindowMaterialContract | None = None,
        history_policy: str = "canonical_causal_imputation",
        train_frac: float = 0.7, val_frac: float = 0.15, stride: int = 1,
        history: int = 12, horizon: int = 12,
    ) -> None:
        if region not in REGIONS or role not in ("source", "target"):
            raise ValueError("reader needs a canonical region and source/target role")
        if (train_frac, val_frac, stride, history, horizon) != (0.7, 0.15, 1, 12, 12):
            raise ValueError("full-window sampling is locked to verified27 .7/.15/1/12/12")
        if history_policy not in HISTORY_POLICIES:
            raise ValueError("unknown clean-history policy")
        self.contract = contract or FullWindowMaterialContract()
        if region not in self.contract.regions:
            raise ValueError("region is outside the material authority")
        self._region, self._role, self._history_policy = region, role, history_policy
        self._counts: Counter[str] = Counter()
        self._recent: deque[dict[str, Any]] = deque(maxlen=128)
        self._access_digest = "0" * 64
        data_dir, graph_dir = Path(data_dir), Path(graph_dir)
        manifest_path = Path(bundle_manifest) if bundle_manifest is not None else data_dir / "cache_bundle_manifest.json"
        manifest_sha, manifest_fp = _file_hash(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_fp != _fingerprint(manifest_path):
            raise RuntimeError("material manifest changed while interpreting identities")
        if (manifest.get("status"), manifest.get("protocol_version"), manifest.get("cache_schema_version")) != ("verified", PROTOCOL, 3):
            raise ValueError("full-window needs a verified schema-three v3 material bundle")
        if (manifest.get("bundle_id"), manifest.get("cache_bundle_digest")) != (self.contract.bundle_id, self.contract.bundle_digest):
            raise ValueError("material manifest differs from frozen release authority")
        reports = manifest.get("regions", {})
        payload = {"protocol_version": PROTOCOL, "cache_schema_version": 3,
                   "bundle_id": self.contract.bundle_id,
                   "regions": {name: report["cache_bundle_digest"] for name, report in sorted(reports.items())}}
        if _json_hash(payload) != self.contract.bundle_digest or region not in reports:
            raise ValueError("material region index does not reproduce the frozen bundle digest")
        report = reports[region]
        provenance = report["cache_provenance"]
        region_payload = {"protocol_version": PROTOCOL, "cache_schema_version": 3,
                          "bundle_id": self.contract.bundle_id,
                          **{key: provenance[key] for key in ("traffic_cache_id", "samples_cache_id", "relation_cache_id", "artifact_sha256")}}
        if _json_hash(region_payload) != report["cache_bundle_digest"]:
            raise ValueError("region material provenance digest is inconsistent")
        # Sample/relation identities above bind manifest authority only. Their
        # files are never opened, hashed, or interpreted by this adapter.
        self.paths = {"traffic": data_dir / f"{region}_traffic.h5",
                      "graph": graph_dir / f"{GRAPH_KEYS[region]}_sparse_adj.npz"}
        hashes, self._fingerprints = {}, {}
        for name, path in self.paths.items():
            digest, fingerprint = _file_hash(path)
            if digest != provenance["artifact_sha256"].get(name):
                raise ValueError(f"{name} bytes differ from the formal raw-material authority")
            hashes[name], self._fingerprints[name] = digest, fingerprint
            self._record("hash_bytes_only", name, "*", "material_identity")
        self._T = self.contract.total_steps
        self._T_h, self._T_p = 12, 12
        self._train_end, self._val_end = int(self.T * 0.7), int(self.T * 0.85)
        with h5py.File(self.paths["traffic"], "r") as handle:
            expected = {"protocol_version": PROTOCOL, "cache_schema_version": 3,
                        "cache_role": "traffic", "cache_mode": "formal",
                        "bundle_id": self.contract.bundle_id,
                        "traffic_cache_id": provenance["traffic_cache_id"],
                        "T": self.T, "train_end": self.train_end, "val_end": self.val_end,
                        "imputation_method": "causal_forward_fill_zero_prefix"}
            if any(handle.attrs.get(key) != value for key, value in expected.items()) or bool(handle.attrs.get("git_dirty", True)):
                raise ValueError("traffic attributes disagree with the frozen clean material contract")
            self._N = int(handle.attrs["N"])
            if self.N <= 0:
                raise ValueError("traffic must have a positive node count")
            if any(handle[key].shape != (self.T, self.N, 3) for key in ("flow_series_imputed", "flow_mask")):
                raise ValueError("raw traffic dimensions disagree with metadata")
            if handle["flow_series_imputed"].dtype != np.dtype(np.float32) or handle["flow_mask"].dtype != np.dtype(bool):
                raise ValueError("raw traffic must be float32 with original boolean masks")
            self._times_ns = _immutable(self._dataset(handle, "times_ns", Ellipsis, "calendar_metadata"))
            self._region_idx = _immutable(self._dataset(handle, "region_idx", Ellipsis, "node_identity"))
        if self.times_ns.shape != (self.T,) or self.times_ns.dtype.kind not in "iu" or not np.all(np.diff(self.times_ns) == FIVE_MINUTES_NS):
            raise ValueError("calendar must be a strictly increasing five-minute grid")
        if self.region_idx.shape != (self.N,) or self.region_idx.dtype.kind not in "iu" or np.unique(self.region_idx).size != self.N:
            raise ValueError("traffic node identity must be a unique integer vector")
        self._dow = _immutable(((self.times_ns // DAY_NS + 3) % 7).astype(np.int64))
        self._tod = _immutable(((self.times_ns // FIVE_MINUTES_NS) % 288).astype(np.int64))
        with np.load(self.paths["graph"], allow_pickle=False) as graph:
            nodes, edges = graph["region_idx"], graph["edge_index"]
        if not np.array_equal(nodes, self.region_idx):
            raise ValueError("static graph does not match exact traffic node order")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu" or (edges.size and (edges.min() < 0 or edges.max() >= self.N)):
            raise ValueError("static graph must contain aligned integer roads")
        self._edge_index = _immutable(edges.astype(np.int64, copy=False))
        self._record("dataset_read", "graph", "region_idx", "node_identity")
        self._record("dataset_read", "graph", "edge_index", "static_roads")
        for name, path in self.paths.items():
            if _fingerprint(path) != self._fingerprints[name]:
                raise RuntimeError("raw material changed during metadata admission")
        starts, splits = build_full_window_index(self.T, 12, 12, 1, self.train_end, self.val_end)
        self._sample_start, self._split = _immutable(starts), _immutable(splits)
        counts = {name: int(np.count_nonzero(splits == code)) for name, code in SPLIT_CODES.items()}
        counts["total"] = int(starts.size)
        if self.contract.formal and counts != FULLWINDOW_COUNTS:
            raise ValueError("full-window count identity differs from verified27")
        algorithm = inspect.getsource(build_full_window_index) + inspect.getsource(classify_forecast_window)
        self._sampling = {"sampling_protocol": "full_window", "history": 12, "horizon": 12,
                          "stride": 1, "train_frac": 0.7, "val_frac": 0.15,
                          "T": self.T, "train_end": self.train_end, "val_end": self.val_end,
                          "counts": counts, "origin_sha256": legacy_array_digest(starts),
                          "split_sha256": legacy_array_digest(splits),
                          "split_origin_sha256": {name: legacy_array_digest(starts[splits == code]) for name, code in SPLIT_CODES.items()},
                          "array_hash_framing": "shape_then_dtype.str_then_contiguous_bytes",
                          "algorithm_sha256": hashlib.sha256(algorithm.encode()).hexdigest(),
                          "unique_origins": True, "target_disjoint_between_splits": True,
                          "adjacent_origins_share_targets_within_split": True,
                          "event_sample_identity_used": False}
        self._sampling["sampling_digest"] = _json_hash(self._sampling)
        material = {"bundle_id": self.contract.bundle_id, "bundle_digest": self.contract.bundle_digest,
                    "manifest_sha256": manifest_sha, "region_digest": report["cache_bundle_digest"],
                    "artifact_sha256": hashes, "traffic_cache_id": provenance["traffic_cache_id"],
                    "formal_release": self.contract.formal, "authority_role": "raw_traffic_and_static_graph_only"}
        self._identity = {"region": region, "role": role, "protocol_version": PROTOCOL,
                          "cache_schema_version": 3, "sampling_protocol": "full_window",
                          "material": material, "sampling": self._sampling,
                          "node_order_sha256": legacy_array_digest(self.region_idx),
                          "calendar_sha256": legacy_array_digest(self.times_ns),
                          "history_policy": history_policy,
                          "input_whitelist": sorted(self.allowed_datasets),
                          "incident_or_affected_values_read": False,
                          "test_numeric_api_available": False}
        self._identity["identity_digest"] = _json_hash(self._identity)
        self._preloaded: tuple[np.ndarray, np.ndarray] | None = None

    region = property(lambda self: self._region)
    region_name = property(lambda self: self._region)
    role = property(lambda self: self._role)
    history_policy = property(lambda self: self._history_policy)
    T = property(lambda self: self._T)
    N = property(lambda self: self._N)
    T_h = property(lambda self: self._T_h)
    T_p = property(lambda self: self._T_p)
    train_end = property(lambda self: self._train_end)
    val_end = property(lambda self: self._val_end)
    times_ns = property(lambda self: self._times_ns)
    region_idx = property(lambda self: self._region_idx)
    dow = property(lambda self: self._dow)
    tod = property(lambda self: self._tod)
    edge_index = property(lambda self: self._edge_index)
    sample_start = property(lambda self: self._sample_start)
    split = property(lambda self: self._split)
    identity = property(lambda self: copy.deepcopy(self._identity))
    sampling_identity = property(lambda self: copy.deepcopy(self._sampling))

    def _record(self, kind: str, artifact: str, dataset: str, purpose: str, **extra: Any) -> None:
        event = {"kind": kind, "artifact": artifact, "dataset": dataset, "purpose": purpose, **extra}
        self._counts[f"{kind}:{artifact}:{dataset}:{purpose}"] += 1
        self._access_digest = _json_hash({"previous": self._access_digest, "event": event})
        self._recent.append(event)

    def access_audit(self) -> dict[str, Any]:
        return {"region": self.region, "role": self.role, "counts": dict(sorted(self._counts.items())),
                "rolling_access_sha256": self._access_digest, "recent_access": copy.deepcopy(list(self._recent)),
                "sampling_protocol": "full_window", "forbidden_dataset_values_read": False,
                "test_numeric_values_read": False}

    def _dataset(self, handle: h5py.File, name: str, selection: Any, purpose: str) -> np.ndarray:
        if name not in self.allowed_datasets:
            raise PermissionError("dataset is outside the full-window whitelist")
        if name in ("flow_series_imputed", "flow_mask"):
            if not isinstance(selection, tuple) or len(selection) != 3 or selection[2] != 0 or selection[1] != slice(None):
                raise PermissionError("full-window numerical access is restricted to flow channel zero")
            interval = selection[0]
            if not isinstance(interval, slice) or interval.step not in (None, 1):
                raise PermissionError("numerical reads require a contiguous declared interval")
            self._authorize_flow(interval.start, interval.stop, purpose=purpose)
        values = np.asarray(handle[name][selection])
        self._record("dataset_read", "traffic", name, purpose)
        return values

    def indices(self, split: str) -> np.ndarray:
        if split not in ("train", "val"):
            raise PermissionError("development indices are restricted to train/val")
        return _immutable(np.flatnonzero(self.split == SPLIT_CODES[split]))

    indices_for_split = indices

    def _authorize_flow(self, start: int, stop: int, *, purpose: str) -> None:
        if self.role != "source":
            raise PermissionError("target readers expose metadata only until a future frozen evaluator")
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= self.val_end:
            raise PermissionError("development traffic cannot cross the validation boundary")
        if purpose == "source_fit":
            if stop > self.train_end:
                raise PermissionError("fitting cannot cross the unique raw train prefix")
        elif purpose == "source_train_val_preload":
            if (start, stop) != (0, self.val_end):
                raise PermissionError("preload must be the exact train/val prefix")
        elif purpose in ("window:train", "window:val"):
            name = purpose.split(":", 1)[1]
            anchor = start + self.T_h - 1
            if stop - start != 24 or classify_forecast_window(anchor, 12, self.train_end, self.val_end, self.T) != SPLIT_CODES[name]:
                raise PermissionError("window targets cross the declared development split")
        else:
            raise PermissionError("unrecognized full-window numerical read purpose")

    def _read_flow(self, start: int, stop: int, *, purpose: str) -> tuple[np.ndarray, np.ndarray]:
        self._authorize_flow(start, stop, purpose=purpose)
        if _fingerprint(self.paths["traffic"]) != self._fingerprints["traffic"]:
            raise RuntimeError("traffic changed after raw-material admission")
        if self._preloaded is not None:
            flow, mask = (values[start:stop].copy() for values in self._preloaded)
        else:
            with h5py.File(self.paths["traffic"], "r") as handle:
                selection = (slice(start, stop), slice(None), 0)
                flow = self._dataset(handle, "flow_series_imputed", selection, purpose)
                mask = self._dataset(handle, "flow_mask", selection, purpose)
        if _fingerprint(self.paths["traffic"]) != self._fingerprints["traffic"]:
            raise RuntimeError("traffic changed during whitelisted numerical access")
        if not np.isfinite(flow).all() or (flow < 0).any():
            raise FloatingPointError("canonical cached flow must be finite and nonnegative")
        self._record("flow_bounds", "traffic", "channel_zero", purpose, start=start, stop=stop, channel=0)
        return flow, mask

    def iter_source_training(self, block_steps: int = 4096) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        if self.role != "source":
            raise PermissionError("target readers cannot expose a statistics fit iterator")
        if type(block_steps) is not int or block_steps <= 0:
            raise ValueError("statistics block size must be positive")
        for start in range(0, self.train_end, block_steps):
            stop = min(start + block_steps, self.train_end)
            flow, mask = self._read_flow(start, stop, purpose="source_fit")
            yield start, stop, np.where(mask, flow, np.float32(0)), mask

    def preload_source_train_val(self) -> dict[str, Any]:
        if self.role != "source":
            raise PermissionError("target readers cannot preload numerical traffic")
        if self._preloaded is None:
            flow, mask = self._read_flow(0, self.val_end, purpose="source_train_val_preload")
            self._preloaded = (_immutable(flow), _immutable(mask))
        return {"loading_policy": "fullwindow_source_channel_zero_prefix_to_val_end",
                "prefix_start": 0, "prefix_end": self.val_end, "channels": [0],
                "flow_sha256": legacy_array_digest(self._preloaded[0]),
                "mask_sha256": legacy_array_digest(self._preloaded[1]),
                "history_policy": self.history_policy, "reader_identity_digest": self._identity["identity_digest"]}

    def get_batch(
        self, indices: Sequence[int] | np.ndarray,
        preprocessing: FullWindowStatistics | TransferPreprocessing, *, split: str,
    ) -> dict[str, np.ndarray]:
        if self.role != "source" or split not in ("train", "val"):
            raise PermissionError("full-window development batches require source train/val")
        _validate_preprocessing(preprocessing, self)
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu" or not selected.size or np.unique(selected).size != selected.size:
            raise ValueError("batch indices must be a nonempty unique integer vector")
        if np.any(selected < 0) or np.any(selected >= self.sample_start.size) or not np.all(self.split[selected] == SPLIT_CODES[split]):
            raise ValueError("batch indices do not belong to declared full-window split")
        rows = []
        for index in selected:
            anchor = int(self.sample_start[index])
            start, stop = anchor - 11, anchor + 13
            flow, mask = self._read_flow(start, stop, purpose=f"window:{split}")
            history_flow = flow[:12] if self.history_policy == "canonical_causal_imputation" else np.where(mask[:12], flow[:12], np.float32(0))
            rows.append({"flow_hist": history_flow.T,
                         "history_mask": mask[:12].T,
                         "x_baseline": preprocessing.lookup(self.dow[start:anchor + 1], self.tod[start:anchor + 1], self.N).T,
                         "y_baseline": preprocessing.lookup(self.dow[anchor + 1:stop], self.tod[anchor + 1:stop], self.N).T,
                         "time_feat": np.stack((self.tod[start:anchor + 1] / 288.0, self.dow[start:anchor + 1] / 7.0), axis=-1).astype(np.float32),
                         "target": np.where(mask[12:], flow[12:], np.float32(0)).T,
                         "target_mask": mask[12:].T})
        result = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        result.update(node_scale=preprocessing.scales(self.N), edge_index=self.edge_index.copy(),
                      sample_start=self.sample_start[selected].copy())
        return result


@dataclass(frozen=True)
class FullWindowStatistics:
    """Regional NDR weekly7 statistics fitted on unique observed train rows."""

    baseline: np.ndarray
    flow_mean: float
    flow_std: float
    residual_std: float
    residual_std_by_node: np.ndarray
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        baseline = _immutable(np.asarray(self.baseline, dtype=np.float32))
        scales = _immutable(np.asarray(self.residual_std_by_node, dtype=np.float32))
        identity = copy.deepcopy(dict(self.identity))
        if baseline.ndim != 3 or baseline.shape[:2] != (7, 288) or scales.shape != (baseline.shape[-1],):
            raise ValueError("regional statistics have invalid weekly/node dimensions")
        if not np.isfinite(baseline).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("regional statistics must be finite with positive scales")
        values = {"baseline_sha256": legacy_array_digest(baseline),
                  "node_scale_sha256": legacy_array_digest(scales),
                  "flow_mean": self.flow_mean, "flow_std": self.flow_std, "residual_std": self.residual_std}
        if any(identity.get(key) != value for key, value in values.items()) or identity.get("identity_digest") != _json_hash({k: v for k, v in identity.items() if k != "identity_digest"}):
            raise ValueError("regional statistics identity does not bind numeric values")
        if not all(np.isfinite(v) for v in (self.flow_mean, self.flow_std, self.residual_std)) or min(self.flow_std, self.residual_std) <= 0:
            raise ValueError("regional scalar scales must be finite and positive")
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "residual_std_by_node", scales)
        object.__setattr__(self, "identity", MappingProxyType(identity))

    def validate_reader(self, reader: FullWindowRegionReader) -> None:
        if self.identity.get("sampling_protocol") != "full_window" or self.identity.get("reader_identity_digest") != reader.identity["identity_digest"] or reader.role != "source":
            raise ValueError("regional statistics are bound to another source reader")

    def lookup(self, dow: np.ndarray, tod: np.ndarray, nodes: int) -> np.ndarray:
        if nodes != self.baseline.shape[-1]:
            raise ValueError("weekly statistics node order differs from reader")
        return self.baseline[dow, tod].copy()

    def scales(self, nodes: int) -> np.ndarray:
        if nodes != len(self.residual_std_by_node):
            raise ValueError("regional residual scales have different node dimensions")
        return self.residual_std_by_node.copy()

    def state_dict(self) -> dict[str, Any]:
        return {"baseline": self.baseline.copy(), "flow_mean": self.flow_mean,
                "flow_std": self.flow_std, "residual_std": self.residual_std,
                "residual_std_by_node": self.residual_std_by_node.copy(),
                "identity": copy.deepcopy(dict(self.identity))}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FullWindowStatistics:
        return cls(**state)


def fit_fullwindow_statistics(reader: FullWindowRegionReader) -> FullWindowStatistics:
    """Match weekly7 NDR semantics without any event data or cached baseline."""
    if not isinstance(reader, FullWindowRegionReader) or reader.role != "source":
        raise PermissionError("regional fitting requires a full-window source reader")
    flow = np.empty((reader.train_end, reader.N), dtype=np.float32)
    mask = np.empty(flow.shape, dtype=bool)
    for start, stop, values, observed in reader.iter_source_training():
        flow[start:stop], mask[start:stop] = values, observed
    observed = flow[mask].astype(np.float64)
    if not observed.size:
        raise ValueError("regional training prefix has no observed points")
    mean = float(observed.mean())
    std = float(observed.std())
    if std <= 1e-6:
        std = 1.0
    dow, tod = reader.dow[:reader.train_end], reader.tod[:reader.train_end]
    baseline = np.zeros((7, 288, reader.N), dtype=np.float32)
    for day in range(7):
        for slot in range(288):
            selected = np.flatnonzero((dow == day) & (tod == slot))
            if selected.size:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    median = np.nanmedian(np.where(mask[selected], flow[selected], np.nan), axis=0)
                baseline[day, slot] = np.where(np.isfinite(median), median, np.float32(0))
    # Subtract in float32 before casting, matching the current weekly7 fitter.
    residual = (flow - baseline[dow, tod]).astype(np.float64)
    residual_std = float(residual[mask].std())
    if residual_std <= 1e-6:
        residual_std = 1.0
    scales = np.ones(reader.N, dtype=np.float32)
    for node in range(reader.N):
        values = residual[:, node][mask[:, node]]
        if values.size and float(values.std()) > 1e-6:
            scales[node] = values.std()
    identity = {"mode": "regional_supervised", "sampling_protocol": "full_window",
                "reader_identity_digest": reader.identity["identity_digest"],
                "sampling_digest": reader.sampling_identity["sampling_digest"],
                "fit_start": 0, "fit_end": reader.train_end,
                "unique_raw_timestamps": reader.train_end, "valid_points": int(observed.size),
                "fit_values": "raw_observed_flow_at_unique_train_timestamps",
                "climatology": "weekly7_per_node_masked_median_zero_for_empty_cells",
                "scalar_reduction_dtype": "float64", "legacy27_scalar_reduction_dtype": "float32",
                "fitter_sha256": hashlib.sha256(inspect.getsource(fit_fullwindow_statistics).encode()).hexdigest(),
                "flow_mean": mean, "flow_std": std, "residual_std": residual_std,
                "baseline_sha256": legacy_array_digest(baseline), "node_scale_sha256": legacy_array_digest(scales)}
    identity["identity_digest"] = _json_hash(identity)
    return FullWindowStatistics(baseline, mean, std, residual_std, scales, identity)


def fit_fullwindow_source_pool(readers: Mapping[str, FullWindowRegionReader] | Sequence[FullWindowRegionReader]) -> TransferPreprocessing:
    """Reuse the pure source-only estimator with full-window identities bound."""
    sources = list(readers.values()) if isinstance(readers, Mapping) else list(readers)
    if any(not isinstance(reader, FullWindowRegionReader) or reader.role != "source" for reader in sources):
        raise PermissionError("pool fitting requires full-window source readers only")
    result = fit_source_only_pool(sources)
    identity = copy.deepcopy(dict(result.identity))
    identity.pop("identity_digest")
    identity.update(sampling_protocol="full_window",
                    sampling_identities={reader.region: reader.sampling_identity["sampling_digest"] for reader in sources},
                    fitter_sha256=hashlib.sha256(inspect.getsource(fit_source_only_pool).encode()).hexdigest())
    identity["identity_digest"] = _json_hash(identity)
    return TransferPreprocessing(result.mode, result.template, result.residual_scale, identity)


def _validate_preprocessing(preprocessing: FullWindowStatistics | TransferPreprocessing, reader: FullWindowRegionReader) -> None:
    if not isinstance(preprocessing, (FullWindowStatistics, TransferPreprocessing)):
        raise TypeError("full-window batches require bound regional or source-only preprocessing")
    if isinstance(preprocessing, TransferPreprocessing):
        # The reused class exposes a dictionary identity. Revalidate its bound
        # arrays and digest here before any numerical read by this new adapter.
        TransferPreprocessing.from_state(preprocessing.to_state())
        if preprocessing.mode != "source_only" or preprocessing.identity.get("sampling_protocol") != "full_window" or preprocessing.identity.get("sampling_identities", {}).get(reader.region) != reader.sampling_identity["sampling_digest"]:
            raise ValueError("event or unbound source-only preprocessing cannot enter full-window development")
    preprocessing.validate_reader(reader)


class FullWindowBatchLoader:
    """Train/validation-only no-drop-tail loader with strict cursor restoration."""

    def __init__(self, reader: FullWindowRegionReader,
                 preprocessing: FullWindowStatistics | TransferPreprocessing, *,
                 split: str, batch_size: int, seed: int, shuffle: bool | None = None) -> None:
        if reader.role != "source" or split not in ("train", "val"):
            raise PermissionError("full-window development loaders allow only source train/val")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch size must be positive")
        _validate_preprocessing(preprocessing, reader)
        self.reader, self.preprocessing, self.split = reader, preprocessing, split
        self.batch_size = self.bs = batch_size
        self.shuffle = split == "train" if shuffle is None else bool(shuffle)
        if self.shuffle and split != "train":
            raise ValueError("validation order must be stable")
        self.indices = reader.indices(split)
        self.size, self.num_batch = len(self.indices), (len(self.indices) + batch_size - 1) // batch_size
        if not self.size:
            raise ValueError("development split is empty")
        self._rng = np.random.default_rng(seed)
        self.permutation, self.cursor, self.epoch = self.indices.copy(), 0, 0
        self._identity = {"reader": reader.identity["identity_digest"],
                          "sampling_digest": reader.sampling_identity["sampling_digest"],
                          "preprocessing": preprocessing.identity["identity_digest"],
                          "split": split, "batch_size": batch_size, "shuffle": self.shuffle,
                          "origin_selection_sha256": legacy_array_digest(reader.sample_start[self.indices]),
                          "drop_last": False}

    def start_epoch(self, epoch: int | None = None) -> None:
        if epoch is not None and (type(epoch) is not int or epoch < 0):
            raise ValueError("epoch index must be nonnegative")
        self.epoch = self.epoch + 1 if epoch is None else epoch
        self.permutation = self.indices.copy()
        if self.shuffle:
            self._rng.shuffle(self.permutation)
        self.cursor = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.cursor >= self.size:
            return None
        stop = min(self.cursor + self.batch_size, self.size)
        result = self.reader.get_batch(self.permutation[self.cursor:stop], self.preprocessing, split=self.split)
        self.cursor = stop
        return result

    def get_iterator(self) -> Iterator[dict[str, np.ndarray]]:
        while (batch := self.get_next()) is not None:
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": "fullwindow_loader_v1", "identity": copy.deepcopy(self._identity),
                "permutation": self.permutation.copy(), "cursor": self.cursor, "epoch": self.epoch,
                "rng_state": copy.deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "fullwindow_loader_v1" or state.get("identity") != self._identity:
            raise ValueError("loader state has another full-window sampling or recipe identity")
        order = np.asarray(state["permutation"])
        cursor, epoch = state["cursor"], state["epoch"]
        if order.dtype.kind not in "iu" or order.shape != self.indices.shape or not np.array_equal(np.sort(order), self.indices):
            raise ValueError("restore order is not the complete bound split")
        if not self.shuffle and not np.array_equal(order, self.indices):
            raise ValueError("validation restore order must remain canonical")
        if type(cursor) is not int or not 0 <= cursor <= self.size or (cursor != self.size and cursor % self.batch_size):
            raise ValueError("restore cursor is not a committed batch boundary")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("restore epoch must be nonnegative")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self.permutation, self.cursor, self.epoch, self._rng = order.copy(), cursor, epoch, rng
