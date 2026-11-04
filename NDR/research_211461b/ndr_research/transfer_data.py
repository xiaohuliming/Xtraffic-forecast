"""Whitelisted lazy v3 data access and explicitly separated transfer preprocessing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import stat
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np


PROTOCOL = "protocol_v3_noleak_dedup"
FORMAL_BUNDLE_ID = "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4"
FORMAL_BUNDLE_DIGEST = "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5"
REGIONS = ("Alameda", "ContraCosta", "Orange")
GRAPH_KEYS = {"Alameda": "alameda", "ContraCosta": "contra_costa", "Orange": "orange"}
SPLITS = {"train": 0, "val": 1, "test": 2}
FORMAL_COUNTS = {
    "Alameda": {"total": 45760, "train": 32001, "val": 6954, "test": 6805},
    "ContraCosta": {"total": 23337, "train": 16414, "val": 3472, "test": 3451},
    "Orange": {"total": 60015, "train": 42372, "val": 9098, "test": 8545},
}
FIVE_MINUTES_NS = 300_000_000_000
DAY_NS = 86_400_000_000_000


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _fingerprint(path: Path) -> tuple[int, ...]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"cache artifact must be a regular nonsymlink file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_hash(path: Path) -> tuple[str, tuple[int, ...]]:
    before = _fingerprint(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if before != _fingerprint(path):
        raise RuntimeError("cache artifact changed while hashing")
    return digest.hexdigest(), before


@dataclass(frozen=True)
class FrozenBundleContract:
    """Default authority is the exact formal release; tests may supply a fixture release."""

    bundle_id: str = FORMAL_BUNDLE_ID
    bundle_digest: str = FORMAL_BUNDLE_DIGEST
    counts: Mapping[str, Mapping[str, int]] = field(default_factory=lambda: copy.deepcopy(FORMAL_COUNTS))

    @property
    def formal(self) -> bool:
        return self.bundle_id == FORMAL_BUNDLE_ID and self.bundle_digest == FORMAL_BUNDLE_DIGEST and self.counts == FORMAL_COUNTS


@dataclass(frozen=True)
class TestEvaluationGate:
    authorization_id: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.authorization_id, str) or not self.authorization_id.strip():
            raise ValueError("test evaluation requires an explicit authorization identity")
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("test evaluation gate requires a lowercase SHA-256 checkpoint identity")


class TransferRegionReader:
    """Read only origin/split metadata, channel-zero traffic, calendars and static roads."""

    allowed_datasets = {
        "samples": frozenset(("sample_start", "split")),
        "traffic": frozenset(("times_ns", "region_idx", "flow_series_imputed", "flow_mask")),
    }

    def __init__(
        self,
        region: str,
        data_dir: Path | str,
        graph_dir: Path | str,
        *,
        role: str = "source",
        bundle_manifest: Path | str | None = None,
        contract: FrozenBundleContract | None = None,
    ) -> None:
        if region not in REGIONS or role not in ("source", "target"):
            raise ValueError("transfer reader needs a canonical region and source/target role")
        self.region = self.region_name = region
        self._role = role
        self.contract = contract or FrozenBundleContract()
        data_dir, graph_dir = Path(data_dir), Path(graph_dir)
        manifest_path = Path(bundle_manifest) if bundle_manifest is not None else data_dir / "cache_bundle_manifest.json"
        manifest_hash, manifest_fingerprint = _file_hash(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_fingerprint != _fingerprint(manifest_path):
            raise RuntimeError("bundle manifest changed while interpreting identities")
        if (manifest.get("status"), manifest.get("protocol_version"), manifest.get("cache_schema_version")) != ("verified", PROTOCOL, 3):
            raise ValueError("transfer requires an audited schema-three v3 cache bundle")
        if manifest.get("bundle_id") != self.contract.bundle_id or manifest.get("cache_bundle_digest") != self.contract.bundle_digest:
            raise ValueError("cache bundle does not match the frozen release authority")
        regions = manifest.get("regions", {})
        digest_payload = {
            "protocol_version": PROTOCOL, "cache_schema_version": 3,
            "bundle_id": self.contract.bundle_id,
            "regions": {name: report["cache_bundle_digest"] for name, report in sorted(regions.items())},
        }
        if _json_hash(digest_payload) != self.contract.bundle_digest:
            raise ValueError("bundle region index does not reproduce its frozen digest")
        if region not in regions or region not in self.contract.counts:
            raise ValueError("requested region is missing from the frozen authority")
        report = regions[region]
        provenance = report["cache_provenance"]
        region_payload = {
            "protocol_version": PROTOCOL, "cache_schema_version": 3,
            "bundle_id": self.contract.bundle_id,
            "traffic_cache_id": provenance["traffic_cache_id"],
            "samples_cache_id": provenance["samples_cache_id"],
            "relation_cache_id": provenance["relation_cache_id"],
            "artifact_sha256": provenance["artifact_sha256"],
        }
        if _json_hash(region_payload) != report["cache_bundle_digest"]:
            raise ValueError("region artifact identities do not reproduce the audited region digest")
        self.paths = {
            "samples": data_dir / f"{region}_samples.h5",
            "traffic": data_dir / f"{region}_traffic.h5",
            "relation": data_dir / f"{region}_event_rel_feat.npz",
            "graph": graph_dir / f"{GRAPH_KEYS[region]}_sparse_adj.npz",
        }
        self._fingerprints = {}
        hashes = {}
        for artifact_role, path in self.paths.items():
            digest, fingerprint = _file_hash(path)
            if digest != provenance["artifact_sha256"].get(artifact_role):
                raise ValueError(f"{artifact_role} bytes do not match the audited formal artifact")
            hashes[artifact_role] = digest
            self._fingerprints[artifact_role] = fingerprint
        self._audit_counts: Counter[str] = Counter()
        self._recent_access: deque[dict[str, Any]] = deque(maxlen=128)
        self._access_digest = "0" * 64
        self._calibration_grant: tuple[int, str] | None = None
        self._test_grant: TestEvaluationGate | None = None
        self._source_flow_cache: np.ndarray | None = None
        self._source_mask_cache: np.ndarray | None = None
        self._source_loading: dict[str, Any] | None = None
        for artifact_role in self.paths:
            self._record("hash_bytes_only", artifact_role, "*", purpose="identity_verification")
        with h5py.File(self.paths["samples"], "r") as handle:
            self._check_attrs(handle, "samples")
            if handle.attrs.get("samples_cache_id") != provenance["samples_cache_id"]:
                raise ValueError("sample cache identity disagrees with audited manifest")
            sample_metadata = {key: int(handle.attrs[key]) for key in ("N", "T", "T_h", "T_p", "train_end", "val_end")}
            self._record("metadata", "samples", "attributes", purpose="schema_and_bounds")
            self.sample_start = self._dataset(handle, "samples", "sample_start", Ellipsis, "origins")
            self.split = self._dataset(handle, "samples", "split", Ellipsis, "split")
        self.N, self.T = sample_metadata["N"], sample_metadata["T"]
        self.T_h, self.T_p = sample_metadata["T_h"], sample_metadata["T_p"]
        self.train_end, self.val_end = sample_metadata["train_end"], sample_metadata["val_end"]
        if self.T_h != 12 or self.T_p != 12 or not (0 < self.train_end < self.val_end < self.T):
            raise ValueError("transfer requires valid v3 twelve-step history and horizon bounds")
        with h5py.File(self.paths["traffic"], "r") as handle:
            self._check_attrs(handle, "traffic")
            for key in ("N", "T", "train_end", "val_end"):
                if int(handle.attrs[key]) != sample_metadata[key]:
                    raise ValueError("sample and traffic metadata disagree")
            if handle.attrs.get("traffic_cache_id") != provenance["traffic_cache_id"]:
                raise ValueError("traffic cache identity disagrees with audited manifest")
            for name in ("flow_series_imputed", "flow_mask"):
                if handle[name].shape != (self.T, self.N, 3):
                    raise ValueError("formal traffic/mask dimensions disagree with sample metadata")
            if handle["flow_mask"].dtype != np.dtype(bool):
                raise ValueError("flow mask must contain original boolean observation validity")
            self.times_ns = self._dataset(handle, "traffic", "times_ns", Ellipsis, "calendar_metadata")
            self.region_idx = self._dataset(handle, "traffic", "region_idx", Ellipsis, "node_identity")
        if self.times_ns.shape != (self.T,) or self.times_ns.dtype.kind not in "iu" or not np.all(np.diff(self.times_ns) == FIVE_MINUTES_NS):
            raise ValueError("calendar must be a strictly increasing five-minute grid")
        if self.region_idx.shape != (self.N,) or self.region_idx.dtype.kind not in "iu" or np.unique(self.region_idx).size != self.N:
            raise ValueError("traffic node identity must be a unique integer vector")
        self.dow = ((self.times_ns // DAY_NS + 3) % 7).astype(np.int64)
        self.tod = ((self.times_ns // FIVE_MINUTES_NS) % 288).astype(np.int64)
        self._validate_origins(report)
        with np.load(self.paths["graph"], allow_pickle=False) as graph:
            graph_nodes = graph["region_idx"]
            self.edge_index = graph["edge_index"]
        self._record("dataset_read", "graph", "region_idx", purpose="node_identity")
        self._record("dataset_read", "graph", "edge_index", purpose="static_roads")
        if not np.array_equal(graph_nodes, self.region_idx):
            raise ValueError("static graph does not exactly match traffic node order")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2 or self.edge_index.dtype.kind not in "iu":
            raise ValueError("static roads must be a (2,E) integer edge array")
        if self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= self.N):
            raise ValueError("static roads reference a node outside the bound region")
        self.edge_index = self.edge_index.astype(np.int64, copy=False)
        for values in (self.sample_start, self.split, self.times_ns, self.region_idx, self.dow, self.tod, self.edge_index):
            values.setflags(write=False)
        self.identity = {
            "region": region, "role": role, "protocol_version": PROTOCOL,
            "cache_schema_version": 3, "bundle_id": self.contract.bundle_id,
            "bundle_digest": self.contract.bundle_digest,
            "manifest_sha256": manifest_hash, "region_digest": report["cache_bundle_digest"],
            "artifact_sha256": hashes, "formal_release": self.contract.formal,
            "node_order_sha256": _array_hash(self.region_idx),
            "origin_sha256": _array_hash(self.sample_start),
            "split_sha256": _array_hash(self.split),
            "counts": dict(self.contract.counts[region]),
            "N": self.N, "T": self.T, "history_steps": self.T_h, "horizon": self.T_p,
            "train_end": self.train_end, "val_end": self.val_end,
            "active_event_isolation_evidence": "exact_samples_artifact_hash_of_verified_v3_release",
            "whitelist": {key: sorted(value) for key, value in self.allowed_datasets.items()},
        }
        self.identity["identity_digest"] = _json_hash(self.identity)

    @property
    def role(self) -> str:
        return self._role

    def _record(self, kind: str, artifact: str, dataset: str, *, purpose: str, **extra: Any) -> None:
        event = {"kind": kind, "artifact": artifact, "dataset": dataset, "purpose": purpose, **extra}
        self._audit_counts[f"{kind}:{artifact}:{dataset}:{purpose}"] += 1
        self._access_digest = _json_hash({"previous": self._access_digest, "event": event})
        self._recent_access.append(event)

    def access_audit(self) -> dict[str, Any]:
        return {
            "region": self.region, "role": self.role,
            "counts": dict(sorted(self._audit_counts.items())),
            "rolling_access_sha256": self._access_digest,
            "recent_access": list(self._recent_access),
            "forbidden_dataset_values_read": False,
        }

    def _dataset(self, handle: h5py.File, artifact: str, key: str, selection: Any, purpose: str) -> np.ndarray:
        if key not in self.allowed_datasets.get(artifact, ()):
            raise PermissionError("dataset is outside the transfer whitelist")
        result = np.asarray(handle[key][selection])
        self._record("dataset_read", artifact, key, purpose=purpose)
        return result

    def _check_attrs(self, handle: h5py.File, artifact: str) -> None:
        expected = {"protocol_version": PROTOCOL, "cache_schema_version": 3, "cache_role": artifact, "cache_mode": "formal", "bundle_id": self.contract.bundle_id}
        if any(handle.attrs.get(key) != value for key, value in expected.items()) or bool(handle.attrs.get("git_dirty", True)):
            raise ValueError("artifact is not a clean formal schema-three v3 cache")

    def _validate_origins(self, report: Mapping[str, Any]) -> None:
        if self.sample_start.ndim != 1 or self.sample_start.dtype.kind not in "iu" or self.split.shape != self.sample_start.shape:
            raise ValueError("origins and split must be aligned integer vectors")
        if not self.sample_start.size or np.unique(self.sample_start).size != self.sample_start.size or not np.all(self.sample_start[1:] > self.sample_start[:-1]):
            raise ValueError("forecast origins must be strictly increasing and unique")
        if not np.isin(self.split, (0, 1, 2)).all() or np.any(self.sample_start < self.T_h - 1) or np.any(self.sample_start >= self.T - self.T_p):
            raise ValueError("sample split or history origin is invalid")
        start, stop = self.sample_start + 1, self.sample_start + 1 + self.T_p
        expected = np.full(self.split.shape, -1, dtype=np.int8)
        expected[(stop <= self.train_end)] = 0
        expected[(start >= self.train_end) & (stop <= self.val_end)] = 1
        expected[(start >= self.val_end) & (stop <= self.T)] = 2
        if not np.array_equal(expected, self.split):
            raise ValueError("forecast targets cross train/validation/test bounds")
        counts = {name: int((self.split == code).sum()) for name, code in SPLITS.items()}
        counts["total"] = int(self.sample_start.size)
        if counts != self.contract.counts[self.region]:
            raise ValueError("sample counts do not match the frozen authority")
        if report.get("sample_count") != counts["total"] or report.get("unique_forecast_origins") != counts["total"] or report.get("split_counts") != {key: counts[key] for key in SPLITS}:
            raise ValueError("audited count metadata disagree with observed origin vectors")

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError("unknown split")
        result = np.flatnonzero(self.split == SPLITS[split])
        result.setflags(write=False)
        return result

    indices_for_split = indices

    def _read_flow(self, start: int, stop: int, *, purpose: str) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= start < stop <= self.T:
            raise ValueError("traffic slice is outside the timeline")
        if purpose == "source_fit" and self.role != "source":
            raise PermissionError("target readers cannot fit source training statistics")
        if purpose.startswith(("window:", "history:")):
            access_kind, split = purpose.split(":", 1)
            required_length = self.T_h + (self.T_p if access_kind == "window" else 0)
            if split not in SPLITS or stop - start != required_length:
                raise PermissionError("prediction reads must be complete bounded history/target windows")
            target_start = start + self.T_h
            target_stop = target_start + self.T_p
            permitted = {
                "train": target_stop <= self.train_end,
                "val": target_start >= self.train_end and target_stop <= self.val_end,
                "test": target_start >= self.val_end and target_stop <= self.T,
            }[split]
            if not permitted or (self.role == "target" and split == "train"):
                raise PermissionError("prediction read is outside the permitted role and split")
            if access_kind == "window" and (split == "test" or (self.role == "target" and split == "val")) and self._test_grant is None:
                raise PermissionError("heldout labels/test traffic read has no active explicit evaluation gate")
        if purpose == "historical_calibration":
            if self.role != "target" or self._calibration_grant is None or stop > self._calibration_grant[0]:
                raise PermissionError("target prefix read has no matching fixed historical calibration grant")
        if purpose not in ("source_fit", "historical_calibration") and not purpose.startswith(("window:", "history:")):
            raise PermissionError("unrecognized traffic access purpose")
        if purpose == "source_fit" and stop > self.train_end:
            raise PermissionError("source fitting cannot cross the frozen training boundary")
        if _fingerprint(self.paths["traffic"]) != self._fingerprints["traffic"]:
            raise RuntimeError("traffic artifact changed after identity verification")
        if self._source_flow_cache is not None and stop <= self.val_end:
            self._record("memory_slice", "traffic", "channel_zero", purpose=purpose, start=start, stop=stop, channel=0)
            return self._source_flow_cache[start:stop], self._source_mask_cache[start:stop]
        with h5py.File(self.paths["traffic"], "r") as handle:
            selection = (slice(start, stop), slice(None), 0)
            flow = self._dataset(handle, "traffic", "flow_series_imputed", selection, purpose).astype(np.float32, copy=False)
            mask = self._dataset(handle, "traffic", "flow_mask", selection, purpose).astype(bool, copy=False)
        if _fingerprint(self.paths["traffic"]) != self._fingerprints["traffic"]:
            raise RuntimeError("traffic artifact changed during the whitelisted read")
        self._record("flow_bounds", "traffic", "channel_zero", purpose=purpose, start=start, stop=stop, channel=0)
        if not np.isfinite(flow[mask]).all() or (flow[mask] < 0).any():
            raise FloatingPointError("raw observed flow must be finite and nonnegative")
        return np.where(mask, flow, np.float32(0)), mask

    def preload_source_train_val(self) -> dict[str, Any]:
        """Cache only source channel zero up to val_end after source-statistics fitting."""
        if self.role != "source":
            raise PermissionError("target readers cannot preload a historical/training prefix")
        if self._source_loading is not None:
            return copy.deepcopy(self._source_loading)
        if _fingerprint(self.paths["traffic"]) != self._fingerprints["traffic"]:
            raise RuntimeError("traffic artifact changed before source preloading")
        selection = (slice(0, self.val_end), slice(None), 0)
        with h5py.File(self.paths["traffic"], "r") as handle:
            flow = self._dataset(handle, "traffic", "flow_series_imputed", selection, "source_train_val_preload").astype(np.float32, copy=False)
            mask = self._dataset(handle, "traffic", "flow_mask", selection, "source_train_val_preload").astype(bool, copy=False)
        if _fingerprint(self.paths["traffic"]) != self._fingerprints["traffic"]:
            raise RuntimeError("traffic artifact changed during source preloading")
        if not np.isfinite(flow[mask]).all() or (flow[mask] < 0).any():
            raise FloatingPointError("preloaded source observations must be finite and nonnegative")
        flow = np.where(mask, flow, np.float32(0))
        flow.setflags(write=False)
        mask.setflags(write=False)
        self._source_flow_cache, self._source_mask_cache = flow, mask
        self._source_loading = {
            "role": "source", "region": self.region,
            "loading_policy": "source_eager_channel_zero_prefix_to_val_end",
            "prefix_start": 0, "prefix_end": self.val_end, "channels": [0],
            "bytes": int(flow.nbytes + mask.nbytes),
            "flow_sha256": _array_hash(flow), "mask_sha256": _array_hash(mask),
            "reader_identity_digest": self.identity["identity_digest"],
            "numeric_identity_changed": False,
        }
        self._record("flow_bounds", "traffic", "channel_zero", purpose="source_train_val_preload", start=0, stop=self.val_end, channel=0)
        return copy.deepcopy(self._source_loading)

    def iter_source_training(self, block_steps: int = 4096) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        if self.role != "source":
            raise PermissionError("target readers never expose a training-prefix fit iterator")
        if type(block_steps) is not int or block_steps <= 0:
            raise ValueError("source fitting block length must be positive")
        for start in range(0, self.train_end, block_steps):
            stop = min(start + block_steps, self.train_end)
            flow, mask = self._read_flow(start, stop, purpose="source_fit")
            yield start, stop, flow, mask

    def get_batch(
        self,
        indices: Sequence[int] | np.ndarray,
        preprocessing: "TransferPreprocessing",
        *,
        split: str,
        evaluation_gate: TestEvaluationGate | None = None,
    ) -> dict[str, np.ndarray]:
        if split not in SPLITS:
            raise ValueError("unknown prediction split")
        if self.role == "target" and split == "train":
            raise PermissionError("target readers cannot expose supervised training batches")
        needs_gate = split == "test" or (self.role == "target" and split == "val")
        if needs_gate and not isinstance(evaluation_gate, TestEvaluationGate):
            raise PermissionError("heldout labels/test access requires an explicit checkpoint-bound evaluation gate")
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu" or selected.size == 0 or np.unique(selected).size != selected.size:
            raise ValueError("batch selection must be a nonempty unique integer vector")
        if np.any(selected < 0) or np.any(selected >= self.sample_start.size) or not np.all(self.split[selected] == SPLITS[split]):
            raise ValueError("batch origin selection does not belong to the declared split")
        preprocessing.validate_reader(self)
        if preprocessing.mode == "historical_calibrated_transfer" and np.any(self.sample_start[selected] + 1 < preprocessing.identity["prefix_end"]):
            raise PermissionError("historical calibration includes observations unavailable at the forecast origin")
        if needs_gate:
            self._record("evaluation_gate", "samples", "split", purpose=split, authorization_id=evaluation_gate.authorization_id, checkpoint_sha256=evaluation_gate.checkpoint_sha256)
        rows = []
        for index in selected:
            anchor = int(self.sample_start[index])
            low, high = anchor - self.T_h + 1, anchor + 1 + self.T_p
            self._test_grant = evaluation_gate if needs_gate else None
            try:
                flow, mask = self._read_flow(low, high, purpose=f"window:{split}")
            finally:
                self._test_grant = None
            hist_idx = np.arange(low, anchor + 1)
            target_idx = np.arange(anchor + 1, high)
            x_base = preprocessing.lookup(self.dow[hist_idx], self.tod[hist_idx], self.N)
            y_base = preprocessing.lookup(self.dow[target_idx], self.tod[target_idx], self.N)
            rows.append({
                "flow_hist": flow[:self.T_h].T,
                "history_mask": mask[:self.T_h].T,
                "target": flow[self.T_h:].T,
                "target_mask": mask[self.T_h:].T,
                "x_baseline": x_base.T,
                "y_baseline": y_base.T,
                "time_feat": np.stack((self.tod[hist_idx] / 288, self.dow[hist_idx] / 7), -1).astype(np.float32),
            })
        batch = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        batch["node_scale"] = preprocessing.scales(self.N)
        batch["sample_start"] = self.sample_start[selected].copy()
        batch["edge_index"] = self.edge_index.copy()
        return batch

    def get_history_batch(
        self,
        indices: Sequence[int] | np.ndarray,
        preprocessing: "TransferPreprocessing",
        *,
        split: str = "val",
    ) -> dict[str, np.ndarray]:
        """Engineering forecast inputs only; never read or return future traffic labels."""
        if split not in SPLITS or (self.role == "target" and split == "train"):
            raise PermissionError("history engineering access must respect reader role and split")
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu" or not selected.size or np.unique(selected).size != selected.size:
            raise ValueError("history selection must be a nonempty unique integer vector")
        if np.any(selected < 0) or np.any(selected >= self.sample_start.size) or not np.all(self.split[selected] == SPLITS[split]):
            raise ValueError("history selection does not belong to its declared split")
        preprocessing.validate_reader(self)
        if preprocessing.mode == "historical_calibrated_transfer" and np.any(self.sample_start[selected] + 1 < preprocessing.identity["prefix_end"]):
            raise PermissionError("calibration includes observations unavailable at the forecast origin")
        rows = []
        for index in selected:
            anchor = int(self.sample_start[index])
            low, high = anchor - self.T_h + 1, anchor + 1
            flow, mask = self._read_flow(low, high, purpose=f"history:{split}")
            hist_idx = np.arange(low, high)
            future_idx = np.arange(high, high + self.T_p)
            rows.append({
                "flow_hist": flow.T,
                "history_mask": mask.T,
                "x_baseline": preprocessing.lookup(self.dow[hist_idx], self.tod[hist_idx], self.N).T,
                "y_baseline": preprocessing.lookup(self.dow[future_idx], self.tod[future_idx], self.N).T,
                "time_feat": np.stack((self.tod[hist_idx] / 288, self.dow[hist_idx] / 7), -1).astype(np.float32),
            })
        batch = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        batch["node_scale"] = preprocessing.scales(self.N)
        batch["sample_start"] = self.sample_start[selected].copy()
        batch["edge_index"] = self.edge_index.copy()
        return batch


@dataclass(frozen=True)
class TransferPreprocessing:
    mode: str
    template: np.ndarray
    residual_scale: float | np.ndarray
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        template = np.asarray(self.template, dtype=np.float32).copy()
        scales = np.asarray(self.residual_scale, dtype=np.float32)
        if self.mode not in ("source_only", "historical_calibrated_transfer"):
            raise ValueError("unknown transfer preprocessing mode")
        if template.shape[:2] != (7, 288) or template.ndim not in (2, 3) or not np.isfinite(template).all():
            raise ValueError("transfer climatology must be a finite weekly template")
        if not np.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("transfer residual scales must be finite and positive")
        if self.mode == "source_only" and (template.ndim != 2 or scales.ndim != 0):
            raise ValueError("strict source-only preprocessing must contain no target-sized statistics")
        if self.mode == "historical_calibrated_transfer" and (template.ndim != 3 or scales.shape != (template.shape[-1],)):
            raise ValueError("calibrated preprocessing requires aligned per-node historical statistics")
        identity = copy.deepcopy(dict(self.identity))
        payload = {key: value for key, value in identity.items() if key != "identity_digest"}
        if identity.get("mode") != self.mode or identity.get("template_sha256") != _array_hash(template) or identity.get("scale_sha256") != _array_hash(scales):
            raise ValueError("preprocessing identity does not bind its numeric arrays")
        if identity.get("identity_digest") != _json_hash(payload):
            raise ValueError("preprocessing identity digest is inconsistent")
        template.setflags(write=False)
        scales.setflags(write=False)
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "residual_scale", float(scales) if scales.ndim == 0 else scales)
        object.__setattr__(self, "identity", identity)

    def validate_reader(self, reader: TransferRegionReader) -> None:
        if self.identity["bundle_id"] != reader.contract.bundle_id or self.identity["bundle_digest"] != reader.contract.bundle_digest:
            raise ValueError("preprocessing and reader are from different cache bundles")
        if self.mode == "historical_calibrated_transfer":
            if reader.role != "target" or self.identity["target_identity_digest"] != reader.identity["identity_digest"]:
                raise ValueError("historical calibration cannot be reused as strict/source or another target preprocessing")
        elif reader.role == "source":
            source = self.identity["source_identities"].get(reader.region)
            if source != reader.identity["identity_digest"]:
                raise ValueError("source preprocessing was not fitted on this exact source reader identity")
        elif reader.region in self.identity["source_regions"]:
            raise ValueError("heldout target cannot be a region used to fit source preprocessing")

    def lookup(self, dow: np.ndarray, tod: np.ndarray, nodes: int) -> np.ndarray:
        values = self.template[dow, tod]
        if self.mode == "source_only":
            return np.broadcast_to(values[:, None], (len(dow), nodes)).astype(np.float32, copy=True)
        if self.template.shape[-1] != nodes:
            raise ValueError("historical calibration node dimension disagrees with target")
        return np.array(values, copy=True)

    def scales(self, nodes: int) -> np.ndarray:
        if self.mode == "source_only":
            return np.full(nodes, self.residual_scale, dtype=np.float32)
        if np.asarray(self.residual_scale).shape != (nodes,):
            raise ValueError("calibrated scale dimension disagrees with target")
        return np.array(self.residual_scale, dtype=np.float32, copy=True)

    def to_state(self) -> dict[str, Any]:
        return {"mode": self.mode, "template": self.template.copy(), "residual_scale": copy.deepcopy(self.residual_scale), "identity": copy.deepcopy(dict(self.identity))}

    state_dict = to_state

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> "TransferPreprocessing":
        return cls(mode=value["mode"], template=value["template"], residual_scale=value["residual_scale"], identity=value["identity"])


def _weekly_node_medians(flow: np.ndarray, mask: np.ndarray, dow: np.ndarray, tod: np.ndarray) -> np.ndarray:
    count = int(mask.sum())
    if count == 0:
        raise ValueError("statistics prefix has no valid raw observations")
    fallback = float(np.sum(flow, where=mask, dtype=np.float64) / count)
    template = np.full((7, 288, flow.shape[1]), fallback, dtype=np.float32)
    for day in range(7):
        for slot in range(288):
            selected = (dow == day) & (tod == slot)
            if not selected.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(np.where(mask[selected], flow[selected], np.nan), axis=0)
            template[day, slot] = np.where(np.isfinite(median), median, np.float32(fallback))
    return template


def fit_source_only_pool(readers: Mapping[str, TransferRegionReader] | Sequence[TransferRegionReader]) -> TransferPreprocessing:
    sources = list(readers.values()) if isinstance(readers, Mapping) else list(readers)
    if len(sources) != 2 or len({reader.region for reader in sources}) != 2 or any(reader.role != "source" for reader in sources):
        raise PermissionError("strict source-only fitting requires exactly two distinct source-role readers")
    sources.sort(key=lambda reader: reader.region)
    if len({reader.contract.bundle_id for reader in sources}) != 1 or len({reader.contract.bundle_digest for reader in sources}) != 1:
        raise ValueError("source readers must share the same frozen bundle authority")
    regional_templates, ranges = [], {}
    for reader in sources:
        flow = np.empty((reader.train_end, reader.N), dtype=np.float32)
        mask = np.empty(flow.shape, dtype=bool)
        for start, stop, values, observed in reader.iter_source_training():
            flow[start:stop], mask[start:stop] = values, observed
        medians = _weekly_node_medians(flow, mask, reader.dow[:reader.train_end], reader.tod[:reader.train_end])
        regional_templates.append(np.median(medians, axis=-1))
        ranges[reader.region] = {
            "start": 0, "end": reader.train_end, "unique_raw_timestamps": reader.train_end,
            "first_time_ns": int(reader.times_ns[0]), "last_time_ns": int(reader.times_ns[reader.train_end - 1]),
            "valid_points": int(mask.sum()), "node_count": reader.N,
        }
        del flow, mask, medians
    template = np.mean(np.stack(regional_templates).astype(np.float64), axis=0).astype(np.float32)
    means, squares = [], []
    for reader in sources:
        total = square = np.float64(0)
        count = 0
        for start, stop, flow, mask in reader.iter_source_training():
            baseline = template[reader.dow[start:stop], reader.tod[start:stop]][:, None]
            residual = (flow.astype(np.float64) - baseline)[mask]
            total += residual.sum(dtype=np.float64)
            square += np.square(residual).sum(dtype=np.float64)
            count += residual.size
        means.append(float(total / count))
        squares.append(float(square / count))
    variance = max(float(np.mean(squares) - np.mean(means) ** 2), 0.0)
    scale = np.float32(math.sqrt(variance))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = np.float32(1.0)
    identity = {
        "mode": "source_only", "bundle_id": sources[0].contract.bundle_id,
        "bundle_digest": sources[0].contract.bundle_digest,
        "source_regions": [reader.region for reader in sources],
        "source_identities": {reader.region: reader.identity["identity_digest"] for reader in sources},
        "fit_ranges": ranges,
        "fit_values": "unique_raw_source_train_timestamps_with_original_observation_mask",
        "template_estimator": "weekly_per_node_median_then_regional_node_median_then_equal_region_mean",
        "scale_estimator": "equal_region_raw_residual_mixture_standard_deviation_against_pooled_template",
        "empty_weekly_cell_fallback": "source_region_raw_training_mean",
        "target_statistics_used": False,
        "template_sha256": _array_hash(template), "scale_sha256": _array_hash(np.asarray(scale)),
        "access_audits": {reader.region: reader.access_audit() for reader in sources},
    }
    identity["identity_digest"] = _json_hash(identity)
    return TransferPreprocessing("source_only", template, float(scale), identity)


def fit_historical_calibration(
    reader: TransferRegionReader,
    *,
    prefix_end: int,
    authorization_id: str,
) -> TransferPreprocessing:
    if reader.role != "target" or type(prefix_end) is not int or not 0 < prefix_end <= reader.train_end:
        raise PermissionError("historical calibration needs a fixed target prefix within its training boundary")
    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise PermissionError("historical calibration requires explicit separate-mode authorization")
    flow = np.empty((prefix_end, reader.N), dtype=np.float32)
    mask = np.empty(flow.shape, dtype=bool)
    reader._calibration_grant = (prefix_end, authorization_id)
    reader._record("calibration_gate", "traffic", "channel_zero", purpose="historical_calibration", prefix_end=prefix_end, authorization_id=authorization_id)
    try:
        for start in range(0, prefix_end, 4096):
            stop = min(start + 4096, prefix_end)
            flow[start:stop], mask[start:stop] = reader._read_flow(start, stop, purpose="historical_calibration")
    finally:
        reader._calibration_grant = None
    template = _weekly_node_medians(flow, mask, reader.dow[:prefix_end], reader.tod[:prefix_end])
    residual = flow.astype(np.float64) - template[reader.dow[:prefix_end], reader.tod[:prefix_end]]
    scales = np.ones(reader.N, dtype=np.float32)
    for node in range(reader.N):
        values = residual[:, node][mask[:, node]]
        if values.size:
            deviation = values.std(dtype=np.float64)
            if np.isfinite(deviation) and deviation > 1e-6:
                scales[node] = deviation
    identity = {
        "mode": "historical_calibrated_transfer", "bundle_id": reader.contract.bundle_id,
        "bundle_digest": reader.contract.bundle_digest, "target_region": reader.region,
        "target_identity_digest": reader.identity["identity_digest"],
        "authorization_id": authorization_id, "prefix_start": 0, "prefix_end": prefix_end,
        "fit_values": "explicit_fixed_historical_target_prefix_original_observations",
        "target_gradient_updates": False, "target_statistics_used": True,
        "template_sha256": _array_hash(template), "scale_sha256": _array_hash(scales),
        "valid_points": int(mask.sum()), "access_audit": reader.access_audit(),
    }
    identity["identity_digest"] = _json_hash(identity)
    return TransferPreprocessing("historical_calibrated_transfer", template, scales, identity)


class TransferBatchLoader:
    """Region-bucketed batches with checkpointable exact permutation/cursor/RNG state."""

    def __init__(
        self,
        reader: TransferRegionReader,
        preprocessing: TransferPreprocessing,
        *,
        split: str,
        batch_size: int,
        seed: int,
        shuffle: bool | None = None,
        evaluation_gate: TestEvaluationGate | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch size must be positive")
        if reader.role == "target" and split == "train":
            raise PermissionError("target readers cannot become training loaders")
        if (split == "test" or (reader.role == "target" and split == "val")) and not isinstance(evaluation_gate, TestEvaluationGate):
            raise PermissionError("heldout label/test loaders require a checkpoint-bound evaluation gate")
        preprocessing.validate_reader(reader)
        self.reader, self.preprocessing, self.split = reader, preprocessing, split
        self.batch_size = self.bs = batch_size
        self.evaluation_gate = evaluation_gate
        self.shuffle = split == "train" if shuffle is None else bool(shuffle)
        if self.shuffle and split != "train":
            raise ValueError("validation/test order must be stable")
        self.indices = reader.indices(split)
        self.size = int(self.indices.size)
        if not self.size:
            raise ValueError("loader split is empty")
        self.num_batch = (self.size + batch_size - 1) // batch_size
        self.origin_selection_digest = _array_hash(reader.sample_start[self.indices])
        self._rng = np.random.default_rng(seed)
        self.permutation = self.indices.copy()
        self.cursor = 0
        self.epoch = 0
        self._identity = {
            "reader": reader.identity["identity_digest"], "preprocessing": preprocessing.identity["identity_digest"],
            "split": split, "batch_size": batch_size, "shuffle": self.shuffle,
            "origin_selection_digest": self.origin_selection_digest,
            "test_gate": None if evaluation_gate is None else {"authorization_id": evaluation_gate.authorization_id, "checkpoint_sha256": evaluation_gate.checkpoint_sha256},
        }

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
        batch = self.reader.get_batch(self.permutation[self.cursor:stop], self.preprocessing, split=self.split, evaluation_gate=self.evaluation_gate)
        self.cursor = stop
        return batch

    def get_iterator(self) -> Iterator[dict[str, np.ndarray]]:
        while (batch := self.get_next()) is not None:
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "identity": copy.deepcopy(self._identity),
            "epoch": self.epoch, "cursor": self.cursor,
            "permutation": self.permutation.copy(),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or state.get("identity") != self._identity:
            raise ValueError("loader checkpoint belongs to another data/recipe/order identity")
        permutation = np.asarray(state["permutation"])
        cursor, epoch = state["cursor"], state["epoch"]
        if permutation.dtype.kind not in "iu" or permutation.shape != self.indices.shape or not np.array_equal(np.sort(permutation), self.indices):
            raise ValueError("loader checkpoint permutation is not the complete bound split")
        if not self.shuffle and not np.array_equal(permutation, self.indices):
            raise ValueError("validation/test checkpoint permutation must preserve fixed order")
        if type(cursor) is not int or not 0 <= cursor <= self.size or (cursor != self.size and cursor % self.batch_size):
            raise ValueError("loader checkpoint cursor is not a committed batch boundary")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("loader checkpoint epoch is invalid")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        self.permutation, self.cursor, self.epoch, self._rng = permutation.astype(np.int64, copy=True), cursor, epoch, rng


@dataclass(frozen=True)
class TransferSources:
    readers: Mapping[str, TransferRegionReader]
    train_loaders: Mapping[str, TransferBatchLoader]
    val_loaders: Mapping[str, TransferBatchLoader]
    preprocessing: TransferPreprocessing
    identity: Mapping[str, Any]


def load_transfer_sources(
    source_regions: Sequence[str],
    heldout_region: str,
    data_dir: Path | str,
    graph_dir: Path | str,
    batch_size: int,
    seed: int,
    *,
    contract: FrozenBundleContract | None = None,
) -> TransferSources:
    if heldout_region not in REGIONS or len(source_regions) != 2 or set(source_regions) != set(REGIONS) - {heldout_region}:
        raise ValueError("leave-one-region-out requires exactly the other two source regions")
    readers = {region: TransferRegionReader(region, data_dir, graph_dir, role="source", contract=contract) for region in sorted(source_regions)}
    preprocessing = fit_source_only_pool(readers)
    train = {region: TransferBatchLoader(reader, preprocessing, split="train", batch_size=batch_size, seed=seed + index) for index, (region, reader) in enumerate(readers.items())}
    validation = {region: TransferBatchLoader(reader, preprocessing, split="val", batch_size=batch_size, seed=seed, shuffle=False) for region, reader in readers.items()}
    identity = {
        "mode": "source_only", "source_regions": sorted(source_regions), "heldout_region": heldout_region,
        "heldout_reader_instantiated": False, "target_statistics_used": False,
        "preprocessing_identity": preprocessing.identity["identity_digest"],
        "readers": {region: reader.identity for region, reader in readers.items()},
        "origin_selection_digest": {region: {"train": train[region].origin_selection_digest, "val": validation[region].origin_selection_digest} for region in readers},
    }
    identity["identity_digest"] = _json_hash(identity)
    return TransferSources(readers, train, validation, preprocessing, identity)
