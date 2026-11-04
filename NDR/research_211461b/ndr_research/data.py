"""Train/validation-only v3 event data, with resumable numeric batch order."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dist_net.data import RegionData
from dist_net.protocol import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION
from scripts.audit_protocol_v3_cache import EXPECTED_FORMAL_COUNTS
from scripts.protocol_v3_event_extension_data import (
    V3EventBatchLoader,
    WeeklyClimatology,
    fit_weekly_climatology,
    prepare_ndr_batch,
)

FORMAL_BUNDLE_ID = "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4"
FORMAL_BUNDLE_DIGEST = "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


class ResearchEventLoader(V3EventBatchLoader):
    """Use the original adapter, but reject test indices and retain a cursor."""

    def __init__(self, *, split: str, **kwargs: Any) -> None:
        if split not in {"train", "val"}:
            raise ValueError("research loaders accept only train or val")
        if kwargs.get("include_incident") or kwargs.get("sensor_support") is not None:
            raise ValueError("research loaders do not consume incident or sensor support")
        indices = np.asarray(kwargs["indices"], dtype=np.int64)
        region = kwargs["region_data"]
        if region.include_analysis_labels:
            raise ValueError("analysis labels must remain disabled")
        split_code = 0 if split == "train" else 1
        if np.any(indices < 0) or np.any(indices >= len(region.split)):
            raise ValueError("loader indices are outside the cache")
        if np.any(region.split[indices] != split_code):
            raise ValueError("loader indices cross the declared train/val split")
        super().__init__(**kwargs)
        self.split = split

    def start_epoch(self) -> None:
        self.shuffle()
        self.current_ind = 0

    def next_batch(self) -> dict[str, Any]:
        if self.current_ind >= self.num_batch:
            raise StopIteration
        start = self.current_ind * self.bs
        selected = self._order[start:min(start + self.bs, self.size)]
        batch = self._build_batch(selected)
        self.current_ind += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "indices_sha256": array_digest(self.indices),
            "batch_size": self.bs,
            "drop_last": self.drop_last,
            "shuffle": self.should_shuffle,
            "order": self._order.copy(),
            "rng": copy.deepcopy(self._rng.bit_generator.state),
            "next_batch": self.current_ind,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "split": self.split,
            "indices_sha256": array_digest(self.indices),
            "batch_size": self.bs,
            "drop_last": self.drop_last,
            "shuffle": self.should_shuffle,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("loader restore identity mismatch")
        order = np.asarray(state["order"], dtype=np.int64)
        cursor = state["next_batch"]
        if not np.array_equal(np.sort(order), np.sort(self.indices)):
            raise ValueError("loader restore order is not the exact sample permutation")
        if type(cursor) is not int or not 0 <= cursor <= self.num_batch:
            raise ValueError("loader restore batch cursor is invalid")
        self._order = order.copy()
        self._rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.current_ind = cursor


def prepare_research_batch(
    batch: Mapping[str, Any], statistics: WeeklyClimatology,
) -> dict[str, np.ndarray]:
    if {"affected_mask", "analysis_labels"}.intersection(batch):
        raise ValueError("analysis labels entered a research batch")
    prepared = prepare_ndr_batch(batch, statistics)
    history_mask = np.asarray(batch["x_mask"])
    if not np.logical_or(history_mask == 0, history_mask == 1).all():
        raise ValueError("history observation mask must contain zero or one")
    prepared["history_mask"] = history_mask[..., 0].transpose(0, 2, 1).astype(np.bool_)
    prepared["sample_start"] = np.asarray(batch["sample_start"], dtype=np.int64)
    return prepared


def validate_manifest_binding(region: Any, manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("status") != "verified"
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("cache_bundle_id") != FORMAL_BUNDLE_ID
        or manifest.get("cache_bundle_digest") != FORMAL_BUNDLE_DIGEST
    ):
        raise ValueError("formal v3 cache manifest identity mismatch")
    reports = manifest["regions"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "bundle_id": FORMAL_BUNDLE_ID,
        "regions": {name: reports[name]["cache_bundle_digest"] for name in sorted(reports)},
    }
    if canonical_digest(payload) != FORMAL_BUNDLE_DIGEST:
        raise ValueError("cache manifest region identities do not match the bundle digest")
    declared = reports[region.region_name]
    actual = region.result_metadata()["cache_provenance"]
    if (
        region.bundle_id != FORMAL_BUNDLE_ID
        or region.cache_bundle_digest != declared["cache_bundle_digest"]
        or actual["artifact_sha256"] != declared["cache_provenance"]["artifact_sha256"]
    ):
        raise ValueError("actual cache artifact hashes differ from the formal manifest")
    expected = EXPECTED_FORMAL_COUNTS[region.region_name]
    counts = {name: int(np.count_nonzero(region.split == code))
              for name, code in (("train", 0), ("val", 1), ("test", 2))}
    if len(region.sample_start) != expected["total"] or counts != expected["split"]:
        raise ValueError("formal event sample counts differ from the locked protocol")


@dataclass
class ResearchData:
    region: Any
    statistics: WeeklyClimatology
    train: ResearchEventLoader
    val: ResearchEventLoader
    identity: dict[str, Any]


def load_research_data(
    *, region_name: str, data_dir: Path | str, graph_dir: Path | str,
    batch_size: int = 16, seed: int = 42, preload: bool = True,
) -> ResearchData:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "cache_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    region = RegionData(
        region_name, data_dir, graph_dir, lazy=True,
        required_protocol=PROTOCOL_VERSION, include_analysis_labels=False,
        require_rel_feat=False, verify_artifact_hashes=True,
    )
    validate_manifest_binding(region, manifest)
    if preload:
        flow, mask = region._slice_traffic(0, region.val_end)
        region.flow_series = np.asarray(flow, dtype=np.float32)
        region.flow_mask = np.asarray(mask, dtype=np.bool_)
        if region._lazy_traffic_f is not None:
            region._lazy_traffic_f.close()
            region._lazy_traffic_f = None
    statistics = fit_weekly_climatology(region)
    loaders = {}
    for name, code in (("train", 0), ("val", 1)):
        loaders[name] = ResearchEventLoader(
            split=name, region_data=region,
            indices=np.flatnonzero(region.split == code), batch_size=batch_size,
            flow_mean=statistics.flow_mean, flow_std=statistics.flow_std,
            sensor_support=None, include_incident=False,
            shuffle=name == "train", drop_last=False, seed=seed,
        )
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_bundle_id": FORMAL_BUNDLE_ID,
        "cache_bundle_digest": FORMAL_BUNDLE_DIGEST,
        "region": region_name,
        "region_cache_digest": region.cache_bundle_digest,
        "artifact_sha256": dict(region.artifact_sha256),
        "manifest_sha256": sha256_file(manifest_path),
        "train_end": region.train_end, "val_end": region.val_end,
        "fit_range": [0, region.train_end],
        "fit_positions": "unique_raw_observed_training_timestamps",
        "weekly_climatology_sha256": array_digest(statistics.baseline),
        "node_residual_std_sha256": array_digest(statistics.residual_std_by_node),
        "flow_mean": statistics.flow_mean, "flow_std": statistics.flow_std,
        "residual_std": statistics.residual_std,
        "valid_training_points": statistics.valid_training_points,
        "split_counts": {name: loader.size for name, loader in loaders.items()},
        "split_origins_sha256": {
            name: array_digest(region.sample_start[loader.indices])
            for name, loader in loaders.items()
        },
        "test_loader_constructed": False,
        "training_analysis_labels": False,
        "preloaded_until_exclusive": region.val_end if preload else None,
    }
    return ResearchData(region, statistics, loaders["train"], loaders["val"], identity)
