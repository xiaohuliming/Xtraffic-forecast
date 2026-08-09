#!/usr/bin/env python3
"""Verify eager and lazy v3 extension loading produce identical batches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dist_net.data import RegionData
from scripts.protocol_v3_event_extension_data import (
    V3EventBatchLoader,
    fit_weekly_climatology,
    load_aligned_sensor_support,
)
from scripts.train_protocol_v3_event_extension import REGION_TO_SUPPORT


REGIONS = ("Alameda", "ContraCosta", "Orange")
_MISSING = object()


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8") + b"\0")
            _update_digest(digest, value[key])
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
        return
    raise TypeError(f"unsupported batch value for loading audit: {type(value)!r}")


def batch_sha256(batch: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, batch)
    return digest.hexdigest()


def compare_loader_streams(
    *,
    split: str,
    lazy_loader: Any,
    eager_loader: Any,
    batch_limit: int,
) -> dict[str, Any]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if type(batch_limit) is not int or batch_limit <= 0:
        raise ValueError("batch_limit must be positive")
    should_shuffle = split == "train"
    if bool(lazy_loader.should_shuffle) != should_shuffle or bool(
        eager_loader.should_shuffle
    ) != should_shuffle:
        raise RuntimeError(f"{split} loader shuffle contract is invalid")
    if should_shuffle:
        lazy_loader.shuffle()
        eager_loader.shuffle()

    digest = hashlib.sha256()
    count = 0
    pairs = zip_longest(
        lazy_loader.get_iterator(), eager_loader.get_iterator(), fillvalue=_MISSING
    )
    for count, (lazy_batch, eager_batch) in enumerate(pairs, start=1):
        if count > batch_limit:
            count -= 1
            break
        if lazy_batch is _MISSING or eager_batch is _MISSING:
            raise RuntimeError(f"{split} loader length differs at batch {count}")
        lazy_sha = batch_sha256(lazy_batch)
        eager_sha = batch_sha256(eager_batch)
        if lazy_sha != eager_sha:
            raise RuntimeError(f"{split} eager/lazy mismatch at batch {count}")
        digest.update(bytes.fromhex(lazy_sha))
    if count == 0:
        raise RuntimeError(f"{split} loading audit produced no batches")
    return {
        "status": "exact_match",
        "split": split,
        "batch_count": count,
        "stream_sha256": digest.hexdigest(),
        "train_shuffle_applied": should_shuffle,
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _compare_statistics(lazy: Any, eager: Any) -> dict[str, Any]:
    scalar_fields = (
        "flow_mean",
        "flow_std",
        "residual_std",
        "valid_training_points",
    )
    for field in scalar_fields:
        if getattr(lazy, field) != getattr(eager, field):
            raise RuntimeError(f"eager/lazy statistic differs: {field}")
    arrays = {
        "baseline": (lazy.baseline, eager.baseline),
        "residual_std_by_node": (
            lazy.residual_std_by_node,
            eager.residual_std_by_node,
        ),
    }
    result = {}
    for field, (lazy_value, eager_value) in arrays.items():
        if not np.array_equal(lazy_value, eager_value, equal_nan=True):
            raise RuntimeError(f"eager/lazy statistic differs: {field}")
        result[f"{field}_sha256"] = _array_sha256(lazy_value)
    return result


def audit_region(
    *,
    region: str,
    data_dir: Path,
    graph_dir: Path,
    support_root: Path,
    node_order: Path,
    batch_limit: int,
) -> dict[str, Any]:
    lazy_data = RegionData(
        region,
        data_dir,
        graph_dir,
        lazy=True,
        require_rel_feat=True,
        include_analysis_labels=False,
        verify_artifact_hashes=True,
    )
    eager_data = RegionData(
        region,
        data_dir,
        graph_dir,
        lazy=False,
        require_rel_feat=True,
        include_analysis_labels=False,
        verify_artifact_hashes=True,
    )
    if lazy_data.flow_series is not None or eager_data.flow_series is None:
        raise RuntimeError("RegionData eager/lazy traffic residency is invalid")
    if lazy_data.cache_bundle_digest != eager_data.cache_bundle_digest:
        raise RuntimeError("RegionData eager/lazy cache identity differs")
    support_dir = support_root / REGION_TO_SUPPORT[region]
    support = load_aligned_sensor_support(
        sensors_path=support_dir / "sensors.csv",
        node_order_path=node_order,
        region_idx=lazy_data.region_idx,
    )
    lazy_stats = fit_weekly_climatology(lazy_data)
    eager_stats = fit_weekly_climatology(eager_data)
    statistics = _compare_statistics(lazy_stats, eager_stats)

    loaders: dict[str, dict[str, V3EventBatchLoader]] = {"lazy": {}, "eager": {}}
    for split, code in (("train", 0), ("val", 1), ("test", 2)):
        for label, rdata in (("lazy", lazy_data), ("eager", eager_data)):
            loaders[label][split] = V3EventBatchLoader(
                region_data=rdata,
                indices=np.flatnonzero(rdata.split == code),
                batch_size=24,
                flow_mean=lazy_stats.flow_mean,
                flow_std=lazy_stats.flow_std,
                sensor_support=support,
                include_incident=True,
                shuffle=split == "train",
                drop_last=False,
                seed=42,
            )
    streams = {
        split: compare_loader_streams(
            split=split,
            lazy_loader=loaders["lazy"][split],
            eager_loader=loaders["eager"][split],
            batch_limit=batch_limit,
        )
        for split in ("train", "val", "test")
    }
    report = {
        "status": "exact_match",
        "region": region,
        "cache_bundle_digest": lazy_data.cache_bundle_digest,
        "eager_traffic_bytes": int(
            eager_data.flow_series.nbytes + eager_data.flow_mask.nbytes
        ),
        "statistics": statistics,
        "streams": streams,
    }
    del loaders, eager_data, lazy_data
    gc.collect()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--batch-limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_limit <= 0:
        raise ValueError("batch-limit must be positive")
    regions = [
        audit_region(
            region=region,
            data_dir=args.data_dir,
            graph_dir=args.graph_dir,
            support_root=args.support_root,
            node_order=args.node_order,
            batch_limit=args.batch_limit,
        )
        for region in REGIONS
    ]
    payload = {
        "schema_version": 1,
        "status": "EAGER_LAZY_EQUIVALENCE_OK",
        "regions": regions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
