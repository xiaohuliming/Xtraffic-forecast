#!/usr/bin/env python3
"""Strictly validate and fingerprint a formal XTraffic protocol-v3 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dist_net.data import RegionData
from dist_net.protocol import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_REGION_DATA_DIR,
    PROTOCOL_VERSION,
)


REGIONS = ("Alameda", "ContraCosta", "Orange")
EXPECTED_FORMAL_COUNTS = {
    "Alameda": {
        "total": 45_760,
        "split": {"train": 32_001, "val": 6_954, "test": 6_805},
        "duplicates_removed": 17_021,
        "mixed_active_removed": 8,
    },
    "ContraCosta": {
        "total": 23_337,
        "split": {"train": 16_414, "val": 3_472, "test": 3_451},
        "duplicates_removed": 3_884,
        "mixed_active_removed": 4,
    },
    "Orange": {
        "total": 60_015,
        "split": {"train": 42_372, "val": 9_098, "test": 8_545},
        "duplicates_removed": 45_464,
        "mixed_active_removed": 11,
    },
}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeError(
            f"formal cache count mismatch for {label}: "
            f"{actual!r} != {expected!r}"
        )


def audit_cache_bundle(
    data_dir: Path,
    graph_dir: Path,
    regions: tuple[str, ...] | list[str] = REGIONS,
    *,
    enforce_expected_counts: bool = True,
    write_manifest: bool = True,
) -> dict[str, object]:
    """Validate every region and return a single bundle release manifest."""
    region_reports: dict[str, object] = {}
    bundle_ids: set[str] = set()

    for region in regions:
        cache = RegionData(
            region,
            data_dir,
            graph_dir,
            lazy=True,
            require_rel_feat=True,
            include_analysis_labels=False,
            verify_artifact_hashes=True,
        )
        bundle_ids.add(cache.bundle_id)
        samples_path = data_dir / f"{region}_samples.h5"
        with h5py.File(samples_path, "r") as samples:
            attributes = {
                "duplicates_removed": int(
                    samples.attrs["n_duplicate_sample_start_dropped"]
                ),
                "mixed_active_removed": int(
                    samples.attrs[
                        "n_mixed_active_event_samples_dropped"
                    ]
                ),
                "mixed_events_removed": int(
                    samples.attrs["n_mixed_events_dropped"]
                ),
                "max_active_seen": int(samples.attrs["max_active_seen"]),
                "truncated_samples": int(
                    samples.attrs["n_truncated_active"]
                ),
            }

        split_counts = {
            "train": int(np.count_nonzero(cache.split == 0)),
            "val": int(np.count_nonzero(cache.split == 1)),
            "test": int(np.count_nonzero(cache.split == 2)),
        }
        report = {
            "sample_count": int(len(cache.sample_start)),
            "split_counts": split_counts,
            "unique_forecast_origins": int(
                np.unique(cache.sample_start).size
            ),
            "event_count": int(len(cache.event_start_idx)),
            "max_active_incidents": int(cache.n_active_incidents.max()),
            "max_packed_incidents": int(cache.n_packed_incidents.max()),
            "cache_bundle_digest": cache.cache_bundle_digest,
            "cache_provenance": cache.result_metadata()[
                "cache_provenance"
            ],
            **attributes,
        }
        if enforce_expected_counts:
            expected = EXPECTED_FORMAL_COUNTS[region]
            _expect_equal(
                f"{region}.total", report["sample_count"], expected["total"]
            )
            _expect_equal(
                f"{region}.split", split_counts, expected["split"]
            )
            _expect_equal(
                f"{region}.duplicates_removed",
                attributes["duplicates_removed"],
                expected["duplicates_removed"],
            )
            _expect_equal(
                f"{region}.mixed_active_removed",
                attributes["mixed_active_removed"],
                expected["mixed_active_removed"],
            )
        region_reports[region] = report

    if len(bundle_ids) != 1:
        raise RuntimeError(
            "all formal regions must share one bundle_id; got "
            + ", ".join(sorted(bundle_ids))
        )
    bundle_id = next(iter(bundle_ids))
    digest_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "regions": {
            region: region_reports[region]["cache_bundle_digest"]
            for region in sorted(region_reports)
        },
    }
    bundle_digest = hashlib.sha256(
        json.dumps(
            digest_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest: dict[str, object] = {
        "status": "verified",
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "cache_bundle_id": bundle_id,
        "cache_bundle_digest": bundle_digest,
        "regions": region_reports,
    }
    if write_manifest:
        _atomic_json(data_dir / "cache_bundle_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_REGION_DATA_DIR)
    parser.add_argument(
        "--graph-dir", type=Path, default=Path("outputs/region_graphs")
    )
    parser.add_argument("--regions", nargs="+", default=list(REGIONS))
    parser.add_argument(
        "--skip-expected-counts",
        action="store_true",
        help="Only for synthetic or local smoke fixtures.",
    )
    parser.add_argument("--no-write-manifest", action="store_true")
    args = parser.parse_args()

    manifest = audit_cache_bundle(
        args.data_dir,
        args.graph_dir,
        regions=tuple(args.regions),
        enforce_expected_counts=not args.skip_expected_counts,
        write_manifest=not args.no_write_manifest,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "bundle_id": manifest["bundle_id"],
                "cache_bundle_digest": manifest["cache_bundle_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
