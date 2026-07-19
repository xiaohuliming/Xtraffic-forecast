#!/usr/bin/env python3
"""Small subprocess used by protocol runner integration tests."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "success",
            "bad_identity",
            "nan_metric",
            "zero_valid",
            "wrong_valid_count",
            "wrong_prediction_metric",
            "bad_prediction_schema",
            "missing_artifact",
            "fail",
        ],
        default="success",
    )
    parser.add_argument("--artifacts", default="ckpt_best.pt,test_predictions.npz")
    args = parser.parse_args()

    print(f"dummy trainer mode={args.mode}", flush=True)
    if args.mode == "fail":
        print("intentional failure", flush=True)
        return 7

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "task_id": os.environ["XTRAFFIC_TASK_ID"],
        "region": os.environ["XTRAFFIC_REGION"],
        "model": os.environ["XTRAFFIC_MODEL"],
        "variant": os.environ["XTRAFFIC_VARIANT"],
        "seed": int(os.environ["XTRAFFIC_SEED"]),
        "protocol_version": os.environ["XTRAFFIC_PROTOCOL_VERSION"],
        "sampling_protocol": os.environ["XTRAFFIC_SAMPLING_PROTOCOL"],
        "cache_schema_version": int(os.environ["XTRAFFIC_CACHE_SCHEMA_VERSION"]),
        "cache_bundle_id": os.environ["XTRAFFIC_CACHE_BUNDLE_ID"],
        "cache_bundle_digest": os.environ["XTRAFFIC_CACHE_BUNDLE_DIGEST"],
        "code_sha": os.environ["XTRAFFIC_CODE_SHA"],
        "config_hash": os.environ["XTRAFFIC_CONFIG_HASH"],
        "seen_staging_root": os.environ["XTRAFFIC_STAGING_ROOT"],
        "all": 1.25,
        "affected": 1.50,
        "unaffected": 1.20,
        "valid_count": 120,
    }
    if args.mode == "bad_identity":
        summary["region"] = "WrongRegion"
    elif args.mode == "nan_metric":
        summary["all"] = math.nan
    elif args.mode == "zero_valid":
        summary["valid_count"] = 0
    elif args.mode == "wrong_valid_count":
        summary["valid_count"] = 119
    elif args.mode == "wrong_prediction_metric":
        summary["all"] = 9.0

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    artifacts = [name for name in args.artifacts.split(",") if name]
    if args.mode == "missing_artifact" and artifacts:
        artifacts.pop()
    for name in artifacts:
        path = args.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "test_predictions.npz":
            n_nodes = 120
            actual = np.full((1, 1, n_nodes), 10.0, dtype=np.float32)
            predicted = actual.copy()
            predicted[..., :20] += 1.5
            predicted[..., 20:] += 1.2
            payload = {
                "region_code": np.array(
                    [
                        {
                            "Alameda": 0,
                            "ContraCosta": 1,
                            "Orange": 2,
                        }[os.environ["XTRAFFIC_REGION"]]
                    ],
                    dtype=np.int64,
                ),
                "sample_start": np.array([100], dtype=np.int64),
                "region_node_idx": np.arange(n_nodes, dtype=np.int64),
                "pred_raw_flow": predicted,
                "actual_future_flow": actual,
                "y_mask_flow": np.ones_like(actual, dtype=np.bool_),
                "affected_mask": np.concatenate(
                    [
                        np.ones((1, 20), dtype=np.bool_),
                        np.zeros((1, 100), dtype=np.bool_),
                    ],
                    axis=1,
                ),
            }
            if args.mode == "bad_prediction_schema":
                payload.pop("y_mask_flow")
            np.savez_compressed(path, **payload)
        else:
            path.write_bytes(f"artifact:{name}\n".encode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
