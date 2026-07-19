#!/usr/bin/env python3
"""Aggregate NDR-Net horizon 3/6/12 metrics across independent seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import h5py
import numpy as np


HORIZONS = (3, 6, 12)
METRICS = ("mae", "rmse", "mape_pct")
EXPECTED_PARAMS = {
    "Alameda": 1_582_308,
    "ContraCosta": 1_558_308,
    "Orange": 2_032_548,
}
FORMAL_CONFIG = {
    "variant": "s0g",
    "epochs": 30,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "lambda_gate": 1.0,
    "lambda_spec": 0.3,
    "num_workers": 0,
    "device": "cuda",
    "strict_causal_cache": True,
}


def parse_args() -> argparse.Namespace:
    env_root = os.environ.get("NDR_OUT_ROOT")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(env_root) if env_root else None,
                        help="Root containing <Region>/s0g_seed*/summary.json")
    parser.add_argument("--region", default=os.environ.get("NDR_REGION", "Alameda"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    parser.add_argument("--method", default="NDR-Net")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-summary", action="store_true",
        help=("Permit summaries without the new provenance contract. Intended only "
              "for displaying preliminary historical runs, not formal results."),
    )
    parser.add_argument(
        "--skip-file-hash-verification", action="store_true",
        help="Skip checking that current cache/graph files still match recorded SHA256 values.",
    )
    args = parser.parse_args()
    if args.root is None:
        parser.error("--root is required (or set NDR_OUT_ROOT)")
    return args


def selected_metrics(summary: dict, horizon: int) -> dict:
    selected = summary.get("selected_horizons") or {}
    if str(horizon) in selected:
        return selected[str(horizon)]
    for row in summary.get("per_horizon", []):
        if int(row["horizon"]) == horizon:
            return row
    raise KeyError(f"summary does not contain horizon {horizon} metrics")


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_provenance(provenance: dict) -> str:
    return json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _seed_independent_config(config: dict) -> str:
    comparable = {key: value for key, value in config.items() if key != "seed"}
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def validate_summary(summary: dict, *, path: Path, region: str, seed: int,
                     allow_legacy: bool = False) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate one seed artifact and return its per-horizon count signatures."""
    if summary.get("variant") != "s0g" or int(summary.get("seed", -1)) != seed:
        raise ValueError(f"unexpected variant/seed in {path}")
    if int(summary.get("summary_schema_version", -1)) != 2 and not allow_legacy:
        raise ValueError(f"unsupported or missing formal summary schema in {path}")
    if summary.get("region") != region and not allow_legacy:
        raise ValueError(f"missing or incorrect region provenance in {path}")
    if summary.get("protocol") != "event" and not allow_legacy:
        raise ValueError(f"missing or incorrect event protocol in {path}")
    for key in ("all", "best_val"):
        value = float(summary.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"non-finite {key} in {path}")

    rows = summary.get("per_horizon")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError(f"expected exactly 12 horizon rows in {path}")
    try:
        by_horizon = {int(row["horizon"]): row for row in rows}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed horizon rows in {path}") from exc
    if set(by_horizon) != set(range(1, 13)) or len(by_horizon) != len(rows):
        raise ValueError(f"horizons must be unique and exactly 1..12 in {path}")

    n_valid, n_mape = [], []
    for horizon in range(1, 13):
        row = by_horizon[horizon]
        for metric in METRICS:
            value = float(row.get(metric, float("nan")))
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric} at horizon {horizon} in {path}")
        nv = int(row.get("n_valid", 0))
        nm = int(row.get("n_mape", 0))
        if nv <= 0 or nm <= 0 or nm > nv:
            raise ValueError(
                f"invalid n_valid/n_mape at horizon {horizon} in {path}: {nv}/{nm}"
            )
        n_valid.append(nv)
        n_mape.append(nm)

    selected = summary.get("selected_horizons")
    if not isinstance(selected, dict):
        raise ValueError(f"missing selected_horizons in {path}")
    for horizon in HORIZONS:
        selected_row = selected.get(str(horizon))
        if not isinstance(selected_row, dict):
            raise ValueError(f"missing selected horizon {horizon} in {path}")
        canonical = by_horizon[horizon]
        for field in (*METRICS, "n_valid", "n_mape"):
            if selected_row.get(field) != canonical.get(field):
                raise ValueError(
                    f"selected_horizons disagrees with per_horizon for H{horizon} "
                    f"field {field} in {path}"
                )

    if not allow_legacy:
        provenance = summary.get("provenance")
        config = summary.get("training_config")
        if not isinstance(provenance, dict) or not isinstance(config, dict):
            raise ValueError(f"missing formal provenance/training_config in {path}")
        if provenance.get("protocol") != "event" or provenance.get("region") != region:
            raise ValueError(f"provenance protocol/region mismatch in {path}")
        if not bool(provenance.get("strict_causal_cache")):
            raise ValueError(f"formal summary did not enforce strict causal cache: {path}")
        cache = provenance.get("cache") or {}
        if cache.get("imputation_method") != "causal_forward_fill":
            raise ValueError(f"non-causal cache provenance in {path}")
        if not math.isclose(float(cache.get("baseline_train_frac", -1)), 0.70,
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"unexpected baseline_train_frac in {path}")
        traffic_id = cache.get("traffic_cache_id")
        if not traffic_id or cache.get("samples_traffic_cache_id") != traffic_id:
            raise ValueError(f"traffic/samples cache identity mismatch in {path}")
        traffic_attrs = cache.get("traffic_attrs") or {}
        samples_attrs = cache.get("samples_attrs") or {}
        source_files = provenance.get("source_files") or {}
        builder_identity = source_files.get("scripts/build_full_county_cache.py") or {}
        current_builder_hash = builder_identity.get("sha256")
        recorded_builder_hashes = {
            traffic_attrs.get("builder_source_sha256"),
            samples_attrs.get("traffic_builder_source_sha256"),
            samples_attrs.get("samples_builder_source_sha256"),
        }
        if (not current_builder_hash or len(recorded_builder_hashes) != 1
                or current_builder_hash not in recorded_builder_hashes):
            raise ValueError(
                f"cache builder source hash does not match the recorded training source in {path}"
            )
        if config.get("region") != region or config.get("variant") != "s0g":
            raise ValueError(f"training configuration mismatch in {path}")
        if int(config.get("seed", -1)) != seed:
            raise ValueError(f"training configuration seed mismatch in {path}")
        for key, expected in FORMAL_CONFIG.items():
            actual = config.get(key)
            if isinstance(expected, float):
                try:
                    matches = math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"unexpected formal config {key}={actual!r}; expected {expected!r} in {path}"
                )

    return tuple(n_valid), tuple(n_mape)


def expected_cache_counts(provenance: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Recompute the exact test observation counts from the recorded cache files."""
    files = provenance["data_files"]
    samples_path = Path(files["samples"]["path"])
    traffic_path = Path(files["traffic"]["path"])
    with h5py.File(samples_path, "r") as f:
        starts = f["sample_start"][...]
        split = f["split"][...]
        T_p = int(f.attrs["T_p"])
    if T_p != 12:
        raise ValueError(f"formal NDR summary requires T_p=12, found {T_p}")
    test_starts = starts[split == 2].astype(np.int64)
    if test_starts.size == 0:
        raise ValueError("test cache contains no samples")
    if np.unique(test_starts).size != test_starts.size:
        raise ValueError("test cache still contains duplicate sample_start values")
    with h5py.File(traffic_path, "r") as f:
        mask = f["flow_mask"][:, :, 0]
        flow = f["flow_series_imputed"][:, :, 0]
    valid_counts, mape_counts = [], []
    for step in range(1, T_p + 1):
        valid = mask[test_starts + step].astype(bool)
        target = flow[test_starts + step]
        if not np.isfinite(target[valid]).all():
            raise ValueError(f"cache contains non-finite valid targets at horizon {step}")
        valid_counts.append(int(valid.sum()))
        mape_counts.append(int((valid & (np.abs(target) > 1e-6)).sum()))
    return tuple(valid_counts), tuple(mape_counts)


def verify_recorded_data_files(provenance: dict) -> None:
    for label, identity in provenance["data_files"].items():
        path = Path(identity["path"])
        if not path.is_file():
            raise FileNotFoundError(f"recorded {label} file no longer exists: {path}")
        if path.stat().st_size != int(identity["size_bytes"]):
            raise ValueError(f"recorded {label} file size changed: {path}")
        actual = _file_sha256(path)
        if actual != identity["sha256"]:
            raise ValueError(f"recorded {label} file SHA256 changed: {path}")


def main() -> None:
    args = parse_args()
    if not args.allow_legacy_summary and args.seeds != [42, 1, 2]:
        raise ValueError(
            f"formal NDR aggregation requires seeds [42, 1, 2], got {args.seeds}"
        )
    output_dir = args.output_dir or (args.root / args.region)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    loaded = []
    count_signatures = []
    provenance_signatures = []
    config_signatures = []
    for seed in args.seeds:
        path = args.root / args.region / f"s0g_seed{seed}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        count_signatures.append(validate_summary(
            summary, path=path, region=args.region, seed=seed,
            allow_legacy=args.allow_legacy_summary,
        ))
        expected_params = EXPECTED_PARAMS.get(args.region)
        if expected_params is None:
            raise ValueError(f"unsupported region: {args.region}")
        if int(summary.get("params", -1)) != expected_params:
            raise ValueError(f"unexpected NDR-Net parameter count in {path}")
        if not args.allow_legacy_summary:
            provenance_signatures.append(_canonical_provenance(summary["provenance"]))
            config_signatures.append(_seed_independent_config(summary["training_config"]))

        row = {"Data": args.region, "Method": args.method, "Seed": seed}
        for horizon in HORIZONS:
            metrics = selected_metrics(summary, horizon)
            for metric in METRICS:
                value = float(metrics[metric])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {metric} at horizon {horizon} in {path}")
                row[f"Horizon {horizon} {metric}"] = value
        seed_rows.append(row)
        loaded.append((seed, path, summary))

    if len(set(count_signatures)) != 1:
        details = {seed: signature for seed, signature in zip(args.seeds, count_signatures)}
        raise ValueError(f"n_valid/n_mape differs across seeds: {details}")
    if not args.allow_legacy_summary:
        if len(set(provenance_signatures)) != 1:
            raise ValueError("seed summaries were not produced from identical data/source provenance")
        if len(set(config_signatures)) != 1:
            raise ValueError("seed summaries used different training configurations (excluding seed)")
        provenance = loaded[0][2]["provenance"]
        if not args.skip_file_hash_verification:
            verify_recorded_data_files(provenance)
        expected_counts = expected_cache_counts(provenance)
        if count_signatures[0] != expected_counts:
            raise ValueError(
                "summary observation counts do not match the recorded test cache: "
                f"summary={count_signatures[0]} cache={expected_counts}"
            )

    mean_row = {"Data": args.region, "Method": args.method, "Seed": "mean"}
    std_row = {"Data": args.region, "Method": args.method, "Seed": "std"}
    for horizon in HORIZONS:
        for metric in METRICS:
            key = f"Horizon {horizon} {metric}"
            values = [float(row[key]) for row in seed_rows]
            mean_row[key] = statistics.mean(values)
            std_row[key] = statistics.stdev(values) if len(values) > 1 else 0.0

    columns = ["Data", "Method", "Seed"] + [
        f"Horizon {horizon} {metric}"
        for horizon in HORIZONS for metric in METRICS
    ]
    detailed_csv = output_dir / "ndrnet_horizon_metrics_by_seed.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(seed_rows + [mean_row, std_row])

    table_columns = ["Data", "Method"] + [
        label
        for horizon in HORIZONS
        for label in (
            f"Horizon {horizon} MAE",
            f"Horizon {horizon} RMSE",
            f"Horizon {horizon} MAPE(%)",
        )
    ]
    table_row = {"Data": args.region, "Method": args.method}
    for horizon in HORIZONS:
        table_row[f"Horizon {horizon} MAE"] = f"{mean_row[f'Horizon {horizon} mae']:.2f}"
        table_row[f"Horizon {horizon} RMSE"] = f"{mean_row[f'Horizon {horizon} rmse']:.2f}"
        table_row[f"Horizon {horizon} MAPE(%)"] = f"{mean_row[f'Horizon {horizon} mape_pct']:.2f}"

    table_csv = output_dir / "ndrnet_horizon_table.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table_columns)
        writer.writeheader()
        writer.writerow(table_row)

    report = {
        "data": args.region,
        "method": args.method,
        "seeds": args.seeds,
        "aggregation": "arithmetic mean and sample standard deviation across seed metrics",
        "horizons": {
            str(h): {
                metric: {
                    "mean": mean_row[f"Horizon {h} {metric}"],
                    "std": std_row[f"Horizon {h} {metric}"],
                }
                for metric in METRICS
            }
            for h in HORIZONS
        },
        "sources": [str(path) for _, path, _ in loaded],
        "provenance": None if args.allow_legacy_summary else loaded[0][2]["provenance"],
        "n_valid_by_horizon": list(count_signatures[0][0]),
        "n_mape_by_horizon": list(count_signatures[0][1]),
        "validation_mode": (
            "legacy-explicit-override" if args.allow_legacy_summary
            else "strict-12-horizon-provenance-and-cache-count-validation"
        ),
    }

    average_report = {}
    for metric in METRICS:
        per_seed_averages = []
        for _, _, summary in loaded:
            values = [
                float(selected_metrics(summary, horizon)[metric])
                for horizon in range(1, 13)
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite average-horizon {metric}")
            per_seed_averages.append(statistics.mean(values))
        average_report[metric] = {
            "mean": statistics.mean(per_seed_averages),
            "std": statistics.stdev(per_seed_averages)
            if len(per_seed_averages) > 1 else 0.0,
        }
    report["average_across_12_horizons"] = average_report

    json_path = output_dir / "ndrnet_horizon_metrics.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{args.region} {args.method}, seeds " + "/".join(map(str, args.seeds)))
    print(f"{'Position':<16}{'MAE':<28}{'RMSE':<28}{'MAPE(%)':<28}")
    for horizon in HORIZONS:
        cells = []
        for metric in METRICS:
            values = report["horizons"][str(horizon)][metric]
            cells.append(f"{values['mean']:.4f} +/- {values['std']:.4f}")
        print(f"{f'Horizon {horizon}':<16}{cells[0]:<28}{cells[1]:<28}{cells[2]:<28}")
    average_cells = [
        f"{average_report[metric]['mean']:.4f} +/- {average_report[metric]['std']:.4f}"
        for metric in METRICS
    ]
    print(f"{'Average':<16}{average_cells[0]:<28}{average_cells[1]:<28}{average_cells[2]:<28}")
    print()

    print("Data,Method,H3 MAE,H3 RMSE,H3 MAPE(%),H6 MAE,H6 RMSE,H6 MAPE(%),"
          "H12 MAE,H12 RMSE,H12 MAPE(%)")
    print(",".join(str(table_row[column]) for column in table_columns))
    print(f"detailed: {detailed_csv}")
    print(f"table:    {table_csv}")
    print(f"json:     {json_path}")


if __name__ == "__main__":
    main()
