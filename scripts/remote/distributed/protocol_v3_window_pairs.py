#!/usr/bin/env python3
"""Read-only window-level paired significance analysis for the verified
protocol-v3 host-stratified 45 matrix.

Designed to run on the result server against the published shards. It never
writes into the results tree: the only outputs are the JSON report and the
optional per-window diff dump given by --output/--dump-diffs.

Every consumed test_predictions.npz is SHA-256 verified against the read-only
composite index before loading, and the composite index itself must match the
published SHA-256 below. Pairing is by sample_start intersection; test math
mirrors scripts/significance_tests.py (paired t normal approximation,
Wilcoxon signed-rank normal approximation, sign-flip permutation on the mean)
but stays in float64.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import socket
from datetime import datetime, timezone
from math import erfc, sqrt
from pathlib import Path

import numpy as np

INDEX_PUBLISHED_SHA256 = (
    "8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe"
)
REGION_ORDER = ["Alameda", "ContraCosta", "Orange"]
SEED_ORDER = [42, 1, 2]
VARIANT_PAIRS = [
    ("deseason_s0a_vs_s0c", "staeformer_deseason", "s0a", "s0c"),
    ("deseason_s0f_vs_s0g", "staeformer_deseason", "s0f", "s0g"),
    ("icsf_base_vs_icsf", "staeformer_icsf", "base", "icsf"),
]
FULL_WINDOW_PAIRS = [
    ("fullwindow_gwn_vs_fdn", "graphwavenet", "fourier_dual_net"),
    ("fullwindow_gwn_vs_staeformer", "graphwavenet", "staeformer"),
    ("fullwindow_fdn_vs_staeformer", "fourier_dual_net", "staeformer"),
]
CHUNK_WINDOWS = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def per_window_sums(npz, mask_kind: str, affected=None):
    """Return (abs_sum, count) per window for the requested cell stratum."""
    pred = npz["pred_raw_flow"]
    actual = npz["actual_future_flow"]
    y_mask = npz["y_mask_flow"]
    sample_count = pred.shape[0]
    abs_sum = np.zeros(sample_count, dtype=np.float64)
    count = np.zeros(sample_count, dtype=np.float64)
    for lo in range(0, sample_count, CHUNK_WINDOWS):
        hi = min(lo + CHUNK_WINDOWS, sample_count)
        mask = y_mask[lo:hi].astype(np.float64)
        if mask_kind != "all":
            aff = affected[lo:hi][:, None, :].astype(np.float64)
            mask = mask * aff if mask_kind == "affected" else mask * (1.0 - aff)
        diff = np.abs(
            pred[lo:hi].astype(np.float64) - actual[lo:hi].astype(np.float64)
        )
        abs_sum[lo:hi] = (diff * mask).sum(axis=(1, 2))
        count[lo:hi] = mask.sum(axis=(1, 2))
    return abs_sum, count


def paired_tests(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int = 0):
    """Paired tests on d = a - b. Negative mean diff = A better."""
    d = a - b
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 8:
        return {"n": n}
    mean_d = float(d.mean())
    med_d = float(np.median(d))
    frac_better = float((d < 0).mean())

    se = d.std(ddof=1) / sqrt(n)
    t = mean_d / se if se > 0 else 0.0
    p_t = erfc(abs(t) / sqrt(2))

    dz = d[d != 0]
    nz = int(dz.size)
    ranks = np.argsort(np.argsort(np.abs(dz))) + 1.0
    w_pos = float(ranks[dz > 0].sum())
    mu = nz * (nz + 1) / 4.0
    sigma = sqrt(nz * (nz + 1) * (2 * nz + 1) / 24.0)
    z = (w_pos - mu) / sigma if sigma > 0 else 0.0
    p_w = erfc(abs(z) / sqrt(2))

    rng = np.random.default_rng(seed)
    obs = abs(mean_d)
    count, chunk = 0, 2000
    for _ in range(n_perm // chunk):
        signs = rng.integers(0, 2, size=(chunk, n)).astype(np.float64) * 2 - 1
        pm = signs @ d / n
        count += int((np.abs(pm) >= obs).sum())
    p_perm = (count + 1) / (n_perm + 1)

    return {"n": n, "mean_diff": mean_d, "median_diff": med_d,
            "frac_A_better": frac_better, "t": t, "p_t": p_t,
            "z_wilcoxon": z, "p_wilcoxon": p_w, "p_perm": p_perm}


def build_pair_specs(tasks_meta):
    by_key = {}
    for meta in tasks_meta:
        by_key[(meta["model"], meta["variant"], meta["region"],
                meta["seed"])] = meta
    specs = []

    def add(label, a_meta, b_meta):
        if a_meta["shard"] != b_meta["shard"]:
            raise RuntimeError(
                f"pair {label} crosses hosts: {a_meta['task_id']} on "
                f"{a_meta['shard']} vs {b_meta['task_id']} on {b_meta['shard']}"
            )
        specs.append({"label": label, "region": a_meta["region"],
                      "shard": a_meta["shard"], "seed": a_meta["seed"],
                      "a": a_meta, "b": b_meta})

    for label, model, a_var, b_var in VARIANT_PAIRS:
        regions = sorted({m["region"] for m in tasks_meta
                          if m["model"] == model}, key=REGION_ORDER.index)
        for region in regions:
            seeds = sorted(
                {m["seed"] for m in tasks_meta
                 if m["model"] == model and m["variant"] == a_var
                 and m["region"] == region}
                & {m["seed"] for m in tasks_meta
                   if m["model"] == model and m["variant"] == b_var
                   and m["region"] == region},
                key=SEED_ORDER.index)
            for seed in seeds:
                add(label, by_key[(model, a_var, region, seed)],
                    by_key[(model, b_var, region, seed)])

    for label, a_model, b_model in FULL_WINDOW_PAIRS:
        for region in REGION_ORDER:
            for a_meta in tasks_meta:
                if a_meta["model"] != a_model or a_meta["region"] != region:
                    continue
                for b_meta in tasks_meta:
                    if (b_meta["model"] == b_model
                            and b_meta["region"] == region
                            and b_meta["seed"] == a_meta["seed"]):
                        add(label, a_meta, b_meta)
    return specs


def analyze_pair(spec, shard_roots: dict, verified: dict, n_perm: int):
    out = {"label": spec["label"], "region": spec["region"],
           "shard": spec["shard"], "seed": spec["seed"],
           "a_task": spec["a"]["task_id"], "b_task": spec["b"]["task_id"]}
    per_model = {}
    for side in ("a", "b"):
        meta = spec[side]
        task_id = meta["task_id"]
        task_dir = Path(meta["summary"]["path"]).parent
        npz_path = (shard_roots[meta["shard"]] / task_dir
                    / "test_predictions.npz")
        expected = meta["artifacts"]["test_predictions.npz"]["sha256"]
        if task_id not in verified:
            actual_sha = sha256_file(npz_path)
            if actual_sha != expected:
                raise RuntimeError(
                    f"npz SHA-256 mismatch for {task_id}: {actual_sha}")
            verified[task_id] = actual_sha
        out[f"{side}_npz_sha256"] = verified[task_id]

        with np.load(npz_path, allow_pickle=False) as npz:
            sample_start = npz["sample_start"].astype(np.int64)
            affected = npz["affected_mask"].astype(bool)
            has_affected = bool(affected.any())
            strata = {}
            for kind in ("all", "unaffected", "affected"):
                if kind == "affected" and not has_affected:
                    strata[kind] = None
                    continue
                abs_sum, count = per_window_sums(
                    npz, kind, affected=affected)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mae = np.where(count > 0,
                                   abs_sum / np.maximum(count, 1), np.nan)
                strata[kind] = mae
        per_model[side] = {"sample_start": sample_start, "strata": strata}
        out[f"n_windows_{side}"] = int(sample_start.size)
        gc.collect()

    a_ss = per_model["a"]["sample_start"]
    b_ss = per_model["b"]["sample_start"]
    a_pos = {int(v): i for i, v in enumerate(a_ss)}
    b_pos = {int(v): i for i, v in enumerate(b_ss)}
    common = sorted(set(a_ss.tolist()) & set(b_ss.tolist()))
    a_idx = np.array([a_pos[v] for v in common], dtype=np.int64)
    b_idx = np.array([b_pos[v] for v in common], dtype=np.int64)
    out["n_matched"] = int(len(common))
    out["matched_fraction_of_smaller"] = (
        len(common) / max(1, min(a_ss.size, b_ss.size))
    )

    strata_out = {}
    for kind in ("all", "unaffected", "affected"):
        a_mae = per_model["a"]["strata"][kind]
        b_mae = per_model["b"]["strata"][kind]
        if a_mae is None or b_mae is None:
            strata_out[kind] = None
            continue
        stats = paired_tests(a_mae[a_idx], b_mae[b_idx], n_perm=n_perm)
        strata_out[kind] = stats
    out["strata"] = strata_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite-dir", required=True, type=Path,
                        help="directory holding composite_index.json")
    parser.add_argument("--shards-root", type=Path, default=None,
                        help="defaults to <composite-dir>/shards")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-perm", type=int, default=20000)
    args = parser.parse_args()

    index_path = args.composite_dir / "composite_index.json"
    index_sha = sha256_file(index_path)
    if index_sha != INDEX_PUBLISHED_SHA256:
        raise RuntimeError(
            f"composite index SHA-256 mismatch: {index_sha}")
    index = json.loads(index_path.read_text())
    if index.get("status") != "verified":
        raise RuntimeError("composite index status is not verified")
    shards_root = args.shards_root or (args.composite_dir / "shards")
    shard_roots = {}
    for shard in index["shards"]:
        root = Path(shard["local_results_root"])
        shard_roots[shard["shard"]] = (
            shards_root / root.parent.name / root.name
        )

    specs = build_pair_specs(index["tasks"])
    verified: dict[str, str] = {}
    pairs = []
    for i, spec in enumerate(specs, 1):
        result = analyze_pair(spec, shard_roots, verified, args.n_perm)
        pairs.append(result)
        print(f"[{i}/{len(specs)}] {result['label']} {result['region']} "
              f"seed{result['seed']} matched={result['n_matched']}",
              flush=True)
        gc.collect()

    script_sha = sha256_file(Path(__file__))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "numpy_version": np.__version__,
        "script_sha256": script_sha,
        "index_sha256": index_sha,
        "n_perm": args.n_perm,
        "verified_npz_count": len(verified),
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
