#!/usr/bin/env python3
"""Collect RGDN variant summary.json into a comparison table + preliminary verdicts.

Single-seed round: judge deltas against the known Alameda seed-noise band (~0.04-0.08
std from prior work). A single seed is suggestive only; confirm with seeds 42/1/2 and
significance_tests.py before claiming a mechanism is real.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

COMPARISONS = [
    ("deseason (v0b-v0a)", "v0b", "v0a"),
    ("headline RGDN (v1-v0b)", "v1", "v0b"),
    ("injection (v1-v2)", "v1", "v2"),
    ("total vs raw (v1-v0a)", "v1", "v0a"),
]
ORDER = ["v0a", "v0b", "v1", "v2", "v3", "v4"]


def compute_deltas(summaries, band, metric="all"):
    rows = []
    for label, a, b in COMPARISONS:
        if a not in summaries or b not in summaries:
            rows.append({"label": label, "delta": None, "verdict": "missing"})
            continue
        d = summaries[a][metric] - summaries[b][metric]
        if d < -band:
            verdict = "below band (improves)"
        elif d > band:
            verdict = "above band (worse)"
        else:
            verdict = "within noise band"
        rows.append({"label": label, "delta": d, "verdict": verdict})
    return rows


def load_summaries(region, seed, variants, root):
    out = {}
    for v in variants:
        p = Path(root) / region / f"{v}_seed{seed}" / "summary.json"
        if p.exists():
            out[v] = json.loads(p.read_text())
    return out


def render(region, seed, summaries, band):
    lines = [f"RGDN round-1  region={region} seed={seed}  noise band={band:.3f}",
             "single seed is suggestive only; confirm with seeds 42/1/2 + significance_tests.py", ""]
    lines.append(f"{'variant':6s} {'params':>9s} {'all':>8s} {'affected':>9s} {'unaffected':>11s}")
    for v in ORDER:
        if v in summaries:
            s = summaries[v]
            lines.append(f"{v:6s} {s.get('params', 0):>9,d} {s['all']:>8.3f} "
                         f"{s['affected']:>9.3f} {s['unaffected']:>11.3f}")
    lines.append("")
    lines.append("comparisons on 'all' MAE (negative = improvement):")
    for r in compute_deltas(summaries, band):
        d = "   n/a" if r["delta"] is None else f"{r['delta']:+.3f}"
        lines.append(f"  {r['label']:24s} {d}   {r['verdict']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+", default=["v0a", "v0b", "v1", "v2"])
    p.add_argument("--band", type=float, default=0.08)
    p.add_argument("--root", default="outputs/rgdn")
    p.add_argument("--out", default="outputs/diagnostics/rgdn_round1_results.txt")
    args = p.parse_args()
    summaries = load_summaries(args.region, args.seed, args.variants, args.root)
    if not summaries:
        print("no summary.json found under", Path(args.root) / args.region)
        return
    text = render(args.region, args.seed, summaries, args.band)
    print(text)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text + "\n")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
