"""Summary-level analysis of the verified protocol-v3 host-stratified 45 matrix.

Inputs (all verified before use):
  - outputs/protocol_v3_noleak_dedup/composite/composite_index.json
    (read-only composite index, published SHA-256 below)
  - outputs/protocol_v3_noleak_dedup/composite/summaries_raw/<shard>/results/
    tasks/<task_id>/summary.json  (45 files, each SHA-256 checked against the
    index before any number is read)

Outputs (written to outputs/protocol_v3_noleak_dedup/analysis/):
  - matrix_45.csv            one row per canonical task, all headline metrics
  - multiseed_summary.csv    mean/std over seeds per (model, variant, region)
  - paired_seed_level.csv    same-host A/B seed-level paired differences
  - analysis_report.md       Chinese report with provenance and all tables

Scope of claims: this script only aggregates summary.json values that were
already validated by the runner and the composite service. Seed-level pairing
has n<=3 and is reported descriptively; window-level significance comes from
the separate read-only window-pair analysis. No paper conclusion is restored
by this script alone.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSITE_DIR = REPO_ROOT / "outputs" / "protocol_v3_noleak_dedup" / "composite"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "protocol_v3_noleak_dedup" / "analysis"

# Published identity of the read-only composite index (see AGENTS.md and
# docs/audits/result_validity.csv, release layer 2026-07-27 18:28:27 +08).
INDEX_PUBLISHED_SHA256 = (
    "8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe"
)

SHARD_DIR = {"5080": "rtx5080_27", "5090": "rtx5090_18"}
REGION_ORDER = ["Alameda", "ContraCosta", "Orange"]
SEED_ORDER = [42, 1, 2]

# Same-host A/B comparison specs. a/b are variant names within one model, or
# model names within the full_window baseline group.
VARIANT_PAIRS = [
    ("deseason_s0a_vs_s0c", "staeformer_deseason", "s0a", "s0c"),
    ("deseason_s0f_vs_s0g", "staeformer_deseason", "s0f", "s0g"),
    ("icsf_base_vs_icsf", "staeformer_icsf", "base", "icsf"),
]
FULL_WINDOW_MODELS = ["graphwavenet", "fourier_dual_net", "staeformer"]
FULL_WINDOW_PAIRS = [
    ("fullwindow_gwn_vs_fdn", "graphwavenet", "fourier_dual_net"),
    ("fullwindow_gwn_vs_staeformer", "graphwavenet", "staeformer"),
    ("fullwindow_fdn_vs_staeformer", "fourier_dual_net", "staeformer"),
]

METRIC_FIELDS = [
    "all", "unaffected", "affected",
    "h3", "h6", "h12",
    "rmse", "rmse_unaffected", "rmse_affected",
    "mape", "mape_unaffected", "mape_affected",
    "best_val", "valid_count", "params",
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_verified_tasks() -> tuple[dict, list[dict]]:
    """Load the composite index and all 45 summaries with hard verification."""
    index_path = COMPOSITE_DIR / "composite_index.json"
    index_bytes = index_path.read_bytes()
    index_sha = _sha256_bytes(index_bytes)
    if index_sha != INDEX_PUBLISHED_SHA256:
        raise RuntimeError(
            f"composite index SHA-256 mismatch: {index_sha} != published "
            f"{INDEX_PUBLISHED_SHA256}; refusing to analyze unverified input"
        )
    index = json.loads(index_bytes)
    if index.get("status") != "verified":
        raise RuntimeError(f"composite index status is {index.get('status')}")
    tasks_meta = index["tasks"]
    if len(tasks_meta) != 45:
        raise RuntimeError(f"expected 45 tasks in index, got {len(tasks_meta)}")

    tasks = []
    for meta in tasks_meta:
        summary_rel = meta["summary"]["path"]
        summary_path = (
            COMPOSITE_DIR
            / "summaries_raw"
            / SHARD_DIR[meta["shard"]]
            / "results"
            / summary_rel
        )
        raw = summary_path.read_bytes()
        digest = _sha256_bytes(raw)
        if digest != meta["summary"]["sha256"]:
            raise RuntimeError(
                f"summary SHA-256 mismatch for {meta['task_id']}: {digest}"
            )
        summary = json.loads(raw)
        for field in ("task_id", "region", "model", "variant", "seed",
                      "sampling_protocol", "config_hash"):
            if summary[field] != meta[field]:
                raise RuntimeError(
                    f"identity mismatch on {field} for {meta['task_id']}"
                )
        row = {
            "task_id": meta["task_id"],
            "idx": int(meta["task_id"].split("_", 1)[0]),
            "host": meta["shard"],
            "region": meta["region"],
            "model": meta["model"],
            "variant": meta["variant"],
            "seed": meta["seed"],
            "sampling_protocol": meta["sampling_protocol"],
            "config_hash": meta["config_hash"],
        }
        for field in METRIC_FIELDS:
            row[field] = summary.get(field)
        tasks.append(row)
    tasks.sort(key=lambda r: r["idx"])
    return index, tasks


def fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_matrix_45(tasks: list[dict]) -> Path:
    path = ANALYSIS_DIR / "matrix_45.csv"
    header = ["idx", "task_id", "host", "region", "model", "variant", "seed",
              "sampling_protocol", *METRIC_FIELDS, "config_hash"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in tasks:
            writer.writerow(
                [row["idx"], row["task_id"], row["host"], row["region"],
                 row["model"], row["variant"], row["seed"],
                 row["sampling_protocol"]]
                + [row[f] for f in METRIC_FIELDS]
                + [row["config_hash"]]
            )
    return path


def aggregate_multiseed(tasks: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in tasks:
        key = (row["model"], row["variant"], row["region"],
               row["sampling_protocol"], row["host"])
        groups.setdefault(key, []).append(row)
    out = []
    for (model, variant, region, protocol, host), rows in sorted(
        groups.items(), key=lambda kv: (REGION_ORDER.index(kv[0][2]),
                                        kv[0][0], kv[0][1])
    ):
        seeds = sorted((r["seed"] for r in rows), key=SEED_ORDER.index)
        entry = {"region": region, "host": host, "model": model,
                 "variant": variant, "protocol": protocol,
                 "n_seeds": len(seeds), "seeds": "/".join(map(str, seeds))}
        for field in ("all", "unaffected", "affected"):
            vals = [r[field] for r in rows if r[field] is not None]
            if vals:
                entry[f"{field}_mean"] = mean(vals)
                entry[f"{field}_std"] = stdev(vals) if len(vals) > 1 else 0.0
                entry[f"{field}_min"] = min(vals)
                entry[f"{field}_max"] = max(vals)
            else:
                entry[f"{field}_mean"] = None
                entry[f"{field}_std"] = None
                entry[f"{field}_min"] = None
                entry[f"{field}_max"] = None
        out.append(entry)
    return out


def write_multiseed(rows: list[dict]) -> Path:
    path = ANALYSIS_DIR / "multiseed_summary.csv"
    header = ["region", "host", "model", "variant", "protocol", "n_seeds",
              "seeds"]
    for f in ("all", "unaffected", "affected"):
        header += [f"{f}_mean", f"{f}_std", f"{f}_min", f"{f}_max"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([r[h] for h in header])
    return path


def build_pairs(tasks: list[dict]) -> list[dict]:
    """Same-host A/B seed-level pairs. Raises if a pair would cross hosts."""
    by_key = {}
    for row in tasks:
        by_key[(row["model"], row["variant"], row["region"], row["seed"])] = row

    results = []

    def add_pair(label, a_row, b_row):
        if a_row["host"] != b_row["host"]:
            raise RuntimeError(
                f"pair {label} would cross hosts: {a_row['task_id']} on "
                f"{a_row['host']} vs {b_row['task_id']} on {b_row['host']}"
            )
        results.append({"label": label, "region": a_row["region"],
                        "host": a_row["host"], "seed": a_row["seed"],
                        "a_task": a_row["task_id"], "b_task": b_row["task_id"],
                        "a_all": a_row["all"], "b_all": b_row["all"],
                        "diff_all": a_row["all"] - b_row["all"],
                        "a_unaffected": a_row["unaffected"],
                        "b_unaffected": b_row["unaffected"],
                        "diff_unaffected": (
                            None if a_row["unaffected"] is None
                            or b_row["unaffected"] is None
                            else a_row["unaffected"] - b_row["unaffected"]),
                        "a_affected": a_row["affected"],
                        "b_affected": b_row["affected"],
                        "diff_affected": (
                            None if a_row["affected"] is None
                            or b_row["affected"] is None
                            else a_row["affected"] - b_row["affected"])})

    for label, model, a_var, b_var in VARIANT_PAIRS:
        regions = sorted({r["region"] for r in tasks if r["model"] == model},
                         key=REGION_ORDER.index)
        for region in regions:
            seeds = sorted(
                {r["seed"] for r in tasks
                 if r["model"] == model and r["variant"] == a_var
                 and r["region"] == region}
                & {r["seed"] for r in tasks
                   if r["model"] == model and r["variant"] == b_var
                   and r["region"] == region},
                key=SEED_ORDER.index)
            for seed in seeds:
                add_pair(label,
                         by_key[(model, a_var, region, seed)],
                         by_key[(model, b_var, region, seed)])

    for label, a_model, b_model in FULL_WINDOW_PAIRS:
        for region in REGION_ORDER:
            a_rows = [r for r in tasks if r["model"] == a_model
                      and r["region"] == region]
            b_rows = [r for r in tasks if r["model"] == b_model
                      and r["region"] == region]
            for a_row in a_rows:
                for b_row in b_rows:
                    if a_row["seed"] == b_row["seed"]:
                        add_pair(label, a_row, b_row)
    return results


def write_pairs(pairs: list[dict]) -> Path:
    path = ANALYSIS_DIR / "paired_seed_level.csv"
    header = ["label", "region", "host", "seed", "a_task", "b_task",
              "a_all", "b_all", "diff_all",
              "a_unaffected", "b_unaffected", "diff_unaffected",
              "a_affected", "b_affected", "diff_affected"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for p in pairs:
            writer.writerow([p[h] for h in header])
    return path


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(index: dict, tasks: list[dict], multiseed: list[dict],
                 pairs: list[dict]) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append("# protocol_v3_noleak_dedup 正式 45 矩阵 summary 层分析")
    lines.append("")
    lines.append("## 数据来源与校验")
    lines.append("")
    lines.append(f"- 分析生成时间:{now}")
    lines.append("- 输入为只读 composite index 与 45 份 summary.json;"
                 "每份 summary 在读取前均按 index 记录的 SHA-256 复验,"
                 "45/45 通过,身份字段与 config_hash 逐项一致。")
    lines.append(f"- composite index SHA-256:`{INDEX_PUBLISHED_SHA256}`"
                 "(与发布值逐字节一致)")
    lines.append(f"- index status:`{index['status']}`,矩阵状态 "
                 f"`verified_host_stratified_45`,verified_at "
                 f"`{index.get('verified_at')}`")
    lines.append(f"- 训练代码 SHA:`{index['code_sha']}`")
    lines.append(f"- cache bundle:`{index['cache_bundle_id']}`")
    lines.append("- 口径:masked MAE;种子约定 42、1、2;所有直接 A/B 比较组同机。")
    lines.append("- 本报告只聚合已验证 summary,不恢复任何论文结论;"
                 "窗口级显著性见 window_pairs 分析。")
    lines.append("")

    lines.append("## 多种子聚合(masked MAE,mean ± std)")
    lines.append("")
    header = ["region", "host", "model", "variant", "protocol", "seeds",
              "all", "unaffected", "affected"]
    rows = []
    for r in multiseed:
        def cell(f):
            if r[f"{f}_mean"] is None:
                return "null"
            if r["n_seeds"] > 1:
                return f"{r[f'{f}_mean']:.4f} ± {r[f'{f}_std']:.4f}"
            return f"{r[f'{f}_mean']:.4f}"
        rows.append([r["region"], r["host"], r["model"], r["variant"],
                     r["protocol"], r["seeds"],
                     cell("all"), cell("unaffected"), cell("affected")])
    lines.append(md_table(header, rows))
    lines.append("")

    lines.append("## 同机配对差值(A − B,masked MAE `all`,负值 = A 更好)")
    lines.append("")
    lines.append("种子层 n≤3,只作方向性描述,不给显著性结论;显著性证据"
                 "来自窗口级配对检验。")
    lines.append("")
    header = ["比较", "region", "host", "seed", "A", "B", "A−B(all)",
              "A−B(affected)"]
    rows = []
    for p in pairs:
        a_name = p["a_task"].split("_", 1)[1].rsplit("_seed", 1)[0]
        b_name = p["b_task"].split("_", 1)[1].rsplit("_seed", 1)[0]
        rows.append([p["label"], p["region"], p["host"], str(p["seed"]),
                     a_name, b_name,
                     f"{p['diff_all']:+.4f}",
                     "null" if p["diff_affected"] is None
                     else f"{p['diff_affected']:+.4f}"])
    lines.append(md_table(header, rows))
    lines.append("")

    # per-comparison seed-level roll-up
    lines.append("## 配对差值按比较组汇总(`all` 口径)")
    lines.append("")
    header = ["比较", "region", "host", "n_seeds", "mean diff", "std",
              "方向一致(seed 数)"]
    rows = []
    groups: dict[tuple, list[dict]] = {}
    for p in pairs:
        groups.setdefault((p["label"], p["region"], p["host"]), []).append(p)
    for (label, region, host), ps in sorted(
            groups.items(), key=lambda kv: (kv[0][0],
                                            REGION_ORDER.index(kv[0][1]))):
        diffs = [p["diff_all"] for p in ps]
        neg = sum(1 for d in diffs if d < 0)
        rows.append([label, region, host, str(len(diffs)),
                     f"{mean(diffs):+.4f}",
                     f"{stdev(diffs):.4f}" if len(diffs) > 1 else "—",
                     f"{neg}/{len(diffs)} 为负"])
    lines.append(md_table(header, rows))
    lines.append("")

    lines.append("## 全窗口基线(seed 42,同机排序)")
    lines.append("")
    lines.append("full_window 协议的 summary 只含 all/unaffected/best_val/"
                 "valid_count/params,无分层 horizon 与 RMSE/MAPE 口径。")
    lines.append("")
    header = ["region", "host", "model", "all", "best_val", "valid_count",
              "params"]
    rows = []
    for region in REGION_ORDER:
        subset = sorted(
            (t for t in tasks if t["region"] == region
             and t["sampling_protocol"] == "full_window"),
            key=lambda t: t["all"])
        for t in subset:
            rows.append([region, t["host"], t["model"], fmt(t["all"]),
                         fmt(t["best_val"]),
                         str(t["valid_count"]) if t["valid_count"] is not None
                         else "",
                         str(t["params"]) if t["params"] is not None
                         else ""])
    lines.append(md_table(header, rows))
    lines.append("")

    lines.append("## 备注")
    lines.append("")
    lines.append("- 空单元格表示该模型 kind 的源 summary 本就不含此字段,"
                 "不是合并缺漏:`staeformer_deseason` 报告全部指标;"
                 "`staeformer_icsf` 无 h3/h6/h12 与 RMSE/MAPE;`full_window` "
                 "协议仅 all/unaffected/valid_count(其中 `staeformer` 另有 "
                 "best_val/params,`graphwavenet` 与 `fourier_dual_net` 无)。")
    lines.append("- `affected` 为 null 的行属于 full_window 协议,无事件锚定"
                 "受影响掩码。")
    lines.append("- 完整 45 行指标见 `matrix_45.csv`;多种子聚合见 "
                 "`multiseed_summary.csv`;逐种子配对见 "
                 "`paired_seed_level.csv`。")
    lines.append("- 下一步:窗口级配对显著性(只读,服务器端计算),输出 "
                 "`window_pairs.json` 后并入本报告。")

    path = ANALYSIS_DIR / "analysis_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    index, tasks = load_verified_tasks()
    p1 = write_matrix_45(tasks)
    multiseed = aggregate_multiseed(tasks)
    p2 = write_multiseed(multiseed)
    pairs = build_pairs(tasks)
    p3 = write_pairs(pairs)
    p4 = write_report(index, tasks, multiseed, pairs)
    for p in (p1, p2, p3, p4):
        print(f"wrote {p.relative_to(REPO_ROOT)}")
    print(f"tasks={len(tasks)} multiseed_groups={len(multiseed)} "
          f"pairs={len(pairs)}")


if __name__ == "__main__":
    main()
