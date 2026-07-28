"""Merge server-side window-level paired significance results into the
protocol-v3 analysis outputs.

Input: outputs/protocol_v3_noleak_dedup/analysis/window_pairs.json produced
by scripts/remote/distributed/protocol_v3_window_pairs.py running read-only
on the result server. The merge verifies that the embedded composite index
SHA-256 equals the published value and that script_sha256 matches the repo
script that generated it (or a file explicitly passed via --script).

Outputs:
  - window_pairs.csv        one row per pair x stratum
  - window_pairs_report.md  Chinese significance report
  - rewrites the trailing pointer line of analysis_report.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "outputs" / "protocol_v3_noleak_dedup" / "analysis"
REPO_SCRIPT = (REPO_ROOT / "scripts" / "remote" / "distributed"
               / "protocol_v3_window_pairs.py")
MAIN_REPORT = ANALYSIS_DIR / "analysis_report.md"
PENDING_LINE = ("- 下一步:窗口级配对显著性(只读,服务器端计算),输出 "
                "`window_pairs.json` 后并入本报告。")
DONE_LINE = ("- 窗口级配对显著性已完成(服务器端只读,npz 逐份按 composite "
             "index 复验),见 `window_pairs_report.md` 与 "
             "`window_pairs.csv`。")

INDEX_PUBLISHED_SHA256 = (
    "8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe"
)
STRATUM_LABEL = {"all": "全部单元", "unaffected": "未受影响",
                 "affected": "受影响"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sig_flag(stats: dict) -> str:
    p = max(stats["p_t"], stats["p_wilcoxon"], stats["p_perm"])
    if p < 0.01:
        return "显著(p<0.01)"
    if p < 0.05:
        return "显著(p<0.05)"
    return "不显著"


def md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def fmt_p(p: float) -> str:
    if p < 1e-300:
        return "<1e-300"
    return f"{p:.2e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ANALYSIS_DIR / "window_pairs.json")
    parser.add_argument("--script", type=Path, default=REPO_SCRIPT)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    index_sha = payload["index_sha256"]
    if index_sha != INDEX_PUBLISHED_SHA256:
        raise RuntimeError(
            f"window_pairs index SHA-256 mismatch: {payload['index_sha256']}")
    script_sha = sha256_file(args.script)
    if payload["script_sha256"] != script_sha:
        raise RuntimeError(
            f"producer script SHA-256 mismatch: {payload['script_sha256']} "
            f"!= repo script {script_sha}; refusing to merge results from an "
            "unreviewed script version")
    pairs = payload["pairs"]

    # CSV: one row per pair x stratum
    csv_path = ANALYSIS_DIR / "window_pairs.csv"
    header = ["label", "region", "shard", "seed", "a_task", "b_task",
              "stratum", "n_windows", "n_matched",
              "matched_fraction_of_smaller", "mean_diff", "median_diff",
              "frac_A_better", "t", "p_t", "z_wilcoxon", "p_wilcoxon",
              "p_perm"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for p in pairs:
            for stratum in ("all", "unaffected", "affected"):
                stats = p["strata"].get(stratum)
                if not stats or "mean_diff" not in stats:
                    continue
                writer.writerow([
                    p["label"], p["region"], p["shard"], p["seed"],
                    p["a_task"], p["b_task"], stratum, stats["n"],
                    p["n_matched"],
                    f"{p['matched_fraction_of_smaller']:.6f}",
                    stats["mean_diff"], stats["median_diff"],
                    stats["frac_A_better"], stats["t"], stats["p_t"],
                    stats["z_wilcoxon"], stats["p_wilcoxon"],
                    stats["p_perm"],
                ])

    # Markdown report
    lines = []
    lines.append("# protocol_v3 正式 45 矩阵:窗口级配对显著性")
    lines.append("")
    lines.append("## 来源与校验")
    lines.append("")
    lines.append(f"- 生成时间(服务器端):`{payload['generated_at']}`,主机 "
                 f"`{payload['host']}`,numpy `{payload['numpy_version']}`")
    lines.append(f"- composite index SHA-256:`{index_sha}`(与发布值一致)")
    lines.append(f"- 分析脚本 SHA-256:`{script_sha}`(与仓库脚本一致)")
    lines.append(f"- 逐任务 test_predictions.npz 已在读取前按 index 复验,"
                 f"共 {payload['verified_npz_count']} 份;配对数 "
                 f"{payload['pair_count']},置换次数 {payload['n_perm']}")
    lines.append("- 检验口径与旧 diagnostics 脚本一致:配对 t(正态近似)、"
                 "Wilcoxon 符号秩(正态近似、无并列修正)、符号翻转置换"
                 "(均值);全程 float64。判定沿用旧约定:三个 p 的最大值 "
                 "<0.01 记显著(p<0.01),<0.05 记显著(p<0.05),否则不显著。")
    lines.append("- 滑窗之间相互重叠、存在自相关,p 值偏乐观;多种子方向"
                 "一致性是配套的佐证。")
    lines.append("- 聚合口径差异:summary 的 masked MAE 按全部单元池化,"
                 "窗口级为先逐窗求 MAE 再对窗口等权平均,两者数值可有"
                 "万分位差异(如 +0.3143 对 +0.3140),均属正常。")
    lines.append("- 统计 n 小于匹配窗口数的差值来自空掩码窗口(掩码内"
                 "无有效单元的窗口不参与检验)。")
    lines.append("")

    for stratum in ("all", "affected"):
        lines.append(f"## {STRATUM_LABEL[stratum]}口径(A − B,负值 = A 更好)")
        lines.append("")
        header = ["比较", "region", "host", "seed", "n", "mean diff",
                  "A 更优窗口占比", "p_t", "p_wilcoxon", "p_perm", "判定"]
        rows = []
        for p in pairs:
            stats = p["strata"].get(stratum)
            if not stats or "mean_diff" not in stats:
                continue
            rows.append([
                p["label"], p["region"], p["shard"], str(p["seed"]),
                str(stats["n"]), f"{stats['mean_diff']:+.4f}",
                f"{stats['frac_A_better'] * 100:.1f}%",
                fmt_p(stats["p_t"]), fmt_p(stats["p_wilcoxon"]),
                fmt_p(stats["p_perm"]), sig_flag(stats),
            ])
        lines.append(md_table(header, rows))
        lines.append("")

    lines.append("## 窗口匹配情况")
    lines.append("")
    header = ["比较", "region", "seed", "n_A", "n_B", "n_matched", "匹配率"]
    rows = []
    for p in pairs:
        rows.append([p["label"], p["region"], str(p["seed"]),
                     str(p["n_windows_a"]), str(p["n_windows_b"]),
                     str(p["n_matched"]),
                     f"{p['matched_fraction_of_smaller'] * 100:.2f}%"])
    lines.append(md_table(header, rows))
    lines.append("")
    lines.append("注:未受影响(unaffected)口径的逐对数值见 "
                 "`window_pairs.csv`,结论口径为 all 与 affected。")

    report_path = ANALYSIS_DIR / "window_pairs_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if MAIN_REPORT.is_file():
        text = MAIN_REPORT.read_text(encoding="utf-8")
        if PENDING_LINE in text:
            text = text.replace(PENDING_LINE, DONE_LINE)
            MAIN_REPORT.write_text(text, encoding="utf-8")
            print("analysis_report.md 指针行已更新")

    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"wrote {report_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
