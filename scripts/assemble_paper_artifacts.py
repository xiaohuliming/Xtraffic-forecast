#!/usr/bin/env python3
"""Assemble paper-ready tables, figures, and draft text from existing outputs."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "paper_artifacts"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

ABLATION_CSV = (
    ROOT
    / "outputs"
    / "impact_guided_next_stage"
    / "ablation_summary"
    / "learned_normal_ablation_table.csv"
)
SEED_RUNS_CSV = (
    ROOT
    / "outputs"
    / "impact_guided_next_stage"
    / "dual_branch_gate_no_aux_seed_robustness"
    / "seed_robustness_runs.csv"
)
SEED_SUMMARY_CSV = (
    ROOT
    / "outputs"
    / "impact_guided_next_stage"
    / "dual_branch_gate_no_aux_seed_robustness"
    / "seed_robustness_summary.csv"
)
SEVERITY_CSV = (
    ROOT
    / "outputs"
    / "impact_guided_next_stage"
    / "decay_group_analysis"
    / "severity_group_metrics.csv"
)
RECOVERY_CSV = (
    ROOT
    / "outputs"
    / "impact_guided_next_stage"
    / "decay_group_analysis"
    / "recovery_group_metrics.csv"
)
HORIZON_CSV = (
    ROOT
    / "outputs"
    / "impact_guided_next_stage"
    / "decay_group_analysis"
    / "horizon_comparison.csv"
)
EXTRA_MAIN_MODELS = [
    (
        "residual temporal decay no-aux",
        "learned normal STGNN",
        "normal_delta + abs(normal_delta) + dual history + temporal gate; no aux labels",
        ROOT / "outputs" / "impact_guided_next_stage" / "full_candidate_stgnn_learned_normal_decay_no_aux" / "metrics.json",
    ),
    (
        "dual-branch gate",
        "learned normal STGNN",
        "normal-style residual branch + incident graph branch + gate",
        ROOT / "outputs" / "impact_guided_next_stage" / "dual_branch_gate_full" / "metrics.json",
    ),
    (
        "dual-branch gate no-aux",
        "learned normal STGNN",
        "normal-style residual branch + incident graph branch + gate; no aux labels",
        ROOT / "outputs" / "impact_guided_next_stage" / "dual_branch_gate_full_no_aux" / "metrics.json",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fval(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mae(value: float | str) -> str:
    return f"{float(value):.4f}"


def pct(value: float | str) -> str:
    return f"{float(value):.2f}"


def int_text(value: float | str) -> str:
    return f"{int(float(value))}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def read_metric_row(model: str, normal_branch: str, inputs: str, metrics_path: Path) -> dict[str, str]:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = data["metrics"]["test"]
    return {
        "model": model,
        "normal_branch": normal_branch,
        "incident_branch_inputs": inputs,
        "test_all_baseline_robust_mae": str(test["all_candidates_baseline_robust_mae"]),
        "test_all_model_robust_mae": str(test["all_candidates_model_robust_mae"]),
        "test_all_improvement_pct": str(test["all_candidates_improvement_pct"]),
        "test_affected_baseline_robust_mae": str(test["affected_candidates_baseline_robust_mae"]),
        "test_affected_model_robust_mae": str(test["affected_candidates_model_robust_mae"]),
        "test_affected_improvement_pct": str(test["affected_candidates_improvement_pct"]),
        "test_unaffected_model_robust_mae": str(test["unaffected_candidates_model_robust_mae"]),
        "residual_beta": str(data["residual_beta"]),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
    }


def build_main_result_table(ablation: list[dict[str, str]]) -> str:
    rows = []
    for row in ablation:
        rows.append(
            [
                row["model"],
                row["normal_branch"],
                row["incident_branch_inputs"],
                f"{mae(row['test_all_baseline_robust_mae'])} -> {mae(row['test_all_model_robust_mae'])}",
                pct(row["test_all_improvement_pct"]),
                f"{mae(row['test_affected_baseline_robust_mae'])} -> {mae(row['test_affected_model_robust_mae'])}",
                pct(row["test_affected_improvement_pct"]),
            ]
        )
    return "# Table 1. Main forecasting results\n\n" + md_table(
        [
            "Model",
            "Normal branch",
            "Incident residual inputs",
            "All robust MAE",
            "All gain (%)",
            "Affected robust MAE",
            "Affected gain (%)",
        ],
        rows,
    )


def build_ablation_table(ablation: list[dict[str, str]]) -> str:
    rows = []
    prev_all = None
    prev_aff = None
    learned_rows = [row for row in ablation if row["normal_branch"] == "learned normal STGNN"]
    for row in learned_rows:
        cur_all = fval(row, "test_all_model_robust_mae")
        cur_aff = fval(row, "test_affected_model_robust_mae")
        if prev_all is None:
            step_all = "-"
            step_aff = "-"
        else:
            step_all = pct((prev_all - cur_all) / prev_all * 100.0)
            step_aff = pct((prev_aff - cur_aff) / prev_aff * 100.0)
        rows.append(
            [
                row["model"],
                row["incident_branch_inputs"],
                mae(cur_all),
                mae(cur_aff),
                step_all,
                step_aff,
                mae(row["residual_beta"]),
            ]
        )
        prev_all = cur_all
        prev_aff = cur_aff
    return "# Table 2. Learned-normal and gated-residual component ablation\n\n" + md_table(
        [
            "Variant",
            "Added signal",
            "All MAE",
            "Affected MAE",
            "Step gain all (%)",
            "Step gain affected (%)",
            "Best beta",
        ],
        rows,
    )


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def latex_table(caption: str, label: str, headers: list[str], rows: list[list[str]]) -> str:
    colspec = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(latex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}%", "\\end{table}", ""])
    return "\n".join(lines)


def markdown_rows_from_table(text: str) -> tuple[list[str], list[list[str]]]:
    lines = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| "):
            lines.append(line)
            in_table = True
        elif in_table:
            break
    if len(lines) < 2:
        return [], []
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return headers, rows


def write_latex_from_markdown(md_path: Path, tex_path: Path, caption: str, label: str) -> None:
    headers, rows = markdown_rows_from_table(md_path.read_text(encoding="utf-8"))
    write_text(tex_path, latex_table(caption, label, headers, rows))


def write_seed_latex(runs: list[dict[str, str]], path: Path) -> None:
    rows = []
    for row in runs:
        rows.append(
            [
                int_text(row["seed"]),
                mae(row["test_all_model_robust_mae"]),
                mae(row["test_affected_model_robust_mae"]),
                mae(row["test_unaffected_model_robust_mae"]),
                mae(row["horizon_06_test_affected_model_robust_mae"]),
                mae(row["horizon_12_test_affected_model_robust_mae"]),
            ]
        )
    write_text(
        path,
        latex_table(
            "Seed robustness of the dual-branch gated residual no-aux model.",
            "tab:seed_robustness",
            ["Seed", "All MAE", "Affected MAE", "Unaffected MAE", "H6 aff.", "H12 aff."],
            rows,
        ),
    )


def write_group_latex(severity: list[dict[str, str]], recovery: list[dict[str, str]], path: Path) -> None:
    rows = []
    for row in severity:
        rows.append(
            [
                "Severity",
                row["label"],
                int_text(row["samples"]),
                mae(row["no_decay_affected_model_robust_mae"]),
                mae(row["decay_affected_model_robust_mae"]),
                pct(row["affected_decay_gain_pct"]),
                pct(row["h12_affected_decay_gain_pct"]),
            ]
        )
    for row in recovery:
        rows.append(
            [
                "Recovery",
                row["label"],
                int_text(row["samples"]),
                mae(row["no_decay_affected_model_robust_mae"]),
                mae(row["decay_affected_model_robust_mae"]),
                pct(row["affected_decay_gain_pct"]),
                pct(row["h12_affected_decay_gain_pct"]),
            ]
        )
    write_text(
        path,
        latex_table(
            "Temporal-decay gains by incident severity and recovery duration.",
            "tab:decay_groups",
            ["Dimension", "Group", "Samples", "No decay", "Decay", "Gain (%)", "H12 gain (%)"],
            rows,
        ),
    )


def build_seed_table(runs: list[dict[str, str]], summary: list[dict[str, str]]) -> str:
    run_rows = []
    for row in runs:
        run_rows.append(
            [
                int_text(row["seed"]),
                mae(row["test_all_model_robust_mae"]),
                mae(row["test_affected_model_robust_mae"]),
                mae(row["test_unaffected_model_robust_mae"]),
                mae(row["horizon_06_test_affected_model_robust_mae"]),
                mae(row["horizon_12_test_affected_model_robust_mae"]),
                mae(row["residual_beta"]),
            ]
        )

    by_stat = {row["stat"]: row for row in summary}
    mean = by_stat["mean"]
    std = by_stat["std"]
    summary_rows = [
        [
            "All candidates",
            f"{mae(mean['test_all_model_robust_mae'])} +/- {mae(std['test_all_model_robust_mae'])}",
            f"{pct(mean['test_all_improvement_pct'])} +/- {pct(std['test_all_improvement_pct'])}",
        ],
        [
            "Affected candidates",
            f"{mae(mean['test_affected_model_robust_mae'])} +/- {mae(std['test_affected_model_robust_mae'])}",
            f"{pct(mean['test_affected_improvement_pct'])} +/- {pct(std['test_affected_improvement_pct'])}",
        ],
        [
            "Unaffected candidates",
            f"{mae(mean['test_unaffected_model_robust_mae'])} +/- {mae(std['test_unaffected_model_robust_mae'])}",
            "-",
        ],
        [
            "Affected horizon 6",
            f"{mae(mean['horizon_06_test_affected_model_robust_mae'])} +/- {mae(std['horizon_06_test_affected_model_robust_mae'])}",
            "-",
        ],
        [
            "Affected horizon 12",
            f"{mae(mean['horizon_12_test_affected_model_robust_mae'])} +/- {mae(std['horizon_12_test_affected_model_robust_mae'])}",
            "-",
        ],
    ]

    return (
        "# Table 3. Seed robustness of the dual-branch gated residual no-aux model\n\n"
        + md_table(
            [
                "Seed",
                "All MAE",
                "Affected MAE",
                "Unaffected MAE",
                "H6 affected MAE",
                "H12 affected MAE",
                "Best beta",
            ],
            run_rows,
        )
        + "\n## Mean and standard deviation\n\n"
        + md_table(["Metric", "Mean +/- std", "Gain (%)"], summary_rows)
    )


def compact_group_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    out = []
    for row in rows:
        out.append(
            [
                row["label"],
                int_text(row["samples"]),
                mae(row["no_decay_affected_model_robust_mae"]),
                mae(row["decay_affected_model_robust_mae"]),
                pct(row["affected_decay_gain_pct"]),
                pct(row["h06_affected_decay_gain_pct"]),
                pct(row["h12_affected_decay_gain_pct"]),
            ]
        )
    return out


def build_group_table(severity: list[dict[str, str]], recovery: list[dict[str, str]]) -> str:
    headers = [
        "Group",
        "Samples",
        "No-decay affected MAE",
        "Decay affected MAE",
        "Gain (%)",
        "H6 gain (%)",
        "H12 gain (%)",
    ]
    return (
        "# Table 4. Where temporal decay helps\n\n"
        "## Severity groups\n\n"
        + md_table(headers, compact_group_rows(severity))
        + "\n## Recovery groups\n\n"
        + md_table(headers, compact_group_rows(recovery))
    )


def build_horizon_table(horizon: list[dict[str, str]]) -> str:
    rows = []
    for row in horizon:
        rows.append(
            [
                int_text(row["horizon"]),
                mae(row["no_decay_affected_model_robust_mae"]),
                mae(row["decay_affected_model_robust_mae"]),
                pct(row["affected_decay_gain_pct"]),
                mae(row["no_decay_all_model_robust_mae"]),
                mae(row["decay_all_model_robust_mae"]),
                pct(row["all_decay_gain_pct"]),
            ]
        )
    return "# Table 5. Horizon-wise decay comparison\n\n" + md_table(
        [
            "Horizon",
            "No-decay affected MAE",
            "Decay affected MAE",
            "Affected gain (%)",
            "No-decay all MAE",
            "Decay all MAE",
            "All gain (%)",
        ],
        rows,
    )


def save_plots(
    ablation: list[dict[str, str]],
    severity: list[dict[str, str]],
    recovery: list[dict[str, str]],
    horizon: list[dict[str, str]],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except Exception as exc:  # pragma: no cover - depends on local environment
        write_text(FIG_DIR / "README.md", f"Matplotlib unavailable; skipped figures: {exc}\n")
        return []

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    written: list[Path] = []

    def box(ax, xy, w, h, text, face, edge):
        patch = FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9.4,
            color="#1f2933",
            linespacing=1.18,
        )

    def arrow(ax, start, end, color="#546A7B", rad=0.0):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(patch)

    fig, ax = plt.subplots(figsize=(11.6, 5.4), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(
        0.2,
        5.25,
        "Latent-incident mediated dual-branch residual gating",
        fontsize=13,
        fontweight="bold",
        color="#1f2933",
        ha="left",
    )
    ax.text(
        0.2,
        4.92,
        "Two residual-space branches compete at each node and horizon before being added to the normal counterfactual forecast.",
        fontsize=9.2,
        color="#52616B",
        ha="left",
    )
    blue_face, blue_edge = "#E7F0FF", "#3B6EA8"
    orange_face, orange_edge = "#FFF1DE", "#C77700"
    green_face, green_edge = "#E8F5E9", "#3B8F4A"
    gray_face, gray_edge = "#F3F4F6", "#667085"

    box(ax, (0.25, 3.65), 1.75, 0.75, "Historical\ntraffic X", gray_face, gray_edge)
    box(ax, (0.25, 1.35), 1.75, 0.75, "Incident\ncontext c", gray_face, gray_edge)
    box(ax, (2.35, 3.65), 1.75, 0.75, "Normal STGNN\nregular dynamics", blue_face, blue_edge)
    box(ax, (4.45, 3.65), 1.8, 0.75, "Counterfactual\nnormal forecast", blue_face, blue_edge)
    box(ax, (2.35, 1.35), 1.75, 0.75, "Full candidate\nsensor graph", orange_face, orange_edge)
    box(
        ax,
        (4.55, 1.0),
        2.05,
        1.35,
        "Residual construction\nstatistical residual\nlearned residual\nnormal_delta / abs",
        orange_face,
        orange_edge,
    )
    box(ax, (7.05, 2.45), 1.85, 0.85, "Normal-style\nresidual branch", blue_face, blue_edge)
    box(ax, (7.05, 0.75), 1.85, 1.1, "Incident graph\nresidual branch", orange_face, orange_edge)
    box(ax, (9.45, 1.55), 1.85, 0.95, "Node-horizon gate\nalpha = sigmoid(g)", green_face, green_edge)
    box(
        ax,
        (8.95, 3.65),
        2.8,
        0.75,
        "Final forecast\nY_hat = Y_normal + beta Delta_gated",
        green_face,
        green_edge,
    )

    arrow(ax, (2.0, 4.02), (2.35, 4.02))
    arrow(ax, (4.1, 4.02), (4.45, 4.02))
    arrow(ax, (6.25, 4.02), (8.95, 4.02), color=blue_edge)
    arrow(ax, (2.0, 1.72), (2.35, 1.72))
    arrow(ax, (4.1, 1.72), (4.55, 1.72), color=orange_edge)
    arrow(ax, (6.6, 2.02), (7.05, 2.85), color=blue_edge, rad=0.12)
    arrow(ax, (6.6, 1.52), (7.05, 1.28), color=orange_edge, rad=-0.05)
    arrow(ax, (2.0, 3.82), (4.55, 2.2), rad=-0.18)
    arrow(ax, (5.35, 3.65), (5.45, 2.35), color=blue_edge, rad=0.12)
    arrow(ax, (8.9, 2.88), (9.45, 2.25), color=blue_edge, rad=-0.12)
    arrow(ax, (8.9, 1.28), (9.45, 1.82), color=orange_edge, rad=0.10)
    arrow(ax, (10.38, 2.50), (10.35, 3.65), color=green_edge, rad=-0.12)

    ax.text(3.25, 4.55, "normal branch", fontsize=8.7, color=blue_edge, ha="center")
    ax.text(6.78, 3.45, "residual target: Y - Y_normal", fontsize=8.2, color="#52616B", ha="center")
    ax.text(8.0, 0.48, "latent incident impact", fontsize=8.7, color=orange_edge, ha="center")
    ax.text(10.55, 3.35, "gated residual fusion", fontsize=8.2, color=green_edge, ha="center")

    fig.tight_layout(pad=0.35)
    path = FIG_DIR / "method_architecture.png"
    fig.savefig(path)
    written.append(path)
    path_pdf = FIG_DIR / "method_architecture.pdf"
    fig.savefig(path_pdf)
    written.append(path_pdf)
    plt.close(fig)

    xs = [int(float(row["horizon"])) for row in horizon]
    affected_gain = [float(row["affected_decay_gain_pct"]) for row in horizon]
    all_gain = [float(row["all_decay_gain_pct"]) for row in horizon]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.plot(xs, affected_gain, marker="o", label="Affected candidates")
    ax.plot(xs, all_gain, marker="s", label="All candidates")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Decay gain (%)")
    ax.set_title("Temporal decay gain grows on affected candidates")
    ax.legend(frameon=True)
    fig.tight_layout()
    path = FIG_DIR / "horizon_decay_gain_pct.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    no_decay_aff = [float(row["no_decay_affected_model_robust_mae"]) for row in horizon]
    decay_aff = [float(row["decay_affected_model_robust_mae"]) for row in horizon]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.plot(xs, no_decay_aff, marker="o", label="No decay")
    ax.plot(xs, decay_aff, marker="s", label="Temporal decay")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Affected robust MAE")
    ax.set_title("Affected-candidate error across horizons")
    ax.legend(frameon=True)
    fig.tight_layout()
    path = FIG_DIR / "horizon_affected_mae.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    labels = [row["model"] for row in ablation]
    values = [float(row["test_affected_model_robust_mae"]) for row in ablation]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=180)
    ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Affected robust MAE")
    ax.set_title("Ablation on affected candidates")
    fig.tight_layout()
    path = FIG_DIR / "ablation_affected_mae.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    sev_labels = [row["group"].replace("severity_", "") for row in severity]
    sev_gain = [float(row["affected_decay_gain_pct"]) for row in severity]
    rec_labels = [
        row["group"]
        .replace("recovery_", "")
        .replace("_lt30", "<30")
        .replace("_30_90", "30-90")
        .replace("_ge90", ">=90")
    for row in recovery]
    rec_gain = [float(row["affected_decay_gain_pct"]) for row in recovery]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=180, sharey=True)
    axes[0].bar(sev_labels, sev_gain, color="#59A14F")
    axes[0].set_title("Severity groups")
    axes[0].set_ylabel("Affected decay gain (%)")
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[1].bar(rec_labels, rec_gain, color="#F28E2B")
    axes[1].set_title("Recovery groups")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    fig.suptitle("Temporal decay helps most on severe and long-recovery incidents")
    fig.tight_layout()
    path = FIG_DIR / "severity_recovery_decay_gain.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    return written


def build_draft() -> str:
    return textwrap.dedent(
        """\
        # Paper Draft Notes: Latent-Incident Mediated Traffic Forecasting

        ## Working Title

        Latent-Incident Mediated Spatio-Temporal Traffic Forecasting under Incidents

        ## Core Problem

        Conventional traffic forecasting models mainly learn regular traffic dynamics. Their error increases under incidents because the observed future state is no longer explained only by periodic and spatial traffic patterns. XTraffic also shows that inferring incident type from traffic state is unreliable, suggesting that the incident type is not the right supervision target for forecasting. We therefore model the traffic impact induced by an incident, rather than treating the incident label itself as the central prediction target.

        ## Method Overview

        The model decomposes future traffic into a normal counterfactual component and an incident-induced residual:

        ```text
        Delta_gated = (1 - alpha) * Delta_normal_style + alpha * Delta_incident_style
        Y_hat = Y_normal + beta * Delta_gated
        ```

        `Y_normal` is produced by a learned normal STGNN trained on normal or weak-incident traffic windows. The final model uses two residual-space branches: a normal-style residual branch and an incident graph branch. A learned gate fuses these two residual embeddings before adding the result back to the learned normal forecast. This keeps the advisor's original dual-branch gate idea, while forcing the branches to operate in incident-impact residual space rather than both predicting the full traffic state.

        ## Architecture

        1. Normal branch: a lightweight STGNN predicts the counterfactual future traffic state under regular conditions.
        2. Residual construction: both residual branches receive the learned normal residual, the statistical residual, the normalized normal-forecast disagreement `normal_delta`, and `abs(normal_delta)` as a disagreement proxy.
        3. Candidate graph: for each incident, the model builds a full candidate sensor graph around the incident location instead of selecting only label-known affected top-k nodes.
        4. Normal-style and incident graph branches: the normal-style branch handles mild residual corrections, while node, event, temporal, and graph features are encoded to predict incident-style residual embeddings.
        5. Dual-branch gate: a learned node-horizon gate fuses the normal-style residual branch and incident graph branch in residual space.

        ## Experimental Story

        The learned-normal residual model reduces robust MAE over the learned normal baseline. Adding `normal_delta` improves the residual branch by exposing the disagreement between the learned normal forecaster and the statistical normal reference. Dual historical residuals further improve the alignment between input residual history and future residual target. The no-aux temporal-decay model verifies that the gains are not caused by future-derived auxiliary labels. The best variant is the dual-branch gated residual no-aux model, which improves over the single residual branch by allowing normal-style and incident-style residual explanations to compete at each candidate node and forecast horizon.

        ## Key Results to Report

        - Best model: dual-branch gated residual no-aux.
        - Test all robust MAE: 0.8328 -> 0.7181, 13.78% improvement.
        - Test affected robust MAE: 1.2938 -> 1.1234, 13.17% improvement.
        - Across three seeds, all robust MAE is 0.7182 +/- 0.0016 and affected robust MAE is 1.1240 +/- 0.0054.
        - The no-aux setting shows that the improvement does not depend on future-derived auxiliary impact labels.
        - Branch interpretability: learned gate affected MAE is 1.1234, better than fixed gate 0.5 at 1.1706, normal-style residual only at 1.2478, and incident-graph residual only at 1.3562.
        - Gate selection alignment: on affected elements, the mean incident-branch gate is 0.3921 when the incident branch has lower local error and 0.3511 when the normal-style branch has lower local error.
        - Case studies: selected incidents where learned gate improves most over fixed gate show stronger gate weights around high residual affected nodes; sample 192208 improves affected MAE from 5.4605 to 3.9825. A mixed success/neutral/failure case set further shows that the gate can over-trust a locally poor incident branch on short-recovery or single-node-impact cases.

        ## Interpretation

        The improvement appears in the main full test split, affected-candidate subset, no-aux setting, and seed robustness. This matters because the comparison uses the same HDF5 cache, time split, learned normal branch, residual target, and robust MAE metric. The only structural change is the gated residual fusion, so the gain is attributable to the model's ability to choose between normal-style and incident-style residual explanations.

        The gate should be interpreted as a local residual-explanation selector, not as a global incident-severity indicator. It does not monotonically increase with incident severity or recovery duration, but it does assign higher incident-branch weights when the incident branch is locally more accurate and when residual magnitude is larger on affected candidates.

        ## Current Caveat

        Aggregate gate analysis and mixed case-level visualizations are now done. Preliminary confidence-aware gate and hard-example reweighting variants did not outperform the current no-aux learned gate. The main remaining model gap is therefore not just branch confidence or loss weighting, but incident-branch expressiveness and calibrated reliability: future work should consider ST-TIS-style incident modeling, uncertainty-aware gating, or stronger branch-confidence calibration.

        ## Recommended Figure and Table Placement

        - Table 1: main forecasting results.
        - Table 2: component ablation.
        - Table 3: seed robustness.
        - Table 4: gate and branch interpretability.
        - Table 5: severity and recovery group analysis.
        - Figure 1: method diagram.
        - Figure 2: branch ablation for learned gate vs fixed gate and single branches.
        - Figure 3: gate selection alignment.
        - Figure 4: case study heatmap.
        - Figure 5: horizon-wise affected-candidate MAE.
        - Figure 6: severity and recovery gains from temporal decay.
        """
    )


def build_readme(figures: list[Path]) -> str:
    figure_lines = "\n".join(f"- `{path.relative_to(OUT_DIR)}`" for path in figures)
    return (
        "# Paper Artifacts\n\n"
        "Generated from existing experiment outputs.\n\n"
        "## Tables\n\n"
        "- `tables/main_result_table.md`\n"
        "- `tables/component_ablation_table.md`\n"
        "- `tables/seed_robustness_table.md`\n"
        "- `tables/temporal_decay_group_table.md`\n"
        "- `tables/horizon_decay_table.md`\n"
        "- `tables/gate_branch_interpretability_table.md`\n"
        "- `tables/gate_case_study_table.md`\n"
        "- `tables/gate_case_study_mixed_table.md`\n\n"
        "LaTeX snippets are also written next to the Markdown tables as `.tex` files.\n\n"
        "## Figures\n\n"
        f"{figure_lines if figure_lines else '- No figures were generated.'}\n\n"
        "Additional interpretability figures:\n\n"
        "- `figures/gate_branch_ablation_mae.png`\n"
        "- `figures/gate_selection_alignment.png`\n"
        "- `figures/gate_by_horizon.png`\n"
        "- `figures/gate_by_event_group.png`\n"
        "- `figures/case_studies/case_01_sample_192208.png`\n"
        "- `figures/case_studies/case_02_sample_56226.png`\n"
        "- `figures/case_studies/case_03_sample_184513.png`\n"
        "- `figures/case_studies/case_04_sample_184542.png`\n"
        "- `figures/case_studies_mixed/case_01_success_sample_192208.png`\n"
        "- `figures/case_studies_mixed/case_02_success_sample_56226.png`\n"
        "- `figures/case_studies_mixed/case_03_neutral_sample_195028.png`\n"
        "- `figures/case_studies_mixed/case_04_neutral_sample_187753.png`\n"
        "- `figures/case_studies_mixed/case_05_failure_sample_88134.png`\n"
        "- `figures/case_studies_mixed/case_06_failure_sample_60576.png`\n\n"
        "## Draft\n\n"
        "- `method_experiment_draft.md`\n"
        "- `manuscript_draft_en.md`\n"
        "- `manuscript_draft_zh.md`\n\n"
        "## Experiment Audit\n\n"
        "- `../impact_guided_next_stage/experiment_audit/status_zh.md`\n"
        "- `../impact_guided_next_stage/dual_branch_gate_interpretability/report_zh.md`\n"
        "- `case_study_report.md`\n"
        "- `case_study_report_mixed.md`\n"
        "- `confidence_gate_experiment_report_zh.md`\n"
        "- `hard_mining_experiment_report_zh.md`\n"
        "- `tables/source_gated_residual_extension.csv`\n"
        "- `tables/source_selected_gate_case_studies.csv`\n"
        "- `tables/source_selected_gate_case_studies_mixed.csv`\n\n"
        "## Writing Helpers\n\n"
        "- `paper_outline_zh.md`\n"
        "- `section_draft_zh.md`\n"
        "- `related_work_draft_en.md`\n"
        "- `method_diagram.md`\n\n"
        "## LaTeX\n\n"
        "- `latex/main.tex`\n"
        "- `latex/main.pdf`\n"
        "- `latex_ieee/main.tex`\n"
        "- `latex_ieee/main.pdf`\n"
        "- `latex_zh/main.tex`\n"
        "- `latex_zh/main.pdf`\n\n"
        "## References\n\n"
        "- `references_todo.bib`\n\n"
        "Source script:\n\n"
        "```bash\n"
        "python3 scripts/assemble_paper_artifacts.py\n"
        "```\n"
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ablation_base = read_csv(ABLATION_CSV)
    extra_rows = [
        read_metric_row(model, normal_branch, inputs, metrics_path)
        for model, normal_branch, inputs, metrics_path in EXTRA_MAIN_MODELS
    ]
    ablation = ablation_base + extra_rows
    seed_runs = read_csv(SEED_RUNS_CSV)
    seed_summary = read_csv(SEED_SUMMARY_CSV)
    severity = read_csv(SEVERITY_CSV)
    recovery = read_csv(RECOVERY_CSV)
    horizon = read_csv(HORIZON_CSV)

    main_md = TABLE_DIR / "main_result_table.md"
    ablation_md = TABLE_DIR / "component_ablation_table.md"
    seed_md = TABLE_DIR / "seed_robustness_table.md"
    group_md = TABLE_DIR / "temporal_decay_group_table.md"
    horizon_md = TABLE_DIR / "horizon_decay_table.md"

    write_text(main_md, build_main_result_table(ablation))
    write_text(ablation_md, build_ablation_table(ablation))
    write_text(seed_md, build_seed_table(seed_runs, seed_summary))
    write_text(group_md, build_group_table(severity, recovery))
    write_text(horizon_md, build_horizon_table(horizon))

    write_latex_from_markdown(
        main_md,
        TABLE_DIR / "main_result_table.tex",
        "Main forecasting results on incident-centered candidate graphs.",
        "tab:main_results",
    )
    write_latex_from_markdown(
        ablation_md,
        TABLE_DIR / "component_ablation_table.tex",
        "Component ablation within the learned-normal residual model.",
        "tab:component_ablation",
    )
    write_seed_latex(seed_runs, TABLE_DIR / "seed_robustness_table.tex")
    write_group_latex(severity, recovery, TABLE_DIR / "temporal_decay_group_table.tex")
    write_latex_from_markdown(
        horizon_md,
        TABLE_DIR / "horizon_decay_table.tex",
        "Horizon-wise comparison between no-decay and temporal-decay models.",
        "tab:horizon_decay",
    )

    copy_csv(ABLATION_CSV, TABLE_DIR / "source_learned_normal_ablation_table.csv")
    write_text(
        TABLE_DIR / "source_gated_residual_extension.csv",
        "\n".join(
            [
                ",".join(extra_rows[0].keys()),
                *[",".join(row[key] for key in extra_rows[0].keys()) for row in extra_rows],
            ]
        )
        + "\n",
    )
    copy_csv(SEED_RUNS_CSV, TABLE_DIR / "source_decay_seed_runs.csv")
    copy_csv(SEED_SUMMARY_CSV, TABLE_DIR / "source_decay_seed_summary.csv")
    copy_csv(SEVERITY_CSV, TABLE_DIR / "source_severity_group_metrics.csv")
    copy_csv(RECOVERY_CSV, TABLE_DIR / "source_recovery_group_metrics.csv")
    copy_csv(HORIZON_CSV, TABLE_DIR / "source_horizon_comparison.csv")

    figures = save_plots(ablation, severity, recovery, horizon)
    write_text(OUT_DIR / "method_experiment_draft.md", build_draft())
    write_text(OUT_DIR / "README.md", build_readme(figures))

    print(f"Wrote paper artifacts to {OUT_DIR}")
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
