#!/usr/bin/env python3
"""Render the verified FDN benchmark v1 MAE comparison as a visual table."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATASETS = ("pems04", "pems08")
SEEDS = (42, 1, 2)
VARIANTS = (
    ("gwn", "GWN"),
    ("dual_gwn", "DualGWN"),
    ("fdn_fixed_k3", "FDN fixed-K3"),
    ("fdn_learnable_k3", "FDN learnable-K3"),
)
COMPARISONS = (
    ("DualGWN - GWN", "dual_gwn", "gwn"),
    ("fixed-K3 - DualGWN", "fdn_fixed_k3", "dual_gwn"),
    ("learnable-K3 - fixed-K3", "fdn_learnable_k3", "fdn_fixed_k3"),
    ("learnable-K3 - DualGWN", "fdn_learnable_k3", "dual_gwn"),
    ("learnable-K3 - GWN", "fdn_learnable_k3", "gwn"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def read_runs(path: Path) -> dict[tuple[str, int, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[tuple[str, int, str], float] = {}
    for row in rows:
        key = (row["dataset"], int(row["seed"]), row["variant"])
        if key in values:
            raise ValueError(f"duplicate run cell: {key}")
        values[key] = float(row["mae"])
    expected = {
        (dataset, seed, variant)
        for dataset in DATASETS
        for seed in SEEDS
        for variant, _ in VARIANTS
    }
    if set(values) != expected:
        raise ValueError("runs.csv does not contain the exact 24-cell matrix")
    return values


def blend_with_white(color: tuple[float, ...], amount: float = 0.28) -> tuple[float, ...]:
    rgb = np.asarray(color[:3], dtype=float)
    blended = rgb * (1.0 - amount) + amount
    return (*blended.tolist(), 1.0)


def draw_performance_table(
    ax: mpl.axes.Axes,
    dataset: str,
    values: dict[tuple[str, int, str], float],
) -> None:
    model_keys = [key for key, _ in VARIANTS]
    model_labels = [label for _, label in VARIANTS]
    seed_values = np.asarray(
        [[values[(dataset, seed, variant)] for seed in SEEDS] for variant in model_keys],
        dtype=float,
    )
    means = seed_values.mean(axis=1)
    stds = seed_values.std(axis=1, ddof=1)
    display = [
        [
            *(f"{value:.4f}" for value in row),
            f"{mean:.4f} ± {std:.4f}",
        ]
        for row, mean, std in zip(seed_values, means, stds, strict=True)
    ]
    color_values = np.column_stack([seed_values, means])
    norm = mpl.colors.Normalize(vmin=float(color_values.min()), vmax=float(color_values.max()))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "verified_mae",
        ["#0072B2", "#D8EEF8", "#FFFFFF"],
    )
    ax.axis("off")
    ax.set_title(dataset.upper(), fontsize=15, fontweight="bold", pad=12)
    table = ax.table(
        cellText=display,
        rowLabels=model_labels,
        colLabels=("Seed 42", "Seed 1", "Seed 2", "Mean ± SD"),
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=(0.17, 0.17, 0.17, 0.26),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.15)
    for column in range(4):
        best_row = int(np.argmin(color_values[:, column]))
        for row in range(4):
            cell = table[(row + 1, column)]
            face = blend_with_white(cmap(norm(color_values[row, column])))
            cell.set_facecolor(face)
            cell.set_edgecolor("#AAB7C4")
            cell.set_linewidth(1.8 if row == best_row else 0.8)
            if row == best_row:
                cell.get_text().set_fontweight("bold")
                cell.set_edgecolor("#111111")
    for column in range(4):
        cell = table[(0, column)]
        cell.set_facecolor("#E9EEF3")
        cell.set_edgecolor("#7A8793")
        cell.get_text().set_fontweight("bold")
    for row in range(1, 5):
        cell = table[(row, -1)]
        cell.set_facecolor("#F6F8FA")
        cell.set_edgecolor("#AAB7C4")
        cell.get_text().set_fontweight("bold")
    ax.text(
        0.5,
        0.04,
        "Bold outline marks the lowest MAE in each column.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4A5560",
        transform=ax.transAxes,
    )


def draw_component_table(
    ax: mpl.axes.Axes,
    values: dict[tuple[str, int, str], float],
) -> None:
    rows: list[list[str]] = []
    numeric: list[list[float]] = []
    for _, left, right in COMPARISONS:
        row_text: list[str] = []
        row_numeric: list[float] = []
        for dataset in DATASETS:
            differences = np.asarray(
                [values[(dataset, seed, left)] - values[(dataset, seed, right)] for seed in SEEDS]
            )
            mean = float(differences.mean())
            wins = int((differences < 0).sum())
            direction = "better" if mean < 0 else "worse"
            row_text.append(f"{mean:+.4f}\n{wins}/3 seeds better\n{direction} on mean")
            row_numeric.append(mean)
        rows.append(row_text)
        numeric.append(row_numeric)
    numeric_array = np.asarray(numeric)
    limit = float(np.abs(numeric_array).max())
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = mpl.colormaps["RdBu_r"]
    ax.axis("off")
    ax.set_title("Component effects: ΔMAE", fontsize=14, fontweight="bold", pad=10)
    table = ax.table(
        cellText=rows,
        rowLabels=[label for label, _, _ in COMPARISONS],
        colLabels=("PEMS04", "PEMS08"),
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.27, 0.27),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.55)
    for row in range(len(COMPARISONS)):
        for column in range(2):
            cell = table[(row + 1, column)]
            cell.set_facecolor(blend_with_white(cmap(norm(numeric_array[row, column])), 0.48))
            cell.set_edgecolor("#AAB7C4")
            cell.get_text().set_fontweight("bold")
    for column in range(2):
        cell = table[(0, column)]
        cell.set_facecolor("#E9EEF3")
        cell.set_edgecolor("#7A8793")
        cell.get_text().set_fontweight("bold")
    for row in range(1, len(COMPARISONS) + 1):
        cell = table[(row, -1)]
        cell.set_facecolor("#F6F8FA")
        cell.set_edgecolor("#AAB7C4")
        cell.get_text().set_fontweight("bold")
    ax.text(
        0.5,
        0.0,
        "Δ = left model − right model. Negative values are better.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4A5560",
        transform=ax.transAxes,
    )


def draw_decision_panel(ax: mpl.axes.Axes) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.96,
        "Pre-registered Go / No-Go",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
        transform=ax.transAxes,
    )
    lines = (
        ("Dataset means < 0", "1 / 2", "required 2 / 2"),
        ("Negative seed pairs", "3 / 6", "required ≥ 5 / 6"),
        ("CI upper bound < 0", "0 / 2", "required ≥ 1 / 2"),
    )
    y = 0.76
    for label, observed, required in lines:
        ax.text(
            0.03,
            y,
            label,
            fontsize=11,
            fontweight="bold",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.64,
            y,
            observed,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="right",
            transform=ax.transAxes,
        )
        ax.text(
            0.98,
            y,
            required,
            fontsize=9.5,
            color="#59636E",
            va="center",
            ha="right",
            transform=ax.transAxes,
        )
        ax.plot(
            [0.03, 0.98],
            [y - 0.085, y - 0.085],
            color="#D5DBE1",
            lw=1,
            transform=ax.transAxes,
        )
        y -= 0.22
    ax.text(
        0.5,
        0.08,
        "STOP independent method paper",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#8B1E1E",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#FDECEC", "edgecolor": "#B73A3A"},
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        -0.03,
        "The decision follows the frozen threshold; no criterion passed.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4A5560",
        transform=ax.transAxes,
    )


def main() -> int:
    args = parse_args()
    runs_path = args.runs.resolve()
    prefix = args.output_prefix.resolve()
    outputs = (prefix.with_suffix(".png"), prefix.with_suffix(".pdf"))
    if any(path.exists() for path in outputs):
        raise FileExistsError("visual table output already exists")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    values = read_runs(runs_path)
    style = {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(16, 10), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), width_ratios=(1.35, 0.65))
        top = grid[0, :].subgridspec(1, 2, wspace=0.08)
        draw_performance_table(fig.add_subplot(top[0, 0]), "pems04", values)
        draw_performance_table(fig.add_subplot(top[0, 1]), "pems08", values)
        draw_component_table(fig.add_subplot(grid[1, 0]), values)
        draw_decision_panel(fig.add_subplot(grid[1, 1]))
        fig.suptitle(
            "FDN benchmark v1: verified MAE comparison",
            fontsize=19,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.006,
            "24 runs · 3 training seeds · 100 epochs · lower MAE is better · ± is sample SD across seeds",
            ha="center",
            fontsize=9.5,
            color="#45515C",
        )
        fig.savefig(
            outputs[0],
            dpi=220,
            facecolor="white",
            metadata={
                "Title": "FDN benchmark v1 verified MAE comparison",
                "Description": f"Source data: {runs_path}",
            },
        )
        fig.savefig(
            outputs[1],
            facecolor="white",
            metadata={
                "Title": "FDN benchmark v1 verified MAE comparison",
                "Subject": f"Source data: {runs_path}",
            },
        )
        plt.close(fig)
    temporary_png = outputs[0].with_name(f".{outputs[0].name}.rgb.tmp")
    try:
        with Image.open(outputs[0]) as raster:
            raster.convert("RGB").save(
                temporary_png,
                format="PNG",
                dpi=(220, 220),
                optimize=True,
            )
        os.replace(temporary_png, outputs[0])
    finally:
        temporary_png.unlink(missing_ok=True)
    print(outputs[0])
    print(outputs[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
