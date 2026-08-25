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
    ("gwn", "Graph WaveNet\n单分支基线"),
    ("dual_gwn", "DualGWN\n双分支容量对照"),
    ("fdn_fixed_k3", "FDN Fixed-Mask\n固定掩码消融"),
    ("fdn_learnable_k3", "FDN Learnable-Mask\n完整模型"),
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
        colLabels=("种子 42", "种子 1", "种子 2", "均值 ± 标准差"),
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
        "font.family": ("Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), layout="constrained")
        draw_performance_table(axes[0], "pems04", values)
        draw_performance_table(axes[1], "pems08", values)
        fig.suptitle(
            "FDN 消融实验：三种子 MAE 对比",
            fontsize=19,
            fontweight="bold",
        )
        fig.savefig(
            outputs[0],
            dpi=220,
            facecolor="white",
            metadata={
                "Title": "FDN 消融实验：三种子 MAE 对比",
                "Description": f"Source data: {runs_path}",
            },
        )
        fig.savefig(
            outputs[1],
            facecolor="white",
            metadata={
                "Title": "FDN 消融实验：三种子 MAE 对比",
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
