#!/usr/bin/env python3
"""Build a paper-style PDF table from the verified IGSTGNN Table 4 CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REGION_ORDER = ("Alameda", "Contra_Costa", "Orange")
REGION_LABELS = {
    "Alameda": "Alameda",
    "Contra_Costa": "Contra<br/>Costa",
    "Orange": "Orange",
}
METHOD_ORDER = (
    "hl",
    "lstm",
    "dcrnn",
    "agcrn",
    "stgcn",
    "gwnet",
    "astgcn",
    "sttn",
    "dstagnn",
    "dgcrn",
    "d2stgnn",
    "bist",
    "igstgnn",
    "ndr_net",
)
METHOD_LABELS = {
    "hl": "HL",
    "lstm": "LSTM",
    "dcrnn": "DCRNN",
    "agcrn": "AGCRN",
    "stgcn": "STGCN",
    "gwnet": "GWNET",
    "astgcn": "ASTGCN",
    "sttn": "STTN",
    "dstagnn": "DSTAGNN",
    "dgcrn": "DGCRN",
    "d2stgnn": "D<super>2</super>STGNN",
    "bist": "BiST",
    "igstgnn": "IGSTGNN",
    "ndr_net": "NDR-Net*",
}
METRICS = (
    ("h3_mae", "Horizon 3", "MAE"),
    ("h3_rmse", "Horizon 3", "RMSE"),
    ("h3_mape_percent", "Horizon 3", "MAPE(%)"),
    ("h6_mae", "Horizon 6", "MAE"),
    ("h6_rmse", "Horizon 6", "RMSE"),
    ("h6_mape_percent", "Horizon 6", "MAPE(%)"),
    ("h12_mae", "Horizon 12", "MAE"),
    ("h12_rmse", "Horizon 12", "RMSE"),
    ("h12_mape_percent", "Horizon 12", "MAPE(%)"),
    ("avg_mae", "Average", "MAE"),
    ("avg_rmse", "Average", "RMSE"),
    ("avg_mape_percent", "Average", "MAPE(%)"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_rows(csv_path: Path, manifest_path: Path) -> tuple[list[dict[str, str]], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "verified_table4_with_ndr_extension":
        raise ValueError("Table manifest is not verified")
    expected = manifest["outputs"]["table4_with_ndr_extension.csv"]["sha256"]
    actual = sha256_file(csv_path)
    if actual != expected:
        raise ValueError(f"CSV SHA-256 mismatch: expected {expected}, got {actual}")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 42:
        raise ValueError(f"Expected 42 display rows, got {len(rows)}")
    return rows, actual


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def rank_metrics(
    rows_by_region: dict[str, dict[str, dict[str, str]]],
) -> dict[tuple[str, str, str], str]:
    ranks: dict[tuple[str, str, str], str] = {}
    for region in REGION_ORDER:
        for metric, _, _ in METRICS:
            ordered = sorted(
                (
                    (float(rows_by_region[region][method][metric]), method)
                    for method in METHOD_ORDER
                ),
                key=lambda item: (item[0], METHOD_ORDER.index(item[1])),
            )
            ranks[(region, ordered[0][1], metric)] = "best"
            ranks[(region, ordered[1][1], metric)] = "second"
    return ranks


def metric_text(row: dict[str, str], metric: str) -> str:
    value = float(row[metric])
    if row["method"] == "ndr_net":
        std = float(row[f"{metric}_sample_std"])
        return f"{value:.2f} &#177; {std:.2f}"
    return f"{value:.2f}"


def decorate_rank(text: str, rank: str | None) -> str:
    if rank == "best":
        return f"<b>{text}</b>"
    if rank == "second":
        return f"<u>{text}</u>"
    return text


def build_pdf(csv_path: Path, manifest_path: Path, output_path: Path) -> None:
    registerFont(UnicodeCIDFont("STSong-Light"))
    rows, source_sha = load_verified_rows(csv_path, manifest_path)
    rows_by_region: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        rows_by_region[row["region"]][row["method"]] = row
    for region in REGION_ORDER:
        missing = set(METHOD_ORDER) - set(rows_by_region[region])
        if missing:
            raise ValueError(f"{region} is missing methods: {sorted(missing)}")

    ranks = rank_metrics(rows_by_region)
    page_size = landscape(A3)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=16,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "subtitle",
        fontName="STSong-Light",
        fontSize=7.6,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
    )
    header_style = ParagraphStyle(
        "header",
        fontName="Times-Roman",
        fontSize=8.8,
        leading=10,
        alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "cell",
        fontName="Times-Roman",
        fontSize=8.0,
        leading=9.2,
        alignment=TA_CENTER,
    )
    method_style = ParagraphStyle(
        "method",
        parent=cell_style,
        alignment=TA_LEFT,
        leftIndent=2,
    )
    note_style = ParagraphStyle(
        "note",
        fontName="STSong-Light",
        fontSize=7,
        leading=9,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#333333"),
    )

    table_data: list[list[Paragraph | str]] = [
        [
            paragraph("Data", header_style),
            paragraph("Method", header_style),
            paragraph("Horizon 3", header_style),
            "",
            "",
            paragraph("Horizon 6", header_style),
            "",
            "",
            paragraph("Horizon 12", header_style),
            "",
            "",
            paragraph("Average", header_style),
            "",
            "",
        ],
        [
            "",
            "",
            *[paragraph(metric[2], header_style) for metric in METRICS],
        ],
    ]

    region_ranges: dict[str, tuple[int, int]] = {}
    method_rows: dict[tuple[str, str], int] = {}
    for region in REGION_ORDER:
        start = len(table_data)
        for method_index, method in enumerate(METHOD_ORDER):
            row = rows_by_region[region][method]
            table_row: list[Paragraph | str] = [
                paragraph(REGION_LABELS[region], cell_style) if method_index == 0 else "",
                paragraph(METHOD_LABELS[method], method_style),
            ]
            for metric, _, _ in METRICS:
                rendered = metric_text(row, metric)
                rendered = decorate_rank(rendered, ranks.get((region, method, metric)))
                table_row.append(paragraph(rendered, cell_style))
            method_rows[(region, method)] = len(table_data)
            table_data.append(table_row)
        region_ranges[region] = (start, len(table_data) - 1)

    page_width, _ = page_size
    usable_width = page_width - 16 * mm
    data_width = 21 * mm
    method_width = 27 * mm
    metric_width = (usable_width - data_width - method_width) / 12
    col_widths = [data_width, method_width] + [metric_width] * 12
    row_heights = [7.6 * mm, 6.7 * mm] + [4.9 * mm] * 42

    table = Table(
        table_data,
        colWidths=col_widths,
        rowHeights=row_heights,
        hAlign="CENTER",
        repeatRows=2,
    )
    table_style = [
        ("SPAN", (0, 0), (0, 1)),
        ("SPAN", (1, 0), (1, 1)),
        ("SPAN", (2, 0), (4, 0)),
        ("SPAN", (5, 0), (7, 0)),
        ("SPAN", (8, 0), (10, 0)),
        ("SPAN", (11, 0), (13, 0)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.2),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 1), (-1, 1), 0.65, colors.black),
        ("LINEAFTER", (1, 0), (1, -1), 0.55, colors.HexColor("#444444")),
        ("LINEAFTER", (4, 0), (4, -1), 0.55, colors.HexColor("#444444")),
        ("LINEAFTER", (7, 0), (7, -1), 0.55, colors.HexColor("#444444")),
        ("LINEAFTER", (10, 0), (10, -1), 0.55, colors.HexColor("#444444")),
        ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor("#F5F5F5")),
    ]
    for region in REGION_ORDER:
        start, end = region_ranges[region]
        table_style.extend(
            [
                ("SPAN", (0, start), (0, end)),
                ("LINEABOVE", (0, start), (-1, start), 0.7, colors.black),
                (
                    "LINEABOVE",
                    (1, method_rows[(region, "igstgnn")]),
                    (-1, method_rows[(region, "igstgnn")]),
                    0.45,
                    colors.HexColor("#555555"),
                ),
                (
                    "LINEABOVE",
                    (1, method_rows[(region, "ndr_net")]),
                    (-1, method_rows[(region, "ndr_net")]),
                    0.45,
                    colors.HexColor("#777777"),
                ),
                (
                    "BACKGROUND",
                    (1, method_rows[(region, "ndr_net")]),
                    (-1, method_rows[(region, "ndr_net")]),
                    colors.HexColor("#F1F4F7"),
                ),
            ]
        )
    table_style.append(
        ("LINEBELOW", (0, region_ranges["Orange"][1]), (-1, -1), 1.0, colors.black)
    )
    table.setStyle(TableStyle(table_style))

    def page_canvas(canvas, _doc) -> None:
        canvas.saveState()
        canvas.setTitle("IGSTGNN Table 4 Reproduction Results with NDR-Net")
        canvas.setAuthor("XTraffic Project")
        canvas.setSubject("Verified IGSTGNN Table 4 public-material reproduction")
        canvas.setFont("Times-Roman", 6.2)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(
            page_width - 8 * mm,
            5 * mm,
            f"Verified source SHA-256: {source_sha}   |   Page 1",
        )
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=page_size,
        leftMargin=8 * mm,
        rightMargin=8 * mm,
        topMargin=7 * mm,
        bottomMargin=10 * mm,
        title="IGSTGNN Table 4 Reproduction Results with NDR-Net",
        author="XTraffic Project",
    )
    story = [
        paragraph(
            "IGSTGNN Table 4 Public-Material Reproduction with NDR-Net Extension",
            title_style,
        ),
        paragraph(
            "三地区正式复现结果。Horizon 3、6、12 分别对应 15、30、60 分钟，Average 为全部 12 个预测步的平均。",
            subtitle_style,
        ),
        Spacer(1, 2.2 * mm),
        table,
        Spacer(1, 2.2 * mm),
        paragraph(
            "<b>说明：</b>数值越低越好，粗体为当前表内最优点估计，下划线为次优点估计。"
            "前 13 个方法为公开材料最大忠实度的单次复现。NDR-Net* 是项目方法扩展，"
            "报告种子 42、1、2 的均值 &#177; 样本标准差。两类数字仅作描述性并列表达，"
            "不构成同等重复次数下的统计比较。",
            note_style,
        ),
        paragraph(
            "<b>有效性：</b>本表使用 IGSTGNN 官方协议，保留已知预测时间重叠，"
            "标签为 official_protocol_with_known_overlap。它不是 protocol_v3_noleak_dedup 无泄露结果。",
            note_style,
        ),
    ]
    document.build(story, onFirstPage=page_canvas, onLaterPages=page_canvas)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build_pdf(arguments.csv, arguments.manifest, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
