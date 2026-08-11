#!/usr/bin/env python3
"""Render the verified 45-row v3 no-leak BiST-NDR supplement as PDF."""

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


EXPECTED_STATUS = "verified_no_leak_bist_ndr_supplement_generated"
EXPECTED_VALIDITY = "target_disjoint_event_no_leak_host_and_code_stratified"
REGION_ORDER = ("Alameda", "Contra Costa", "Orange")
REGION_LABELS = {
    "Alameda": "Alameda",
    "Contra Costa": "Contra<br/>Costa",
    "Orange": "Orange",
}
METHOD_ORDER = (
    "HL",
    "LSTM",
    "DCRNN",
    "AGCRN",
    "STGCN",
    "GWNET",
    "ASTGCN",
    "STTN",
    "DSTAGNN",
    "DGCRN",
    "D²STGNN",
    "BiST",
    "IGSTGNN",
    "NDR-Net",
    "BiST-NDR",
)
METHOD_LABELS = {
    **{name: name for name in METHOD_ORDER},
    "D²STGNN": "D<super>2</super>STGNN",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_verified_rows(
    csv_path: Path, manifest_path: Path
) -> tuple[list[dict[str, str]], str, dict[str, object]]:
    if csv_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("symlinked PDF inputs are forbidden")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != EXPECTED_STATUS:
        raise ValueError("supplement manifest is not verified")
    if manifest.get("validity_label") != EXPECTED_VALIDITY:
        raise ValueError("supplement validity label is not reviewed")
    proof = manifest.get("outputs", {}).get("table.csv", {})
    expected = proof.get("sha256") if isinstance(proof, dict) else None
    actual = _sha256(csv_path)
    if actual != expected:
        raise ValueError(f"CSV SHA-256 mismatch: expected {expected}, got {actual}")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 45 or manifest.get("row_count") != 45:
        raise ValueError(f"expected 45 verified rows, got {len(rows)}")
    return rows, actual, manifest


def _rank_metrics(
    rows: dict[str, dict[str, dict[str, str]]],
) -> dict[tuple[str, str, str], str]:
    ranks: dict[tuple[str, str, str], str] = {}
    for region in REGION_ORDER:
        for metric, _, _ in METRICS:
            ordered = sorted(
                (
                    (float(rows[region][method][metric]), method)
                    for method in METHOD_ORDER
                ),
                key=lambda item: (item[0], METHOD_ORDER.index(item[1])),
            )
            ranks[(region, ordered[0][1], metric)] = "best"
            ranks[(region, ordered[1][1], metric)] = "second"
    return ranks


def _decorate(value: str, rank: str | None) -> str:
    if rank == "best":
        return f"<b>{value}</b>"
    if rank == "second":
        return f"<u>{value}</u>"
    return value


def build_pdf(csv_path: Path, manifest_path: Path, output_path: Path) -> None:
    registerFont(UnicodeCIDFont("STSong-Light"))
    rows, source_sha, manifest = _load_verified_rows(csv_path, manifest_path)
    by_region: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_region[row["region"]][row["method"]] = row
    for region in REGION_ORDER:
        if set(by_region[region]) != set(METHOD_ORDER):
            raise ValueError(f"{region} method set mismatch")
    ranks = _rank_metrics(by_region)

    page_size = landscape(A3)
    page_width, _ = page_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "subtitle",
        fontName="STSong-Light",
        fontSize=7.2,
        leading=9.2,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
    )
    header_style = ParagraphStyle(
        "header",
        fontName="Times-Roman",
        fontSize=8.3,
        leading=9.2,
        alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "cell",
        fontName="Times-Roman",
        fontSize=7.5,
        leading=8.3,
        alignment=TA_CENTER,
    )
    method_style = ParagraphStyle(
        "method", parent=cell_style, alignment=TA_LEFT, leftIndent=2
    )
    note_style = ParagraphStyle(
        "note",
        fontName="STSong-Light",
        fontSize=6.6,
        leading=8.1,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#333333"),
    )

    def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
        return Paragraph(text, style)

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
        ["", "", *[paragraph(metric[2], header_style) for metric in METRICS]],
    ]
    region_ranges: dict[str, tuple[int, int]] = {}
    method_rows: dict[tuple[str, str], int] = {}
    for region in REGION_ORDER:
        start = len(table_data)
        for index, method in enumerate(METHOD_ORDER):
            source = by_region[region][method]
            current: list[Paragraph | str] = [
                paragraph(REGION_LABELS[region], cell_style) if index == 0 else "",
                paragraph(METHOD_LABELS[method], method_style),
            ]
            for metric, _, _ in METRICS:
                value = f"{float(source[metric]):.2f}"
                current.append(
                    paragraph(
                        _decorate(value, ranks.get((region, method, metric))),
                        cell_style,
                    )
                )
            method_rows[(region, method)] = len(table_data)
            table_data.append(current)
        region_ranges[region] = (start, len(table_data) - 1)

    usable_width = page_width - 16 * mm
    data_width = 21 * mm
    method_width = 26 * mm
    metric_width = (usable_width - data_width - method_width) / 12
    table = Table(
        table_data,
        colWidths=[data_width, method_width] + [metric_width] * 12,
        rowHeights=[7.0 * mm, 6.0 * mm] + [4.45 * mm] * 45,
        hAlign="CENTER",
        repeatRows=2,
    )
    style: list[tuple] = [
        ("SPAN", (0, 0), (0, 1)),
        ("SPAN", (1, 0), (1, 1)),
        ("SPAN", (2, 0), (4, 0)),
        ("SPAN", (5, 0), (7, 0)),
        ("SPAN", (8, 0), (10, 0)),
        ("SPAN", (11, 0), (13, 0)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.0),
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
        style.extend(
            [
                ("SPAN", (0, start), (0, end)),
                ("LINEABOVE", (0, start), (-1, start), 0.7, colors.black),
                (
                    "LINEABOVE",
                    (1, method_rows[(region, "BiST")]),
                    (-1, method_rows[(region, "BiST")]),
                    0.45,
                    colors.HexColor("#555555"),
                ),
                (
                    "BACKGROUND",
                    (1, method_rows[(region, "NDR-Net")]),
                    (-1, method_rows[(region, "BiST-NDR")]),
                    colors.HexColor("#F1F4F7"),
                ),
            ]
        )
    style.append(("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black))
    table.setStyle(TableStyle(style))

    def page_canvas(canvas, _doc) -> None:
        canvas.saveState()
        canvas.setTitle("Protocol v3 No-Leak Event Comparison with BiST-NDR")
        canvas.setAuthor("XTraffic Project")
        canvas.setSubject("Verified target-disjoint event comparison supplement")
        canvas.setFont("Times-Roman", 6.1)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(
            page_width - 8 * mm,
            5 * mm,
            f"Verified table.csv SHA-256: {source_sha}   |   Page 1",
        )
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=page_size,
        leftMargin=8 * mm,
        rightMargin=8 * mm,
        topMargin=5 * mm,
        bottomMargin=9 * mm,
        title="Protocol v3 No-Leak Event Comparison with BiST-NDR",
        author="XTraffic Project",
    )
    story = [
        paragraph("Protocol v3 No-Leak Event Comparison with BiST-NDR", title_style),
        paragraph(
            "三地区 15 方法补充结果。全部使用 seed 42、最多 30 epoch、target-disjoint event split 与显式观测 mask。",
            subtitle_style,
        ),
        Spacer(1, 1.2 * mm),
        table,
        Spacer(1, 1.2 * mm),
        paragraph(
            "<b>说明：</b>数值越低越好，粗体为表内最优，下划线为次优。BiST-NDR 是将 NDR 插件接入 BiST backbone 的项目扩展。",
            note_style,
        ),
        paragraph(
            "<b>有效性：</b>标签为 target_disjoint_event_no_leak_host_and_code_stratified。原 42 行保持只读，新增三行来自 RTX 5080 独立补跑。该表为单种子描述性比较。",
            note_style,
        ),
    ]
    if manifest.get("seed_policy") != "single_seed_42":
        raise ValueError("unexpected seed policy")
    document.build(story, onFirstPage=page_canvas, onLaterPages=page_canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pdf(args.csv, args.manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
