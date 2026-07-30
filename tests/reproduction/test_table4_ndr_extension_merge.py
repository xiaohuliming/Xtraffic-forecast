from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from scripts.reproduction.igstgnn_table4.ndr_extension_merge import (
    build_table4_with_ndr_extension,
)
from scripts.reproduction.igstgnn_table4.ndr_extension_report import (
    build_ndr_extension_report,
)
from tests.reproduction.test_table4_ndr_extension_report import (
    _write_ndr_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_TABLE = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "paper_table4.reference_v1.csv"
)


def test_combined_table_keeps_39_rows_and_appends_three_ndr_rows(
    tmp_path: Path,
) -> None:
    snapshot, receipt_sha256 = _write_ndr_snapshot(tmp_path)
    ndr_dir = tmp_path / "ndr"
    build_ndr_extension_report(
        snapshot_root=snapshot,
        expected_receipt_sha256=receipt_sha256,
        output_dir=ndr_dir,
    )
    ndr_index = ndr_dir / "ndr_extension_index.json"
    output_dir = tmp_path / "combined"

    result = build_table4_with_ndr_extension(
        base_table_csv=BASE_TABLE,
        expected_base_table_sha256=hashlib.sha256(
            BASE_TABLE.read_bytes()
        ).hexdigest(),
        ndr_extension_index=ndr_index,
        expected_ndr_index_sha256=hashlib.sha256(
            ndr_index.read_bytes()
        ).hexdigest(),
        output_dir=output_dir,
    )

    assert result["status"] == "verified_table4_with_ndr_extension"
    assert result["base_row_count"] == 39
    assert result["extension_row_count"] == 3
    with (output_dir / "table4_with_ndr_extension.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 42
    assert rows[-3]["region"] == "Alameda"
    assert rows[-3]["method"] == "ndr_net"
    assert rows[-3]["avg_mae"] == "2.010000"
    assert rows[-3]["avg_mae_sample_std"] == "1.000000"
    markdown = (
        output_dir / "table4_with_ndr_extension.md"
    ).read_text(encoding="utf-8")
    assert "NDR-Net*" in markdown
    assert "2.01 ± 1.00" in markdown
    assert (
        output_dir / "table4_with_ndr_extension.tex"
    ).is_file()
