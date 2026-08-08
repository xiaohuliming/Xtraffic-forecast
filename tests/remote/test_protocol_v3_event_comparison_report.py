from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tests.remote.test_protocol_v3_event_extension_postflight import (
    _publish_extension_host,
)
from tests.remote.test_protocol_v3_event_multi_gpu_postflight import (
    _publish_verified_host,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_indexes(tmp_path: Path):
    from scripts.remote.linux.protocol_v3_event_full33_composite import (
        build_extension9_composite_index,
        build_full33_composite_index,
    )

    baseline_root, _ = _publish_verified_host(tmp_path / "baseline")
    extension_root = _publish_extension_host(tmp_path / "extension")
    baseline_index = tmp_path / "indexes" / "baseline.json"
    extension_index = tmp_path / "indexes" / "extension.json"
    build_full33_composite_index(
        published_root=baseline_root,
        output_path=baseline_index,
        expected_matrix_id="protocol_v3_event_cc_orange_test",
        expected_host_id="event_test_multi_gpu",
    )
    build_extension9_composite_index(
        published_root=extension_root,
        output_path=extension_index,
        expected_matrix_id="protocol_v3_event_extensions_test",
        expected_host_id="event_extensions_test",
        expected_sources=4,
    )
    return baseline_root, baseline_index, extension_root, extension_index


def test_verified_indexes_generate_one_42_row_no_leak_table(tmp_path: Path) -> None:
    from scripts.protocol_v3_event_comparison_report import (
        build_event_comparison_report,
    )

    baseline_root, baseline_index, extension_root, extension_index = _build_indexes(
        tmp_path
    )
    output = tmp_path / "table"
    manifest = build_event_comparison_report(
        baseline_index_path=baseline_index,
        baseline_index_sha256=_sha256(baseline_index),
        baseline_published_root=baseline_root,
        extension_index_path=extension_index,
        extension_index_sha256=_sha256(extension_index),
        extension_published_root=extension_root,
        output_dir=output,
    )

    with (output / "table.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 42
    assert [row["method"] for row in rows[:14]] == [
        "HL", "LSTM", "DCRNN", "AGCRN", "STGCN", "GWNET", "ASTGCN",
        "STTN", "DSTAGNN", "DGCRN", "D²STGNN", "BiST", "IGSTGNN", "NDR-Net",
    ]
    assert {row["region"] for row in rows} == {"Alameda", "Contra Costa", "Orange"}
    assert manifest["validity_label"] == "target_disjoint_event_no_leak"
    assert manifest["row_count"] == 42
    assert (output / "table.md").is_file()
    assert (output / "table.tex").is_file()


def test_merge_rejects_different_cache_bundle_bindings(tmp_path: Path) -> None:
    from scripts.protocol_v3_event_comparison_report import (
        EventComparisonReportError,
        build_event_comparison_report,
    )

    baseline_root, baseline_index, extension_root, extension_index = _build_indexes(
        tmp_path
    )
    payload = json.loads(extension_index.read_text(encoding="utf-8"))
    payload["cache_bundle_digest"] = "d" * 64
    changed = tmp_path / "indexes" / "extension_changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EventComparisonReportError, match="cache bundle"):
        build_event_comparison_report(
            baseline_index_path=baseline_index,
            baseline_index_sha256=_sha256(baseline_index),
            baseline_published_root=baseline_root,
            extension_index_path=changed,
            extension_index_sha256=_sha256(changed),
            extension_published_root=extension_root,
            output_dir=tmp_path / "table",
        )
