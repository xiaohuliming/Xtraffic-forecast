from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tests.remote.test_protocol_v3_event_multi_gpu_postflight import (
    _publish_verified_host,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hostinger_post_publish_wrapper_is_memory_limited_and_identity_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (
        root / "scripts/remote/linux/protocol_v3_event_full33_post_publish.sh"
    ).read_text(encoding="utf-8")

    assert "systemd-run" in script
    assert "MemoryMax=3221225472" in script
    assert "MemorySwapMax=4294967296" in script
    assert "protocol_v3_event_full33_5090x4_v1" in script
    assert "event_rtx5090x4_full33" in script
    assert "protocol_v3_event_full33_composite" in script
    assert "protocol_v3_event_baseline_report" in script


def test_full33_composite_revalidates_the_published_four_source_tree(
    tmp_path: Path,
) -> None:
    from scripts.remote.linux.protocol_v3_event_full33_composite import (
        build_full33_composite_index,
    )

    published, _ = _publish_verified_host(tmp_path)
    output = tmp_path / "composite" / "composite_index.json"

    index = build_full33_composite_index(
        published_root=published,
        output_path=output,
        expected_matrix_id="protocol_v3_event_cc_orange_test",
        expected_host_id="event_test_multi_gpu",
    )

    assert index["status"] == "verified_event_full33_single_host_33"
    assert index["execution_layout"] == "single_host_four_gpu_interleaved"
    assert index["task_count"] == 33
    assert index["source_count"] == 4
    assert [item["task_id"] for item in index["tasks"]] == [
        f"v3e{number:03d}_" + item["task_id"].split("_", 1)[1]
        for number, item in enumerate(index["tasks"], start=1)
    ]
    graphwavenet = next(
        item for item in index["tasks"] if item["model"] == "graphwavenet"
    )
    for group in ("h3", "h6", "h12", "average"):
        assert graphwavenet["metrics"][group] == pytest.approx(
            {"mae": 1.0, "rmse": 1.0, "mape_percent": 10.0}
        )
    assert output.is_file()


def test_full33_composite_rejects_a_result_changed_after_source_postflight(
    tmp_path: Path,
) -> None:
    from scripts.remote.linux.protocol_v3_event_full33_composite import (
        Full33CompositeError,
        build_full33_composite_index,
    )

    published, _ = _publish_verified_host(tmp_path)
    summary = next((published / "results").rglob("summary.json"))
    summary.write_bytes(summary.read_bytes() + b"\n")

    with pytest.raises(Full33CompositeError, match="hash|size"):
        build_full33_composite_index(
            published_root=published,
            output_path=tmp_path / "index.json",
            expected_matrix_id="protocol_v3_event_cc_orange_test",
            expected_host_id="event_test_multi_gpu",
        )


def test_full33_composite_rejects_the_wrong_launch_identity(tmp_path: Path) -> None:
    from scripts.remote.linux.protocol_v3_event_full33_composite import (
        Full33CompositeError,
        build_full33_composite_index,
    )

    published, _ = _publish_verified_host(tmp_path)
    with pytest.raises(Full33CompositeError, match="matrix_id"):
        build_full33_composite_index(
            published_root=published,
            output_path=tmp_path / "index.json",
            expected_matrix_id="wrong-matrix",
            expected_host_id="event_test_multi_gpu",
        )


def test_verified_full33_index_generates_the_paper_shape_table(
    tmp_path: Path,
) -> None:
    from scripts.protocol_v3_event_baseline_report import build_event_baseline_report
    from scripts.remote.linux.protocol_v3_event_full33_composite import (
        build_full33_composite_index,
    )

    published, _ = _publish_verified_host(tmp_path)
    index_path = tmp_path / "composite" / "composite_index.json"
    build_full33_composite_index(
        published_root=published,
        output_path=index_path,
        expected_matrix_id="protocol_v3_event_cc_orange_test",
        expected_host_id="event_test_multi_gpu",
    )
    output_dir = tmp_path / "table"

    manifest = build_event_baseline_report(
        composite_index_path=index_path,
        expected_index_sha256=_sha256(index_path),
        published_root=published,
        output_dir=output_dir,
    )

    with (output_dir / "table.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 33
    assert rows[0]["region"] == "Alameda"
    assert rows[0]["method"] == "HL"
    assert rows[-1]["region"] == "Orange"
    assert rows[-1]["method"] == "D²STGNN"
    assert set(rows[0]) == {
        "region", "method",
        "h3_mae", "h3_rmse", "h3_mape_percent",
        "h6_mae", "h6_rmse", "h6_mape_percent",
        "h12_mae", "h12_rmse", "h12_mape_percent",
        "avg_mae", "avg_rmse", "avg_mape_percent",
    }
    assert manifest["protocol_version"] == "protocol_v3_noleak_dedup"
    assert manifest["seed_policy"] == "single_seed_42"
    assert (output_dir / "table.md").is_file()
    assert (output_dir / "table.tex").is_file()
