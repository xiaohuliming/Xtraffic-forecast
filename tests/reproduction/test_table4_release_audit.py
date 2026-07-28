import json

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4.release_audit import (
    REQUIRED_RELEASE_FILES,
    audit_official_release,
)


def _write_header_only_region(
    root,
    *,
    region,
    node_count,
    sample_count,
    description_count,
):
    region_dir = root / region
    region_dir.mkdir()
    np.save(
        region_dir / "adj_matrix.npy",
        np.zeros((node_count, node_count), dtype=np.float64, order="F"),
    )
    np.save(
        region_dir / "incident_all.npy",
        np.empty(sample_count, dtype=object),
    )
    np.savez(
        region_dir / "incident_stats.npz",
        mean=np.float32(1.0),
        std=np.float32(2.0),
        normalized=np.array(True),
    )
    sensor_rows = [
        "order,Sensor Type,Surface,Roadway Use,Road Width,Design Speed Limit"
    ]
    sensor_rows.extend(
        f"{42 + index},loops,Concrete,No Special Features,80 ft,70 mph"
        for index in range(node_count)
    )
    (region_dir / "sensors.csv").write_text(
        "\n".join(sensor_rows) + "\n",
        encoding="utf-8",
    )
    (region_dir / "desc_mapping.json").write_text(
        json.dumps(
            {f"description_{index}": index for index in range(description_count)}
        ),
        encoding="utf-8",
    )
    (region_dir / "type_mapping.json").write_text(
        json.dumps({f"type_{index}": index for index in range(6)}),
        encoding="utf-8",
    )


def _write_header_only_release(root):
    for region, node_count, sample_count, description_count in (
        ("Alameda", 521, 14_687, 30),
        ("Contra_Costa", 496, 5_585, 29),
        ("Orange", 990, 18_700, 32),
    ):
        _write_header_only_region(
            root,
            region=region,
            node_count=node_count,
            sample_count=sample_count,
            description_count=description_count,
        )


def test_release_audit_hashes_all_18_files_and_reports_official_sample_counts(
    tmp_path,
):
    _write_header_only_release(tmp_path)

    manifest = audit_official_release(tmp_path)

    assert manifest["status"] == "safe_structure_verified"
    assert manifest["release_id"] == "kaggle_lixiangfan_data4igstgnn_v1"
    assert len(manifest["release_digest"]) == 64
    assert set(manifest["regions"]) == {
        "Alameda",
        "Contra_Costa",
        "Orange",
    }
    assert sum(len(region["files"]) for region in manifest["regions"].values()) == 18
    assert manifest["regions"]["Alameda"]["sample_count"] == 14_687
    assert manifest["regions"]["Contra_Costa"]["sample_count"] == 5_585
    assert manifest["regions"]["Orange"]["sample_count"] == 18_700
    assert manifest["regions"]["Contra_Costa"]["split_counts"] == {
        "train": 3_909,
        "val": 837,
        "test": 839,
    }
    assert set(REQUIRED_RELEASE_FILES) == set(manifest["regions"]["Orange"]["files"])
    assert manifest["regions"]["Alameda"]["safe_structure"] == {
        "adjacency": {
            "diagonal_zero": True,
            "dtype": "float64",
            "finite": True,
            "fortran_order": True,
            "maximum": 0.0,
            "minimum": 0.0,
            "nonnegative": True,
            "symmetric": True,
        },
        "incident_stats": {
            "mean": 1.0,
            "normalized": True,
            "std": 2.0,
        },
        "mappings": {
            "description_count": 30,
            "type_count": 6,
        },
        "sensors": {
            "order_is_unique": True,
            "road_width_fallback_count": 521,
            "row_count": 521,
            "speed_limit_fallback_count": 521,
        },
    }


def test_release_audit_writes_the_verified_manifest_as_strict_json(tmp_path):
    _write_header_only_release(tmp_path)
    output = tmp_path / "release_manifest.json"

    manifest = audit_official_release(tmp_path, write_manifest=output)

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert output.read_bytes().endswith(b"\n")


def test_release_audit_rejects_symlinked_required_files(tmp_path):
    _write_header_only_release(tmp_path)
    external = tmp_path / "external.csv"
    external.write_text("order\n0\n", encoding="utf-8")
    sensors = tmp_path / "Alameda" / "sensors.csv"
    sensors.unlink()
    sensors.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        audit_official_release(tmp_path)
