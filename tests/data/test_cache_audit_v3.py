import json

import h5py
import pytest

from scripts.audit_protocol_v3_cache import audit_cache_bundle
from tests.data.test_region_data_v2 import _write_v3_fixture


def test_cache_audit_writes_verified_bundle_manifest(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)

    manifest = audit_cache_bundle(
        data_dir,
        graph_dir,
        regions=("Alameda",),
        enforce_expected_counts=False,
        write_manifest=True,
    )

    assert manifest["status"] == "verified"
    assert manifest["bundle_id"] == "bundle-test"
    assert len(manifest["cache_bundle_digest"]) == 64
    assert manifest["regions"]["Alameda"]["sample_count"] == 3
    assert manifest["regions"]["Alameda"]["unique_forecast_origins"] == 3
    on_disk = json.loads(
        (data_dir / "cache_bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk == manifest


def test_cache_audit_rejects_dirty_formal_artifact(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    with h5py.File(data_dir / "Alameda_samples.h5", "r+") as samples:
        samples.attrs["git_dirty"] = True

    with pytest.raises(RuntimeError, match="dirty.*formal|formal.*dirty"):
        audit_cache_bundle(
            data_dir,
            graph_dir,
            regions=("Alameda",),
            enforce_expected_counts=False,
            write_manifest=False,
        )
