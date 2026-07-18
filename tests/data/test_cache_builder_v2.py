from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

from scripts.build_full_county_cache import build_samples_h5
from dist_net.protocol import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION


def test_sample_cache_drops_mixed_event_groups_and_separates_analysis_labels(
    tmp_path,
):
    events = pd.DataFrame(
        {
            "start_idx": [30, 57, 85],
            "anchor_region_idx": [0, 0, 1],
            "incident_ids": ["a", "b", "c"],
            "primary_type": ["Fire", "Hazard", "NoInj"],
            "latitude": [37.0, 37.1, 37.2],
            "longitude": [-122.0, -122.1, -122.2],
            "incident_abs_pm": [1.0, 2.0, 3.0],
            "duration_min": [20, 200, 40],
            "severity_any_z_auc_topk": [1.0, 99.0, 2.0],
        }
    )
    args = SimpleNamespace(
        smoke_events=0,
        sample_offsets=[0, 6, 12],
        t_h=12,
        t_p=12,
        active_window=24,
        m_max=4,
        split_train_end=0.70,
        split_val_end=0.85,
    )
    output = tmp_path / "Alameda_samples.h5"

    build_samples_h5(
        region_name="Alameda",
        region_node_idx=np.array([0, 1], dtype=np.int64),
        events=events,
        affected_lookup={"a": np.array([0]), "b": np.array([1])},
        total_steps=120,
        args=args,
        out_path=output,
        train_end=70,
        val_end=85,
        baseline_fit_end=70,
        manifest=[],
        git_info={"git_sha": "test", "git_branch": "test", "git_dirty": False},
    )

    with h5py.File(output, "r") as h5:
        assert h5.attrs["cache_schema_version"] == CACHE_SCHEMA_VERSION
        assert h5.attrs["protocol_version"] == PROTOCOL_VERSION
        assert h5.attrs["train_end"] == 70
        assert h5.attrs["baseline_fit_end"] == 70
        assert h5.attrs["imputation_method"] == "causal_forward_fill_zero_prefix"
        assert h5.attrs["baseline_method"] == "label_free_train_only_robust_median"
        assert h5.attrs["source_manifest_json"] == "[]"
        assert h5.attrs["git_sha"] == "test"
        assert "affected_mask" not in h5
        assert "analysis_labels/affected_mask" in h5
        assert h5["incident_feat"].shape[-1] == 12
        assert 1 not in set(h5["event_idx"][:].tolist())
        active = h5["active_event_idx"][:]
        slots = h5["primary_incident_slot"][:]
        rows = np.arange(len(slots))
        np.testing.assert_array_equal(
            active[rows, slots],
            h5["event_idx"][:],
        )
