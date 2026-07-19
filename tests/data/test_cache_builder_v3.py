from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from scripts.build_full_county_cache import (
    build_samples_h5,
    resolve_cache_mode,
    write_protocol_attrs,
)


def _builder_args() -> SimpleNamespace:
    return SimpleNamespace(
        smoke_events=0,
        sample_offsets=[0, 6, 12],
        t_h=12,
        t_p=12,
        active_window=24,
        m_max=8,
        split_train_end=0.70,
        split_val_end=0.85,
    )


def test_dirty_builds_cannot_be_marked_formal(tmp_path):
    assert resolve_cache_mode(smoke_events=0, git_dirty=False) == "formal"
    assert resolve_cache_mode(smoke_events=1, git_dirty=False) == "smoke"
    assert resolve_cache_mode(smoke_events=0, git_dirty=True) == "debug"
    assert resolve_cache_mode(smoke_events=1, git_dirty=True) == "smoke"

    dirty_git = {
        "git_sha": "a" * 40,
        "git_branch": "v3",
        "git_dirty": True,
    }
    with h5py.File(tmp_path / "formal.h5", "w") as h5:
        with pytest.raises(ValueError, match="dirty.*formal"):
            write_protocol_attrs(
                h5,
                train_end=70,
                val_end=85,
                baseline_fit_end=70,
                manifest=[],
                git_info=dirty_git,
                bundle_id="bundle-test",
                artifact_id="traffic-test",
                cache_role="traffic",
                traffic_cache_id="traffic-test",
                cache_mode="formal",
            )

    with h5py.File(tmp_path / "debug.h5", "w") as h5:
        write_protocol_attrs(
            h5,
            train_end=70,
            val_end=85,
            baseline_fit_end=70,
            manifest=[],
            git_info=dirty_git,
            bundle_id="bundle-test",
            artifact_id="traffic-test",
            cache_role="traffic",
            traffic_cache_id="traffic-test",
            cache_mode="debug",
        )
        assert h5.attrs["cache_mode"] == "debug"
        assert bool(h5.attrs["git_dirty"])


def test_forecast_origin_is_unique_and_prefers_nearest_anchor_event(tmp_path):
    events = pd.DataFrame(
        {
            "agg_id": ["event-30", "event-36", "event-42"],
            "start_idx": [30, 36, 42],
            "anchor_region_idx": [0, 1, 2],
            "incident_ids": ["a", "b", "c"],
            "primary_type": ["Fire", "Hazard", "NoInj"],
            "latitude": [37.0, 37.1, 37.2],
            "longitude": [-122.0, -122.1, -122.2],
            "incident_abs_pm": [1.0, 2.0, 3.0],
        }
    )
    output = tmp_path / "Alameda_samples.h5"

    build_samples_h5(
        region_name="Alameda",
        region_node_idx=np.arange(3, dtype=np.int64),
        events=events,
        affected_lookup={},
        total_steps=120,
        args=_builder_args(),
        out_path=output,
        train_end=84,
        val_end=102,
        baseline_fit_end=84,
        manifest=[],
        git_info={"git_sha": "test", "git_branch": "test", "git_dirty": False},
    )

    with h5py.File(output, "r") as h5:
        origins = h5["sample_start"][:]
        assert origins.tolist() == [30, 36, 42, 48, 54]
        assert len(origins) == len(np.unique(origins))

        row = int(np.flatnonzero(origins == 42)[0])
        assert int(h5["sample_offset"][row]) == 0
        assert int(h5["event_idx"][row]) == 2
        assert int(h5["anchor_region_idx"][row]) == 2


def test_origin_keeps_every_event_that_generated_the_forecast(tmp_path):
    events = pd.DataFrame(
        {
            "agg_id": ["event-30", "event-36", "event-42"],
            "start_idx": [30, 36, 42],
            "anchor_region_idx": [0, 1, 2],
            "incident_ids": ["a", "b", "c"],
            "primary_type": ["Fire", "Hazard", "NoInj"],
            "latitude": [37.0, 37.1, 37.2],
            "longitude": [-122.0, -122.1, -122.2],
            "incident_abs_pm": [1.0, 2.0, 3.0],
        }
    )
    output = tmp_path / "Alameda_samples.h5"

    build_samples_h5(
        region_name="Alameda",
        region_node_idx=np.arange(3, dtype=np.int64),
        events=events,
        affected_lookup={},
        total_steps=120,
        args=_builder_args(),
        out_path=output,
        train_end=84,
        val_end=102,
        baseline_fit_end=84,
        manifest=[],
        git_info={"git_sha": "test", "git_branch": "test", "git_dirty": False},
    )

    with h5py.File(output, "r") as h5:
        row = int(np.flatnonzero(h5["sample_start"][:] == 42)[0])
        active = h5["active_event_idx"][row]
        anchor_mask = h5["anchor_event_mask"][row]
        assert set(active[anchor_mask].tolist()) == {0, 1, 2}
        assert int(h5["n_origin_events"][row]) == 3


def test_analysis_union_and_active_count_include_truncated_events(tmp_path):
    events = pd.DataFrame(
        {
            "agg_id": [f"event-{start}" for start in range(30, 34)],
            "start_idx": list(range(30, 34)),
            "anchor_region_idx": list(range(4)),
            "incident_ids": list("abcd"),
            "primary_type": ["Fire"] * 4,
            "latitude": [37.0] * 4,
            "longitude": [-122.0] * 4,
            "incident_abs_pm": list(range(4)),
        }
    )
    args = _builder_args()
    args.sample_offsets = [0]
    args.m_max = 2
    output = tmp_path / "Alameda_samples.h5"

    build_samples_h5(
        region_name="Alameda",
        region_node_idx=np.arange(4, dtype=np.int64),
        events=events,
        affected_lookup={
            event_id: np.asarray([idx], dtype=np.int32)
            for idx, event_id in enumerate("abcd")
        },
        total_steps=120,
        args=args,
        out_path=output,
        train_end=84,
        val_end=102,
        baseline_fit_end=84,
        manifest=[],
        git_info={"git_sha": "test", "git_branch": "test", "git_dirty": False},
    )

    with h5py.File(output, "r") as h5:
        row = int(np.flatnonzero(h5["sample_start"][:] == 33)[0])
        assert h5["analysis_labels/affected_mask"][row].tolist() == [
            True,
            True,
            True,
            True,
        ]
        assert int(h5["n_active_incidents"][row]) == 4
        assert int(h5["n_packed_incidents"][row]) == 2


def test_active_event_identity_is_disjoint_across_splits(tmp_path):
    events = pd.DataFrame(
        {
            "agg_id": ["train-event", "boundary-val-event", "val-event"],
            "start_idx": [77, 90, 110],
            "anchor_region_idx": [0, 1, 2],
            "incident_ids": ["a", "b", "c"],
            "primary_type": ["Fire", "Hazard", "NoInj"],
            "latitude": [37.0, 37.1, 37.2],
            "longitude": [-122.0, -122.1, -122.2],
            "incident_abs_pm": [1.0, 2.0, 3.0],
        }
    )
    args = _builder_args()
    args.sample_offsets = [0]
    output = tmp_path / "Alameda_samples.h5"

    build_samples_h5(
        region_name="Alameda",
        region_node_idx=np.arange(3, dtype=np.int64),
        events=events,
        affected_lookup={},
        total_steps=160,
        args=args,
        out_path=output,
        train_end=90,
        val_end=130,
        baseline_fit_end=90,
        manifest=[],
        git_info={"git_sha": "test", "git_branch": "test", "git_dirty": False},
    )

    with h5py.File(output, "r") as h5:
        splits = h5["split"][:]
        active = h5["active_event_idx"][:]
        event_sets = [
            set(active[splits == code][active[splits == code] >= 0].tolist())
            for code in range(3)
        ]
        assert event_sets[0].isdisjoint(event_sets[1])
        assert event_sets[0].isdisjoint(event_sets[2])
        assert event_sets[1].isdisjoint(event_sets[2])
        assert int(h5.attrs["n_mixed_active_event_samples_dropped"]) == 1


def test_cache_persists_stable_event_and_bundle_identities(tmp_path):
    events = pd.DataFrame(
        {
            "agg_id": ["event-b", "event-a"],
            "start_idx": [30, 30],
            "anchor_region_idx": [1, 0],
            "incident_ids": ["b", "a"],
            "primary_type": ["Fire", "Hazard"],
            "latitude": [37.1, 37.0],
            "longitude": [-122.1, -122.0],
            "incident_abs_pm": [2.0, 1.0],
        }
    )

    def build(path, frame):
        build_samples_h5(
            region_name="Alameda",
            region_node_idx=np.arange(2, dtype=np.int64),
            events=frame,
            affected_lookup={},
            total_steps=120,
            args=_builder_args(),
            out_path=path,
            train_end=84,
            val_end=102,
            baseline_fit_end=84,
            manifest=[],
            git_info={
                "git_sha": "test",
                "git_branch": "test",
                "git_dirty": False,
            },
            bundle_id="bundle-test",
            traffic_cache_id="traffic-test",
            samples_cache_id="samples-test",
        )

    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    build(first, events)
    build(second, events.sample(frac=1.0, random_state=9))

    with h5py.File(first, "r") as left, h5py.File(second, "r") as right:
        assert left.attrs["bundle_id"] == "bundle-test"
        assert left.attrs["traffic_cache_id"] == "traffic-test"
        assert left.attrs["samples_cache_id"] == "samples-test"
        assert left.attrs["cache_role"] == "samples"
        assert left["event_id"].asstr()[:].tolist() == [
            "agg:event-a",
            "agg:event-b",
        ]
        for name in (
            "event_id",
            "sample_start",
            "event_idx",
            "active_event_idx",
            "primary_incident_slot",
        ):
            if name == "event_id":
                assert left[name].asstr()[:].tolist() == right[name].asstr()[
                    :
                ].tolist()
            else:
                np.testing.assert_array_equal(left[name][:], right[name][:])


def test_duplicate_stable_event_identity_is_rejected(tmp_path):
    events = pd.DataFrame(
        {
            "agg_id": ["same", "same"],
            "start_idx": [30, 36],
            "anchor_region_idx": [0, 1],
            "incident_ids": ["a", "b"],
            "primary_type": ["Fire", "Hazard"],
            "latitude": [37.0, 37.1],
            "longitude": [-122.0, -122.1],
            "incident_abs_pm": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="event identity"):
        build_samples_h5(
            region_name="Alameda",
            region_node_idx=np.arange(2, dtype=np.int64),
            events=events,
            affected_lookup={},
            total_steps=120,
            args=_builder_args(),
            out_path=tmp_path / "duplicate.h5",
            train_end=84,
            val_end=102,
            baseline_fit_end=84,
            manifest=[],
            git_info={
                "git_sha": "test",
                "git_branch": "test",
                "git_dirty": False,
            },
            bundle_id="bundle-test",
            traffic_cache_id="traffic-test",
            samples_cache_id="samples-test",
        )
