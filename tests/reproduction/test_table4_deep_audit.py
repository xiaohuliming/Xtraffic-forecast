import copy

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4.deep_audit import (
    audit_loaded_samples,
)


def _sample(node_count, *, marker):
    ticks = (144 + marker * 24 + np.arange(24)) % (7 * 288)
    x_data = np.empty((12, node_count, 3), dtype=np.float32)
    x_data[..., 0] = marker
    x_data[..., 1] = (ticks[:12] % 288)[:, None] / 288
    x_data[..., 2] = (ticks[:12] // 288)[:, None] / 7
    y_data = np.empty((12, node_count, 3), dtype=np.float32)
    y_data[..., 0] = marker + 1
    y_data[..., 1] = (ticks[12:] % 288)[:, None] / 288
    y_data[..., 2] = (ticks[12:] // 288)[:, None] / 7
    return {
        "x_data": x_data,
        "y_data": y_data,
        "incident_features": {
            "Incident Time": float(marker),
            "Description": marker % 3,
            "Type": marker % 2,
            "Holiday": marker % 2,
        },
        "incident_position": marker % 12,
        "incident_distances": np.full(
            (node_count, 3),
            marker + 2,
            dtype=np.float32,
        ),
        "durations": np.full(12, marker + 3, dtype=np.float32),
    }


def test_deep_audit_validates_every_loaded_record_and_reports_duplicates():
    samples = np.asarray(
        [
            _sample(5, marker=0),
            _sample(5, marker=1),
            _sample(5, marker=1),
        ],
        dtype=object,
    )

    audit = audit_loaded_samples(
        samples,
        region="Synthetic",
        node_count=5,
        expected_sample_count=3,
        description_count=3,
        type_count=2,
    )

    assert audit["status"] == "records_structurally_verified"
    assert audit["split_integrity_status"] == "cross_split_duplicates_detected"
    assert audit["validity_label"] == "official_protocol_with_known_overlap"
    assert audit["region"] == "Synthetic"
    assert audit["sample_count"] == 3
    assert audit["records_validated"] == 3
    assert audit["x_data"] == {
        "dtypes": {"float32": 3},
        "finite": True,
        "shape": [12, 5, 3],
    }
    assert audit["y_data"] == {
        "consumed_shape": [12, 5, 1],
        "dtypes": {"float32": 3},
        "discarded_feature_count": 2,
        "finite": True,
        "stored_shape": [12, 5, 3],
    }
    assert audit["incident_features"]["layouts"] == {"dict": 3}
    assert audit["incident_features"]["description_range"] == [0, 1]
    assert audit["incident_features"]["type_range"] == [0, 1]
    assert audit["incident_features"]["holiday_range"] == [0, 1]
    assert audit["incident_position_range"] == [0, 1]
    assert audit["incident_distances"] == {
        "dtypes": {"float32": 3},
        "finite": True,
        "shape": [5, 3],
    }
    assert audit["durations"] == {
        "consumed_by_official_model": False,
        "dtypes": {"float32": 3},
        "finite": True,
        "shapes": {"[12]": 3},
    }
    assert audit["duplicate_full_record_count"] == 1
    assert audit["duplicate_xy_window_count"] == 1
    assert audit["duplicate_model_consumed_record_count"] == 1
    assert audit["full_record_multiplicity"] == {"1": 1, "2": 1}
    assert audit["xy_window_multiplicity"] == {"1": 1, "2": 1}
    assert audit["model_consumed_record_multiplicity"] == {"1": 1, "2": 1}
    assert len(audit["content_digest"]) == 64
    assert len(audit["model_consumed_digest"]) == 64


def test_deep_audit_requires_the_released_twelve_step_duration_vector():
    sample = _sample(5, marker=0)
    sample["durations"] = np.float32(3)

    with pytest.raises(ValueError, match=r"durations shape mismatch.*\(12,\)"):
        audit_loaded_samples(
            np.asarray([sample], dtype=object),
            region="Synthetic",
            node_count=5,
            expected_sample_count=1,
            description_count=3,
            type_count=2,
        )


def test_deep_audit_requires_consecutive_five_minute_time_steps():
    sample = _sample(5, marker=0)
    sample["x_data"][5, :, 1] = 151 / 288

    with pytest.raises(ValueError, match="time channels are not consecutive"):
        audit_loaded_samples(
            np.asarray([sample], dtype=object),
            region="Synthetic",
            node_count=5,
            expected_sample_count=1,
            description_count=3,
            type_count=2,
        )


def test_deep_audit_requires_x_to_y_time_continuity():
    sample = _sample(5, marker=0)
    sample["y_data"][..., 1] += np.float32(1 / 288)

    with pytest.raises(ValueError, match="x_data to y_data time discontinuity"):
        audit_loaded_samples(
            np.asarray([sample], dtype=object),
            region="Synthetic",
            node_count=5,
            expected_sample_count=1,
            description_count=3,
            type_count=2,
        )


def test_deep_audit_accepts_consecutive_time_steps_across_week_rollover():
    sample = _sample(5, marker=0)
    ticks = (7 * 288 - 6 + np.arange(24)) % (7 * 288)
    sample["x_data"][..., 1] = (ticks[:12] % 288)[:, None] / 288
    sample["x_data"][..., 2] = (ticks[:12] // 288)[:, None] / 7
    sample["y_data"][..., 1] = (ticks[12:] % 288)[:, None] / 288
    sample["y_data"][..., 2] = (ticks[12:] // 288)[:, None] / 7

    audit = audit_loaded_samples(
        np.asarray([sample], dtype=object),
        region="Synthetic",
        node_count=5,
        expected_sample_count=1,
        description_count=3,
        type_count=2,
    )

    assert audit["status"] == "records_structurally_verified"


def test_model_consumed_digest_excludes_the_unused_duration_vector():
    original = _sample(5, marker=0)
    changed = copy.deepcopy(original)
    changed["durations"][:] = 999

    original_audit = audit_loaded_samples(
        np.asarray([original], dtype=object),
        region="Synthetic",
        node_count=5,
        expected_sample_count=1,
        description_count=3,
        type_count=2,
    )
    changed_audit = audit_loaded_samples(
        np.asarray([changed], dtype=object),
        region="Synthetic",
        node_count=5,
        expected_sample_count=1,
        description_count=3,
        type_count=2,
    )

    assert original_audit["content_digest"] != changed_audit["content_digest"]
    assert (
        original_audit["model_consumed_digest"]
        == changed_audit["model_consumed_digest"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda sample: sample.__setitem__("unexpected", 1),
            "top-level keys mismatch",
        ),
        (
            lambda sample: sample["incident_features"].__setitem__(
                "Description",
                0.5,
            ),
            "Description must be exactly integral",
        ),
        (
            lambda sample: sample["x_data"].__setitem__(
                (0, 0, 1),
                0.123,
            ),
            "time-of-day channel is not quantized",
        ),
    ],
)
def test_deep_audit_rejects_values_the_official_model_would_silently_clamp(
    mutation,
    message,
):
    sample = _sample(5, marker=0)
    mutation(sample)

    with pytest.raises(ValueError, match=message):
        audit_loaded_samples(
            np.asarray([sample], dtype=object),
            region="Synthetic",
            node_count=5,
            expected_sample_count=1,
            description_count=3,
            type_count=2,
        )
