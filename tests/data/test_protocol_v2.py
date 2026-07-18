import numpy as np

from dist_net.protocol import (
    build_deployable_incident_feature,
    build_full_window_index,
    build_training_fit_mask,
    causal_forward_fill,
    classify_event_group,
    classify_forecast_window,
    forecast_window_bounds,
    select_active_events,
)


def test_future_changes_do_not_change_filled_history():
    traffic = np.array(
        [
            [[np.nan], [4.0]],
            [[10.0], [np.nan]],
            [[np.nan], [np.nan]],
            [[np.nan], [8.0]],
            [[20.0], [np.nan]],
        ],
        dtype=np.float32,
    )
    changed_future = traffic.copy()
    changed_future[3:] = 999.0

    filled, observed = causal_forward_fill(traffic)
    filled_changed, observed_changed = causal_forward_fill(changed_future)

    np.testing.assert_array_equal(filled[:3], filled_changed[:3])
    np.testing.assert_array_equal(observed[:3], observed_changed[:3])
    np.testing.assert_array_equal(
        filled[:, 0, 0],
        np.array([0.0, 10.0, 10.0, 10.0, 20.0], dtype=np.float32),
    )


def test_forecast_window_must_fit_entirely_inside_one_split():
    train_end, val_end, total = 70, 85, 100

    assert classify_forecast_window(57, 12, train_end, val_end, total) == 0
    assert classify_forecast_window(58, 12, train_end, val_end, total) is None
    assert classify_forecast_window(69, 12, train_end, val_end, total) == 1
    assert classify_forecast_window(73, 12, train_end, val_end, total) is None
    assert classify_forecast_window(84, 12, train_end, val_end, total) == 2


def test_one_event_cannot_contribute_samples_to_multiple_splits():
    assert classify_event_group(
        event_start=57,
        offsets=(0, 6, 12),
        history=12,
        horizon=12,
        train_end=70,
        val_end=85,
        total_steps=100,
    ) is None

    assert classify_event_group(
        event_start=30,
        offsets=(0, 6, 12),
        history=12,
        horizon=12,
        train_end=70,
        val_end=85,
        total_steps=100,
    ) == 0


def test_full_window_splits_have_disjoint_target_timestamps():
    starts, splits = build_full_window_index(
        total_steps=100,
        history=12,
        horizon=12,
        stride=1,
        train_end=70,
        val_end=85,
    )

    targets = {
        split: {
            target
            for anchor in starts[splits == split]
            for target in range(int(anchor) + 1, int(anchor) + 13)
        }
        for split in (0, 1, 2)
    }
    assert all(targets.values())
    assert targets[0].isdisjoint(targets[1])
    assert targets[0].isdisjoint(targets[2])
    assert targets[1].isdisjoint(targets[2])


def test_label_free_fit_mask_uses_only_the_training_boundary():
    fit_mask = build_training_fit_mask((100, 3), train_end=70)

    assert fit_mask.shape == (100, 3)
    assert fit_mask[:70].all()
    assert not fit_mask[70:].any()


def test_deployable_incident_feature_ignores_post_event_fields():
    type_to_idx = {"Crash": 0, "Other": 1}
    event = {
        "primary_type": "Crash",
        "start_idx": 40,
        "latitude": 37.7,
        "longitude": -122.2,
        "incident_abs_pm": 12.5,
        "duration_min": 25,
        "severity_any_z_auc_topk": 1.2,
        "affected": 0,
    }
    changed_post_event = {
        **event,
        "duration_min": 240,
        "severity_any_z_auc_topk": 99.0,
        "affected": 1,
    }

    feature = build_deployable_incident_feature(
        event, anchor_step=46, type_to_idx=type_to_idx
    )
    changed = build_deployable_incident_feature(
        changed_post_event, anchor_step=46, type_to_idx=type_to_idx
    )

    np.testing.assert_array_equal(feature, changed)
    assert feature.shape == (len(type_to_idx) + 4,)
    assert feature[len(type_to_idx)] == 0.5


def test_primary_event_survives_active_event_truncation():
    chosen, primary_slot = select_active_events(
        active_indices=np.arange(10, dtype=np.int64),
        primary_index=2,
        max_events=4,
    )

    np.testing.assert_array_equal(chosen, np.array([2, 7, 8, 9]))
    assert primary_slot == 0


def test_window_bounds_align_history_and_forecast_at_the_anchor():
    assert forecast_window_bounds(anchor=20, history=12, horizon=12) == (
        9,
        21,
        21,
        33,
    )
