from types import SimpleNamespace

import pandas as pd

from dist_net.data import RegionBucketedSampler
from dist_net.protocol import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_REGION_DATA_DIR,
    DEFAULT_RESULT_ROOT,
    PROTOCOL_VERSION,
    canonical_event_id,
)


def test_v3_protocol_has_new_schema_and_output_roots():
    assert CACHE_SCHEMA_VERSION == 3
    assert PROTOCOL_VERSION == "protocol_v3_noleak_dedup"
    assert DEFAULT_REGION_DATA_DIR.as_posix().endswith(
        "outputs/dist_net/region_data_v3_noleak_dedup"
    )
    assert DEFAULT_RESULT_ROOT.as_posix().endswith(
        "outputs/protocol_v3_noleak_dedup"
    )
    assert "v2" not in DEFAULT_REGION_DATA_DIR.as_posix()
    assert "v2" not in DEFAULT_RESULT_ROOT.as_posix()


def test_event_identity_is_stable_and_prefers_aggregate_id():
    base = {
        "agg_id": "agg-42",
        "start_idx": 100,
        "incident_ids": "b,a",
        "anchor_region_idx": 7,
    }
    changed_non_identity_fields = {
        **base,
        "duration_min": 999,
        "severity_any_z_auc_topk": 123.0,
    }

    assert canonical_event_id(base) == "agg:agg-42"
    assert canonical_event_id(changed_non_identity_fields) == "agg:agg-42"

    no_agg_a = {**base, "agg_id": "", "incident_ids": "b,a"}
    no_agg_b = {**base, "agg_id": None, "incident_ids": "a;b"}
    assert canonical_event_id(no_agg_a) == canonical_event_id(no_agg_b)


def test_bucketed_sampler_changes_order_by_epoch_but_is_reproducible():
    dataset = SimpleNamespace(
        index_table=[
            *[("Alameda", index) for index in range(12)],
            *[("Orange", index) for index in range(12)],
        ]
    )
    sampler_a = RegionBucketedSampler(
        dataset, batch_size=3, shuffle=True, seed=42
    )
    epoch_zero_a = list(iter(sampler_a))
    epoch_one_a = list(iter(sampler_a))

    sampler_b = RegionBucketedSampler(
        dataset, batch_size=3, shuffle=True, seed=42
    )
    epoch_zero_b = list(iter(sampler_b))
    epoch_one_b = list(iter(sampler_b))

    assert epoch_zero_a != epoch_one_a
    assert epoch_zero_a == epoch_zero_b
    assert epoch_one_a == epoch_one_b

    sampler_b.set_epoch(0)
    assert list(iter(sampler_b)) == epoch_zero_a


def test_event_identity_does_not_depend_on_dataframe_row_order():
    events = pd.DataFrame(
        [
            {"agg_id": "z", "start_idx": 10, "incident_ids": "3"},
            {"agg_id": "a", "start_idx": 10, "incident_ids": "1"},
            {"agg_id": "m", "start_idx": 9, "incident_ids": "2"},
        ]
    )

    def canonical_order(frame: pd.DataFrame) -> list[str]:
        identities = frame.apply(canonical_event_id, axis=1)
        ordered = frame.assign(_event_id=identities).sort_values(
            ["start_idx", "_event_id"], kind="mergesort"
        )
        return ordered["_event_id"].tolist()

    assert canonical_order(events) == canonical_order(
        events.sample(frac=1.0, random_state=7)
    )
