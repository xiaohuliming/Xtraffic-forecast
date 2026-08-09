from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_protocol_v3_event_extension_loading import compare_loader_streams


class _Loader:
    def __init__(self, values: list[int], *, shuffle: bool) -> None:
        self.values = values
        self.should_shuffle = shuffle
        self.shuffle_calls = 0

    def shuffle(self) -> None:
        self.shuffle_calls += 1
        if self.should_shuffle:
            self.values = list(reversed(self.values))

    def get_iterator(self):
        for value in self.values:
            yield {
                "x_data": np.asarray([[value]], dtype=np.float32),
                "y_mask": np.asarray([[value % 2 == 0]], dtype=np.bool_),
                "sensor_data": {
                    "sensor_type": np.asarray([[value]], dtype=np.int64)
                },
            }


@pytest.mark.parametrize("split,shuffle", [("train", True), ("val", False), ("test", False)])
def test_loading_audit_compares_exact_streams_and_split_order(
    split: str, shuffle: bool
) -> None:
    lazy = _Loader([1, 2, 3], shuffle=shuffle)
    eager = _Loader([1, 2, 3], shuffle=shuffle)
    report = compare_loader_streams(
        split=split,
        lazy_loader=lazy,
        eager_loader=eager,
        batch_limit=3,
    )
    assert report["status"] == "exact_match"
    assert report["batch_count"] == 3
    assert len(report["stream_sha256"]) == 64
    assert lazy.shuffle_calls == (1 if shuffle else 0)
    assert eager.shuffle_calls == (1 if shuffle else 0)


def test_loading_audit_fails_closed_on_any_array_difference() -> None:
    lazy = _Loader([1, 2, 3], shuffle=False)
    eager = _Loader([1, 9, 3], shuffle=False)
    with pytest.raises(RuntimeError, match="batch 2"):
        compare_loader_streams(
            split="val",
            lazy_loader=lazy,
            eager_loader=eager,
            batch_limit=3,
        )
