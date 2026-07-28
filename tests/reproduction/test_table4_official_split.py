import numpy as np

from scripts.reproduction.igstgnn_table4.official_split import (
    split_official_samples,
)


def test_official_split_preserves_release_order_and_truncates_ratio_counts():
    samples = np.arange(5_585)

    splits = split_official_samples(samples)

    assert [len(splits[name]) for name in ("train", "val", "test")] == [
        3_909,
        837,
        839,
    ]
    assert splits["train"][0] == 0
    assert splits["train"][-1] == 3_908
    assert splits["val"][0] == 3_909
    assert splits["test"][0] == 4_746
    assert splits["test"][-1] == 5_584
