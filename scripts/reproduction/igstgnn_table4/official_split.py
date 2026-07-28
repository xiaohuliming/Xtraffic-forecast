from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


SampleT = TypeVar("SampleT")


def split_official_samples(
    samples: Sequence[SampleT],
) -> dict[str, Sequence[SampleT]]:
    total_samples = len(samples)
    train_end = int(total_samples * 0.70)
    val_end = train_end + int(total_samples * 0.15)
    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }
