"""Shared no-leakage data protocol helpers."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np


CACHE_SCHEMA_VERSION = 2
PROTOCOL_VERSION = "protocol_v2_noleak"


def git_worktree_provenance(repo_root: str | Path) -> dict[str, object]:
    """Return the current Git revision and dirty state for result metadata."""
    root = Path(repo_root)

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "git_sha": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "git_dirty": bool(run("status", "--porcelain")),
    }


def forecast_window_bounds(
    anchor: int, history: int, horizon: int
) -> tuple[int, int, int, int]:
    """Return exclusive history and target bounds around a forecast anchor."""
    history_start = anchor - history + 1
    history_stop = anchor + 1
    target_start = history_stop
    target_stop = target_start + horizon
    return history_start, history_stop, target_start, target_stop


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def build_deployable_incident_feature(
    event: Mapping[str, object],
    anchor_step: int,
    type_to_idx: Mapping[str, int],
    other_type: str = "Other",
    steps_per_hour: int = 12,
) -> np.ndarray:
    """Pack only incident fields available at the forecast origin."""
    n_types = len(type_to_idx)
    feature = np.zeros(n_types + 4, dtype=np.float32)
    event_type = str(event.get("primary_type", other_type)).strip()
    type_idx = type_to_idx.get(event_type, type_to_idx[other_type])
    feature[type_idx] = 1.0

    start_step = int(_finite_float(event.get("start_idx"), anchor_step))
    elapsed_steps = max(0, int(anchor_step) - start_step)
    feature[n_types] = elapsed_steps / max(1, int(steps_per_hour))
    feature[n_types + 1] = _finite_float(event.get("latitude"))
    feature[n_types + 2] = _finite_float(event.get("longitude"))
    feature[n_types + 3] = _finite_float(event.get("incident_abs_pm"))
    return feature


def build_training_fit_mask(
    shape: tuple[int, int], train_end: int
) -> np.ndarray:
    """Return a label-free fit mask limited to raw training timestamps."""
    total_steps, n_nodes = shape
    if not 0 < train_end <= total_steps:
        raise ValueError(
            f"train_end must be within (0, {total_steps}], got {train_end}"
        )
    valid = np.zeros((total_steps, n_nodes), dtype=bool)
    valid[:train_end] = True
    return valid


def select_active_events(
    active_indices: np.ndarray,
    primary_index: int,
    max_events: int,
) -> tuple[np.ndarray, int]:
    """Keep recent active events while guaranteeing the primary event a slot."""
    active = np.asarray(active_indices, dtype=np.int64)
    if max_events < 1:
        raise ValueError("max_events must be positive")
    if primary_index not in active:
        raise ValueError(f"primary event {primary_index} is not active")

    chosen = active[-max_events:].copy()
    if primary_index not in chosen:
        chosen[0] = primary_index
        chosen.sort()
    primary_slot = int(np.flatnonzero(chosen == primary_index)[0])
    return chosen, primary_slot


def classify_forecast_window(
    anchor: int,
    horizon: int,
    train_end: int,
    val_end: int,
    total_steps: int,
) -> int | None:
    """Return the split whose interval fully contains an anchor's targets."""
    target_start = anchor + 1
    target_stop = target_start + horizon
    if target_start < 0 or target_stop > total_steps:
        return None
    if target_stop <= train_end:
        return 0
    if target_start >= train_end and target_stop <= val_end:
        return 1
    if target_start >= val_end:
        return 2
    return None


def classify_event_group(
    event_start: int,
    offsets: tuple[int, ...] | list[int],
    history: int,
    horizon: int,
    train_end: int,
    val_end: int,
    total_steps: int,
) -> int | None:
    """Assign an event only when all usable offsets belong to one split."""
    splits: set[int] = set()
    for offset in offsets:
        anchor = event_start + int(offset)
        if anchor < history - 1:
            continue
        split = classify_forecast_window(
            anchor, horizon, train_end, val_end, total_steps
        )
        if split is not None:
            splits.add(split)
    if len(splits) == 1:
        return splits.pop()
    return None


def build_full_window_index(
    total_steps: int,
    history: int,
    horizon: int,
    stride: int,
    train_end: int,
    val_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window anchors with purged, target-disjoint splits."""
    anchors = np.arange(
        history - 1,
        total_steps - horizon,
        max(1, int(stride)),
        dtype=np.int64,
    )
    kept_anchors: list[int] = []
    kept_splits: list[int] = []
    for anchor in anchors:
        split = classify_forecast_window(
            int(anchor), horizon, train_end, val_end, total_steps
        )
        if split is not None:
            kept_anchors.append(int(anchor))
            kept_splits.append(split)
    return (
        np.asarray(kept_anchors, dtype=np.int64),
        np.asarray(kept_splits, dtype=np.int8),
    )


def causal_forward_fill(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill missing traffic values without consulting future rows."""
    data = np.asarray(values, dtype=np.float32)
    observed = np.isfinite(data)
    filled = data.copy()

    last = np.zeros(data.shape[1:], dtype=np.float32)
    seen = np.zeros(data.shape[1:], dtype=bool)
    for row in range(data.shape[0]):
        valid = observed[row]
        last[valid] = data[row][valid]
        seen |= valid
        filled[row] = np.where(valid, data[row], np.where(seen, last, 0.0))

    return filled, observed
