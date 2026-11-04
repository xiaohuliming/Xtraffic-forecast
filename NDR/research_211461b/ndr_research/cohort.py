"""Paired three-seed summaries with shared calendar-day resampling."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

SEEDS = (42, 1, 2)


def training_recipe_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Retain optimization semantics, omitting per-run provenance and seed."""
    omitted = {"seed", "runtime", "source_snapshot", "data_identity", "source_statistics_identity",
               "threads", "log_interval", "output_dir", "data_dir", "graph_dir", "training_process_id"}
    return {key: value for key, value in config.items() if key not in omitted}


def preprocessing_semantics(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Numerical fitting policy, independent of a gate's authorization receipt."""
    omitted = {"authorization_id", "access_audit", "access_audits", "identity_digest"}
    return {key: value for key, value in identity.items() if key not in omitted}


def _blocks(report: Mapping[str, Any], scenario: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    current = report["scenarios"][scenario]
    if current["collection"] != report["collection"] or not current["collection"]["all_declared_split_samples"]:
        raise ValueError("cohort requires one complete declared collection")
    blocks = current["day_blocks"]
    days = sorted(blocks)
    if len(days) < 2:
        raise ValueError("cohort uncertainty requires at least two real calendar days")
    errors, counts = [], []
    for day in days:
        if str(np.datetime64(day, "D")) != day:
            raise ValueError("cohort block key is not an ISO calendar day")
        row = blocks[day]
        error = np.asarray(row["absolute_error_sum"], dtype=np.float64)
        count = np.asarray(row["valid_count"])
        if (error.shape != (12,) or count.shape != (12,) or count.dtype.kind not in "iu"
                or not np.isfinite(error).all() or (error < 0).any() or (count < 0).any()):
            raise ValueError("day blocks require finite nonnegative sums and integer horizon counts")
        errors.append(error)
        counts.append(count)
    errors = np.asarray(errors)
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() <= 0:
        raise ValueError("cohort contains no valid target observations")
    totals = current["sums"]
    if (not np.array_equal(counts.sum(axis=0), totals["valid_count"])
            or not np.allclose(errors.sum(axis=0), totals["absolute_error_sum"], rtol=1e-10, atol=1e-8)
            or int(counts.sum()) != current["valid_count"]
            or not np.isclose(errors.sum() / counts.sum(), current["global_masked_mae"], rtol=1e-10, atol=1e-10)):
        raise ValueError("day blocks do not reproduce the report's point-weighted totals")
    return days, errors.sum(axis=1), counts.sum(axis=1)


def _interval(left: np.ndarray, right: np.ndarray, counts: np.ndarray, *, samples: int,
              seed: int, resample_seeds: bool) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = []
    seeds, days = left.shape
    for _ in range(samples):
        day_draw = rng.integers(0, days, size=days)
        seed_draw = rng.integers(0, seeds, size=seeds) if resample_seeds else np.arange(seeds)
        # The same day draw is shared by every paired seed and both models.
        denominator = counts[day_draw].sum() * len(seed_draw)
        if denominator:
            difference = (left[seed_draw][:, day_draw] - right[seed_draw][:, day_draw]).sum()
            values.append(float(difference / denominator))
    if not values:
        raise ValueError("cohort bootstrap produced no observed samples")
    low, high = np.quantile(values, [0.025, 0.975])
    return {"ci95": [float(low), float(high)], "valid_resamples": len(values),
            "bootstrap_samples": samples, "seed": seed,
            "seed_resampling": "paired_seed_with_replacement" if resample_seeds else "fixed_training_seed_cohort",
            "day_resampling": "one_common_calendar_day_draw_shared_across_seeds_and_models"}


def paired_seed_day_cohort(left: Mapping[int, Mapping[str, Any]],
                           right: Mapping[int, Mapping[str, Any]], *, scenario: str = "clean",
                           samples: int = 10000, bootstrap_seed: int = 42) -> dict[str, Any]:
    """Keep discovery seed 42 visible and confirmation seeds 1/2 separately."""
    if set(left) != set(SEEDS) or set(right) != set(SEEDS):
        raise ValueError("paired cohort requires exactly training seeds 42, 1 and 2")
    if type(samples) is not int or samples < 100 or type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**32:
        raise ValueError("bootstrap needs at least 100 draws and a uint32 seed")
    collections = []
    families = {"left": [], "right": []}
    left_errors, right_errors = [], []
    common_days, common_counts = None, None
    perturbation = None
    metadata = None
    training_scopes = {"left": [], "right": []}
    for seed in SEEDS:
        first, second = left[seed], right[seed]
        for side, report in (("left", first), ("right", second)):
            if type(report["training_seed"]) is not int or report["training_seed"] != seed:
                raise ValueError("cohort report training seed differs from its declaration")
            families[side].append(report["family_identity"])
            current_metadata = {key: report[key] for key in ("region", "split", "sampling_protocol", "evaluation_mode", "comparison_scope")}
            if metadata is not None and current_metadata != metadata:
                raise ValueError("cohort mixes regions, splits, protocols or transfer modes")
            metadata = current_metadata
            collections.append(report["collection"])
            training_scopes[side].append({key: report[key] for key in
                                         ("training_scope", "training_regions", "trained_on_evaluated_region")})
            current = report["scenarios"][scenario]
            mask_identity = tuple(current[key] for key in ("perturbation_seed", "history_observed_points_original", "history_observed_points_removed"))
            if perturbation is not None and mask_identity != perturbation:
                raise ValueError("cohort does not use a common historical perturbation")
            perturbation = mask_identity
        first_days, first_errors, first_counts = _blocks(first, scenario)
        second_days, second_errors, second_counts = _blocks(second, scenario)
        if first_days != second_days or not np.array_equal(first_counts, second_counts):
            raise ValueError("paired models have different calendar days or observation denominators")
        if common_days is not None and (first_days != common_days or not np.array_equal(first_counts, common_counts)):
            raise ValueError("training seeds do not share the exact observed day collection")
        common_days, common_counts = first_days, first_counts
        left_errors.append(first_errors)
        right_errors.append(second_errors)
    if any(collection != collections[0] for collection in collections):
        raise ValueError("cohort origins, targets or masks differ across models or seeds")
    for side in families:
        if any(family != families[side][0] for family in families[side]):
            raise ValueError("cohort mixes model recipes or source identities within a family")
        if any(scope != training_scopes[side][0] for scope in training_scopes[side]):
            raise ValueError("cohort mixes training supervision scopes within a model family")
    first = np.asarray(left_errors)
    second = np.asarray(right_errors)
    count = int(common_counts.sum())
    differences = (first.sum(axis=1) - second.sum(axis=1)) / count
    per_seed = [{"training_seed": seed, "left_global_mae": float(first[index].sum() / count),
                 "right_global_mae": float(second[index].sum() / count),
                 "left_minus_right_global_mae": float(differences[index]),
                 "role": "discovery" if seed == 42 else "frozen_confirmation"}
                for index, seed in enumerate(SEEDS)]
    groups = {}
    for label, rows in (("all_three_seeds_descriptive", np.arange(3)), ("confirmation_seeds_1_2", np.asarray([1, 2]))):
        current_left, current_right = first[rows], second[rows]
        groups[label] = {
            "training_seeds": [SEEDS[index] for index in rows],
            "left_minus_right_global_mae": float(differences[rows].mean()),
            "between_seed_difference_std": float(differences[rows].std(ddof=1)),
            "conditional_day_interval": _interval(current_left, current_right, common_counts,
                                                    samples=samples, seed=bootstrap_seed, resample_seeds=False),
            "hierarchical_seed_day_interval": _interval(current_left, current_right, common_counts,
                                                          samples=samples, seed=bootstrap_seed, resample_seeds=True),
        }
    return {"schema_version": 1, "status": "paired_frozen_cohort_analyzed", **metadata,
            "scenario": scenario, "metric": "left_minus_right_global_masked_mae",
            "collection": collections[0], "forecast_day_blocks": len(common_days),
            "valid_target_points_per_seed": count, "per_seed": per_seed, "groups": groups,
            "family_identity": {side: values[0] for side, values in families.items()},
            "model_training_scopes": {side: values[0] for side, values in training_scopes.items()},
            "resampling_does_not_treat_seed_times_day_as_independent": True,
            "claim_scope": "descriptive_paired_cohort_with_two_new_confirmation_seeds",
            "limitations": ["Three training seeds provide limited initialization uncertainty evidence.",
                            "Calendar-day resampling does not preserve dependence between adjacent days.",
                            "Seed 42 participated in discovery and is not a new independent confirmation."],
            "sota_or_winner_claim": False}
