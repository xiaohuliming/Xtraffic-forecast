"""Shared days preserve paired seed dependence and observed-point weights."""

import copy
import json

import numpy as np
import pytest

from ndr_research.cohort import (
    SEEDS,
    paired_seed_day_cohort,
    preprocessing_semantics,
    training_recipe_identity,
)
from ndr_research.transfer_evaluation import SCENARIOS, file_sha256
from scripts.analyze_ndr_cohort import analyze_manifest


def report(seed, family, errors):
    counts = [10, 1, 1]
    blocks = {
        f"2024-01-0{index + 1}": {
            "valid_count": [count] + [0] * 11,
            "absolute_error_sum": [error] + [0.0] * 11,
        }
        for index, (count, error) in enumerate(zip(counts, errors))
    }
    collection = {
        "samples": 7,
        "forecast_origins_sha256": "a" * 64,
        "targets_sha256": "b" * 64,
        "target_masks_sha256": "c" * 64,
        "all_declared_split_samples": True,
    }
    scenario = {
        "collection": collection,
        "day_blocks": blocks,
        "valid_count": 12,
        "sums": {
            "valid_count": [12] + [0] * 11,
            "absolute_error_sum": [sum(errors)] + [0.0] * 11,
        },
        "global_masked_mae": sum(errors) / 12,
        "perturbation_seed": 42,
        "history_observed_points_original": 20,
        "history_observed_points_removed": 0,
    }
    return {
        "training_seed": seed,
        "family_identity": {"model": family},
        "region": "Orange",
        "split": "test",
        "sampling_protocol": "event",
        "evaluation_mode": "source_only",
        "comparison_scope": "source_only_transfer_with_fixed_source_controls",
        "training_scope": "source_only_trained_weights",
        "training_regions": ["Alameda", "ContraCosta"],
        "trained_on_evaluated_region": False,
        "collection": collection,
        "scenarios": {"clean": scenario},
    }


@pytest.fixture
def cohorts():
    left = {
        seed: report(
            seed,
            "candidate",
            [30, 3, 3] if seed == 42 else [10, 1, 1] if seed == 1 else [0, 0, 0],
        )
        for seed in SEEDS
    }
    right = {seed: report(seed, "reference", [10, 1, 1]) for seed in SEEDS}
    return left, right


def test_common_day_draw_preserves_seed_correlation_and_separates_discovery(cohorts):
    result = paired_seed_day_cohort(*cohorts, samples=1000, bootstrap_seed=91)
    full = result["groups"]["all_three_seeds_descriptive"]
    confirmation = result["groups"]["confirmation_seeds_1_2"]
    assert full["left_minus_right_global_mae"] == pytest.approx(1 / 3)
    assert full["conditional_day_interval"]["ci95"] == pytest.approx([1 / 3, 1 / 3])
    assert full["hierarchical_seed_day_interval"]["ci95"] == pytest.approx([-1, 2])
    assert confirmation["left_minus_right_global_mae"] == -0.5
    assert confirmation["training_seeds"] == [1, 2]
    assert result["per_seed"][0]["role"] == "discovery"
    assert result["sota_or_winner_claim"] is False
    assert result == paired_seed_day_cohort(*cohorts, samples=1000, bootstrap_seed=91)


def test_point_weights_are_retained_in_the_cohort_estimate():
    left = {seed: report(seed, "candidate", [90, 1, 1]) for seed in SEEDS}
    right = {seed: report(seed, "reference", [0, 0, 0]) for seed in SEEDS}
    result = paired_seed_day_cohort(left, right, samples=100)
    estimate = result["groups"]["all_three_seeds_descriptive"][
        "left_minus_right_global_mae"
    ]
    assert estimate == pytest.approx(92 / 12)
    assert estimate != pytest.approx(np.mean([9, 1, 1]))


def test_missing_seed_cannot_be_presented_as_three_seed_evidence(cohorts):
    del cohorts[0][2]
    with pytest.raises(ValueError, match="exactly training seeds"):
        paired_seed_day_cohort(*cohorts, samples=100)


@pytest.mark.parametrize(
    "field", ["targets_sha256", "forecast_origins_sha256", "target_masks_sha256"]
)
def test_seed_collection_drift_is_rejected(cohorts, field):
    cohorts[0][2]["collection"][field] = "d" * 64
    with pytest.raises(ValueError, match="origins, targets or masks"):
        paired_seed_day_cohort(*cohorts, samples=100)


def test_model_recipe_drift_is_rejected(cohorts):
    cohorts[0][2]["family_identity"] = {"model": "another_width"}
    with pytest.raises(ValueError, match="model recipes"):
        paired_seed_day_cohort(*cohorts, samples=100)


def test_cross_region_cohorts_cannot_be_pooled_implicitly(cohorts):
    cohorts[1][1]["region"] = "Alameda"
    with pytest.raises(ValueError, match="mixes regions"):
        paired_seed_day_cohort(*cohorts, samples=100)


def test_new_missingness_draw_cannot_be_called_a_paired_comparison(cohorts):
    cohorts[1][2]["scenarios"]["clean"]["perturbation_seed"] = 1
    with pytest.raises(ValueError, match="common historical perturbation"):
        paired_seed_day_cohort(*cohorts, samples=100)


def test_inconsistent_report_sums_are_rejected(cohorts):
    cohorts[0][42]["scenarios"]["clean"]["day_blocks"]["2024-01-01"][
        "absolute_error_sum"
    ][0] += 1
    with pytest.raises(ValueError, match="point-weighted totals"):
        paired_seed_day_cohort(*cohorts, samples=100)


def test_new_day_denominators_are_rejected_even_if_global_totals_match(cohorts):
    current = cohorts[1][2]["scenarios"]["clean"]
    current["day_blocks"]["2024-01-01"]["valid_count"][0] = 9
    current["day_blocks"]["2024-01-02"]["valid_count"][0] = 2
    with pytest.raises(ValueError, match="denominators"):
        paired_seed_day_cohort(*cohorts, samples=100)


def test_analysis_does_not_mutate_the_source_reports(cohorts):
    before = copy.deepcopy(cohorts)
    paired_seed_day_cohort(*cohorts, samples=100)
    assert cohorts == before


def manifest_files(tmp_path, cohorts):
    manifest = {
        "schema_version": 1,
        "selection_frozen": True,
        "scenarios": list(SCENARIOS),
        "bootstrap_samples": 100,
        "bootstrap_seed": 42,
        "evidence_role": "indomain_performance",
        "pairs": [],
    }
    for seed in SEEDS:
        pair = {"training_seed": seed}
        for side, reports in zip(("left", "right"), cohorts):
            current = copy.deepcopy(reports[seed])
            current.update(
                status="frozen_indomain_evaluation_complete",
                model_id=side + str(seed),
                model_family=side,
                evaluation_mode="indomain",
                comparison_scope="matched_indomain_training",
                training_scope="regional_supervised",
                training_regions=["Orange"],
                trained_on_evaluated_region=True,
                checkpoint_sha256="d" * 64,
                schema_version=1,
                provenance={
                    "model_config": {"hidden_dim": 64},
                    "source_files_digest": "e" * 64,
                    "training_recipe_identity": {"teacher": "original", "epochs": 30},
                    "training_runtime": {"device": "cuda"},
                    "training_regions": ["Orange"],
                },
                evaluation_source_snapshot={"files_digest": "f" * 64},
            )
            clean = current["scenarios"]["clean"]
            current["scenarios"] = {
                scenario: copy.deepcopy(clean) for scenario in SCENARIOS
            }
            path = tmp_path / (side + str(seed) + ".json")
            path.write_text(json.dumps(current))
            pair[side] = {"report": path.name, "report_sha256": file_sha256(path)}
        manifest["pairs"].append(pair)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_analyzes_all_frozen_scenarios_without_label_reader(tmp_path, cohorts):
    result = analyze_manifest(manifest_files(tmp_path, cohorts))
    assert set(result["scenarios"]) == set(SCENARIOS)
    assert result["traffic_labels_opened"] is False
    assert result["sota_or_winner_claim"] is False


def test_report_mutation_after_manifest_freeze_is_rejected(tmp_path, cohorts):
    path = manifest_files(tmp_path, cohorts)
    (tmp_path / "left42.json").write_text("{}")
    with pytest.raises(ValueError, match="report hash"):
        analyze_manifest(path)


def test_indomain_result_cannot_be_called_primary_zero_shot_confirmation(
    tmp_path, cohorts
):
    path = manifest_files(tmp_path, cohorts)
    manifest = json.loads(path.read_text())
    manifest["evidence_role"] = "primary_orange_confirmation"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="primary unseen-region"):
        analyze_manifest(path)


def test_training_recipe_retains_teacher_optimizer_and_budget():
    config = {
        "teacher": "original",
        "epochs": 30,
        "seed": 42,
        "learning_rate": 0.001,
        "optimizer": {"name": "Adam"},
        "runtime": {"device": "cuda"},
        "source_snapshot": {"git": "old"},
        "training_process_id": 123,
    }
    other = copy.deepcopy(config)
    other.update(seed=1, source_snapshot={"git": "new"}, training_process_id=456)
    assert training_recipe_identity(config) == training_recipe_identity(other)
    other["teacher"] = "projected"
    assert training_recipe_identity(config) != training_recipe_identity(other)
    other["teacher"] = "original"
    other["epochs"] = 100
    assert training_recipe_identity(config) != training_recipe_identity(other)


def test_preprocessing_auth_receipt_changes_are_not_numeric_recipe_changes():
    first = {
        "mode": "historical_calibrated_transfer",
        "prefix_end": 100,
        "template_sha256": "a" * 64,
        "scale_sha256": "b" * 64,
        "authorization_id": "seed42",
        "identity_digest": "c" * 64,
        "access_audit": ["read"],
    }
    other = first | {
        "authorization_id": "seed1",
        "identity_digest": "d" * 64,
        "access_audit": ["read", "gate"],
    }
    assert preprocessing_semantics(first) == preprocessing_semantics(other)
    assert preprocessing_semantics(first) != preprocessing_semantics(
        other | {"prefix_end": 200}
    )


def test_frozen_reports_with_different_teacher_are_not_one_model_family(
    tmp_path, cohorts
):
    path = manifest_files(tmp_path, cohorts)
    report_path = tmp_path / "left1.json"
    member = json.loads(report_path.read_text())
    member["provenance"]["training_recipe_identity"]["teacher"] = "projected"
    report_path.write_text(json.dumps(member))
    manifest = json.loads(path.read_text())
    next(row for row in manifest["pairs"] if row["training_seed"] == 1)["left"][
        "report_sha256"
    ] = file_sha256(report_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model recipes"):
        analyze_manifest(path)


def test_mixed_supervision_scope_survives_cohort_aggregation(cohorts):
    for side, members in zip(("left", "right"), cohorts):
        for member in members.values():
            member["evaluation_mode"] = "indomain"
            member["comparison_scope"] = "declared_mixed_training_scope"
            if side == "right":
                member.update(
                    training_scope="regional_supervised",
                    training_regions=["Orange"],
                    trained_on_evaluated_region=True,
                )
    result = paired_seed_day_cohort(*cohorts, samples=100)
    assert result["comparison_scope"] == "declared_mixed_training_scope"
    assert (
        result["model_training_scopes"]["left"]["trained_on_evaluated_region"] is False
    )
    assert result["model_training_scopes"]["right"]["training_regions"] == ["Orange"]


def test_within_family_supervision_scope_drift_is_rejected(cohorts):
    cohorts[0][1]["training_regions"] = ["Orange"]
    with pytest.raises(ValueError, match="training supervision scopes"):
        paired_seed_day_cohort(*cohorts, samples=100)


def test_source_cohort_report_selects_real_model_per_seed_and_preserves_class(
    tmp_path, cohorts
):
    from scripts.analyze_ndr_cohort import load_member

    matrices = {}
    for seed in SEEDS:
        members = {}
        for side, reports, model_class in zip(
            ("left", "right"), cohorts, ("SharedGraphNDR", "SharedSTAEformer")
        ):
            member = copy.deepcopy(reports[seed])
            member.update(
                model_class=model_class,
                model_family=side,
                checkpoint_sha256="a" * 64,
                source_only_control=False,
                training_scope="two_source_supervised_target_disjoint_event",
            )
            members[side + str(seed)] = member
        matrix = {
            "status": "frozen_source_cohort_evaluation_complete",
            "region": "Orange",
            "source_regions": ["Alameda", "ContraCosta"],
            "split": "test",
            "evaluation_mode": "source_only",
            "sampling_protocol": "event",
            "comparison_scope": "matched_two_source_training",
            "collection": cohorts[0][seed]["collection"],
            "models": members,
            "evaluation_source_snapshot": {"files_digest": "b" * 64},
            "preprocessing_identity": {
                "mode": "source_only",
                "template_sha256": "c" * 64,
            },
        }
        path = tmp_path / f"matrix{seed}.json"
        path.write_text(json.dumps(matrix))
        matrices[seed] = path
    left, right = {}, {}
    for seed, path in matrices.items():
        declaration = {"report": path.name, "report_sha256": file_sha256(path)}
        left[seed], _ = load_member(
            declaration | {"model": "left" + str(seed)},
            training_seed=seed,
            base=tmp_path,
        )
        right[seed], _ = load_member(
            declaration | {"model": "right" + str(seed)},
            training_seed=seed,
            base=tmp_path,
        )
    result = paired_seed_day_cohort(left, right, samples=100)
    assert result["comparison_scope"] == "matched_two_source_training"
    assert (
        result["model_training_scopes"]["left"]["trained_on_evaluated_region"] is False
    )
    with pytest.raises(ValueError, match="different seed or true model class"):
        load_member(
            {
                "report": matrices[42].name,
                "report_sha256": file_sha256(matrices[42]),
                "model": "left42",
            },
            training_seed=1,
            base=tmp_path,
        )
