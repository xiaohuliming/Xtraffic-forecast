#!/usr/bin/env python3
"""Analyze existing frozen reports, without opening any traffic labels."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ndr_research.cohort import (
    SEEDS,
    paired_seed_day_cohort,
    preprocessing_semantics,
    training_recipe_identity,
)  # noqa: E402
from ndr_research.transfer_evaluation import SCENARIOS, canonical_sha256, file_sha256  # noqa: E402

STATUSES = {
    "frozen_transfer_evaluation_complete",
    "frozen_indomain_evaluation_complete",
    "frozen_indomain_model_evaluation_complete",
    "frozen_source_cohort_evaluation_complete",
    "frozen_fullwindow_evaluation_complete",
}


def _path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _fullwindow_family(report: dict, current: dict, training_seed: int) -> dict:
    classes = {
        "baseline": ("fullwindow_ndr", "STAEformerDeseason", None),
        "projected_oracle": ("fullwindow_ndr", "STAEformerDeseason", None),
        "sanitized_control": ("fullwindow_ndr", "SanitizedNDR", None),
        "mask_signed": ("fullwindow_ndr", "MaskSignedNDR", None),
        "shared_graph": ("fullwindow_ndr", "SharedGraphNDR", None),
        "staeformer": ("fullwindow_strong", "StrongReference", "STAEformer"),
        "bist": ("fullwindow_strong", "StrongReference", "BiST"),
    }
    expected = classes.get(current.get("method"))
    if (
        type(current.get("training_seed")) is not int
        or current["training_seed"] != training_seed
        or expected is None
        or (
            current.get("kind"),
            current.get("model_class"),
            current.get("native_model_class"),
        )
        != expected
        or current.get("checkpoint_kind") != expected[0]
    ):
        raise ValueError(
            "full-window selected model has a different seed or true model class"
        )
    if (
        report.get("schema_version") != "fullwindow_frozen_eval_v1"
        or report.get("sampling_protocol") != "full_window"
        or current.get("sampling_protocol") != "full_window"
        or report.get("mode") != "same_region_temporal_heldout"
        or report.get("comparison_scope") != "same_region_full_window_temporal_holdout"
        or report.get("scalar_fit_precision") != "float64"
        or report.get("legacy27_scalar_fit_precision") != "float32"
        or report.get("legacy27_scores_directly_comparable") is not False
    ):
        raise ValueError("full-window sampling and scalar precision boundaries differ")
    for field in ("reader_identity", "sampling_identity", "statistics_identity"):
        if current.get(field) != report.get(field) or not current.get(field):
            raise ValueError(
                "full-window model and report reader, sampling or statistics identities differ"
            )
        if current.get("data_identity", {}).get(field) != current[field]:
            raise ValueError(
                "full-window model data identity is not bound to its saved statistics and sampling"
            )
    reader, sampling, statistics = (
        current["reader_identity"],
        current["sampling_identity"],
        current["statistics_identity"],
    )
    family = current["family_identity"]
    model_family = current["method"]
    if model_family == "shared_graph":
        model_family += f"_h{family.get('architecture', {}).get('hidden_dim')}"
    if (
        reader.get("region") != report["region"]
        or reader.get("sampling_protocol") != "full_window"
        or reader.get("sampling") != sampling
        or statistics.get("sampling_protocol") != "full_window"
        or statistics.get("scalar_reduction_dtype") != "float64"
        or statistics.get("reader_identity_digest") != reader.get("identity_digest")
        or statistics.get("sampling_digest") != sampling.get("sampling_digest")
        or family.get("sampling_digest") != sampling.get("sampling_digest")
        or family.get("statistics_identity_digest") != statistics.get("identity_digest")
        or family.get("checkpoint_kind") != expected[0]
        or family.get("model_class") != expected[1]
        or family.get("method") != current["method"]
        or current.get("model_family") != model_family
        or family.get("sampling_protocol") != "full_window"
        or family.get("training_recipe_identity") != current.get("training_recipe")
        or family.get("training_runtime") != current.get("training_runtime")
        or not current.get("source_snapshot", {}).get("files_digest")
        or family.get("source_files_digest")
        != current["source_snapshot"]["files_digest"]
        or current.get("training_regions") != [report["region"]]
        or current.get("training_scope")
        != "same_region_full_window_supervised_temporal_split"
        or report["collection"].get("sampling_protocol") != "full_window"
    ):
        raise ValueError(
            "full-window family, training scope, source or numerical identities disagree"
        )
    return {
        "fullwindow_model_family": family,
        "model_family": current["model_family"],
        "model_class": current["model_class"],
        "native_model_class": current.get("native_model_class"),
        "reader_identity": reader,
        "sampling_identity": sampling,
        "statistics_identity": statistics,
        "data_identity": current["data_identity"],
        "evaluation_source": report["evaluation_source_snapshot"]["files_digest"],
        "scalar_fit_precision": "float64",
        "legacy27_scalar_fit_precision": "float32",
        "legacy27_scores_directly_comparable": False,
    }


def load_member(
    declaration: dict, *, training_seed: int, base: Path
) -> tuple[dict, dict]:
    path = _path(declaration["report"], base)
    digest = file_sha256(path)
    if digest != declaration["report_sha256"]:
        raise ValueError(
            "frozen evaluation report hash differs from the cohort declaration"
        )
    report = json.loads(path.read_text())
    if report["status"] not in STATUSES or report["split"] != "test":
        raise ValueError("cohort accepts only completed frozen test reports")
    source_hash = report.get("evaluation_source_snapshot", {}).get("files_digest")
    if not source_hash:
        source_hash = report.get("provenance", {}).get("evaluation_source_files_digest")
    if not source_hash:
        raise ValueError("report omits its evaluation source identity")
    if report["status"] == "frozen_transfer_evaluation_complete":
        # Older transfer reports record perturbation seed, not the training seed.
        # Bind the latter to the already frozen original checkpoint instead.
        checkpoint_path = _path(declaration["checkpoint"], base)
        if file_sha256(checkpoint_path) != report["checkpoint_sha256"]:
            raise ValueError(
                "transfer checkpoint hash differs from the completed report"
            )
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint["config"]["seed"] != training_seed
            or canonical_sha256(checkpoint["config"]) != checkpoint["config_digest"]
            or checkpoint["config"]["source_snapshot"]["files_digest"]
            != report["candidate_sha"]
        ):
            raise ValueError(
                "transfer checkpoint training seed or source identity mismatch"
            )
        kind = declaration.get("model", "shared_graph_ndr")
        current = report if kind == "shared_graph_ndr" else report["controls"][kind]
        family = {
            "kind": kind,
            "candidate_source": report["candidate_sha"],
            "model_config": report["model_config"]
            if kind == "shared_graph_ndr"
            else None,
            "training_recipe": training_recipe_identity(checkpoint["config"])
            if kind == "shared_graph_ndr"
            else None,
            "training_runtime": checkpoint["config"].get("runtime")
            if kind == "shared_graph_ndr"
            else None,
            "preprocessing": preprocessing_semantics(
                report["evaluation_preprocessing_identity"]
                if kind == "shared_graph_ndr"
                else report["control_preprocessing_identity"]
            ),
            "evaluation_source": source_hash,
        }
        region, mode, protocol = (
            report["heldout_region"],
            report["evaluation_mode"],
            "event",
        )
        training_regions = report["source_regions"]
        training_scope = (
            "source_only_trained_weights"
            if kind == "shared_graph_ndr"
            else "fixed_source_only_control"
        )
        trained_on_region, comparison_scope = (
            False,
            "source_only_transfer_with_fixed_source_controls",
        )
    elif report["status"] == "frozen_source_cohort_evaluation_complete":
        kind = declaration["model"]
        current = report["models"][kind]
        if current.get("source_only_control"):
            family = {
                "kind": kind,
                "source_preprocessing": preprocessing_semantics(
                    report["source_preprocessing_identity"]
                ),
                "evaluation_source": source_hash,
            }
            training_scope = "fixed_source_only_control"
        else:
            if current["training_seed"] != training_seed or current[
                "model_class"
            ] not in ("SharedGraphNDR", "SharedSTAEformer"):
                raise ValueError(
                    "source cohort selected model has a different seed or true model class"
                )
            family = {
                "source_model_family": current["family_identity"],
                "evaluation_preprocessing": preprocessing_semantics(
                    report["preprocessing_identity"]
                ),
                "evaluation_source": source_hash,
            }
            training_scope = current["training_scope"]
        training_regions = report["source_regions"]
        region, mode, protocol = (
            report["region"],
            report["evaluation_mode"],
            report["sampling_protocol"],
        )
        trained_on_region, comparison_scope = False, report["comparison_scope"]
    elif report["status"] == "frozen_fullwindow_evaluation_complete":
        kind = declaration["model"]
        current = report["models"][kind]
        family = _fullwindow_family(report, current, training_seed)
        region, mode, protocol = report["region"], "indomain", "full_window"
        training_regions = current["training_regions"]
        training_scope = current["training_scope"]
        trained_on_region = True
        comparison_scope = "matched_fullwindow_regional_supervision"
    else:
        if report["training_seed"] != training_seed:
            raise ValueError(
                "in-domain report training seed differs from its declaration"
            )
        kind = report["model_id"]
        if declaration.get("model", kind) != kind:
            raise ValueError(
                "in-domain model id differs from the frozen cohort declaration"
            )
        current = report
        provenance = report["provenance"]
        family = {
            "kind": report["model_family"],
            "model_config": provenance["model_config"],
            "training_recipe": provenance["training_recipe_identity"],
            "training_runtime": provenance["training_runtime"],
            "candidate_source": provenance["source_files_digest"],
            "evaluation_source": source_hash,
        }
        region, mode, protocol = (
            report["region"],
            report.get("evaluation_mode", "indomain"),
            report["sampling_protocol"],
        )
        training_regions = provenance["training_regions"]
        training_scope = report["training_scope"]
        trained_on_region = report["trained_on_evaluated_region"]
        comparison_scope = report["comparison_scope"]
    if (
        not isinstance(training_regions, list)
        or not training_regions
        or len(set(training_regions)) != len(training_regions)
        or type(trained_on_region) is not bool
        or trained_on_region != (region in training_regions)
    ):
        raise ValueError(
            "reported training regions and evaluated-region supervision flag disagree"
        )
    normalized = {
        "training_seed": training_seed,
        "family_identity": family,
        "region": region,
        "split": report["split"],
        "evaluation_mode": mode,
        "comparison_scope": comparison_scope,
        "training_scope": training_scope,
        "training_regions": training_regions,
        "trained_on_evaluated_region": trained_on_region,
        "sampling_protocol": protocol,
        "collection": report["collection"],
        "scenarios": current["scenarios"],
    }
    identity = {
        "report": str(path),
        "report_sha256": digest,
        "model": kind,
        "checkpoint_sha256": current.get(
            "checkpoint_sha256", report.get("checkpoint_sha256")
        ),
    }
    return normalized, identity


def analyze_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != 1 or manifest["selection_frozen"] is not True:
        raise ValueError("cohort needs an explicit frozen analysis declaration")
    rows = manifest["pairs"]
    if (
        len(rows) != 3
        or any(type(row["training_seed"]) is not int for row in rows)
        or {row["training_seed"] for row in rows} != set(SEEDS)
    ):
        raise ValueError(
            "cohort declaration requires exactly the three recorded training seeds"
        )
    if tuple(manifest["scenarios"]) != SCENARIOS:
        raise ValueError(
            "cohort must report all five predeclared historical observation scenarios"
        )
    left, right, identities = {}, {}, []
    for row in rows:
        seed = row["training_seed"]
        left[seed], first = load_member(
            row["left"], training_seed=seed, base=path.parent
        )
        right[seed], second = load_member(
            row["right"], training_seed=seed, base=path.parent
        )
        identities.append({"training_seed": seed, "left": first, "right": second})
    results = {
        scenario: paired_seed_day_cohort(
            left,
            right,
            scenario=scenario,
            samples=manifest["bootstrap_samples"],
            bootstrap_seed=manifest["bootstrap_seed"],
        )
        for scenario in SCENARIOS
    }
    region, mode = left[42]["region"], left[42]["evaluation_mode"]
    if manifest["evidence_role"] not in (
        "primary_orange_confirmation",
        "supplementary_frozen_loco",
        "indomain_performance",
    ):
        raise ValueError(
            "cohort evidence role must distinguish confirmation from supplemental folds"
        )
    if manifest["evidence_role"] == "primary_orange_confirmation" and (
        region != "Orange" or mode == "indomain"
    ):
        raise ValueError(
            "primary unseen-region confirmation is the frozen Orange transfer fold"
        )
    if (manifest["evidence_role"] == "indomain_performance") != (mode == "indomain"):
        raise ValueError(
            "cohort evidence role does not match its regional or transfer evaluation scope"
        )
    return {
        "schema_version": 1,
        "status": "frozen_cohort_reports_analyzed",
        "analysis_manifest_sha256": file_sha256(path),
        "reports": identities,
        "evidence_role": manifest["evidence_role"],
        "region": region,
        "evaluation_mode": mode,
        "comparison_scope": left[42]["comparison_scope"],
        "sampling_protocol": left[42]["sampling_protocol"],
        "model_family_identities": results["clean"]["family_identity"],
        "model_training_scopes": results["clean"]["model_training_scopes"],
        "scenarios": results,
        "traffic_labels_opened": False,
        "test_driven_tuning": False,
        "sota_or_winner_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_manifest(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "region": result["region"],
                "traffic_labels_opened": False,
                "output": str(args.output.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
