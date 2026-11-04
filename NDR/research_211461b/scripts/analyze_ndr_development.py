"""Admit a complete, independently verified NDR development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

REGIONS = ("Alameda", "ContraCosta")
VARIANTS = ("baseline", "mask_signed", "sanitized_control", "projected_oracle")
BUNDLE = "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_inventory(root: Path, postflight: dict) -> None:
    declared = set()
    for row in postflight["files"]:
        relative = PurePosixPath(row["path"].replace("\\", "/"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] != "tasks"
        ):
            raise ValueError("postflight contains an unsafe or non-task path")
        if str(relative) in declared:
            raise ValueError("duplicate postflight file")
        declared.add(str(relative))
        path = root.joinpath(*relative.parts)
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
        ):
            raise ValueError(
                "postflight file is missing or escapes the copied task tree"
            )
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("copied task bytes differ from the source postflight")
    actual = {
        p.relative_to(root).as_posix()
        for p in (root / "tasks").rglob("*")
        if p.is_file()
    }
    if actual != declared:
        raise ValueError("copied task tree differs from the exact source inventory")


def admit_matrix(root: Path, plan: dict) -> dict:
    claimed = plan["plan_digest"]
    if canonical_hash({k: v for k, v in plan.items() if k != "plan_digest"}) != claimed:
        raise ValueError("plan digest mismatch")
    queue = json.loads((root / "results/queue_status.json").read_text())
    if queue["status"] != "completed" or queue["plan_digest"] != claimed:
        raise ValueError("the complete development queue is not source-verified")
    expected = [t["task_id"] for t in plan["tasks"]]
    if queue["completed"] != expected or len(expected) != 8:
        raise ValueError("development matrix task union or fixed order differs")
    postflight = json.loads((root / "results/postflight.json").read_text())
    if (
        postflight["status"] != "verified_development_train_val_only"
        or postflight["plan_digest"] != claimed
        or postflight["tasks"] != expected
        or postflight["test_evaluated"]
    ):
        raise ValueError(
            "source postflight does not certify the declared no-test matrix"
        )
    verify_inventory(root, postflight)
    rows = []
    common = {}
    recipes = set()
    source_digests = set()
    runtimes = set()
    for task in plan["tasks"]:
        directory = root / "tasks" / task["task_id"]
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        verified = json.loads((directory / "verification.json").read_text())
        identity = json.loads((directory / "data_identity.json").read_text())
        validation = verified["validation"]
        selected = summary["best_validation"]
        if any(
            config.get(k) != v
            for k, v in {
                "region": task["region"],
                "variant": task["variant"],
                "epochs": 30,
                "seed": 42,
                "precision": "FP32",
                "claim_scope": "validation_development_only",
            }.items()
        ):
            raise ValueError("task recipe or smoke/formal scope differs from the plan")
        if (
            config.get("max_train_batches") is not None
            or config.get("max_val_batches") is not None
        ):
            raise ValueError("development matrix contains a limited-batch smoke")
        if (
            summary["completed_epochs"] != 30
            or summary["test_evaluated"]
            or verified["test_evaluated"]
        ):
            raise ValueError("task is incomplete or consumed test during development")
        if (
            verified["status"] != "verified_checkpoint_complete_validation"
            or not validation["selection"]["all_validation_samples"]
        ):
            raise ValueError("independent verification covers only a validation subset")
        if (
            summary["config_digest"] != canonical_hash(config)
            or verified["config_digest"] != summary["config_digest"]
        ):
            raise ValueError("task configuration identity mismatch")
        if verified["checkpoint_sha256"] != sha(directory / "best.pt"):
            raise ValueError(
                "selected checkpoint bytes differ from independent verification"
            )
        if (
            verified["cache_bundle_digest"] != BUNDLE
            or identity["cache_bundle_digest"] != BUNDLE
            or identity != config["data_identity"]
        ):
            raise ValueError("validation data identity differs from the formal bundle")
        if verified["source_files_digest"] != config["source_snapshot"]["files_digest"]:
            raise ValueError("task source identity mismatch")
        for key in [
            "valid_count",
            "selection",
            "observed_targets_sha256",
            "observation_masks_sha256",
        ]:
            if selected[key] != validation[key]:
                raise ValueError(
                    "checkpoint selection and verification collections differ"
                )
        if not math.isclose(
            summary["best_val_mae"],
            validation["global_masked_mae"],
            rel_tol=2e-6,
            abs_tol=2e-5,
        ):
            raise ValueError("independent MAE differs from the selected checkpoint")
        count = validation["valid_count"]
        errors = validation["absolute_error_sum"]
        mae = validation["global_masked_mae"]
        if (
            not count > 0
            or not math.isfinite(mae)
            or not math.isclose(errors / count, mae, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(
                "validation metric does not use the full observed-point denominator"
            )
        paired = {
            k: validation[k]
            for k in [
                "valid_count",
                "selection",
                "observed_targets_sha256",
                "observation_masks_sha256",
            ]
        }
        region = task["region"]
        if region in common and common[region] != paired:
            raise ValueError(
                "regional methods do not share exact validation targets/masks/origins"
            )
        common[region] = paired
        snapshot_files = config["source_snapshot"]["files"]
        if canonical_hash(snapshot_files) != config["source_snapshot"]["files_digest"]:
            raise ValueError("source snapshot content digest is inconsistent")
        for filename, declaration in snapshot_files.items():
            if declaration["sha256"] != plan["source_files"].get(filename):
                raise ValueError("trained source bytes differ from the frozen plan")
        recipe_fields = (
            "epochs",
            "seed",
            "batch_size",
            "microbatch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip",
            "eta_min",
            "lambda_gate",
            "lambda_specialization",
            "hidden_dim",
            "reliable_anchor",
            "max_reference_age",
            "precision",
            "early_stopping",
            "test_evaluation",
            "log_interval",
            "threads",
        )
        recipes.add(canonical_hash({key: config[key] for key in recipe_fields}))
        if (
            config["early_stopping"]
            or config["test_evaluation"]
            or config["reliable_anchor"]
        ):
            raise ValueError(
                "frozen development task changed stopping, test or anchor policy"
            )
        source_digests.add(verified["source_files_digest"])
        runtimes.add(canonical_hash(config["runtime"]))
        rows.append(
            {
                "region": region,
                "variant": task["variant"],
                "seed": 42,
                "global_validation_mae": mae,
                "average_horizon_mae": validation["average_horizon_mae"],
                "horizon_mae": validation["horizon_mae"],
                "best_epoch": summary["best_epoch"],
                "epochs": 30,
                "valid_count": count,
                "elapsed_seconds": summary["elapsed_seconds"],
                "checkpoint_sha256": verified["checkpoint_sha256"],
                "config_digest": summary["config_digest"],
            }
        )
    if len(source_digests) != 1 or len(runtimes) != 1 or len(recipes) != 1:
        raise ValueError(
            "direct development comparison requires a single code/runtime/optimizer snapshot"
        )
    if {(row["region"], row["variant"]) for row in rows} != {
        (r, v) for r in REGIONS for v in VARIANTS
    }:
        raise ValueError("matrix is not the complete two-region/four-variant design")
    indexed = {(row["region"], row["variant"]): row for row in rows}
    decisions = {}
    for variant in ("mask_signed", "projected_oracle"):
        deltas = {
            region: indexed[region, variant]["global_validation_mae"]
            - indexed[region, "baseline"]["global_validation_mae"]
            for region in REGIONS
        }
        sanitation = {
            region: indexed[region, variant]["global_validation_mae"]
            - indexed[region, "sanitized_control"]["global_validation_mae"]
            for region in REGIONS
        }
        improvements = all(value < 0 for value in deltas.values())
        mechanism = variant != "mask_signed" or all(
            value < 0 for value in sanitation.values()
        )
        decisions[variant] = {
            "candidate_minus_baseline_mae": deltas,
            "candidate_minus_sanitized_control_mae": sanitation,
            "both_regions_improved": improvements,
            "beyond_sanitization_in_both_regions": mechanism,
            "eligible_for_frozen_multiseed_confirmation": improvements and mechanism,
            "test_or_sota_claim": False,
        }
    return {
        "status": "verified_complete_development_comparison",
        "plan_digest": claimed,
        "rows": rows,
        "decisions": decisions,
        "source_files_digest": next(iter(source_digests)),
        "runtime_digest": next(iter(runtimes)),
        "test_evaluated": False,
        "claim_scope": "single_seed_fixed_budget_validation_development_only",
        "postflight_sha256": sha(root / "results/postflight.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = admit_matrix(args.root, json.loads(args.plan.read_text()))
    with args.output.open("x", encoding="utf8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(
        json.dumps(
            {"status": report["status"], "decisions": report["decisions"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
