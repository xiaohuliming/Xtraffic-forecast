#!/usr/bin/env python3
"""Inspect frozen checkpoint structure or evaluate a closed regional test cohort."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndr_research.indomain_evaluation import (
    KINDS,
    admit_regional_gate,
    evaluate_regional_models,
    fit_regional_training_preprocessing,
    inspect_selected_model,
    model_preprocessing_from_admission,
    rebuild_regional_model,
    RegionalEvaluationLedger,
    state_digest,
)
from ndr_research.transfer_data import TransferRegionReader, TestEvaluationGate
from ndr_research.transfer_data import REGIONS
from ndr_research.transfer_evaluation import (
    canonical_sha256,
    file_sha256,
    write_new_json,
)
from scripts.evaluate_ndr_transfer import evaluation_source_snapshot
from scripts.train_ndr_research import require_snapshot
from scripts.train_ndr_transfer import (
    configure_transfer_runtime,
    transfer_runtime_identity,
)


def argument_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use one shared regional-family ledger. Three seeds plus one independent inference repeat require a predeclared budget of six seed-inference slots. Failures consume slots. Native clean inputs are preserved; added missingness never reuses removed observations or old cache fills.",
    )
    parser.add_argument(
        "--mode",
        choices=("inspect-checkpoint", "create-gate", "evaluate"),
        default="evaluate",
    )
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-kind", choices=KINDS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--bist-checkout", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup",
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=ROOT / "outputs/region_graphs"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--region", choices=REGIONS)
    parser.add_argument("--authorization-id")
    parser.add_argument("--structure-development-closed", action="store_true")
    parser.add_argument(
        "--comparison-scope",
        choices=("matched_indomain_training", "declared_mixed_training_scope"),
        default="matched_indomain_training",
    )
    parser.add_argument("--evaluation-ledger", type=Path)
    parser.add_argument("--max-test-evaluations", type=int, default=3)
    parser.add_argument("--perturbation-seed", type=int, default=42)
    return parser


def create_explicit_gate(args):
    """A model manifest selects already verified checkpoints, never architectures."""
    from ndr_research.transfer_evaluation import SCENARIOS
    from ndr_research.transfer_data import PROTOCOL

    if (
        args.model_manifest is None
        or args.gate_json is None
        or args.region is None
        or args.evaluation_ledger is None
        or not args.authorization_id
        or not args.structure_development_closed
    ):
        raise ValueError(
            "gate creation requires a checkpoint manifest, region, authorization, shared ledger and explicit closed-development declaration"
        )
    if args.output is not None:
        raise ValueError("gate creation cannot create a target report directory")
    manifest = json.loads(args.model_manifest.read_text())
    declarations = []
    bundle = None
    for row in manifest["models"]:
        paths = {
            name: Path(row[name]) for name in ("checkpoint", "verification", "summary")
        }
        paths = {
            name: value
            if value.is_absolute()
            else args.model_manifest.resolve().parent / value
            for name, value in paths.items()
        }
        accepted = inspect_selected_model(
            paths["checkpoint"],
            paths["verification"],
            paths["summary"],
            kind=row["checkpoint_kind"],
        )
        if bundle is not None and bundle != accepted["bundle_digest"]:
            raise ValueError("checkpoint manifest mixes formal cache bundles")
        bundle = accepted["bundle_digest"]
        declarations.append(
            {
                "model_id": row["model_id"],
                "checkpoint_kind": row["checkpoint_kind"],
                **{name: str(value.resolve()) for name, value in paths.items()},
                **{
                    name + "_sha256": file_sha256(value)
                    for name, value in paths.items()
                },
                **{
                    key: accepted[key]
                    for key in (
                        "validation_selection_sha256",
                        "data_identity_sha256",
                        "config_digest",
                        "family_identity",
                        "training_seed",
                    )
                },
            }
        )
    gate = {
        "schema_version": 1,
        "region": args.region,
        "split": "test",
        "sampling_protocol": "event",
        "protocol_version": PROTOCOL,
        "structure_development_closed": True,
        "selection_frozen": True,
        "test_used_for_structure_selection": False,
        "authorization_id": args.authorization_id,
        "cache_bundle_digest": bundle,
        "scenarios": list(SCENARIOS),
        "input_policy": "native_clean_and_retained_observation_causal_corruption",
        "perturbation_seed": args.perturbation_seed,
        "comparison_scope": args.comparison_scope,
        "evaluation_ledger": str(args.evaluation_ledger.resolve()),
        "max_test_evaluations": args.max_test_evaluations,
        "models": declarations,
    }
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.gate_json, gate)
    try:
        admit_regional_gate(args.gate_json)
    except BaseException:
        args.gate_json.unlink()
        raise
    return {
        "status": "frozen_indomain_gate_created",
        "gate_json": str(args.gate_json.resolve()),
        "gate_sha256": file_sha256(args.gate_json),
        "checkpoint_count": len(declarations),
        "target_reader_created": False,
        "test_evaluated": False,
    }


def _evaluation_snapshot():
    snapshot = evaluation_source_snapshot()
    for name in (
        "ndr_research/indomain_evaluation.py",
        "scripts/evaluate_ndr_indomain.py",
        "ndr_research/strong_references.py",
        "scripts/train_ndr_strong_reference.py",
        "scripts/protocol_v3_event_extension_data.py",
        "ndr_research/cohort.py",
    ):
        path = ROOT / name
        snapshot["files"][name] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    snapshot["files_digest"] = canonical_sha256(snapshot["files"])
    snapshot["scope"] = "frozen_indomain_forecast_and_preprocessing_implementation"
    return snapshot


def run_evaluation(args):
    if args.gate_json is None or args.output is None:
        raise ValueError(
            "regional evaluation requires --gate-json and a new --output directory"
        )
    gate, admitted = admit_regional_gate(args.gate_json, source_root=ROOT)
    device = configure_transfer_runtime(
        args.device, gate["perturbation_seed"], args.threads
    )
    snapshot = _evaluation_snapshot()
    require_snapshot(snapshot)
    models = {
        name: rebuild_regional_model(spec, device, bist_checkout=args.bist_checkout)
        for name, spec in admitted.items()
    }
    before = {name: state_digest(model) for name, model in models.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "gate.json").open("xb") as handle:
        handle.write(args.gate_json.read_bytes())
    write_new_json(args.output / "evaluation_source_snapshot.json", snapshot)
    ledger = RegionalEvaluationLedger(gate, admitted, args.gate_json, args.output)
    reservation = ledger.reserve()
    reader = None
    try:
        reader = TransferRegionReader(
            gate["region"], args.data_dir, args.graph_dir, role="source"
        )
        if reader.contract.bundle_digest != gate["cache_bundle_digest"]:
            raise ValueError(
                "regional test reader differs from the frozen cache bundle"
            )
        regional = fit_regional_training_preprocessing(reader)
        preprocessing = {
            name: model_preprocessing_from_admission(spec, regional)
            for name, spec in admitted.items()
        }
        result = evaluate_regional_models(
            models,
            reader=reader,
            preprocessing=regional,
            model_preprocessing=preprocessing,
            evaluation_grant=TestEvaluationGate(
                gate["authorization_id"], gate["gate_sha256"]
            ),
            batch_size=args.batch_size,
            perturbation_seed=gate["perturbation_seed"],
            device=device,
        )
        require_snapshot(snapshot)
        current_gate, _ = admit_regional_gate(args.gate_json, source_root=ROOT)
        if current_gate["gate_sha256"] != gate["gate_sha256"]:
            raise RuntimeError("regional freeze gate changed during evaluation")
        if before != {name: state_digest(model) for name, model in models.items()}:
            raise RuntimeError(
                "regional evaluation changed model parameters or registered buffers"
            )
        index = []
        for name, current in result["models"].items():
            spec = admitted[name]
            trained_here = gate["region"] in spec["training_regions"]
            training_scope = (
                "source_only_transfer"
                if not trained_here
                else "regional_supervised"
                if len(spec["training_regions"]) == 1
                else "multisource_including_evaluated_region"
            )
            provenance = {
                "source_files_digest": spec["family_identity"]["source_files_digest"],
                "model_config": spec["family_identity"]["architecture"],
                "training_recipe_identity": spec["family_identity"][
                    "training_recipe_identity"
                ],
                "training_runtime": spec["checkpoint"]["config"]["runtime"],
                "training_regions": list(spec["training_regions"]),
                "training_scope": training_scope,
                "trained_on_evaluated_region": trained_here,
                "source_validation_verification_sha256": spec["verification_sha256"],
                "checkpoint_config_digest": spec["config_digest"],
                "validation_selection_sha256": spec["validation_selection_sha256"],
                "preprocessing_identity": dict(preprocessing[name].identity),
                "original_model_architecture_preserved": True,
            }
            report = {
                "schema_version": 1,
                "status": "frozen_indomain_model_evaluation_complete",
                "region": gate["region"],
                "split": "test",
                "sampling_protocol": "event",
                "evaluation_mode": "indomain",
                "training_seed": spec["training_seed"],
                "model_id": name,
                "model_family": spec["model_family"],
                "family_identity": spec["family_identity"],
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "gate_sha256": gate["gate_sha256"],
                "provenance": provenance,
                "comparison_scope": gate["comparison_scope"],
                "training_scope": training_scope,
                "trained_on_evaluated_region": trained_here,
                "evaluation_source_snapshot": snapshot,
                "runtime": transfer_runtime_identity(device),
                "state_sha256": before[name],
                "target_access_audit": reader.access_audit(),
                "evaluation_ledger": str(ledger.path),
                "family_test_budget_sha256": ledger.family_sha,
                "family_test_budget": gate["max_test_evaluations"],
                "consumed_test_inference_slots": reservation["consumed_slots_after"],
                "claim_scope": "frozen_paired_regional_test_no_automatic_sota_claim",
                **current,
            }
            destination = args.output / (name + ".json")
            write_new_json(destination, report)
            index.append(
                {
                    "model_id": name,
                    "path": destination.name,
                    "sha256": file_sha256(destination),
                    "training_seed": spec["training_seed"],
                    "family_identity": spec["family_identity"],
                }
            )
        summary = {
            "schema_version": 1,
            "status": "frozen_indomain_paired_evaluation_complete",
            "gate_sha256": gate["gate_sha256"],
            "region": gate["region"],
            "evaluation_mode": "indomain",
            "collection": result["collection"],
            "models": index,
            "evaluation_region_train_statistics_used_for_regional_controls": True,
            "comparison_scope": gate["comparison_scope"],
            "wallclock_seconds": result["wallclock_seconds"],
            "evaluation_ledger": str(ledger.path),
            "family_test_budget_sha256": ledger.family_sha,
            "consumed_test_inference_slots": reservation["consumed_slots_after"],
            "target_windows_read_once_for_all_models": True,
            "evaluation_source_snapshot": snapshot,
            "sota_or_winner_claim": False,
        }
        write_new_json(args.output / "paired_matrix_report.json", summary)
        ledger.finish(
            "completed",
            paired_matrix_sha256=file_sha256(args.output / "paired_matrix_report.json"),
        )
        return summary
    except BaseException as error:
        failure = {
            "status": "frozen_indomain_evaluation_failed",
            "error_type": type(error).__name__,
            "gate_sha256": gate["gate_sha256"],
            "test_gate_consumed": True,
            "target_access_audit": None if reader is None else reader.access_audit(),
        }
        write_new_json(args.output / "failure.json", failure)
        ledger.finish("failed", error_type=type(error).__name__)
        raise


def main(argv=None):
    args = argument_parser().parse_args(argv)
    if args.mode == "inspect-checkpoint":
        if any(
            value is None
            for value in (
                args.checkpoint,
                args.verification,
                args.summary,
                args.checkpoint_kind,
            )
        ):
            raise ValueError(
                "structure inspection needs checkpoint, kind, verification and summary"
            )
        inspected = inspect_selected_model(
            args.checkpoint, args.verification, args.summary, kind=args.checkpoint_kind
        )
        configure_transfer_runtime("cpu", inspected["training_seed"], args.threads)
        model = rebuild_regional_model(
            inspected, "cpu", bist_checkout=args.bist_checkout
        )
        result = {
            "status": "complete_checkpoint_structure_verified_only",
            "checkpoint_sha256": inspected["checkpoint_sha256"],
            "model_family": inspected["model_family"],
            "training_seed": inspected["training_seed"],
            "parameters": sum(value.numel() for value in model.parameters()),
            "model_state_sha256": state_digest(model),
            "target_reader_created": False,
            "accuracy_evaluated": False,
            "gpu_used": False,
        }
    elif args.mode == "create-gate":
        result = create_explicit_gate(args)
    else:
        result = run_evaluation(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
