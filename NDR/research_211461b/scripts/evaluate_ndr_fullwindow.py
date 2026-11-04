#!/usr/bin/env python3
"""Inspect or evaluate a frozen same-region independent full-window cohort."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ndr_research.fullwindow_data import FullWindowStatistics
from ndr_research.fullwindow_evaluation import (
    FAMILY_SCOPE,
    GATE_SCHEMA,
    KINDS,
    REPORT_STATUS,
    SEEDS,
    FullWindowEvaluationLedger,
    FullWindowForecastAdapter,
    FullWindowFrozenGrant,
    FullWindowFrozenReader,
    admit_fullwindow_gate,
    evaluate_fullwindow_models,
    inspect_fullwindow_model,
    rebuild_fullwindow_model,
)
from ndr_research.transfer_data import PROTOCOL, REGIONS
from ndr_research.transfer_evaluation import (
    SCENARIOS,
    canonical_sha256,
    file_sha256,
    write_new_json,
)
from scripts.evaluate_ndr_source_cohort import evaluator_snapshot as base_snapshot
from scripts.train_ndr_research import require_snapshot
from scripts.train_ndr_transfer import (
    configure_transfer_runtime,
    transfer_runtime_identity,
)


def public_spec(spec):
    return {
        name: copy.deepcopy(value)
        for name, value in spec.items()
        if name not in {"checkpoint", "paths"}
    }


def evaluator_snapshot():
    snapshot = base_snapshot()
    for name in (
        "ndr_research/fullwindow_evaluation.py",
        "scripts/evaluate_ndr_fullwindow.py",
        "ndr_research/fullwindow_data.py",
        "ndr_research/fullwindow_training.py",
        "scripts/train_ndr_fullwindow.py",
        "scripts/train_strong_fullwindow.py",
    ):
        path = ROOT / name
        snapshot["files"][name] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    snapshot["files_digest"] = canonical_sha256(snapshot["files"])
    return snapshot


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inspect-checkpoint", "create-gate", "evaluate"),
        default="evaluate",
    )
    parser.add_argument("--checkpoint-kind", choices=KINDS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--bist-checkout", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--region", choices=REGIONS)
    parser.add_argument("--authorization-id")
    parser.add_argument("--structure-development-closed", action="store_true")
    parser.add_argument("--evaluation-ledger", type=Path)
    parser.add_argument("--max-test-evaluations", type=int, default=6)
    parser.add_argument("--perturbation-seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup",
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=ROOT / "outputs/region_graphs"
    )
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def create_gate(args):
    if (
        not args.model_manifest
        or not args.gate_json
        or not args.region
        or not args.authorization_id
        or not args.structure_development_closed
        or not args.evaluation_ledger
        or args.output is not None
    ):
        raise ValueError(
            "full-window gate requires an explicit model manifest, region, authorization, closed structure and shared ledger"
        )
    models = []
    for row in json.loads(args.model_manifest.read_text())["models"]:
        paths = {
            field: Path(row[field])
            for field in ("checkpoint", "verification", "summary")
        }
        paths = {
            field: value
            if value.is_absolute()
            else args.model_manifest.resolve().parent / value
            for field, value in paths.items()
        }
        spec = inspect_fullwindow_model(
            paths["checkpoint"],
            paths["verification"],
            paths["summary"],
            kind=row["checkpoint_kind"],
            bist_checkout=args.bist_checkout,
        )
        models.append(
            {
                "model_id": row["model_id"],
                "checkpoint_kind": row["checkpoint_kind"],
                **{field: str(value.resolve()) for field, value in paths.items()},
                **{
                    field + "_sha256": file_sha256(value)
                    for field, value in paths.items()
                },
                **{
                    field: spec[field]
                    for field in (
                        "family_identity",
                        "training_seed",
                        "model_class",
                        "config_digest",
                        "validation_selection_sha256",
                        "model_state_sha256",
                    )
                },
            }
        )
    gate = {
        "schema_version": GATE_SCHEMA,
        "protocol_version": PROTOCOL,
        "sampling_protocol": "full_window",
        "split": "test",
        "evaluation_role": "same_region_temporal_heldout",
        "region": args.region,
        "authorization_id": args.authorization_id,
        "structure_development_closed": True,
        "selection_frozen": True,
        "test_used_for_structure_selection": False,
        "training_seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "perturbation_seed": args.perturbation_seed,
        "models": models,
        "max_test_evaluations": args.max_test_evaluations,
        "evaluation_ledger": str(args.evaluation_ledger.resolve()),
        **{
            field: spec[field]
            for field in ("reader_identity", "sampling_identity", "statistics_identity")
        },
    }
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.gate_json, gate)
    try:
        accepted, admitted = admit_fullwindow_gate(
            args.gate_json, bist_checkout=args.bist_checkout
        )
    except BaseException:
        args.gate_json.unlink()
        raise
    ledger = FullWindowEvaluationLedger(
        accepted, admitted, args.gate_json, args.gate_json.parent
    )
    return {
        "status": "frozen_fullwindow_gate_created",
        "gate_sha256": accepted["gate_sha256"],
        "family_sha256": ledger.family_sha,
        "target_data_read": False,
    }


def run_evaluation(args):
    if (
        not args.gate_json
        or not args.output
        or args.batch_size <= 0
        or args.threads <= 0
    ):
        raise ValueError(
            "full-window evaluation requires a frozen gate, new output and positive batch/thread values"
        )
    if any(
        getattr(args, field) is not None
        for field in (
            "checkpoint",
            "verification",
            "summary",
            "checkpoint_kind",
            "model_manifest",
            "region",
            "authorization_id",
            "evaluation_ledger",
        )
    ):
        raise ValueError(
            "full-window evaluation obtains all model and authorization identities from its gate"
        )
    gate, admitted = admit_fullwindow_gate(
        args.gate_json, bist_checkout=args.bist_checkout
    )
    device = configure_transfer_runtime(
        args.device, gate["perturbation_seed"], args.threads
    )
    snapshot = evaluator_snapshot()
    require_snapshot(snapshot)
    raw_models = {
        name: rebuild_fullwindow_model(spec, device, args.bist_checkout)
        for name, spec in admitted.items()
    }
    models = {
        name: FullWindowForecastAdapter(
            raw_models[name], kind=spec["kind"], method=spec["method"]
        ).eval()
        for name, spec in admitted.items()
    }
    statistics = FullWindowStatistics.from_state(
        next(iter(admitted.values()))["checkpoint"]["statistics_state"]
    )
    statistics_before = dict(statistics.identity)
    args.output.mkdir(parents=True, exist_ok=False)
    gate_bytes = args.gate_json.read_bytes()
    if file_sha256(args.gate_json) != gate["gate_sha256"]:
        raise RuntimeError("full-window gate changed after admission")
    with (args.output / "gate.json").open("xb") as handle:
        handle.write(gate_bytes)
    write_new_json(args.output / "evaluation_source_snapshot.json", snapshot)
    ledger = FullWindowEvaluationLedger(gate, admitted, args.gate_json, args.output)
    reservation = ledger.reserve()
    reader = None
    try:
        grant = FullWindowFrozenGrant(
            gate["authorization_id"],
            gate["gate_sha256"],
            canonical_sha256(
                {name: spec["checkpoint_sha256"] for name, spec in admitted.items()}
            ),
            gate["region"],
            gate["reader_identity"]["identity_digest"],
            gate["sampling_identity"]["sampling_digest"],
            gate["statistics_identity"]["identity_digest"],
        )
        reader = FullWindowFrozenReader(
            gate["region"],
            args.data_dir,
            args.graph_dir,
            grant=grant,
            statistics=statistics,
            source_reader_identity=gate["reader_identity"],
            sampling_identity=gate["sampling_identity"],
            bundle_manifest=args.bundle_manifest,
        )
        result = evaluate_fullwindow_models(
            models,
            reader=reader,
            grant=grant,
            batch_size=args.batch_size,
            perturbation_seed=gate["perturbation_seed"],
            device=device,
        )
        if result["model_state_sha256"] != {
            name: spec["model_state_sha256"] for name, spec in admitted.items()
        }:
            raise RuntimeError(
                "full-window inference model state differs from its admitted checkpoint"
            )
        after_gate, after_models = admit_fullwindow_gate(
            args.gate_json, bist_checkout=args.bist_checkout
        )
        if after_gate != gate or {
            name: spec["model_state_sha256"] for name, spec in after_models.items()
        } != {name: spec["model_state_sha256"] for name, spec in admitted.items()}:
            raise RuntimeError(
                "full-window gate or saved model state changed during test access"
            )
        require_snapshot(snapshot)
        for spec in admitted.values():
            require_snapshot(spec["source_snapshot"])
        if dict(statistics.identity) != statistics_before:
            raise RuntimeError("saved train-only full-window statistics changed")
        reader.require_unchanged()
        for name, value in result["models"].items():
            value.update(public_spec(admitted[name]))
        result.update(
            schema_version=GATE_SCHEMA,
            status=REPORT_STATUS,
            family_scope=FAMILY_SCOPE,
            family_sha256=ledger.family_sha,
            family_identity=ledger.family_payload,
            region=gate["region"],
            sampling_protocol="full_window",
            protocol_version=PROTOCOL,
            cache_bundle_digest=gate["reader_identity"]["material"]["bundle_digest"],
            comparison_scope="same_region_full_window_temporal_holdout",
            test_evaluation_budget_scope=FAMILY_SCOPE,
            split="test",
            mode="same_region_temporal_heldout",
            evaluation_role="same_region_temporal_heldout",
            training_seeds=list(SEEDS),
            reader_identity=gate["reader_identity"],
            sampling_identity=gate["sampling_identity"],
            statistics_identity=statistics_before,
            evaluation_reader_identity=reader.identity,
            gate_sha256=gate["gate_sha256"],
            budget_reservation=reservation,
            evaluation_runtime=transfer_runtime_identity(device),
            evaluation_source_snapshot=snapshot,
            access_audit=reader.access_audit(),
            scalar_fit_precision="float64",
            legacy27_scalar_fit_precision="float32",
            legacy27_scores_directly_comparable=False,
            test_driven_tuning=False,
            method_comparison_is_gate_ablation=False,
            test_evaluated=True,
        )
        write_new_json(args.output / "report.json", result)
        ledger.finish(
            "completed", report_sha256=file_sha256(args.output / "report.json")
        )
        return {
            "status": REPORT_STATUS,
            "report": str((args.output / "report.json").resolve()),
            "family_sha256": ledger.family_sha,
        }
    except BaseException as error:
        if reader is not None:
            write_new_json(
                args.output / "failure_access_audit.json", reader.access_audit()
            )
        ledger.finish("failed", error_type=type(error).__name__)
        raise


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    if args.mode == "evaluate" and any(
        value == flag or value.startswith(flag + "=")
        for value in arguments
        for flag in (
            "--max-test-evaluations",
            "--perturbation-seed",
            "--structure-development-closed",
        )
    ):
        raise ValueError(
            "full-window evaluation cannot override frozen budget, seed or structure"
        )
    if args.mode == "inspect-checkpoint":
        if (
            not args.checkpoint
            or not args.verification
            or not args.summary
            or not args.checkpoint_kind
        ):
            raise ValueError(
                "inspection needs checkpoint, verification, summary and actual checkpoint kind"
            )
        result = public_spec(
            inspect_fullwindow_model(
                args.checkpoint,
                args.verification,
                args.summary,
                kind=args.checkpoint_kind,
                bist_checkout=args.bist_checkout,
            )
        )
    elif args.mode == "create-gate":
        result = create_gate(args)
    else:
        result = run_evaluation(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
