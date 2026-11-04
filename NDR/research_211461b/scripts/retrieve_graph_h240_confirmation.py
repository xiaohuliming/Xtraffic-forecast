"""Opaque two-frame preservation and complete six-model Graph confirmation.

The unchanged collector library supplies exact tree, source-generation and
original Handle binding checks. No model or Torch imports occur on the Mac.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid

LIBRARY_SHA = "f2ad686f65ada30eb707b2da102c773208067060e45aaeaeb81f10825a5ae9c7"
_LIBRARY_PATH = Path(__file__).with_name("retrieve_shared_sta_norm_stage.py")
if os.name != "nt" and not _LIBRARY_PATH.exists():
    _LIBRARY_PATH = Path(__file__).resolve().parents[1] / "outputs/confirmation_cpu_dependencies/opaque_collector.py"
_CONTROLLER_PATH = Path(__file__).parent / "remote/windows/graph_h240_confirmation.py"
PRESERVED_SCHEMA = "graph_h240_confirmation_preservation_v1"
FAMILY_SCHEMA = "graph_h240_confirmed_family6_v1"
SOURCES = ("Alameda", "ContraCosta")


def controller(plan=None):
    spec = importlib.util.spec_from_file_location("_graph_confirmation_controller_" + uuid.uuid4().hex, _CONTROLLER_PATH)
    result = importlib.util.module_from_spec(spec); sys.modules[spec.name] = result
    previous = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try: spec.loader.exec_module(result)
    finally: sys.dont_write_bytecode = previous
    if plan is not None: result.need(result.sha(_CONTROLLER_PATH) == plan["controller"]["sha256"], "running Mac controller bytes differ")
    return result


def library(plan=None):
    q = controller(plan)
    result = q.load({"path": str(_LIBRARY_PATH), "sha256": LIBRARY_SHA}, LIBRARY_SHA)
    result.__file__ = __file__
    result.INVENTORY_SCHEMA = q.INVENTORY_SCHEMA; result.NATIVE_SCHEMA = q.NATIVE_SCHEMA
    result.PRESERVED_SCHEMA = PRESERVED_SCHEMA
    result.ARMS = tuple(row[0] for row in q.CONFIGS)
    result._controller = controller
    result._receipt_artifacts = lambda receipt, plan, files: q.native_units_validate(receipt, plan, files)
    result.source_native_validate = q.source_native_validate
    return result


def source_inventory(plan_path, expected_plan_sha256, *, exit_probe=None):
    return library().source_inventory(plan_path, expected_plan_sha256, exit_probe=exit_probe)


def validate_inventory(frame, plan):
    return library(plan).validate_inventory(frame, plan)


def admit_copied(root, frame, plan):
    return library(plan).admit_copied(root, frame, plan)


def source_native_validate(plan, **kwargs):
    return controller(plan).source_native_validate(plan, **kwargs)


def collect_stage(plan, **kwargs):
    """Always preserve the complete eight-child tree before family arithmetic."""
    return library(plan).collect_stage(plan, **kwargs)


def _metric(proof, expected, need):
    need(proof.get("status") == "verified_transfer_source_checkpoint_complete_validation"
         and proof.get("target_data_read") is False and proof.get("test_evaluated") is False
         and proof.get("config_digest") == expected["config_digest"]
         and proof.get("checkpoint_sha256") == expected["checkpoint_sha256"]["best.pt"], "complete source-only verification bytes differ")
    regions = proof["source_validation"]["regions"]
    values = {name: regions[name]["global_masked_mae"] for name in SOURCES}
    need(all(type(value) in {int, float} and math.isfinite(value) and value >= 0 for value in values.values()), "nonfinite validation metric")
    return values


def compare_family6(values):
    q = controller()
    names = {base + "_seed" + str(seed) for base in q.BASE_IDS for seed in (42, 1, 2)}
    q.need(type(values) is dict and set(values) == names and all(type(row) is dict and set(row) == set(SOURCES)
           and all(type(value) in {int, float} and math.isfinite(value) and value >= 0 for value in row.values()) for row in values.values()),
           "exact complete finite six-model source family required")
    improved = {str(seed): {region: values[q.BASE_IDS[0] + "_seed" + str(seed)][region] < values[q.BASE_IDS[1] + "_seed" + str(seed)][region]
                            for region in SOURCES} for seed in (42, 1, 2)}
    return improved


def family6(preservation_path, *, seed42_root, output):
    """Open metrics only after re-admitting complete Mac preservation and family.

    seed42_root is an explicit local copied tree. Foreign receipt paths are never
    interpreted as local paths. The source42 native proof is reused as JSON.
    """
    q = controller(); u = library()
    receipt = u.read(preservation_path)
    q.need(receipt.get("schema_version") == PRESERVED_SCHEMA and receipt.get("status") == "complete_success_preserved"
           and receipt.get("model_deserialization") is False and receipt.get("validation_inference") is False,
           "complete success preservation required before any family metric")
    root = Path(receipt["source_host_tree"]["path"])
    output_path = Path(output).absolute()
    q.need(all(not output_path.is_relative_to(protected.resolve()) for protected in (root, Path(seed42_root))),
           "family output cannot mutate preserved source trees")
    plan = u.read(root / "control/plan.json"); q.validate_plan(plan)
    q.need(plan.get("synthetic_only") is not True, "synthetic CPU protocol evidence cannot establish a scientific family")
    q.need(q.sha(_CONTROLLER_PATH) == plan["controller"]["sha256"], "family controller bytes differ")
    before = u.read(q.pin(receipt["source_before"])); after = u.read(q.pin(receipt["source_after"]))
    q.need(u.stable_frame(before) == u.stable_frame(after)
           and q.libraries().utc(before["observed_at_utc"]) < q.libraries().utc(after["observed_at_utc"]), "two ordered identical complete frames required")
    first = u.admit_copied(root, before, plan); second = u.admit_copied(root, after, plan)
    q.need(first == second and first["success"] is True and len(first["accepted_job_ids"]) == 8
           and receipt["terminal"] == second and receipt["source_native_receipt_sha256"] == after["source_native_receipt_sha256"]
           and receipt["source_host_tree"]["files"] == after["files"] and receipt["source_host_tree"]["files_digest"] == after["files_digest"], "complete8 preservation receipt or composite differs")
    files = u.validate_inventory(after, plan)
    # The original seven-result native composite is re-used, never Torch loaded.
    seed42_receipt = u.read(q.pin(_local_record(root, plan["seed42_preservation"], u), q.SEED42_SHA))
    analysis = u.read(q.pin(_local_record(root, plan["discovery_analysis"], u), q.ANALYSIS_SHA))
    seed42 = Path(seed42_root)
    q.need(seed42.is_absolute() and u.inventory_tree(seed42), "explicit actual Mac seed42 root required")
    q.need(seed42_receipt.get("status") == "complete30_verify_only_composite_admitted"
           and seed42_receipt.get("source_native_checkpoint_schema_admission_bound_to_copied_bytes") is True
           and seed42_receipt.get("composite") == analysis["complete_recovery_lineage"]["composite_counts"] | {"lifecycle_finished_at_utc": seed42_receipt["composite"].get("lifecycle_finished_at_utc")},
           "original complete7 native composite differs")
    # Exact original seven-result source file inventory is available in the
    # copied source_after frame. Keep source labels opaque on the Mac.
    original_after_record = analysis["evidence"]["complete_bridge_bindings"]["source_after"]
    original_after_name = Path(seed42_receipt["source_after"]).name
    original_after_path = seed42.parent / "receipts" / original_after_name
    q.need(u.sha(original_after_path) == original_after_record["sha256"] == seed42_receipt["source_after_sha256"], "original seed42 complete source frame missing")
    original_after = u.read(original_after_path)
    original_before_path = seed42.parent / "receipts" / Path(seed42_receipt["source_before"]).name
    q.need(u.sha(original_before_path) == seed42_receipt["source_before_sha256"]
           == analysis["evidence"]["complete_bridge_bindings"]["source_before"]["sha256"], "original seed42 before frame missing")
    original_before = u.read(original_before_path)
    q.need(original_before["files"] == original_after["files"]
           and original_before["files_digest"] == original_after["files_digest"] == seed42_receipt["source_inventory_files_digest"]
           and q.libraries().utc(original_before["observed_at_utc"]) < q.libraries().utc(original_after["observed_at_utc"]),
           "original seed42 two exact complete inventories differ")
    original_files = {row["path"]: row for row in original_after["files"]}
    u.verify_tree(seed42, original_files)
    values, identities, selections = {}, {}, {}
    for seed in (42, 1, 2):
        for base in q.BASE_IDS:
            name = base + "_seed" + str(seed)
            if seed == 42:
                task = seed42 / "results/tasks" / base
                selected = analysis["selected_checkpoint_values"][base]
                expected = {"config_digest": selected["config_digest"], "checkpoint_sha256": {"best.pt": selected["checkpoint_sha256"]}}
            else:
                task = root / "results/tasks" / name
                expected = after["native_admission"]["native_units"][name]
            proof = u.read(task / "independent_verify/verification.json")
            config = u.read(task / "config.json")
            q.need(config.get("seed") == seed and config.get("precision") == "FP32"
                   and config.get("epochs") == 30 and config.get("batch_size") == 16
                   and config.get("microbatch_size") == (16 if base == q.BASE_IDS[0] else 2)
                   and config.get("test_evaluation") is False and config.get("target_data_read") is False
                   and config["source_snapshot"]["git"]["git_sha"] == q.SOURCE_GIT
                   and config["source_snapshot"]["git"]["git_dirty"] is False,
                   "family uses a different seed/source/precision/recipe")
            current = _metric(proof, expected, q.need)
            for region in SOURCES:
                leaf = proof["source_validation"]["regions"][region]
                collection = {"valid_count": leaf["valid_count"], "samples": leaf["selection"]["samples"],
                    "forecast_origins_sha256": leaf["selection"]["forecast_origins_sha256"],
                    "observed_targets_sha256": leaf["observed_targets_sha256"], "observation_masks_sha256": leaf["observation_masks_sha256"]}
                q.need(collection == analysis["validation_collections"][region], "family6 validation point collection differs from original complete gate")
            identity = {"data_identity": config["data_identity"], "source_statistics_identity": config["source_statistics_identity"], "runtime": config["runtime"]}
            identities[name] = q.canonical(identity)
            values[name] = current
            selections[name] = {"config_digest": expected["config_digest"], "checkpoint_sha256": expected["checkpoint_sha256"]["best.pt"],
                                "verification_sha256": u.sha(task / "independent_verify/verification.json")}
    q.need(len(set(identities.values())) == 1, "family6 data/statistics/runtime producer identities differ")
    improved = compare_family6(values)
    result = {"schema_version": FAMILY_SCHEMA, "status": "confirmed_each_seed_both_sources" if all(all(row.values()) for row in improved.values()) else "confirmation_failed_preserve_negative_result",
              "confirmation_preservation": {"path": str(Path(preservation_path).resolve()), "sha256": u.sha(preservation_path)},
              "source42_preservation_sha256": q.SEED42_SHA, "source42_native_proof_reused_without_Torch": True,
              "models": 6, "seeds": [42, 1, 2], "new_original_owned_wait0": 8, "metric": "global_masked_mae_all_observed_validation_targets",
              "values": values, "per_seed_both_region_strict": improved, "paired_seed_composites": first["paired_seed_composites"], "selections": selections, "common_identity_sha256": next(iter(identities.values())),
              "failed_region_offset_by_mean": False, "original8_NoGo_changed": False, "Orange_evaluated": False, "test_evaluated": False,
              "SOTA_established": False, "convergence_established": False, "automatic_successor_dispatch": False, "production_launchable": False}
    # Recheck exact opaque bytes after arithmetic before publishing the family.
    u.admit_copied(root, after, plan); u.verify_tree(seed42, original_files)
    u.write_new(output, result)
    return result


def _local_record(root, original, library):
    pins = library._plan_pins({"record": original})
    name = next(iter(pins))
    return {"path": str(root / name), "sha256": original["sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path); parser.add_argument("--expected-plan-sha")
    parser.add_argument("--family-preservation", type=Path); parser.add_argument("--seed42-root", type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.source_plan:
        value = source_inventory(args.source_plan, args.expected_plan_sha); print(json.dumps(value)); return value
    if args.family_preservation:
        if args.seed42_root is None or args.output is None: parser.error("family requires explicit seed42 root and new output")
        value = family6(args.family_preservation, seed42_root=args.seed42_root, output=args.output)
        print(json.dumps(value)); return value
    parser.error("collection requires explicit authorized callbacks; use the Python API")


if __name__ == "__main__": main()
