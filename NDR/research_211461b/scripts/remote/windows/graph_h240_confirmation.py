"""Selected pooled Graph H240 paired confirmation, with unchanged producer source43.

The pinned serial engine supplies original Handle waits and external owned-Job
supervision. This module supplies only the selected schedule, source admission,
and explicit seeds1/2 native schema adaptation. No plan writer is provided.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import uuid
from types import FunctionType

ENGINE_SHA = "a2e6eceb2c1d1db7696b8daae3abf177fecfdbe6d1f4ed8b075620ff40d920a1"
COLLECTOR_LIBRARY_SHA = "f2ad686f65ada30eb707b2da102c773208067060e45aaeaeb81f10825a5ae9c7"
PREDECESSOR_SHA = "a81b0db249192f8fa8accd0539f8fb2c74612701555371203f613af98561597a"
ADAPTER_SHA = "5f951b3d376f44ef0b78f21c28336470f68cda9ce0ec251750bbfb42ac30d585"
NATIVE_CORE_SHA = "6363d026700b6fe52a8fa035472cdc4c0b2c1ae8a3e1152550c0b3e74d12763c"
FORMAL_PLAN_SHA = "b4f11e80274b5b9463acfc3e945b3f68421e28d21d71df70a1e771b9f9c6f180"
PREPARATION_SHA = "53cdcac4e7986a9158b186a9d26baaee76d5149aad6e97922f3fe1b5ddbb0a12"
ANALYSIS_SHA = "62cd73ddfc3fcadf3dec9987b534ff07cb8d3b0d164e0ba7337ebc49044d67eb"
SEED42_SHA = "a6ad860c638967aade1d3876531ca1062c22aa17eb542ba38b5c8baec2b9469d"
POLICY_SHA = "2ccd8c9add31623bb78208b655fd22d3c3e73b5875ad941d105b22f2f4e875b5"
TEMPLATES_SHA = "71a8e5a97fb6dc8509d9fd38afe075dafbea471c7112ff82fa7d3534e0c2a4e3"
SOURCE43_FILES_DIGEST = "0f4a980b701328cfb364fd4dd2b5451b94de6b5cc37c748dc959ca34a02ae3c2"
ORIGINAL_ROOT = "C:/XTrafficResearch/ndr_formal_20261015/development_complete30_20261015T050849Z_6b9501f945"
SOURCE_GIT = "3b09dbf199372b356889a21ae9cfa8a95de37ff6"
SOURCE_ORIGIN = "97440a52708f2c3f03169ed4a385080c2129ea77"
PLAN_SCHEMA = "graph_h240_confirmation_serial_plan_v1"
STAGE = "pooled_h240_seeds1_2"
STATE_SCHEMA = "graph_h240_confirmation_serial_state_v1"
POST_SCHEMA = "graph_h240_confirmation_postflight_v1"
NATIVE_SCHEMA = "graph_h240_confirmation_native_v1"
INVENTORY_SCHEMA = "graph_h240_confirmation_source_inventory_v1"
SCOPE = "selected_pooled_graph_h240_confirmation"
CONFIGS = [(f"{name}_seed{seed}", kind, None, hidden, seed)
           for seed in (1, 2) for name, kind, hidden in
           (("pooled_graph_h240", "pooled_graph", 240), ("pooled_shared_sta", "pooled_sta", None))]
STAGES = {STAGE: tuple((name + suffix, "GPU_verify" if suffix else "GPU_train")
                      for name, *_ in CONFIGS for suffix in ("", "_verify"))}
BASE_IDS = ("pooled_graph_h240", "pooled_shared_sta")
OUTPUTS = {"status": "status.json", "journal": "process_journal.jsonl", "postflight": "postflight.json",
           "source_native_receipt": "source_native_admission.json"}
_ENGINE_LIBRARY_PATH = Path(__file__).with_name("shared_sta_norm_serial_queue.py")
if os.name != "nt" and not _ENGINE_LIBRARY_PATH.exists():
    _ENGINE_LIBRARY_PATH = Path(__file__).resolve().parents[3] / "outputs/confirmation_cpu_dependencies/serial_engine.py"


def need(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON key"); result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and all(not p.is_symlink() for p in path.parents), "real immutable regular file required")
    before = path.stat(); result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8388608), b""): result.update(block)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), "file changed while hashing")
    return result.hexdigest()


def pin(record, expected=None, bound=None):
    need(type(record) is dict and set(record) == {"path", "sha256"} and re.fullmatch("[0-9a-f]{64}", record["sha256"])
         and (expected is None or record["sha256"] == expected), "exact file record required")
    path = Path(record["path"])
    need(path.is_absolute() and sha(path) == record["sha256"], "immutable file differs")
    if bound is not None: bound[str(path.resolve())] = record["sha256"]
    return path


def load(record, expected=None, bound=None):
    path = pin(record, expected, bound)
    spec = importlib.util.spec_from_file_location("_graph_confirmation_" + uuid.uuid4().hex, path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    old = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try: spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode = old
    return module


def authority(plan):
    """Authority is byte-pinned full discovery, never partial rankings."""
    values = {}
    for key, digest in (("formal7_plan", FORMAL_PLAN_SHA), ("source_preparation", PREPARATION_SHA),
                        ("discovery_analysis", ANALYSIS_SHA), ("seed42_preservation", SEED42_SHA), ("discovery_policy", POLICY_SHA)):
        values[key] = read(pin(plan[key], digest))
    old, prep, analysis = values["formal7_plan"], values["source_preparation"], values["discovery_analysis"]
    selected = analysis["tracks"]["source_only_transfer_readiness"]
    need(selected["selected_width_for_confirmation"] == 240 and selected["eligible_for_seed1_seed2_confirmation"] is True
         and selected["confirmation_seeds"] == [1, 2]
         and analysis["tracks"]["regional_accuracy"]["selected_width_for_confirmation"] is None
         and all(analysis[key] is False for key in ("previous_eight_NoGo_changed", "heldout_Orange_evaluated", "test_evaluated", "SOTA_success_established")),
         "only selected pooled H240 route is eligible")
    need(values["seed42_preservation"].get("status") == "complete30_verify_only_composite_admitted"
         and values["seed42_preservation"].get("source_native_checkpoint_schema_admission_bound_to_copied_bytes") is True,
         "original complete seed42 native preservation required")
    need(prep.get("source_remote_git_sha") == SOURCE_GIT and prep.get("source_origin_git_sha") == SOURCE_ORIGIN
         and prep.get("source_remote_git_dirty") is False and len(prep.get("source_files", {})) == 43
         and prep["source_files"] == plan["shared_producer_files"], "unchanged actual source43 required")
    need(all(plan[key] == old[key] for key in ("python", "repo", "data_dir", "graph_dir")), "producer runtime paths must equal original b4f plan")
    return values


def expected_jobs(plan, prepared_sha=None):
    """Clone only the two original commands; change seed and output paths."""
    jobs = plan["original_job_templates"]
    need(canonical(jobs) == TEMPLATES_SHA, "original b4f four command templates differ")
    result = []
    for seed in (1, 2):
        for base in BASE_IDS:
            original = ORIGINAL_ROOT + "/tasks/" + base
            target = plan["run_root"] + "/tasks/" + base + "_seed" + str(seed)
            for suffix in ("", "_verify"):
                command = list(jobs[base + suffix])
                for index, value in enumerate(command):
                    if value == original or value.startswith(original + "/"): command[index] = target + value[len(original):]
                if not suffix:
                    need(command.count("--seed") == 1, "original seed flag differs")
                    command[command.index("--seed") + 1] = str(seed)
                result.append(command)
    return result


def validate_plan(plan):
    need(plan.get("schema_version") == PLAN_SCHEMA and plan.get("stage") == STAGE and plan.get("budget_frozen") is True
         and plan.get("plan_digest") == canonical({k: v for k, v in plan.items() if k != "plan_digest"}), "explicit frozen confirmation plan required; drafts authorize nothing")
    finite = lambda value: type(value) in {int, float} and math.isfinite(value) and value > 0
    need(finite(plan.get("total_budget_seconds")), "explicit total budget required; code supplies none")
    for key in ("absolute_deadline_utc", "current_boot_utc"):
        instant = datetime.fromisoformat(plan[key].replace("Z", "+00:00"))
        need(instant.utcoffset() is not None and instant.utcoffset().total_seconds() == 0, "explicit UTC required")
    wanted = STAGES[STAGE]
    need(type(plan.get("jobs")) is list and len(plan["jobs"]) == 8 and all(set(row) == {"job_id", "timeout_seconds"}
         and row["job_id"] == job[0] and finite(row["timeout_seconds"]) and row["timeout_seconds"] <= plan["total_budget_seconds"]
         for row, job in zip(plan["jobs"], wanted)), "eight fixed ordered jobs and explicit timeouts required")
    need(all(plan.get(key) is False for key in ("automatic_retry", "automatic_resume", "automatic_successor_dispatch", "automatic_shutdown",
         "automatic_transfer", "test_evaluated", "Orange_evaluated")), "no retries, heldout access or successor dispatch")
    need("commands" not in plan and "tasks" not in plan and plan.get("minimum_free_gpu_mib", 0) >= 14000, "arbitrary commands or weaker ownership threshold forbidden")
    need(set(plan.get("predecessors", {})) == {"whole7_verify_only", "dualcast_numerical", "dualcast_complete_cost", "strong4_fixed30"}, "all four actual predecessor closures required")
    for rows in plan["predecessors"].values():
        need(type(rows) is dict and rows, "typed original raw predecessor records required")
        for row in rows.values():
            need(type(row) is dict and set(row) == {"path", "sha256"} and type(row["path"]) is str
                 and re.fullmatch("[0-9a-f]{64}", row.get("sha256", "")), "predecessor records cannot be status or admission booleans")
    for key, digest in (("serial_engine", ENGINE_SHA), ("collector_library", COLLECTOR_LIBRARY_SHA),
                        ("predecessor_helper", PREDECESSOR_SHA), ("successor_adapter", ADAPTER_SHA), ("native_core", NATIVE_CORE_SHA)):
        need(plan[key].get("sha256") == digest, "reviewed library bytes differ: " + key)
    for key in ("controller", "collector", "serial_engine", "collector_library", "predecessor_helper", "successor_adapter", "native_core",
                "process_helper", "windows_supervisor_primitives", "gpu_provider", "gui_implementation", "gui_policy"):
        row = plan[key]; need(plan["control_files"].get(row["path"]) == row["sha256"], "critical immutable closure omitted: " + key)
    source = plan.get("source_identity", {})
    need(set(source) == {"git_sha", "git_tree", "git_dirty"} and source["git_sha"] == SOURCE_GIT and source["git_dirty"] is False
         and re.fullmatch("[0-9a-f]{40}", source.get("git_tree", "")), "exact clean source43 Git required")
    for key in ("python", "repo", "data_dir", "graph_dir", "run_root", "supervision_root"):
        need(type(plan.get(key)) is str and (PureWindowsPath(plan[key]).is_absolute() or plan.get("synthetic_only") is True and Path(plan[key]).is_absolute()), "absolute producer paths required")
    need(plan.get("outputs") == {key: plan["run_root"].rstrip("/") + "/" + name for key, name in OUTPUTS.items()}, "fixed output paths differ")
    need(canonical(plan.get("original_job_templates")) == TEMPLATES_SHA
         and canonical(plan.get("shared_producer_files")) == SOURCE43_FILES_DIGEST, "original b4f commands or exact43 file set differs")
    for key, digest in (("formal7_plan", FORMAL_PLAN_SHA), ("source_preparation", PREPARATION_SHA), ("discovery_analysis", ANALYSIS_SHA),
                        ("seed42_preservation", SEED42_SHA), ("discovery_policy", POLICY_SHA)):
        need(plan[key].get("sha256") == digest, "frozen discovery authority differs")
    primitive_pins = {"process_helper": "808abd383d6d40fb5dd5f8b6aa02af9d39f49f3ac1e8b41967d4baab78dd1d60",
        "windows_supervisor_primitives": "5d78666a4053b03231c6a96475021209f55de574fe52704075d808a604d7c7c0",
        "gpu_provider": "23c999a6e4321da1d3298d6eb9e9682798e112648b5e2f6751a3850a9e052015",
        "gui_implementation": "d59ec43e58fe0b9575ebe8ee9fa8d454aa100cc2931ebe8bd1375b8279dc24ba",
        "gui_policy": "d9dcd0c20aadb6cdf39b90fcb9b8963f7251a4f9c7fc100364ceea27648463c9"}
    need(all(plan[key].get("sha256") == digest for key, digest in primitive_pins.items()), "strict OS/ownership primitive authority differs")
    return plan


def completed_predecessor_functions(helper):
    """Private function globals; preserve the loaded module and function code."""
    need(helper.SUCCESSOR_SHA == "eec22a844575b95bb3d8874e95061fa207b531f560845c4ee3372001400a196c", "original predecessor helper authority differs")
    namespace = helper.predecessor_admission.__globals__.copy()
    namespace["SUCCESSOR_SHA"] = ADAPTER_SHA
    def clone(function):
        result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
        result.__kwdefaults__ = function.__kwdefaults__
        return result
    completed = clone(helper.strong4_completed_admission)
    predecessor = clone(helper.predecessor_admission)
    namespace["strong4_completed_admission"] = completed
    namespace["predecessor_admission"] = predecessor
    return predecessor, completed


def predecessor_admission(plan, bound, remaining_seconds):
    helper = load(plan["predecessor_helper"], PREDECESSOR_SHA, bound)
    pin(plan["successor_adapter"], ADAPTER_SHA, bound)
    need(plan["predecessors"]["dualcast_complete_cost"]["admission_helper"] == plan["successor_adapter"], "cost chain must use actual v5 adapter")
    # Explicit compatibility: retain the exact native pure predecessor function,
    # substituting only its byte-pinned successor adapter authority.
    predecessor, _completed = completed_predecessor_functions(helper)
    return predecessor(plan, bound, helper.Budget(min(remaining_seconds, 1800)))


def admit(plan, remaining_seconds):
    validate_plan(plan); bound = {}
    for path, digest in plan["control_files"].items(): pin({"path": path, "sha256": digest}, bound=bound)
    values = authority(plan)
    for key in values: pin(plan[key], bound=bound)
    for relative, digest in plan["shared_producer_files"].items():
        need(not Path(relative).is_absolute() and ":" not in relative and "\\" not in relative and ".." not in Path(relative).parts, "unsafe source relative path")
        pin({"path": str(Path(plan["repo"]) / relative), "sha256": digest}, bound=bound)
    engine = libraries()
    need(pin(plan["serial_engine"]).resolve() == _ENGINE_LIBRARY_PATH.resolve(), "actual serial engine path differs")
    need(engine._git(plan["repo"]) == plan["source_identity"], "actual source43 changed")
    predecessor = predecessor_admission(plan, bound, remaining_seconds)
    need(predecessor["admissions"]["strong4_fixed30"].get("status") == "admitted_complete_source_native_strong4_fixed30", "actual completed strong4 native admission missing")
    manifest_path = Path(plan["data_dir"]) / "cache_bundle_manifest.json"
    need(str(manifest_path) in plan["control_files"], "cache manifest pin required")
    manifest = read(manifest_path)
    need((manifest.get("status"), manifest.get("protocol_version"), manifest.get("cache_schema_version")) == ("verified", "protocol_v3_noleak_dedup", 3)
         and manifest.get("cache_bundle_digest") == "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5", "schema3 source cache differs")
    for region, graph in (("Alameda", "alameda"), ("ContraCosta", "contra_costa")):
        hashes = manifest["regions"][region]["cache_provenance"]["artifact_sha256"]
        for role, path in {"traffic": Path(plan["data_dir"]) / (region + "_traffic.h5"), "samples": Path(plan["data_dir"]) / (region + "_samples.h5"),
                           "relation": Path(plan["data_dir"]) / (region + "_event_rel_feat.npz"), "graph": Path(plan["graph_dir"]) / (graph + "_sparse_adj.npz")}.items():
            pin({"path": str(path), "sha256": hashes[role]}, bound=bound)
    return {"bound_files": bound, "protected_roots": predecessor["protected_roots"],
            "predecessor_admissions": predecessor["admissions"], "cost_prerequisite_admission": None}


def preliminary(plan, root, job_id):
    name = job_id.removesuffix("_verify"); prefix = "tasks/" + name + "/"
    names = ["independent_verify/verification.json"] if job_id.endswith("_verify") else ["best.pt", "last.pt", "best.pt.sha256.json", "last.pt.sha256.json", "summary.json"]
    return {"scope": "pre_native_artifact_identity_only", "artifacts": {prefix + name: sha(root / (prefix + name)) for name in names}}


def native_units_validate(receipt, plan, files=None):
    need(receipt.get("schema_version") == NATIVE_SCHEMA and receipt.get("stage") == STAGE and receipt.get("plan_digest") == plan["plan_digest"]
         and receipt.get("source_identity") == plan["source_identity"] and receipt.get("shared_producer_files") == plan["shared_producer_files"]
         and receipt.get("metric_values_extracted_for_selection") is False, "source native receipt identity differs")
    units = receipt.get("native_units")
    need(type(units) is dict and set(units) == {row[0] for row in CONFIGS}, "four native units required")
    for name, kind, _, hidden, seed in CONFIGS:
        unit = units[name]
        need(type(unit.get("seed")) is int and unit["seed"] == seed and unit.get("kind") == kind and unit.get("completed_epochs") == 30
             and unit.get("native_checkpoint_schemas_loaded_on_cpu") is True and unit.get("inference_repeated_by_controller") is False
             and unit.get("test_evaluated") is False and unit.get("producer_emits_formal_pid_fields") is False
             and type(unit.get("constructor")) is dict and unit["constructor"]
             and type(unit.get("config_digest")) is str and re.fullmatch("[0-9a-f]{64}", unit["config_digest"]), "native unit schema or seed differs")
        need(type(unit.get("checkpoint_sha256")) is dict and set(unit["checkpoint_sha256"]) == {"best.pt", "last.pt"}, "both native checkpoint identities required")
        artifacts = unit.get("artifacts", {})
        required = {"best": "best.pt", "last": "last.pt", "verification": "independent_verify/verification.json", "summary": "summary.json",
                    "best_marker": "best.pt.sha256.json", "last_marker": "last.pt.sha256.json", "config": "config.json", "history": "training_diagnostics.jsonl",
                    "data_identity": "data_identity.json", "snapshot": "source_snapshot.json"}
        need(set(artifacts) == set(required), "native artifact roles differ")
        need(all(artifacts[role].get("sha256") == unit["checkpoint_sha256"][suffix] for role, suffix in (("best", "best.pt"), ("last", "last.pt"))), "checkpoint native role digest mismatch")
        for role, suffix in required.items():
            row = artifacts[role]; need(row.get("path") == "tasks/" + name + "/" + suffix and re.fullmatch("[0-9a-f]{64}", row.get("sha256", "")), "native artifact paths differ")
            if files is not None:
                path = "results/" + row["path"]
                need(path in files and files[path]["sha256"] == row["sha256"], "native artifact is not bound to copied postflight bytes")
    return units


def terminal_validate(raw_validator, plan, status, journal, postflight, native_receipt=None):
    # Reuse every original raw attempt/Handle/wait check. The raw-only projection
    # deliberately cannot claim success; completed/native facts are checked below.
    need(status.get("status") in {"running", "failed", "completed"}, "unknown terminal control state")
    projection = {**status, "status": "failed"}
    raw_validator(plan, projection, journal, None, None)
    success = status.get("status") == "completed"
    if success:
        ids = [row[0] for row in STAGES[STAGE]]
        need(len(status["jobs"]) == len(status["attempts"]) == 8 and status.get("completed") == ids
             and status.get("worker_pid") is None and status.get("current_job") is None
             and type(postflight) is dict and postflight.get("schema_version") == POST_SCHEMA and postflight.get("status") == "completed"
             and postflight.get("stage") == STAGE and postflight.get("plan_digest") == plan["plan_digest"]
             and postflight.get("accepted_job_ids") == ids and postflight.get("owned_wait_rows_sha256") == canonical(status["jobs"]), "complete 8 original waits and postflight required")
        need(type(native_receipt) is dict, "complete source native receipt required")
        native_units_validate(native_receipt, plan)
        need(native_receipt.get("accepted_job_ids") == ids and native_receipt.get("owned_wait_rows_sha256") == canonical(status["jobs"])
             and native_receipt.get("predecessor_admissions") == status.get("predecessor_admissions")
             and type(status.get("predecessor_admissions")) is dict
             and set(status["predecessor_admissions"]) == set(plan["predecessors"]), "native receipt must bind exact waits and actual predecessors")
        prior = status["predecessor_admissions"]
        need(prior["strong4_fixed30"].get("status") == "admitted_complete_source_native_strong4_fixed30"
             and prior["strong4_fixed30"].get("pairs") == 4 and prior["strong4_fixed30"].get("native_owned_Popen_rows") == 8
             and prior["strong4_fixed30"].get("original_held_wait_proof_synthesized") is False
             and prior["dualcast_complete_cost"].get("status") == "admitted_preserved_dualcast_memory_cost_terminal",
             "complete raw predecessor native closures cannot be replaced by status booleans")
        for index, (name, *_rest) in enumerate(CONFIGS):
            unit, train, verify = native_receipt["native_units"][name], status["jobs"][2 * index], status["jobs"][2 * index + 1]
            need(unit.get("training_pid") == train["pid"] and unit.get("verification_pid") == verify["pid"]
                 and unit.get("independence_evidence") == "distinct_owned_serial_popen_handles", "native source schema is not bound to actual original train/verify instances")
            captured = train["native_admission"]["artifacts"]
            need(all(captured.get(unit["artifacts"][role]["path"]) == unit["artifacts"][role]["sha256"]
                     for role in ("best", "last", "best_marker", "last_marker", "summary"))
                 and verify["native_admission"]["artifacts"].get(unit["artifacts"]["verification"]["path"])
                     == unit["artifacts"]["verification"]["sha256"], "native final bytes differ from original train/verify boundary artifact captures")
        rows = {row["path"]: row for row in postflight.get("files", [])}
        need("postflight.json" not in rows and rows.get("source_native_admission.json", {}).get("sha256") == status.get("source_native_receipt_sha256")
             == postflight.get("source_native_receipt_sha256"), "native receipt byte binding missing")
    elif postflight is not None:
        need(postflight.get("schema_version") == POST_SCHEMA and postflight.get("status") == "failed" and postflight.get("stage") == STAGE
             and postflight.get("plan_digest") == plan["plan_digest"], "failed postflight differs")
    composites = {str(seed): {"seed": seed, "owned_job_ids": [row[0] for row in STAGES[STAGE] if "_seed" + str(seed) in row[0]],
                  "native_unit_ids": [name for name, *_rest, item_seed in CONFIGS if item_seed == seed],
                  "original_wait_rows_sha256": canonical([row for row in status["jobs"] if "_seed" + str(seed) in row["job_id"]])}
                  for seed in (1, 2)} if success else None
    return {"paired_seed_composites": composites, "success": success, "incomplete_control_status": status["status"] == "running", "stage": STAGE,
            "accepted_job_ids": status.get("completed", []), "owned_wait_rows_sha256": canonical(status["jobs"]),
            "native_admission": native_receipt if success else None}


def libraries():
    engine = load({"path": str(_ENGINE_LIBRARY_PATH), "sha256": ENGINE_SHA}, ENGINE_SHA)
    raw_validator = engine.validate_terminal
    engine.__file__ = __file__
    for name, value in {"PLAN_SCHEMA": PLAN_SCHEMA, "STATE_SCHEMA": STATE_SCHEMA, "POST_SCHEMA": POST_SCHEMA, "NATIVE_SCHEMA": NATIVE_SCHEMA,
                        "INVENTORY_SCHEMA": INVENTORY_SCHEMA, "SCOPE": SCOPE, "STAGES": STAGES,
                        "ENV": "XTRAFFIC_GRAPH_H240_CONFIRMATION_SUPERVISION_V1", "validate_plan": validate_plan,
                        "expected_jobs": expected_jobs, "admit": admit, "_preliminary": preliminary, "source_native_validate": source_native_validate}.items():
        setattr(engine, name, value)
    engine._collector = lambda plan: load({"path": str(Path(__file__).resolve().parents[2] / "retrieve_graph_h240_confirmation.py"), "sha256": plan["collector"]["sha256"]})
    engine.validate_terminal = lambda *args, **kwargs: terminal_validate(raw_validator, *args, **kwargs)
    # Defaults are function objects, so substitute the two narrow extension hooks
    # explicitly without modifying the original code or model/validator literals.
    original_engine = engine.engine
    def execute(plan, **kwargs):
        kwargs.setdefault("admission", admit); kwargs.setdefault("native_validator", source_native_validate)
        return original_engine(plan, **kwargs)
    engine.engine = execute
    return engine


def source_native_validate(plan, *, code_root=None, results_root=None):
    sys.dont_write_bytecode = True
    validate_plan(plan)
    need(plan.get("synthetic_only") is not True, "synthetic engine cannot create production native admission")
    root = Path(plan["run_root"] if results_root is None else results_root)
    repo = Path(plan["repo"] if code_root is None else code_root)
    need(repo.resolve() == Path(plan["repo"]).resolve(), "native source repo cannot change")
    engine = libraries(); before = engine.inventory(root)
    need(engine._git(repo) == plan["source_identity"], "actual native source Git differs")
    for name, digest in plan["shared_producer_files"].items(): need(sha(repo / name) == digest, "native source43 bytes differ")
    need(pin(plan["controller"]).resolve() == Path(__file__).resolve(), "actual source controller bytes or location differ")
    native = load(plan["native_core"], NATIVE_CORE_SHA)
    # The original helper imports its standard-library schema helpers from its
    # own frozen package. All imported files must belong to control_files.
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file() and Path(path).resolve().is_relative_to(Path(native.CONTROL_ROOT).resolve()):
            need(plan["control_files"].get(str(Path(path).resolve())) == sha(path), "transitive native control import is not pinned")
    state = read(root / "status.json"); need(len(state["jobs"]) == 8 and state.get("status") == "running", "native admission only after all eight original waits")
    context = {"plan": plan, "root": root, "source_files": plan["shared_producer_files"], "native_core": native}
    units = {}
    for index, (name, kind, _, hidden, seed) in enumerate(CONFIGS):
        entry = "scripts/train_ndr_transfer.py" if kind == "pooled_graph" else "scripts/train_shared_staeformer_transfer.py"
        train, verify = state["jobs"][index * 2:index * 2 + 2]
        value = validate_native_pair(context, {"job_id": name, "kind": kind, "entrypoint": entry}, train["pid"], verify["pid"], checkpoint_loader=native.native_checkpoint)
        value.pop("selection_mae", None)
        names = {"best": "best.pt", "last": "last.pt", "verification": "independent_verify/verification.json", "summary": "summary.json", "best_marker": "best.pt.sha256.json",
                 "last_marker": "last.pt.sha256.json", "config": "config.json", "history": "training_diagnostics.jsonl", "data_identity": "data_identity.json", "snapshot": "source_snapshot.json"}
        value.update(seed=seed, kind=kind, constructor=read(root / "tasks" / name / "config.json")["model_config"], artifacts={role: {"path": "tasks/" + name + "/" + suffix, "sha256": sha(root / "tasks" / name / suffix)} for role, suffix in names.items()})
        units[name] = value
    need(engine.inventory(root) == before and engine._git(repo) == plan["source_identity"], "source artifacts changed during native admission")
    return {"schema_version": NATIVE_SCHEMA, "stage": STAGE, "plan_digest": plan["plan_digest"], "source_identity": plan["source_identity"],
            "shared_producer_files": plan["shared_producer_files"], "native_units": units, "metric_values_extracted_for_selection": False}


def validate_terminal(plan, status, journal, postflight, native_receipt=None):
    return libraries().validate_terminal(plan, status, journal, postflight, native_receipt)


def source_terminal_observations(plan, status):
    return libraries().source_terminal_observations(plan, status)


def validate_source_terminal_observations(plan, status, observations, **kwargs):
    return libraries().validate_source_terminal_observations(plan, status, observations, **kwargs)


def policy(plan):
    return libraries().policy(plan)


def readiness():
    return {"schema_version": "graph_h240_confirmation_engineering_readiness_v1", "production_launchable": False,
            "new_training_plan_created": False, "budget_frozen_here": False, "pending_real_predecessors": ["dualcast_complete_cost", "strong4_fixed30"],
            "job_order": [row[0] for row in STAGES[STAGE]], "scope": "selected_pooled_h240_only_no_Orange_or_test"}


def main(argv=None):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if argv == ["--readiness"] or argv is None and sys.argv[1:] == ["--readiness"]:
        print(json.dumps(readiness())); return readiness()
    args = sys.argv[1:] if argv is None else argv
    if "--dispatch" in args:
        parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--plan", type=Path)
        parsed, _ = parser.parse_known_args(args)
        need(parsed.plan is not None, "explicit source plan required")
        plan = validate_plan(read(parsed.plan))
        need(datetime.fromisoformat(plan["absolute_deadline_utc"].replace("Z", "+00:00")) > datetime.now(timezone.utc), "new dispatch deadline must be future")
    return libraries().main(argv)


def validate_native_pair(context, job, training_pid, verify_pid=None, *, checkpoint_loader=None):
    q = context["native_core"]
    read_json, canonical_digest, file_sha = q.read_json, q.canonical_digest, q.file_sha
    FORMAL_COUNTS, BUNDLE_DIGEST = q.FORMAL_COUNTS, q.BUNDLE_DIGEST
    _snapshot, _source_identity, _validation, _same_validation = q._snapshot, q._source_identity, q._validation, q._same_validation
    regional_validation = q.regional_validation
    need(callable(checkpoint_loader), "actual original native checkpoint loader required")
    directory = context["root"] / "tasks" / job["job_id"]
    config, summary = read_json(directory / "config.json"), read_json(directory / "summary.json")
    identity, snapshot = read_json(directory / "data_identity.json"), read_json(directory / "source_snapshot.json")
    history = [json.loads(line) for line in (directory / "training_diagnostics.jsonl").read_text().splitlines()]
    digest, pooled = canonical_digest(config), job["kind"] != "regional_graph"
    expected = CONFIGS[[row[0] for row in CONFIGS].index(job["job_id"])]
    _, kind, region, hidden, seed = expected
    need(type(seed) is int and seed in (1, 2), "confirmation adapter permits only selected seeds1/2")
    need(kind in {"pooled_graph", "pooled_sta"} and job["kind"] == kind
         and job["entrypoint"] == ("scripts/train_ndr_transfer.py" if kind == "pooled_graph" else "scripts/train_shared_staeformer_transfer.py"),
         "only selected original pooled producers are permitted")
    need(type(config.get("seed")) is int, "native confirmation seed must be an integer")
    fixed = {"epochs": 30, "seed": seed, "batch_size": 16, "microbatch_size": 2 if kind == "pooled_sta" else 16,
        "threads": 3, "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip": 5.0, "eta_min": 0.00001,
        "precision": "FP32", "max_train_batches": None, "max_val_batches": None, "early_stopping": False, "test_evaluation": False,
        "claim_scope": "source_validation_development_only" if pooled else "validation_development_only"}
    if hidden:
        fixed.update(hidden_dim=hidden, teacher="original", lambda_gate=1.0, lambda_specialization=0.3)
    if (any(config.get(key) != value for key, value in fixed.items()) or config.get("profile_closeout", False)
            or config["data_identity"] != identity or config["source_snapshot"] != snapshot
            or snapshot["git"].get("git_sha") != SOURCE_GIT or snapshot["git"].get("git_dirty") is not False
            or config["runtime"].get("device_type") != "cuda" or "5080" not in config["runtime"].get("device_name", "")
            or summary.get("completed_epochs") != 30 or summary.get("config_digest") != digest or summary.get("test_evaluated") is not False
            or summary.get("status") != ("training_completed_source_validation_pending_independent_verification" if pooled else "training_completed_validation_pending_independent_verification")
            or summary.get("claim_scope") != fixed["claim_scope"] or read_json(directory / "run_state.json") != summary
            or len(history) != 30 or [row["epoch"] for row in history] != list(range(1, 31))):
        raise ValueError("formal fit is incomplete, altered, smoke, non-CUDA or consumed test")
    _snapshot(config, {"source_files": context["source_files"]}, None)
    if job["entrypoint"] not in snapshot["files"]:
        raise ValueError("formal source snapshot omits its own frozen entrypoint")
    if hidden and config.get("model_config") != {"history_steps": 12, "horizon": 12, "hidden_dim": hidden,
            "road_layers": 2, "latent_tokens": 8, "num_heads": 4, "dropout": 0.0}:
        raise ValueError("formal Graph architecture differs from its fixed width and recipe")
    selected_key, history_key = ("best_source_validation", "source_validation") if pooled else ("best_validation", "validation")
    selected = summary[selected_key]
    if pooled:
        _source_identity(identity, config["source_statistics_identity"])
        if (config.get("source_regions") != ["Alameda", "ContraCosta"] or config.get("heldout_region") != "Orange"
                or config.get("target_data_read") is not False or config.get("complete_source_train_epoch") is not True):
            raise ValueError("formal pooled fit is not full source-only training")
        for row in history:
            _validation(row[history_key], identity)
            if any(row["train_regions"][name]["batches"] != math.ceil(FORMAL_COUNTS[name]["train"] / 16) for name in FORMAL_COUNTS):
                raise ValueError("formal pooled epoch omitted training batches")
        common = {key: config[key] for key in ("data_identity", "runtime", "source_statistics_identity")}
        if context.setdefault("pooled_common", common) != common:
            raise ValueError("pooled development fits used different source statistics/data/runtime")
    else:
        if (identity.get("region") != region or identity.get("cache_bundle_digest") != BUNDLE_DIGEST
                or identity.get("cache_schema_version") != 3 or identity.get("fit_range") != [0, identity["train_end"]]
                or identity.get("training_analysis_labels") is not False or identity.get("test_loader_constructed") is not False
                or identity.get("split_counts") != {key: FORMAL_COUNTS[region][key] for key in ("train", "val")}
                or config.get("training_process_id") != training_pid):
            raise ValueError("formal regional train-only data/config identity differs")
        for row in history:
            regional_validation(row[history_key], identity)
    def metric(value):
        return value["macro_global_mae" if pooled else "global_masked_mae"]
    best_epoch = min(range(30), key=lambda index: metric(history[index][history_key])) + 1
    if (summary.get("best_epoch") != best_epoch or selected != history[best_epoch - 1][history_key]
            or not math.isclose(summary["best_macro_source_val_mae" if pooled else "best_val_mae"], metric(selected), rel_tol=1e-12, abs_tol=1e-12)
            or summary.get("updates") != history[-1]["updates"]):
        raise ValueError("formal best selection or completed-update history differs")
    checkpoints = {}
    for name, epoch, selection in (("best.pt", best_epoch, selected), ("last.pt", 30, None)):
        path = directory / name
        marker = read_json(path.with_suffix(".pt.sha256.json"))
        if marker.get("sha256") != file_sha(path) or marker.get("size_bytes") != path.stat().st_size or marker.get("config_digest") != digest:
            raise ValueError("formal checkpoint sidecar differs")
        value = checkpoint_loader(context, job["entrypoint"], path)
        progress = value["progress"]
        if (value.get("checkpoint_role") is not None or value["config"] != config or value["data_identity"] != identity
                or value["config_digest"] != digest or value["selection_metric"] != selection
                or value.get("resume_boundary") != "completed_effective_batch_or_epoch"
                or progress.get("epoch") != epoch or progress.get("stage") != "start" or progress.get("history") != history[:epoch]
                or progress.get("updates") != history[epoch - 1]["updates"] or not value["optimizer"].get("state")
                or not value["optimizer"].get("param_groups") or set(value["rng"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}
                or set(value["loaders"]) != {"train", "val"}
                or value["scheduler"].get("last_epoch") != progress["updates"]
                or value["scheduler"].get("T_max") != 30 * sum(math.ceil(FORMAL_COUNTS[name]["train"] / 16) for name in (FORMAL_COUNTS if pooled else [region]))):
            raise ValueError("formal native checkpoint lacks complete epoch/state/history identity")
        for split in ("train", "val"):
            if pooled:
                if set(value["loaders"][split]) != set(FORMAL_COUNTS) or progress.get("plan") != [] or progress.get("plan_cursor") != 0:
                    raise ValueError("formal pooled checkpoint has an unfinished region plan")
                if any(state.get("cursor") != FORMAL_COUNTS[name][split] for name, state in value["loaders"][split].items()):
                    raise ValueError("formal pooled checkpoint omitted split samples")
            elif value["loaders"][split].get("next_batch") != math.ceil(FORMAL_COUNTS[region][split] / 16):
                raise ValueError("formal regional checkpoint omitted split batches")
        checkpoints[name] = marker["sha256"]
    result = {"config_digest": digest, "checkpoint_sha256": checkpoints, "completed_epochs": 30, "test_evaluated": False,
              "native_checkpoint_schemas_loaded_on_cpu": True, "inference_repeated_by_controller": False,
              "claim_scope": fixed["claim_scope"], "selection_mae": metric(selected)}
    if verify_pid is not None:
        verified = read_json(directory / "independent_verify/verification.json")
        if (verify_pid == training_pid or verified.get("status") != ("verified_transfer_source_checkpoint_complete_validation" if pooled else "verified_graph_indomain_checkpoint_complete_validation")
                or verified.get("config_digest") != digest or verified.get("checkpoint_sha256") != checkpoints["best.pt"]
                or verified.get("claim_scope") != fixed["claim_scope"] or verified.get("test_evaluated") is not False
                or verified.get("source_files_digest") != snapshot["files_digest"]):
            raise ValueError("formal independent verification receipt differs")
        if pooled:
            _same_validation(selected, verified["source_validation"], identity)
            if verified.get("target_data_read") is not False or verified.get("source_statistics_identity") != config["source_statistics_identity"]:
                raise ValueError("formal pooled verifier used different source statistics")
        else:
            actual = verified["validation"]
            regional_validation(actual, identity)
            if (verified.get("training_process_id") != training_pid or verified.get("verification_process_id") != verify_pid
                    or any(selected[key] != actual[key] for key in ("selection", "valid_count", "observed_targets_sha256", "observation_masks_sha256"))
                    or not math.isclose(metric(selected), metric(actual), rel_tol=2e-6, abs_tol=2e-5)):
                raise ValueError("formal regional verifier observed another validation collection")
        result.update(training_pid=training_pid, verification_pid=verify_pid, independence_evidence="distinct_owned_serial_popen_handles",
                      producer_emits_formal_pid_fields=not pooled)
    return result



if __name__ == "__main__":
    value = main()
    if type(value) is int: raise SystemExit(value)
