"""Continue the frozen queue only after its fourth train finished without verification.

The failed source tree and an exact remote backup must be preserved, with a
separately SHA-bound receipt of the complete local copy. The fourth task is only
verified. The four remaining tasks retain their original commands and deadline.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
import uuid


PROVIDER_SHA256 = "23c999a6e4321da1d3298d6eb9e9682798e112648b5e2f6751a3850a9e052015"
RETRIEVAL_SHA256 = "6902df283b3e6d4c7034cca952f5523f03271bbf7924964f99f8da1dbbc46664"
ANALYZER_SHA256 = "c9524fea14202d8b88bb05d22fd99fd32752ab9bc304a401fb06cdd2152dec8f"
EXPECTED_PLAN = "ad530e29fdcf2bd0cdabdd739349814e805f12e11cfab6341e9180651807672b"
EXPECTED_DEADLINE = "2026-10-15T08:10:22.442895+00:00"
TASK_IDS = (
    "alameda_baseline_seed42",
    "contracosta_baseline_seed42",
    "alameda_mask_signed_seed42",
    "contracosta_mask_signed_seed42",
    "alameda_sanitized_control_seed42",
    "contracosta_sanitized_control_seed42",
    "alameda_projected_oracle_seed42",
    "contracosta_projected_oracle_seed42",
)
INCIDENT_SCHEMA = "ndr_stage1_continuation_incident_v1"
PRESERVATION_SCHEMA = "ndr_stage1_continuation_preservation_v1"
LOCAL_PRESERVATION_SCHEMA = "ndr_stage1_continuation_local_preservation_v1"
BUSY_ERROR = "the GPU has a competing or unverified owner; none will be stopped"
TRAINED_STATUS = "training_completed_validation_pending_independent_verification"


@dataclass(frozen=True)
class Policy:
    """Fixture overrides have no executable CLI authority."""

    plan_digest: str = EXPECTED_PLAN
    deadline_utc: str = EXPECTED_DEADLINE
    task_ids: tuple[str, ...] = TASK_IDS
    job_seconds: float = 43200.0


def require_provider(path: Path) -> None:
    import hashlib

    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != PROVIDER_SHA256
    ):
        raise ValueError("pinned continuation GPU/helper provider bytes differ")


def load_provider(path: Path) -> Any:
    require_provider(path)
    name = "_pinned_stage1_continuation_provider_" + uuid.uuid4().hex
    specification = importlib.util.spec_from_file_location(name, path)
    provider = importlib.util.module_from_spec(specification)
    sys.modules[name] = provider
    try:
        specification.loader.exec_module(provider)
        require_provider(path)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return provider


def require_dead_handles(probes: Any, incident: dict) -> None:
    # Any live occupant of a recorded PID is rejected, including a reused PID.
    # No recovery code obtains a termination handle for one of these PIDs.
    for process in incident["original_processes"]:
        if process["alive_at_snapshot"] is not False or probes.live(process["pid"]):
            raise PermissionError(
                "an original PID is live or reused; none will be stopped"
            )


def preserved_inventory(provider: Any, manifest: dict, task: dict, plan: dict) -> dict:
    prefix = "tasks/" + task["task_id"] + "/"
    return {
        "task_id": task["task_id"],
        "plan_digest": plan["plan_digest"],
        "files": [
            {**row, "path": provider.safe_relative(row["path"])[len(prefix) :]}
            for row in manifest["files"]
            if provider.safe_relative(row["path"]).startswith(prefix)
        ],
    }


def require_preserved_tasks_unchanged(
    plan: dict, manifest: dict, provider: Any, count: int = 3
) -> None:
    root = Path(plan["run_root"])
    for task in plan["tasks"][:count]:
        directory = Path(task["output_dir"])
        rows = preserved_inventory(provider, manifest, task, plan)["files"]
        expected = {row["path"]: row for row in rows}
        actual = set()
        for path in directory.rglob("*"):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("an immutable task acquired a symlink or escaped path")
            if path.is_file():
                actual.add(path.relative_to(directory).as_posix())
        allowed_new = (
            {"verification.json"} if count == 4 and task == plan["tasks"][3] else set()
        )
        if actual - allowed_new != expected.keys():
            raise ValueError("an immutable task changed its preserved file set")
        for name, row in expected.items():
            path = directory / name
            if (
                path.stat().st_size != row["bytes"]
                or provider.file_sha(path) != row["sha256"]
            ):
                raise ValueError("an immutable task changed preserved bytes")


def audit_trained_fourth(
    directory: Path, plan: dict, task: dict, provider: Any
) -> dict:
    """CPU checkpoint reconstruction only; the original verifier checks live cache/runtime."""
    import torch

    provider.setup_paths(plan["repo"])
    from scripts.train_ndr_research import load_checkpoint
    from ndr_research.models import make_research_model

    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    identity = json.loads((directory / "data_identity.json").read_text())
    config_digest = provider.digest(config)
    if (
        any(config[key] != task[key] for key in ("region", "variant", "seed", "epochs"))
        or config["claim_scope"] != "validation_development_only"
        or config["max_train_batches"] is not None
        or config["max_val_batches"] is not None
        or config["test_evaluation"]
        or config["early_stopping"]
        or config["precision"] != "FP32"
        or config["data_identity"] != identity
        or identity["cache_bundle_digest"] != provider.BUNDLE
        or summary["config_digest"] != config_digest
        or summary["completed_epochs"] != 30
        or summary["status"] != TRAINED_STATUS
        or summary["test_evaluated"] is not False
        or config["runtime"]["device_type"] != "cuda"
    ):
        raise ValueError("the fourth task is not original complete no-test training")
    source = config["source_snapshot"]
    if provider.digest(source["files"]) != source["files_digest"] or any(
        row["sha256"] != plan["source_files"].get(name)
        for name, row in source["files"].items()
    ):
        raise ValueError("the fourth checkpoint source differs from the frozen plan")
    # This incident cannot introduce a different optimizer/runtime cohort.
    matched_fields = (
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
        "runtime",
    )
    for previous in plan["tasks"][:3]:
        prior = json.loads((Path(previous["output_dir"]) / "config.json").read_text())
        if any(config[name] != prior[name] for name in matched_fields):
            raise ValueError(
                "the fourth task changed the frozen runtime or optimizer cohort"
            )
    selected = summary["best_validation"]
    if (
        selected["selection"]["all_validation_samples"] is not True
        or selected["selection"]["samples"] != identity["split_counts"]["val"]
        or selected["valid_count"] <= 0
        or not math.isfinite(selected["global_masked_mae"])
        or not math.isclose(
            selected["absolute_error_sum"] / selected["valid_count"],
            selected["global_masked_mae"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or summary["best_val_mae"] != selected["global_masked_mae"]
    ):
        raise ValueError("the fourth task has no complete observed-point selection")
    hashes = {}
    batches = math.ceil(identity["split_counts"]["train"] / config["batch_size"])
    for name, epoch in (("best.pt", summary["best_epoch"]), ("last.pt", 30)):
        checkpoint = load_checkpoint(directory / name)
        if (
            checkpoint["config_digest"] != config_digest
            or checkpoint["data_identity"] != identity
        ):
            raise ValueError(
                "the fourth checkpoint configuration or data identity differs"
            )
        progress = checkpoint["progress"]
        if (
            progress["epoch"] != epoch
            or len(progress["history"]) != epoch
            or progress["stage"] != "start"
            or checkpoint["scheduler"]["T_max"] != 30 * batches
            or checkpoint["scheduler"]["last_epoch"] != progress["updates"]
            or not checkpoint["optimizer"]["state"]
            or not checkpoint["optimizer"]["param_groups"]
        ):
            raise ValueError(
                "the fourth checkpoint does not represent completed original training"
            )
        if name == "best.pt" and checkpoint["selection_metric"] != selected:
            raise ValueError(
                "the fourth best checkpoint selection differs from its summary"
            )
        model = make_research_model(**checkpoint["model_factory"], device="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        if any(
            not torch.isfinite(value).all()
            for value in model.state_dict().values()
            if value.is_floating_point()
        ):
            raise ValueError("the fourth checkpoint model parameters are nonfinite")
        hashes[name] = provider.file_sha(directory / name)
    return {
        "config_digest": config_digest,
        "best_checkpoint_sha256": hashes["best.pt"],
        "last_checkpoint_sha256": hashes["last.pt"],
        "completed_epochs": 30,
        "runtime": config["runtime"],
        "source_files_digest": source["files_digest"],
        "full_validation_inference_repeated": False,
        "test_evaluated": False,
    }


def admit_continuation(
    plan: dict,
    incident: dict,
    manifest: dict,
    local_receipt: dict,
    *,
    provider: Any,
    now: float,
    policy: Policy = Policy(),
    task_verifier: Callable | None = None,
    trained_auditor: Callable | None = None,
    probes: Any = None,
) -> dict:
    provider.validate_plan(
        plan,
        provider.Policy(
            policy.plan_digest, policy.deadline_utc, policy.task_ids, policy.job_seconds
        ),
    )
    if plan["busy_wait_minutes"] != 30:
        raise ValueError(
            "continuation must retain the original thirty-minute resource wait"
        )
    if now >= provider.utc_seconds(policy.deadline_utc):
        raise TimeoutError("the original absolute queue deadline has expired")
    queue = incident["queue_snapshot"]
    if (
        incident.get("schema_version") != INCIDENT_SCHEMA
        or incident.get("plan_digest") != policy.plan_digest
        or incident.get("absolute_deadline_utc") != policy.deadline_utc
        or incident.get("current_task_id") != policy.task_ids[3]
        or incident.get("failure_boundary") != "post_train_gpu_admission"
        or incident.get("fourth_verification_absent") is not True
        or queue.get("status") != "failed"
        or queue.get("plan_digest") != policy.plan_digest
        or queue.get("absolute_deadline_utc") != policy.deadline_utc
        or queue.get("completed") != list(policy.task_ids[:3])
        or queue.get("current_task") != policy.task_ids[3]
        or queue.get("phase") != "train"
        or queue.get("worker_pid") is not None
        or queue.get("worker_exit_code") != 0
        or queue.get("error") != BUSY_ERROR
        or queue.get("test_evaluated") is not False
        or queue.get("automatic_shutdown") is not False
        or queue.get("automatic_transfer") is not False
    ):
        raise PermissionError(
            "continuation only accepts the exact failed three-of-eight post-train incident"
        )
    processes = incident["original_processes"]
    by_role = {row["role"]: row for row in processes}
    if (
        len(by_role) != len(processes)
        or not {"supervisor", "train_worker"} <= by_role.keys()
        or by_role["supervisor"]["pid"] != queue["pid"]
        or by_role["train_worker"]["pid"] != incident["previous_train_worker_pid"]
        or any(type(row["pid"]) is not int or row["pid"] <= 0 for row in processes)
        or len({row["pid"] for row in processes}) != len(processes)
        or (
            queue.get("original_supervisor_pid") is not None
            and queue["original_supervisor_pid"]
            not in {row["pid"] for row in processes}
        )
    ):
        raise ValueError("incident does not bind all original recorded process handles")
    require_dead_handles(probes or provider.WindowsProbes(), incident)
    if manifest.get("schema_version") != PRESERVATION_SCHEMA:
        raise PermissionError("a complete new continuation preservation is required")
    rows = provider.check_preservation(
        plan, {**manifest, "schema_version": provider.PRESERVATION_SCHEMA}
    )
    if (
        local_receipt.get("schema_version") != LOCAL_PRESERVATION_SCHEMA
        or local_receipt.get("status") != "verified_exact_local_copy"
        or local_receipt.get("files") != manifest["files"]
        or local_receipt.get("inventory_sha256") != manifest["inventory_sha256"]
        or Path(local_receipt.get("source_root", "")).resolve()
        != Path(plan["run_root"]).resolve()
        or not local_receipt.get("local_copy_root")
    ):
        raise PermissionError(
            "the exact complete local preservation receipt is absent or different"
        )
    provider.utc_seconds(local_receipt["verified_at_utc"])
    old_queue = json.loads(
        (Path(manifest["backup_root"]) / "results/queue_status.json").read_text()
    )
    if old_queue != queue:
        raise ValueError("incident queue differs from the complete preserved failure")
    fourth = plan["tasks"][3]
    directory = Path(fourth["output_dir"])
    if (directory / "verification.json").exists():
        raise PermissionError("the fourth task already has independent verification")
    required = {
        "best.pt",
        "best.pt.sha256.json",
        "last.pt",
        "last.pt.sha256.json",
        "config.json",
        "data_identity.json",
        "summary.json",
        "run_state.json",
        "source_snapshot.json",
        "training_diagnostics.jsonl",
    }
    prefix = "tasks/" + fourth["task_id"] + "/"
    if not {prefix + name for name in required} <= rows.keys():
        raise PermissionError("complete fourth training artifacts are not preserved")
    summary = json.loads((directory / "summary.json").read_text())
    if (
        summary.get("completed_epochs") != 30
        or summary.get("status") != TRAINED_STATUS
        or summary.get("test_evaluated") is not False
        or provider.file_sha(directory / "summary.json")
        != incident["current_summary_sha256"]
    ):
        raise ValueError("the fourth task is incomplete or its summary changed")
    if any(Path(task["output_dir"]).exists() for task in plan["tasks"][4:]):
        raise FileExistsError(
            "a later frozen task already has outputs; fresh execution is forbidden"
        )
    verifier = task_verifier or provider.completed_verifier
    completed = [
        verifier(
            Path(task["output_dir"]),
            preserved_inventory(provider, manifest, task, plan),
            plan,
            task,
        )
        for task in plan["tasks"][:3]
    ]
    auditor = trained_auditor or audit_trained_fourth
    trained = auditor(directory, plan, fourth, provider)
    if (
        trained["completed_epochs"] != 30
        or trained["config_digest"] != incident["current_config_digest"]
        or trained["best_checkpoint_sha256"]
        != incident["current_best_checkpoint_sha256"]
        or trained["last_checkpoint_sha256"]
        != incident["current_last_checkpoint_sha256"]
    ):
        raise ValueError("incident does not bind the completed fourth checkpoints")
    require_preserved_tasks_unchanged(plan, manifest, provider)
    return {
        "completed": list(policy.task_ids[:3]),
        "current_task": fourth["task_id"],
        "trained_fourth": trained,
        "verified_completed_tasks": completed,
        "fourth_training_will_be_repeated": False,
    }


def require_helpers(context: dict) -> None:
    provider = context["provider"]
    require_provider(context["provider_path"])
    for path, expected in context["helper_files"].items():
        path = Path(path)
        if path.is_symlink() or provider.file_sha(path) != expected:
            raise ValueError("continuation admission/helper source changed")
    policy = context["policy"]
    provider.validate_plan(
        context["plan"],
        provider.Policy(
            policy.plan_digest, policy.deadline_utc, policy.task_ids, policy.job_seconds
        ),
    )
    require_dead_handles(context["probes"], context["incident"])
    require_preserved_tasks_unchanged(
        context["plan"], context["manifest"], provider, count=4
    )


def wait_for_idle(
    context: dict,
    *,
    now: Callable = time.time,
    sleep: Callable = time.sleep,
    probe_interval_seconds: float = 15.0,
    monotonic: Callable = time.monotonic,
) -> dict:
    provider, plan, state = context["provider"], context["plan"], context["state"]
    absolute = provider.utc_seconds(context["policy"].deadline_utc)
    started = now()
    started_monotonic = monotonic()
    wait_seconds = float(plan["busy_wait_minutes"]) * 60
    end = min(absolute, started + wait_seconds)
    consecutive = 0
    phase = state["phase"]

    class CapturingProbes:
        latest = None

        def live(self, pid):
            return context["probes"].live(pid)

        def gpu(self):
            self.latest = context["probes"].gpu()
            return self.latest

    captured = CapturingProbes()
    while True:
        if now() >= end or monotonic() - started_monotonic >= wait_seconds:
            raise TimeoutError(
                "GPU wait exhausted its original thirty-minute or absolute deadline"
            )
        require_helpers(context)
        ready = True
        try:
            gpu = provider.require_idle(captured, context["incident"], plan)
        except PermissionError as error:
            if str(error) != BUSY_ERROR or captured.latest is None:
                raise
            ready, gpu = False, captured.latest
        require_provider(context["provider_path"])
        if (
            "5080" not in gpu["name"]
            or any(
                not math.isfinite(float(gpu[key])) or float(gpu[key]) < 0
                for key in ("free_mib", "utilization")
            )
            or float(gpu["utilization"]) > 100
        ):
            raise ValueError("the authorized GPU probe is absent or malformed")
        if now() >= end or monotonic() - started_monotonic >= wait_seconds:
            raise TimeoutError(
                "GPU probe completed after its retained admission deadline"
            )
        consecutive = consecutive + 1 if ready else 0
        state.update(
            phase="resource_wait_" + phase,
            gpu=gpu,
            consecutive_idle_probes=consecutive,
            resource_wait_started_at_utc=datetime.fromtimestamp(
                started, timezone.utc
            ).isoformat(),
            resource_wait_deadline_utc=datetime.fromtimestamp(
                end, timezone.utc
            ).isoformat(),
        )
        provider.atomic_json(context["status_path"], state)
        with (context["control"] / "gpu_admission.jsonl").open(
            "a", encoding="utf8"
        ) as stream:
            json.dump(
                {
                    "checked_at_utc": datetime.fromtimestamp(
                        now(), timezone.utc
                    ).isoformat(),
                    "task_id": state["current_task"],
                    "intended_phase": phase,
                    "ready": ready,
                    "consecutive_idle_probes": consecutive,
                    "gpu": gpu,
                },
                stream,
                allow_nan=False,
            )
            stream.write("\n")
        if consecutive >= 2:
            state["phase"] = phase
            provider.atomic_json(context["status_path"], state)
            return gpu
        sleep(
            min(
                probe_interval_seconds,
                max(0.0, end - now()),
                max(0.0, wait_seconds - (monotonic() - started_monotonic)),
            )
        )


def complete_matrix(context: dict) -> dict:
    context["provider"].setup_paths(context["plan"]["repo"])
    from scripts.analyze_ndr_development import admit_matrix

    return admit_matrix(Path(context["plan"]["run_root"]), context["plan"])


def execute_continuation(
    context: dict,
    *,
    now: Callable = time.time,
    sleep: Callable = time.sleep,
    execute_owned: Callable | None = None,
) -> dict:
    provider, plan, state = context["provider"], context["plan"], context["state"]
    absolute = provider.utc_seconds(context["policy"].deadline_utc)
    execute = execute_owned or provider.execute_owned
    verifier = context.get("task_verifier") or provider.completed_verifier
    status = context["status_path"]
    provider.atomic_json(status, state)
    try:
        for index, task in enumerate(plan["tasks"][3:], start=3):
            require_helpers(context)
            state.update(current_task=task["task_id"])
            if index > 3:
                state["phase"] = "train"
                wait_for_idle(context, now=now, sleep=sleep)
                require_helpers(context)
                if Path(task["output_dir"]).exists():
                    raise FileExistsError("a later task output is not fresh")
                started = now()
                hard = min(absolute, started + context["policy"].job_seconds)
                state.update(
                    task_hard_deadline_utc=datetime.fromtimestamp(
                        hard, timezone.utc
                    ).isoformat(),
                    original_task_limit_seconds=context["policy"].job_seconds,
                )
                provider.atomic_json(status, state)
                execute(
                    list(task["train_command"]),
                    repo=Path(plan["repo"]),
                    log=Path(plan["run_root"])
                    / "logs"
                    / (
                        task["task_id"]
                        + "_continuation_"
                        + context["recovery_id"]
                        + "_train.log"
                    ),
                    hard_deadline=hard,
                    state=state,
                    status_path=status,
                    now=now,
                    sleep=sleep,
                )
            summary = json.loads(
                (Path(task["output_dir"]) / "summary.json").read_text()
            )
            if (
                summary.get("completed_epochs") != 30
                or summary.get("status") != TRAINED_STATUS
                or summary.get("test_evaluated") is not False
            ):
                raise RuntimeError(
                    "partial training is not eligible for independent verification"
                )
            state["phase"] = "verify"
            wait_for_idle(context, now=now, sleep=sleep)
            require_helpers(context)
            if index == 3 and (Path(task["output_dir"]) / "verification.json").exists():
                raise FileExistsError(
                    "the fourth verification appeared outside the owned continuation"
                )
            execute(
                list(task["verify_command"]),
                repo=Path(plan["repo"]),
                log=Path(plan["run_root"])
                / "logs"
                / (
                    task["task_id"]
                    + "_continuation_"
                    + context["recovery_id"]
                    + "_verify.log"
                ),
                hard_deadline=absolute,
                state=state,
                status_path=status,
                now=now,
                sleep=sleep,
            )
            require_helpers(context)
            verifier(
                Path(task["output_dir"]),
                provider.task_inventory(Path(plan["run_root"]), task, plan),
                plan,
                task,
            )
            state["completed"].append(task["task_id"])
            provider.atomic_json(status, state)
        require_helpers(context)
        if now() >= absolute:
            raise TimeoutError(
                "source postflight cannot finish after the original deadline"
            )
        files = [
            {**row, "path": "tasks/" + task["task_id"] + "/" + row["path"]}
            for task in plan["tasks"]
            for row in provider.task_inventory(Path(plan["run_root"]), task, plan)[
                "files"
            ]
        ]
        provider.atomic_json(
            Path(plan["run_root"]) / "results/postflight.json",
            {
                "status": "verified_development_train_val_only",
                "plan_digest": plan["plan_digest"],
                "tasks": state["completed"],
                "files": files,
                "test_evaluated": False,
                "source_retained": True,
                "continuation_id": context["recovery_id"],
                "incident_sha256": state["incident_sha256"],
                "continuation_controller_sha256": state[
                    "continuation_controller_sha256"
                ],
            },
        )
        state.update(
            status="completed",
            phase="source_postflight",
            current_task=None,
            finished_at=datetime.fromtimestamp(now(), timezone.utc).isoformat(),
        )
        provider.atomic_json(status, state)
        report = complete_matrix(context)
        if now() >= absolute:
            raise TimeoutError(
                "matrix admission exceeded the original absolute deadline"
            )
        provider.atomic_json(
            context["control"] / "completed_matrix_admission.json", report
        )
        return state
    except BaseException as error:
        state.update(
            status="failed",
            error=str(error),
            worker_pid=None,
            failed_at=datetime.fromtimestamp(now(), timezone.utc).isoformat(),
        )
        provider.atomic_json(status, state)
        provider.atomic_json(context["control"] / "failure.json", state)
        raise


def run(args: argparse.Namespace) -> dict:
    if os.name != "nt":
        raise PermissionError(
            "actual continuation runs only on the authorized Windows host"
        )
    provider_path = args.gpu_provider.resolve()
    provider = load_provider(provider_path)
    plan = json.loads(args.plan.read_text())
    incident = provider.read_bound_json(args.incident, args.incident_sha256)
    manifest = provider.read_bound_json(args.preservation, args.preservation_sha256)
    local_receipt = provider.read_bound_json(
        args.local_preservation_receipt, args.local_preservation_receipt_sha256
    )
    if (
        incident.get("preservation_sha256") != args.preservation_sha256
        or incident.get("local_preservation_receipt_sha256")
        != args.local_preservation_receipt_sha256
    ):
        raise ValueError("incident does not bind both complete preservation receipts")
    code = provider.setup_paths(plan["repo"])
    helper_files = {
        str(provider_path): PROVIDER_SHA256,
        str(code / "scripts/retrieve_ndr_development.py"): RETRIEVAL_SHA256,
        str(code / "scripts/analyze_ndr_development.py"): ANALYZER_SHA256,
        str(Path(__file__).resolve()): provider.file_sha(Path(__file__)),
    }
    for path, expected in helper_files.items():
        if provider.file_sha(Path(path)) != expected:
            raise ValueError(
                "independent continuation helper bytes differ before admission"
            )
    probes = provider.WindowsProbes()
    accepted = admit_continuation(
        plan,
        incident,
        manifest,
        local_receipt,
        provider=provider,
        now=time.time(),
        probes=probes,
    )
    control = args.control_root.resolve()
    if control.is_relative_to(
        Path(plan["run_root"]).resolve()
    ) or control.is_relative_to(Path(plan["repo"]).resolve()):
        raise ValueError(
            "new continuation control must be outside the old result and source trees"
        )
    control.mkdir(parents=True, exist_ok=False)
    recovery_id = uuid.uuid4().hex
    with (Path(plan["run_root"]) / "results/continuation.lock").open(
        "x", encoding="utf8"
    ) as stream:
        json.dump(
            {
                "pid": os.getpid(),
                "continuation_id": recovery_id,
                "incident_sha256": args.incident_sha256,
            },
            stream,
        )
    state = {
        "status": "running",
        "plan_digest": plan["plan_digest"],
        "absolute_deadline_utc": EXPECTED_DEADLINE,
        "started_at": incident["queue_snapshot"]["started_at"],
        "pid": os.getpid(),
        "worker_pid": None,
        "completed": list(accepted["completed"]),
        "current_task": accepted["current_task"],
        "phase": "continuation_admitted",
        "test_evaluated": False,
        "automatic_shutdown": False,
        "automatic_transfer": False,
        "continuation_id": recovery_id,
        "incident_sha256": args.incident_sha256,
        "preservation_sha256": args.preservation_sha256,
        "local_preservation_receipt_sha256": args.local_preservation_receipt_sha256,
        "continuation_controller_sha256": helper_files[str(Path(__file__).resolve())],
        "gpu_provider_sha256": PROVIDER_SHA256,
        "original_supervisor_pid": incident["queue_snapshot"]["pid"],
        "original_deadline_retained": True,
        "fourth_training_will_be_repeated": False,
    }
    provider.atomic_json(control / "admission.json", accepted)
    provider.atomic_json(control / "original_incident.json", incident)
    provider.atomic_json(
        control / "source_provenance.json",
        {
            "continuation_sources": helper_files,
            "original_sources": plan["source_files"],
            "plan_digest": plan["plan_digest"],
            "original_absolute_deadline_utc": EXPECTED_DEADLINE,
        },
    )
    context = {
        "provider": provider,
        "provider_path": provider_path,
        "plan": plan,
        "incident": incident,
        "manifest": manifest,
        "accepted": accepted,
        "helper_files": helper_files,
        "recovery_id": recovery_id,
        "control": control,
        "status_path": Path(plan["run_root"]) / "results/queue_status.json",
        "state": state,
        "probes": probes,
        "policy": Policy(),
    }
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise PermissionError("idle-sleep prevention could not be established")
    try:
        return execute_continuation(context)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--incident-sha256", required=True)
    parser.add_argument("--preservation", type=Path, required=True)
    parser.add_argument("--preservation-sha256", required=True)
    parser.add_argument("--local-preservation-receipt", type=Path, required=True)
    parser.add_argument("--local-preservation-receipt-sha256", required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument(
        "--gpu-provider",
        type=Path,
        default=Path(__file__).with_name("ndr_research_recovery.py"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
