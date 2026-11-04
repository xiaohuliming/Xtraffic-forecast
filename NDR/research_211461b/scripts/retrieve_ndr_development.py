#!/usr/bin/env python3
"""Preserve completed development tasks without modifying the running queue."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_ndr_development import BUNDLE, admit_matrix, canonical_hash, sha  # noqa: E402


def validate_plan(plan: dict) -> None:
    if canonical_hash({k: v for k, v in plan.items() if k != "plan_digest"}) != plan["plan_digest"]:
        raise ValueError("development plan digest mismatch")
    if plan["cache_bundle_digest"] != BUNDLE or plan["test_evaluated"]:
        raise ValueError("retrieval requires the locked no-test development bundle")


def inventory_paths(inventory: dict) -> dict[str, dict]:
    files = {}
    for row in inventory["files"]:
        relative = PurePosixPath(row["path"].replace("\\", "/"))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or ":" in str(relative):
            raise ValueError("unsafe task inventory path")
        key = str(relative)
        if key in files or type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("duplicate or invalid task inventory entry")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("invalid task inventory hash")
        files[key] = row
    required = {"config.json", "data_identity.json", "summary.json", "verification.json",
                "best.pt", "best.pt.sha256.json", "last.pt", "last.pt.sha256.json",
                "training_diagnostics.jsonl"}
    if not required.issubset(files):
        raise ValueError("completed task inventory lacks required artifacts")
    return files


def verify_task_tree(directory: Path, inventory: dict, plan: dict, task: dict,
                     *, load_checkpoints: bool = True) -> dict:
    validate_plan(plan)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("task root must be a real directory")
    if inventory["task_id"] != task["task_id"] or inventory["plan_digest"] != plan["plan_digest"]:
        raise ValueError("task inventory does not bind the frozen plan")
    declarations = inventory_paths(inventory)
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("copied task contains a symlink or escaped path")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(declarations):
        raise ValueError("copied task differs from the exact source file set")
    for relative, row in declarations.items():
        path = directory / relative
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("copied task bytes differ from the source inventory")
    config = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    identity = json.loads((directory / "data_identity.json").read_text())
    digest = canonical_hash(config)
    if any(config[key] != task[key] for key in ("region", "variant", "seed", "epochs")):
        raise ValueError("completed task recipe differs from the frozen plan")
    if (config["claim_scope"] != "validation_development_only"
            or config["max_train_batches"] is not None or config["max_val_batches"] is not None
            or config["test_evaluation"] or summary["test_evaluated"] or verification["test_evaluated"]
            or summary["completed_epochs"] != task["epochs"]):
        raise ValueError("task is incomplete, limited-batch or consumed test")
    if (summary["config_digest"] != digest or verification["config_digest"] != digest
            or config["data_identity"] != identity or identity["cache_bundle_digest"] != BUNDLE
            or verification["cache_bundle_digest"] != BUNDLE):
        raise ValueError("task config or formal data identity mismatch")
    source = config["source_snapshot"]
    if (canonical_hash(source["files"]) != source["files_digest"]
            or verification["source_files_digest"] != source["files_digest"]
            or any(item["sha256"] != plan["source_files"].get(name)
                   for name, item in source["files"].items())):
        raise ValueError("trained source differs from the frozen plan")
    observed = verification["validation"]
    selected = summary["best_validation"]
    if (verification["status"] != "verified_checkpoint_complete_validation"
            or not observed["selection"]["all_validation_samples"]
            or observed["valid_count"] <= 0
            or not math.isclose(observed["absolute_error_sum"] / observed["valid_count"],
                                observed["global_masked_mae"], rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(summary["best_val_mae"], observed["global_masked_mae"],
                                rel_tol=2e-6, abs_tol=2e-5)):
        raise ValueError("task has no independently verified complete validation")
    for key in ("selection", "valid_count", "observed_targets_sha256", "observation_masks_sha256"):
        if observed[key] != selected[key]:
            raise ValueError("checkpoint selection and verification collections differ")
    if verification["checkpoint_sha256"] != declarations["best.pt"]["sha256"]:
        raise ValueError("verification checkpoint hash mismatch")
    for name in ("best.pt", "last.pt"):
        marker = json.loads((directory / (name + ".sha256.json")).read_text())
        if (marker["sha256"] != declarations[name]["sha256"]
                or marker["size_bytes"] != declarations[name]["bytes"]
                or marker["config_digest"] != digest):
            raise ValueError("checkpoint sidecar mismatch")
        if load_checkpoints:
            from scripts.train_ndr_research import load_checkpoint
            from ndr_research.models import make_research_model
            import torch

            checkpoint = load_checkpoint(directory / name)
            if checkpoint["config_digest"] != digest or checkpoint["data_identity"] != identity:
                raise ValueError("loaded checkpoint config or data identity mismatch")
            model = make_research_model(**checkpoint["model_factory"], device="cpu")
            model.load_state_dict(checkpoint["model"], strict=True)
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()
                   if value.is_floating_point()):
                raise ValueError("loaded model parameters are nonfinite")
            if name == "best.pt" and checkpoint["selection_metric"] != selected:
                raise ValueError("best checkpoint selection differs from its summary")
            expected_epoch = summary["best_epoch"] if name == "best.pt" else summary["completed_epochs"]
            if (checkpoint["progress"]["epoch"] != expected_epoch
                    or len(checkpoint["progress"]["history"]) != expected_epoch
                    or checkpoint["progress"]["stage"] != "start"):
                raise ValueError("completed checkpoint epoch or history differs from its summary")
    return {"status": "verified_individual_completed_development_task",
            "task_id": task["task_id"], "files": len(declarations),
            "bytes": sum(row["bytes"] for row in declarations.values()),
            "global_validation_mae": observed["global_masked_mae"],
            "best_epoch": summary["best_epoch"], "completed_epochs": summary["completed_epochs"],
            "checkpoint_structurally_loaded": load_checkpoints,
            "local_full_validation_inference_repeated": False,
            "test_evaluated": False, "matrix_conclusion": False}


class Transport:
    def __init__(self, python: str) -> None:
        self.python = python
        self.env = os.environ.copy()
        self.env["SSHPASS"] = self.env["LAB_5080_PASSWORD"]
        self.destination = self.env["LAB_5080_USER"] + "@" + self.env["LAB_5080_TAILSCALE_HOST"]
        self.options = ["-o", "StrictHostKeyChecking=yes", "-o", "PreferredAuthentications=password",
                        "-o", "PubkeyAuthentication=no", "-o", "NumberOfPasswordPrompts=1",
                        "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
                        "-o", "ServerAliveCountMax=3"]

    def run(self, command: list[str], *, timeout: int = 55) -> str:
        result = subprocess.run(["sshpass", "-e", *command], env=self.env,
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError("SSH or file transfer failed: " + result.stderr[-1000:])
        return result.stdout

    def execute(self, script: str) -> dict:
        payload = base64.b64encode(script.encode()).decode()
        command = self.python + ' -X utf8 -c "import base64;exec(base64.b64decode(\'' + payload + '\'))"'
        return json.loads(self.run(["ssh", *self.options, self.destination, command]))

    def copy(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self.run(["scp", *self.options, self.destination + ":" + remote, str(local)], timeout=120)


def source_inventory(transport: Transport, plan: dict, task: dict) -> dict:
    script = """from pathlib import Path
import hashlib,json
root=Path(RUN_ROOT);task=Path(TASK_ROOT)
queue=json.loads((root/'results/queue_status.json').read_text())
assert queue['plan_digest']==PLAN_DIGEST and TASK_ID in queue['completed']
summary=json.loads((task/'summary.json').read_text())
verified=json.loads((task/'verification.json').read_text())
assert summary['completed_epochs']==EPOCHS and not summary['test_evaluated']
assert verified['status']=='verified_checkpoint_complete_validation' and not verified['test_evaluated']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
files=[]
for path in sorted(task.rglob('*')):
 assert not path.is_symlink() and path.resolve().is_relative_to(task.resolve())
 if path.is_file():files.append({'path':path.relative_to(task).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
print(json.dumps({'task_id':TASK_ID,'source_root':str(task),'plan_digest':PLAN_DIGEST,'files':files}))
"""
    constants = {"RUN_ROOT": plan["run_root"], "TASK_ROOT": task["output_dir"],
                 "PLAN_DIGEST": plan["plan_digest"], "TASK_ID": task["task_id"], "EPOCHS": task["epochs"]}
    return transport.execute("\n".join(key + "=" + repr(value) for key, value in constants.items()) + "\n" + script)


def completed_controls(transport: Transport, plan: dict) -> dict:
    script = """from pathlib import Path
import json,hashlib
root=Path(RUN_ROOT)
queue=json.loads((root/'results/queue_status.json').read_text())
assert queue['status']=='completed' and queue['plan_digest']==PLAN_DIGEST
assert queue['completed']==EXPECTED_TASKS
files={}
for name in ('postflight.json','queue_status.json'):
 path=root/'results'/name;raw=path.read_bytes()
 files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(files))
"""
    constants = {"RUN_ROOT": plan["run_root"], "PLAN_DIGEST": plan["plan_digest"],
                 "EXPECTED_TASKS": [task["task_id"] for task in plan["tasks"]]}
    return transport.execute("\n".join(key + "=" + repr(value) for key, value in constants.items()) + "\n" + script)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--complete-matrix", action="store_true",
                        help="Require all eight copied tasks and immutable source postflight before analysis.")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    tasks = [row for row in plan["tasks"] if row["task_id"] == args.task_id]
    if len(tasks) != 1:
        raise ValueError("requested task is not uniquely in the frozen plan")
    task = tasks[0]
    transport = Transport(plan["python"])
    before = source_inventory(transport, plan, task)
    files = inventory_paths(before)
    destination = args.output_root / "tasks" / task["task_id"]
    stage = args.output_root / ".staging" / (task["task_id"] + "_" + uuid.uuid4().hex)
    copied = not destination.exists()
    if copied:
        stage.mkdir(parents=True)
        for relative in sorted(files):
            transport.copy(task["output_dir"].rstrip("/\\") + "/" + relative, stage / relative)
    check_root = stage if copied else destination
    result = verify_task_tree(check_root, before, plan, task)
    after = source_inventory(transport, plan, task)
    if before != after:
        raise ValueError("source task changed during transport; staged files remain preserved")
    if copied:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, destination)
    receipts = args.output_root / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = result | {"source_unchanged_during_transfer": True, "source_inventory": before,
                        "plan_digest": plan["plan_digest"], "copied_this_invocation": copied,
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    path = receipts / (task["task_id"] + "_" + uuid.uuid4().hex + ".json")
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if args.complete_matrix:
        controls_before = completed_controls(transport, plan)
        results = args.output_root / "results"
        staged_results = args.output_root / ".staging" / ("matrix_" + uuid.uuid4().hex) / "results"
        copy_controls = not results.exists()
        check_results = staged_results if copy_controls else results
        for name, expected in controls_before.items():
            local = check_results / name
            if copy_controls:
                transport.copy(plan["run_root"] + "/results/" + name, local)
            if local.is_symlink() or local.stat().st_size != expected["bytes"] or sha(local) != expected["sha256"]:
                raise ValueError("copied matrix control bytes differ from the source")
        if controls_before != completed_controls(transport, plan):
            raise ValueError("source matrix controls changed during transport")
        if copy_controls:
            os.rename(staged_results, results)
        admitted = admit_matrix(args.output_root, plan)
        analysis = args.output_root / "development_analysis.json"
        encoded = json.dumps(admitted, indent=2) + "\n"
        if analysis.exists():
            if analysis.read_text() != encoded:
                raise ValueError("existing admitted analysis differs; preserve it for audit")
        else:
            with analysis.open("x") as stream:
                stream.write(encoded)
        with (receipts / ("matrix_" + uuid.uuid4().hex + ".json")).open("x") as stream:
            json.dump({"status": "verified_development_matrix_copy_and_admission",
                       "controls": controls_before, "source_unchanged_during_transfer": True,
                       "analysis_sha256": sha(analysis), "test_evaluated": False}, stream, indent=2)
    print(json.dumps(result | {"receipt": str(path)}, allow_nan=False))


if __name__ == "__main__":
    main()
