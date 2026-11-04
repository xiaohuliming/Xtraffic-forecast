"""Copied bytes and partial-task scope remain separate from matrix admission."""

import copy
import json

import pytest

from scripts.analyze_ndr_development import BUNDLE, canonical_hash, sha
from scripts.retrieve_ndr_development import inventory_paths, verify_task_tree


@pytest.fixture
def completed(tmp_path):
    task = {"task_id": "alameda_baseline_seed42", "region": "Alameda", "variant": "baseline",
            "seed": 42, "epochs": 30}
    identity = {"cache_bundle_digest": BUNDLE}
    source_files = {"model.py": {"sha256": "a" * 64, "size_bytes": 2}}
    config = {key: task[key] for key in ("region", "variant", "seed", "epochs")}
    config.update({"claim_scope": "validation_development_only", "max_train_batches": None,
                   "max_val_batches": None, "test_evaluation": False, "data_identity": identity,
                   "source_snapshot": {"files": source_files, "files_digest": canonical_hash(source_files)}})
    selection = {"selection": {"all_validation_samples": True}, "valid_count": 3,
                 "absolute_error_sum": 12.0, "global_masked_mae": 4.0,
                 "observed_targets_sha256": "b" * 64, "observation_masks_sha256": "c" * 64}
    summary = {"test_evaluated": False, "completed_epochs": 30, "best_epoch": 14,
               "config_digest": canonical_hash(config), "best_val_mae": 4.0,
               "best_validation": selection}
    plan = {"cache_bundle_digest": BUNDLE, "test_evaluated": False,
            "source_files": {"model.py": "a" * 64}, "tasks": [task]}
    plan["plan_digest"] = canonical_hash(plan)
    (tmp_path / "best.pt").write_bytes(b"best")
    (tmp_path / "last.pt").write_bytes(b"last")
    verification = {"test_evaluated": False, "status": "verified_checkpoint_complete_validation",
                    "cache_bundle_digest": BUNDLE, "config_digest": canonical_hash(config),
                    "source_files_digest": canonical_hash(source_files), "validation": selection,
                    "checkpoint_sha256": sha(tmp_path / "best.pt")}
    objects = {"config.json": config, "data_identity.json": identity, "summary.json": summary,
               "verification.json": verification}
    for name in ("best.pt", "last.pt"):
        objects[name + ".sha256.json"] = {"sha256": sha(tmp_path / name),
                                           "size_bytes": (tmp_path / name).stat().st_size,
                                           "config_digest": canonical_hash(config)}
    for name, value in objects.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "training_diagnostics.jsonl").write_text("{}\n")
    inventory = {"task_id": task["task_id"], "plan_digest": plan["plan_digest"], "files": []}
    for path in sorted(tmp_path.iterdir()):
        inventory["files"].append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
    return tmp_path, inventory, plan, task


def rewrite(completed, name, change):
    root, inventory, _, _ = completed
    obj = json.loads((root / name).read_text())
    change(obj)
    (root / name).write_text(json.dumps(obj))
    row = next(row for row in inventory["files"] if row["path"] == name)
    row.update({"sha256": sha(root / name), "bytes": (root / name).stat().st_size})


def verify(completed):
    return verify_task_tree(*completed, load_checkpoints=False)


def test_completed_individual_task_is_not_matrix_conclusion(completed):
    result = verify(completed)
    assert result["global_validation_mae"] == 4.0
    assert result["matrix_conclusion"] is False
    assert result["checkpoint_structurally_loaded"] is False


@pytest.mark.parametrize("relative", ["../best.pt", "C:/escape", "/absolute", "..\\escape"])
def test_inventory_rejects_escaped_paths(completed, relative):
    inventory = copy.deepcopy(completed[1])
    inventory["files"][0]["path"] = relative
    with pytest.raises(ValueError, match="unsafe"):
        inventory_paths(inventory)


def test_inventory_rejects_unexpected_local_file(completed):
    (completed[0] / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="exact source file set"):
        verify(completed)


def test_modified_checkpoint_is_rejected_before_deserialization(completed):
    (completed[0] / "best.pt").write_bytes(b"evil")
    with pytest.raises(ValueError, match="source inventory"):
        verify(completed)


def test_success_receipt_cannot_admit_test_consumption(completed):
    rewrite(completed, "verification.json", lambda obj: obj.update(test_evaluated=True))
    with pytest.raises(ValueError, match="consumed test"):
        verify(completed)


def test_complete_summary_cannot_admit_subset_verification(completed):
    rewrite(completed, "verification.json", lambda obj: obj["validation"]["selection"].update(all_validation_samples=False))
    with pytest.raises(ValueError, match="complete validation"):
        verify(completed)


def test_equal_score_cannot_admit_different_target_collection(completed):
    rewrite(completed, "verification.json", lambda obj: obj["validation"].update(observed_targets_sha256="d" * 64))
    with pytest.raises(ValueError, match="collections differ"):
        verify(completed)


def test_wrong_sidecar_is_rejected_even_when_inventory_agrees(completed):
    rewrite(completed, "last.pt.sha256.json", lambda obj: obj.update(sha256="d" * 64))
    with pytest.raises(ValueError, match="sidecar"):
        verify(completed)


def test_copy_cannot_admit_symlink(completed):
    (completed[0] / "alias").symlink_to(completed[0] / "best.pt")
    with pytest.raises(ValueError, match="symlink"):
        verify(completed)
