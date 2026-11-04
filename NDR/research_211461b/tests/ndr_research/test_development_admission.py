"""Completed-matrix admission rejects partial runs and hidden recipe/data drift."""

import json

import pytest

from scripts.analyze_ndr_development import admit_matrix, canonical_hash, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_matrix(root):
    source = {"model.py": {"sha256": "a" * 64, "size_bytes": 12}}
    tasks = []
    for region in ("Alameda", "ContraCosta"):
        for variant, mae in [
            ("baseline", 10.0),
            ("mask_signed", 9.0),
            ("sanitized_control", 9.5),
            ("projected_oracle", 10.1),
        ]:
            tid = region.lower() + "_" + variant
            tasks.append({"task_id": tid, "region": region, "variant": variant})
            task = root / "tasks" / tid
            task.mkdir(parents=True)
            checkpoint = task / "best.pt"
            checkpoint.write_bytes(tid.encode())
            identity = {
                "cache_bundle_digest": "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5",
                "region": region,
            }
            config = dict(
                region=region,
                variant=variant,
                epochs=30,
                seed=42,
                precision="FP32",
                claim_scope="validation_development_only",
                max_train_batches=None,
                max_val_batches=None,
                batch_size=16,
                microbatch_size=8,
                learning_rate=0.001,
                weight_decay=0.0001,
                gradient_clip=5,
                eta_min=0.00001,
                lambda_gate=1,
                lambda_specialization=0.3,
                hidden_dim=16,
                reliable_anchor=False,
                max_reference_age=2,
                early_stopping=False,
                test_evaluation=False,
                log_interval=100,
                threads=3,
                source_snapshot={
                    "files": source,
                    "files_digest": canonical_hash(source),
                },
                runtime={"device": "cuda:0"},
                data_identity=identity,
            )
            validation = dict(
                valid_count=120,
                absolute_error_sum=mae * 120,
                global_masked_mae=mae,
                average_horizon_mae=mae,
                horizon_mae=[mae] * 12,
                selection={
                    "all_validation_samples": True,
                    "samples": 10,
                    "region": region,
                },
                observed_targets_sha256=region + "-targets",
                observation_masks_sha256=region + "-masks",
            )
            summary = dict(
                completed_epochs=30,
                test_evaluated=False,
                config_digest=canonical_hash(config),
                best_validation=validation,
                best_val_mae=mae,
                best_epoch=29,
                elapsed_seconds=100,
            )
            verified = dict(
                validation=validation,
                status="verified_checkpoint_complete_validation",
                test_evaluated=False,
                config_digest=canonical_hash(config),
                checkpoint_sha256=sha(checkpoint),
                cache_bundle_digest=identity["cache_bundle_digest"],
                source_files_digest=canonical_hash(source),
            )
            for name, value in [
                ("config", config),
                ("summary", summary),
                ("verification", verified),
                ("data_identity", identity),
            ]:
                write(task / (name + ".json"), value)
    plan = {"tasks": tasks, "source_files": {"model.py": "a" * 64}}
    plan["plan_digest"] = canonical_hash(plan)
    write(
        root / "results/queue_status.json",
        {
            "status": "completed",
            "plan_digest": plan["plan_digest"],
            "completed": [t["task_id"] for t in tasks],
        },
    )
    renew_postflight(root, plan)
    return plan


def renew_postflight(root, plan):
    files = [
        {
            "path": str(p.relative_to(root)).replace("/", "\\"),
            "bytes": p.stat().st_size,
            "sha256": sha(p),
        }
        for p in (root / "tasks").rglob("*")
        if p.is_file()
    ]
    write(
        root / "results/postflight.json",
        {
            "status": "verified_development_train_val_only",
            "plan_digest": plan["plan_digest"],
            "tasks": [t["task_id"] for t in plan["tasks"]],
            "test_evaluated": False,
            "files": files,
        },
    )


def test_complete_matrix_may_select_for_confirmation_but_never_claim_test_winner(
    tmp_path,
):
    plan = fixture_matrix(tmp_path)
    report = admit_matrix(tmp_path, plan)
    assert report["decisions"]["mask_signed"][
        "eligible_for_frozen_multiseed_confirmation"
    ]
    assert not report["decisions"]["projected_oracle"][
        "eligible_for_frozen_multiseed_confirmation"
    ]
    assert report["test_evaluated"] is False
    assert report["decisions"]["mask_signed"]["test_or_sota_claim"] is False


def test_partial_live_queue_cannot_be_scored_as_a_finished_design(tmp_path):
    plan = fixture_matrix(tmp_path)
    write(
        tmp_path / "results/queue_status.json",
        {"status": "running", "plan_digest": plan["plan_digest"], "completed": []},
    )
    with pytest.raises(ValueError, match="not source-verified"):
        admit_matrix(tmp_path, plan)


def test_a_lower_mae_on_different_masks_cannot_admit_a_candidate(tmp_path):
    plan = fixture_matrix(tmp_path)
    directory = tmp_path / "tasks/alameda_mask_signed"
    for name, key in [
        ("summary.json", "best_validation"),
        ("verification.json", "validation"),
    ]:
        value = json.loads((directory / name).read_text())
        value[key]["observation_masks_sha256"] = "changed-mask"
        write(directory / name, value)
    renew_postflight(tmp_path, plan)
    with pytest.raises(ValueError, match="exact validation"):
        admit_matrix(tmp_path, plan)


def test_complete_validation_after_two_training_steps_is_still_not_formal(tmp_path):
    plan = fixture_matrix(tmp_path)
    directory = tmp_path / "tasks/alameda_mask_signed"
    config = json.loads((directory / "config.json").read_text())
    config["max_train_batches"] = 2
    config["claim_scope"] = "smoke_only"
    write(directory / "config.json", config)
    renew_postflight(tmp_path, plan)
    with pytest.raises(ValueError, match="smoke/formal scope"):
        admit_matrix(tmp_path, plan)


def test_one_favorable_region_is_insufficient_for_the_predeclared_gate(tmp_path):
    plan = fixture_matrix(tmp_path)
    directory = tmp_path / "tasks/contracosta_mask_signed"
    for name, key in [
        ("summary.json", "best_validation"),
        ("verification.json", "validation"),
    ]:
        value = json.loads((directory / name).read_text())
        value[key].update(global_masked_mae=10.01, absolute_error_sum=1201.2)
        if name == "summary.json":
            value["best_val_mae"] = 10.01
        write(directory / name, value)
    renew_postflight(tmp_path, plan)
    report = admit_matrix(tmp_path, plan)
    assert not report["decisions"]["mask_signed"][
        "eligible_for_frozen_multiseed_confirmation"
    ]
