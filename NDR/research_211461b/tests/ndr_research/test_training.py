from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ndr_research.data import (
    ResearchData,
    ResearchEventLoader,
    array_digest,
    canonical_digest,
    prepare_research_batch,
    validate_manifest_binding,
)
from ndr_research.models import make_research_model
from scripts.protocol_v3_event_extension_data import WeeklyClimatology
from scripts import train_ndr_research as trainer


class TinyRegion:
    N = 2
    T_h = 12
    T_p = 12
    include_analysis_labels = False
    region_name = "Alameda"
    train_end = 50
    val_end = 100
    split = np.asarray([0] * 6 + [1] * 3 + [2], dtype=np.int8)
    sample_start = np.asarray([12, 14, 16, 18, 20, 22, 65, 67, 69, 115], dtype=np.int64)

    def get_sample(self, index: int) -> dict:
        if self.split[index] == 2:
            raise AssertionError("a test target was read")
        flow = np.arange(24, dtype=np.float32).reshape(2, 12) + 30 + index
        target = flow + index + 1
        x_mask = np.ones((2, 12, 3), dtype=np.bool_)
        x_mask[0, -2:, 0] = False
        y_mask = np.ones_like(x_mask)
        y_mask[0, :index % 11, 0] = False
        return {
            "x_hist": np.repeat(flow[..., None], 3, axis=-1),
            "x_hist_mask": x_mask,
            "y_true": np.repeat(target[..., None], 3, axis=-1),
            "y_mask": y_mask,
            "time_feat": np.stack((np.arange(12, dtype=np.float32) / 288,
                                   np.zeros(12, dtype=np.float32)), axis=-1),
            "sample_start": self.sample_start[index],
        }


def tiny_data(batch_size=2, seed=42) -> ResearchData:
    region = TinyRegion()
    statistics = WeeklyClimatology(
        baseline=np.full((7, 288, 2), 40.0, dtype=np.float32),
        flow_mean=40.0, flow_std=10.0, residual_std=5.0,
        residual_std_by_node=np.asarray([4.0, 6.0], dtype=np.float32),
        valid_training_points=100,
    )
    loaders = {
        name: ResearchEventLoader(
            split=name, region_data=region, indices=np.flatnonzero(region.split == code),
            batch_size=batch_size, flow_mean=40.0, flow_std=10.0,
            sensor_support=None, include_incident=False, shuffle=name == "train",
            drop_last=False, seed=seed,
        ) for name, code in (("train", 0), ("val", 1))
    }
    return ResearchData(region, statistics, loaders["train"], loaders["val"],
                        {"region_cache_digest": "tiny_cache", "cache_bundle_digest": "tiny_bundle"})


def tiny_model(variant="baseline"):
    model = make_research_model(2, variant=variant, residual_std=5.0,
                               residual_std_by_node=[4.0, 6.0], flow_mean=40.0,
                               flow_std=10.0, device="cpu")
    return model


def config_for(data, *, epochs=2, microbatch=1, max_val=None):
    return {
        "epochs": epochs, "seed": 42, "batch_size": data.train.bs,
        "microbatch_size": microbatch, "learning_rate": 0.001,
        "weight_decay": 0.0001, "eta_min": 0.00001, "gradient_clip": 5.0,
        "lambda_gate": 1.0, "lambda_specialization": 0.3,
        "max_train_batches": None, "max_val_batches": max_val, "log_interval": 100,
        "claim_scope": "smoke_only", "source_snapshot": trainer.source_snapshot(),
        "runtime": trainer.runtime_identity(torch.device("cpu")),
    }


def test_loader_rejects_test_or_analysis_labels():
    data = tiny_data()
    common = dict(region_data=data.region, batch_size=2, flow_mean=40.0, flow_std=10.0,
                  sensor_support=None, include_incident=False, shuffle=False,
                  drop_last=False, seed=42)
    with pytest.raises(ValueError, match="only train or val"):
        ResearchEventLoader(split="test", indices=np.asarray([9]), **common)
    with pytest.raises(ValueError, match="cross"):
        ResearchEventLoader(split="train", indices=np.asarray([0, 9]), **common)
    contaminated = TinyRegion()
    contaminated.include_analysis_labels = True
    with pytest.raises(ValueError, match="analysis labels"):
        ResearchEventLoader(split="train", indices=np.asarray([0]),
                            **(common | {"region_data": contaminated}))


def test_original_batch_preparation_adds_history_mask_and_retains_tail():
    data = tiny_data()
    batch = data.val.next_batch()
    prepared = prepare_research_batch(batch, data.statistics)
    assert prepared["history_mask"].shape == (2, 2, 12)
    assert not prepared["history_mask"][0, 0, -1]
    assert prepared["target_mask"].shape == (2, 2, 12)
    tail = data.val.next_batch()
    assert tail["x_data"].shape[0] == 1
    assert "sensor_data" not in tail and "incident_features" not in tail
    with pytest.raises(ValueError, match="analysis labels"):
        prepare_research_batch(batch | {"affected_mask": np.ones(1)}, data.statistics)


def test_loader_cursor_and_rng_resume_identical_batches_and_next_epoch():
    first, restored = tiny_data(), tiny_data()
    first.train.start_epoch()
    first.train.next_batch()
    state = first.train.state_dict()
    restored.train.load_state_dict(state)
    np.testing.assert_array_equal(first.train.next_batch()["sample_start"],
                                  restored.train.next_batch()["sample_start"])
    first.train.start_epoch()
    restored.train.start_epoch()
    np.testing.assert_array_equal(first.train._order, restored.train._order)
    corrupted = copy.deepcopy(state)
    corrupted["order"][0] = 9
    with pytest.raises(ValueError, match="permutation"):
        restored.train.load_state_dict(corrupted)


@pytest.mark.parametrize("variant", ["baseline", "projected_oracle", "sanitized_control", "mask_signed"])
def test_microbatch_gradient_matches_effective_full_batch(variant):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data(batch_size=4)
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    # Make one microbatch entirely unobserved to test exact empty-chunk handling.
    tensors["target_mask"][0] = False
    tensors["target"][0] = 0
    full = tiny_model(variant)
    for module in full.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    micro = copy.deepcopy(full)
    full.train()
    micro.train()
    expected = trainer.backward_effective_batch(full, tensors, microbatch_size=4)
    actual = trainer.backward_effective_batch(micro, tensors, microbatch_size=1)
    assert actual["valid_count"] == expected["valid_count"]
    assert actual["objective"] == pytest.approx(expected["objective"], rel=2e-6)
    for left, right in zip(full.parameters(), micro.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-6)


def test_validation_accumulates_all_observations_and_all_horizons(monkeypatch):
    data = tiny_data()
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(trainer, "model_forward", lambda _, tensors: torch.zeros_like(tensors["target"]))
    actual = trainer.evaluate(model, data, torch.device("cpu"), microbatch_size=1)
    targets, masks = [], []
    for index in data.val.indices:
        sample = data.region.get_sample(int(index))
        targets.append(sample["y_true"][..., 0])
        masks.append(sample["y_mask"][..., 0])
    targets, masks = np.stack(targets), np.stack(masks)
    assert actual["valid_count"] == int(masks.sum())
    assert actual["mae"] == pytest.approx(targets[masks].mean(dtype=np.float64))
    assert actual["selection"]["samples"] == 3
    assert actual["selection"]["all_validation_samples"]
    expected_horizons = [targets[..., step][masks[..., step]].mean(dtype=np.float64) for step in range(12)]
    np.testing.assert_allclose(actual["horizon_mae"], expected_horizons)
    assert actual["average_horizon_mae"] == pytest.approx(np.mean(expected_horizons))
    assert actual["mae"] != pytest.approx(actual["average_horizon_mae"])


def test_validation_rejects_nan_even_at_unobserved_predictions(monkeypatch):
    data = tiny_data()
    def corrupted(_, tensors):
        prediction = torch.zeros_like(tensors["target"])
        prediction[~tensors["target_mask"]] = float("nan")
        return prediction
    # Exercise the independent metric finite check even if model_forward is replaced.
    monkeypatch.setattr(trainer, "model_forward", corrupted)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.evaluate(torch.nn.Linear(1, 1), data, torch.device("cpu"), microbatch_size=1)


def test_rng_restores_python_numpy_and_torch_sequences():
    trainer.configure_runtime("cpu", 42, 1)
    state = trainer.capture_rng_state()
    import random
    expected = (random.random(), np.random.rand(), torch.rand(3))
    trainer.restore_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_effective_batch_interruption_resumes_identically_and_checkpoint_verifies(tmp_path, monkeypatch):
    device = trainer.configure_runtime("cpu", 42, 1)
    continuous_data = tiny_data()
    continuous_model = tiny_model()
    config = config_for(continuous_data)
    full_output = tmp_path / "full"
    full_output.mkdir()
    trainer.train_run(model=continuous_model, data=continuous_data, device=device,
                      config=config, output=full_output)

    trainer.configure_runtime("cpu", 42, 1)
    interrupted_data = tiny_data()
    interrupted_model = tiny_model()
    output = tmp_path / "resumed"
    output.mkdir()
    # Each completed update consumes three monotonic calls. Stop after two updates.
    clock = iter(range(100))
    with monkeypatch.context() as limited:
        limited.setattr(trainer.time, "monotonic", lambda: float(next(clock)))
        stopped = trainer.train_run(model=interrupted_model, data=interrupted_data, device=device,
                                    config=config, output=output, max_wallclock_seconds=6)
    assert stopped["status"] == "stopped_resumable"
    assert stopped["next_batch"] == 2
    checkpoint = trainer.load_checkpoint(output / "last.pt")
    restored_data = tiny_data()
    restored_model = tiny_model()
    resumed = trainer.train_run(model=restored_model, data=restored_data, device=device,
                                config=config, output=output, checkpoint=checkpoint)
    assert resumed["completed_epochs"] == 2
    for name, expected in continuous_model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[name], expected, rtol=0, atol=0)
    verified = trainer.verify_checkpoint(checkpoint_path=output / "best.pt", data=restored_data,
                                         device=device, output=output,
                                         current_runtime=trainer.runtime_identity(device))
    assert verified["status"] == "verified_smoke_checkpoint_complete_validation"
    assert verified["absolute_mae_difference"] == 0
    assert not verified["test_evaluated"]
    damaged = output / "last.pt"
    with damaged.open("ab") as stream:
        stream.write(b"unexpected")
    with pytest.raises(ValueError, match="hash mismatch"):
        trainer.load_checkpoint(damaged)


def test_resume_rejects_config_or_data_changes(tmp_path):
    device = trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = tiny_model()
    config = config_for(data, epochs=1)
    output = tmp_path / "run"
    output.mkdir()
    trainer.train_run(model=model, data=data, device=device, config=config, output=output)
    checkpoint = trainer.load_checkpoint(output / "last.pt")
    optimizer, scheduler = trainer.create_optimizer(model, config, data.train.num_batch)
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer.restore_training(checkpoint, model, optimizer, scheduler, data,
                                 config | {"epochs": 2})
    data.identity["region_cache_digest"] = "different_cache"
    with pytest.raises(ValueError, match="identity mismatch"):
        trainer.restore_training(checkpoint, model, optimizer, scheduler, data, config)


def test_source_snapshot_detects_imported_source_change(tmp_path, monkeypatch):
    source = tmp_path / "ndr_research" / "routers.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(trainer, "git_worktree_provenance", lambda _: {"git_sha": "fixture"})
    snapshot = trainer.source_snapshot(tmp_path)
    trainer.require_snapshot(snapshot, tmp_path)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="source snapshot changed"):
        trainer.require_snapshot(snapshot, tmp_path)


def test_manifest_binding_rejects_actual_cache_hash_changes():
    root = Path("/Users/xhlm/Desktop/Study/科研实习/outputs/dist_net/region_data_v3_noleak_dedup")
    if not (root / "cache_bundle_manifest.json").exists():
        pytest.skip("formal local manifest is unavailable")
    manifest = json.loads((root / "cache_bundle_manifest.json").read_text())
    declaration = manifest["regions"]["Alameda"]
    region = SimpleNamespace(
        region_name="Alameda", bundle_id=manifest["cache_bundle_id"],
        cache_bundle_digest=declaration["cache_bundle_digest"],
        sample_start=np.arange(45760), split=np.asarray([0] * 32001 + [1] * 6954 + [2] * 6805),
        result_metadata=lambda: {"cache_provenance": declaration["cache_provenance"]},
    )
    validate_manifest_binding(region, manifest)
    region.cache_bundle_digest = "changed"
    with pytest.raises(ValueError, match="actual cache artifact"):
        validate_manifest_binding(region, manifest)


def test_cli_defaults_cuda_and_disallows_subset_verification(monkeypatch):
    parser = trainer.argument_parser()
    assert parser.parse_args([]).device == "cuda:0"
    assert parser.parse_args([]).batch_size == 16
    assert parser.parse_args([]).microbatch_size == 2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        trainer.configure_runtime("cuda:0", 42)
    args = parser.parse_args(["--mode", "verify", "--checkpoint", "best.pt", "--max-val-batches", "1"])
    with pytest.raises(ValueError, match="subset overrides"):
        trainer.validate_arguments(args)
    restored_args = parser.parse_args(["--mode", "train", "--resume", "last.pt", "--epochs", "50"])
    with pytest.raises(ValueError, match="frozen identity"):
        trainer.reject_identity_overrides(restored_args, {"epochs": 30},
                                          ["--mode", "train", "--resume", "last.pt", "--epochs", "50"])


def test_val_selection_signature_changes_with_subset():
    data = tiny_data()
    full = trainer.evaluation_identity(data, None)
    subset = trainer.evaluation_identity(data, 1)
    assert full["sample_indices_sha256"] == array_digest(data.val.indices)
    assert subset["samples"] == 2 and not subset["all_validation_samples"]
    assert canonical_digest(full) != canonical_digest(subset)
