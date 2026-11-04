from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest
import torch

from baselines.STAEformer.STAEformer import STAEformer
from ndr_research.data import ResearchData, ResearchEventLoader
from ndr_research.strong_references import STAE_MODEL_CONFIG, make_strong_reference, reference_recipe
from scripts import train_ndr_strong_reference as trainer
from scripts.protocol_v3_event_extension_data import WeeklyClimatology
from scripts.reproduction.igstgnn_table4 import bist_factory


@pytest.fixture
def bist_checkout():
    candidates = [Path(os.environ.get("XTRAFFIC_BIST_TEST_CHECKOUT", "/nonexistent")),
                  Path(__file__).resolve().parents[2] / ".third_party/igstgnn_table4/pristine/bist_dd94adf",
                  Path("/Users/xhlm/Desktop/Study/科研实习/.third_party/igstgnn_table4/pristine/bist_dd94adf")]
    for path in candidates:
        if (path / "src/models/bist.py").is_file():
            return path
    pytest.skip("locked pristine BiST checkout is not installed for this environment")


class TinyRegion:
    N, T_h, T_p = 2, 12, 12
    train_end, val_end = 50, 100
    include_analysis_labels = False
    region_name = "Alameda"
    split = np.array([0] * 4 + [1] * 3 + [2], dtype=np.int8)
    sample_start = np.array([12, 14, 16, 18, 65, 67, 69, 115], dtype=np.int64)

    def get_sample(self, index):
        if self.split[index] == 2:
            raise AssertionError("strong reference attempted a test read")
        flow = np.arange(24, dtype=np.float32).reshape(2, 12) + 30 + index
        target = flow + 2
        mask = np.ones((2, 12, 3), dtype=bool)
        mask[0, :index % 4, 0] = False
        calendar = np.stack((np.arange(12, dtype=np.float32) / 288, np.full(12, 3 / 7, dtype=np.float32)), -1)
        return {"x_hist": np.repeat(flow[..., None], 3, -1), "x_hist_mask": np.ones_like(mask),
                "y_true": np.repeat(target[..., None], 3, -1), "y_mask": mask,
                "time_feat": calendar, "sample_start": self.sample_start[index]}


def tiny_data():
    region = TinyRegion()
    statistics = WeeklyClimatology(baseline=np.full((7, 288, 2), 40, dtype=np.float32),
                                  flow_mean=40.0, flow_std=10.0, residual_std=5.0,
                                  residual_std_by_node=np.full(2, 5, dtype=np.float32), valid_training_points=100)
    loaders = {name: ResearchEventLoader(split=name, region_data=region, indices=np.flatnonzero(region.split == code),
               batch_size=2, flow_mean=40, flow_std=10, sensor_support=None, include_incident=False,
               shuffle=name == "train", drop_last=False, seed=42) for name, code in (("train", 0), ("val", 1))}
    return ResearchData(region, statistics, loaders["train"], loaders["val"], {"cache_bundle_digest": "fixture", "region_cache_digest": "fixture_region", "test_loader_constructed": False})


def model_for(method, checkout):
    return make_strong_reference(method=method, node_count=2, flow_mean=40, flow_std=10, bist_checkout=checkout)


def config_for(data, method="staeformer"):
    recipe = reference_recipe(method, "matched-ndr", region="Alameda", epochs=30, batch_size=2)
    return {"epochs": 1, "batch_size": 2, "microbatch_size": 2, "training_recipe": recipe,
            "max_train_batches": None, "max_val_batches": None, "claim_scope": "smoke_only",
            "source_snapshot": trainer.source_snapshot(), "runtime": trainer.runtime_identity(torch.device("cpu")),
            "training_process_id": os.getpid()}


@pytest.mark.parametrize("method", ["staeformer", "bist"])
def test_native_calendar_and_normalization_are_preserved(method, bist_checkout):
    model = model_for(method, bist_checkout)
    history = torch.full((2, 2, 12), 50.0)
    calendar = torch.zeros(2, 12, 2)
    calendar[..., 0] = 80 / 288
    calendar[..., 1] = 3 / 7
    inputs = model.build_inputs(history, calendar)
    torch.testing.assert_close(inputs[..., 0], torch.ones(2, 12, 2))
    torch.testing.assert_close(inputs[..., 1], torch.full((2, 12, 2), 80 / 288))
    torch.testing.assert_close(inputs[..., 2], torch.full((2, 12, 2), 3.0 if method == "staeformer" else 3 / 7))


@pytest.mark.parametrize("method", ["staeformer", "bist"])
def test_real_native_controls_return_finite_raw_flow_predictions(method, bist_checkout):
    model = model_for(method, bist_checkout).eval()
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    with torch.no_grad():
        output = trainer.strong_forward(model, tensors)
    assert output.shape == (2, 2, 12) and torch.isfinite(output).all()
    assert model.research_spec["output_units"] == "raw_flow"


def test_raw_staeformer_matches_published_native_model_exactly():
    torch.manual_seed(9)
    model = model_for("staeformer", None).eval()
    torch.manual_seed(9)
    native = STAEformer(num_nodes=2, **STAE_MODEL_CONFIG).eval()
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    with torch.no_grad():
        expected = native(model.build_inputs(tensors["flow_hist"], tensors["time_feat"])).squeeze(-1).permute(0, 2, 1) * 10 + 40
        actual = trainer.strong_forward(model, tensors)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bist_is_locked_pristine_native_source_without_engine_or_test_loader(bist_checkout):
    model = model_for("bist", bist_checkout)
    assert model.upstream["commit"] == "dd94adf7721fcbb9e3feb5d1b44040305199a4cc"
    assert "src/models/bist.py" in model.upstream["python_files_sha256"]
    assert model.upstream["native_engine_created"] is False
    assert model.upstream["test_loader_created"] is False
    assert isinstance(model.native.backcast.norm, torch.nn.BatchNorm2d)


def test_bist_import_does_not_modify_factory_globals_or_runtime_policy(bist_checkout):
    sentinel = ModuleType("src")
    previous = sys.modules.get("src")
    sys.modules["src"] = sentinel
    original_checkout, original_upstream = bist_factory.BIST_CHECKOUT, bist_factory._UPSTREAM
    threads, benchmark, deterministic = torch.get_num_threads(), torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled()
    try:
        model_for("bist", bist_checkout)
        assert sys.modules["src"] is sentinel
        assert bist_factory.BIST_CHECKOUT == original_checkout
        assert bist_factory._UPSTREAM is original_upstream
        assert (torch.get_num_threads(), torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled()) == (threads, benchmark, deterministic)
    finally:
        if previous is None:
            sys.modules.pop("src", None)
        else:
            sys.modules["src"] = previous


@pytest.mark.parametrize("method", ["staeformer", "bist"])
def test_forward_never_accepts_targets_incidents_or_affected(method, bist_checkout):
    model = model_for(method, bist_checkout)
    assert set(inspect.signature(model.forward).parameters) == {"flow_hist", "time_feat"}
    with pytest.raises(TypeError):
        model(torch.ones(2, 2, 12), torch.zeros(2, 12, 2), target=torch.ones(2, 2, 12))


@pytest.mark.parametrize("method,recipe,epochs", [("bist", "native", 30), ("bist", "native", 100), ("bist", "matched-ndr", 30), ("staeformer", "native", 30), ("staeformer", "matched-ndr", 100)])
def test_optimizer_and_epoch_budget_differences_are_explicit(method, recipe, epochs):
    value = reference_recipe(method, recipe, region="Alameda", epochs=epochs)
    if method == "bist" and recipe == "native":
        assert value["batch_size"] == value["microbatch_size"] == 64
        assert value["optimizer"]["learning_rate"] == 0.002
        assert value["optimizer"]["scheduler"]["milestones"] == [30, 60, 80, 100]
        assert value["scheduler_step_unit"] == "completed_epoch"
    else:
        assert value["batch_size"] == 16
        assert value["optimizer"]["scheduler"]["name"] == "CosineAnnealingLR"
    assert not value["early_stopping"]
    assert value["requested_epoch_budget"] == epochs
    assert value["partial_batch_validation"] == "retain"


@pytest.mark.parametrize("method", ["staeformer", "bist"])
def test_microbatch_loss_uses_complete_effective_valid_point_denominator(method, bist_checkout):
    model = model_for(method, bist_checkout).eval()
    tensors = trainer.batch_tensors(tiny_data().train.next_batch(), tiny_data(), torch.device("cpu"))
    tensors["target_mask"][0] = False
    micro = copy.deepcopy(model)
    # Native BiST BN is held in evaluation mode here; training micro-BN is an explicit override.
    full = trainer.backward_reference_batch(model, tensors, microbatch_size=2)
    split = trainer.backward_reference_batch(micro, tensors, microbatch_size=1)
    assert full["objective"] == pytest.approx(split["objective"], rel=2e-6)
    for left, right in zip(model.parameters(), micro.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-6)


def test_empty_effective_batch_does_not_update_model_statistics(bist_checkout):
    model = model_for("bist", bist_checkout).train()
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    tensors["target_mask"][:] = False
    before = copy.deepcopy(model.state_dict())
    result = trainer.backward_reference_batch(model, tensors, microbatch_size=2)
    assert not result["updated"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_validation_covers_every_observed_point_and_retains_tail():
    model = model_for("staeformer", None).eval()
    data = tiny_data()
    output = trainer.evaluate_reference(model, data, torch.device("cpu"), microbatch_size=1)
    count = sum(data.region.get_sample(int(index))["y_mask"][..., 0].sum() for index in data.val.indices)
    assert output["valid_count"] == count
    assert output["selection"]["samples"] == 3
    assert output["selection"]["all_validation_samples"] is True
    assert len(output["horizon_mae"]) == 12


@pytest.mark.parametrize("method", ["staeformer", "bist"])
def test_checkpoint_restores_model_optimizer_scheduler_cursor_and_rng_exactly(tmp_path, method, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = model_for(method, bist_checkout)
    config = config_for(data, method)
    optimizer, scheduler = trainer.create_reference_optimizer(model, config, data.train.num_batch)
    data.train.start_epoch()
    first = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    trainer.optimize_reference(model, optimizer, scheduler, first, config)
    path = tmp_path / "last.pt"
    progress = {"epoch": 0, "stage": "train", "updates": 1}
    trainer.save_reference_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress)
    checkpoint = trainer.load_reference_checkpoint(path)
    next_original = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    trainer.optimize_reference(model, optimizer, scheduler, next_original, config)
    expected = copy.deepcopy(model.state_dict())
    restored_data = tiny_data()
    restored = model_for(method, bist_checkout)
    restored_optimizer, restored_scheduler = trainer.create_reference_optimizer(restored, config, restored_data.train.num_batch)
    assert trainer.restore_reference(checkpoint, restored, restored_optimizer, restored_scheduler, restored_data, config) == progress
    next_restored = trainer.batch_tensors(restored_data.train.next_batch(), restored_data, torch.device("cpu"))
    trainer.optimize_reference(restored, restored_optimizer, restored_scheduler, next_restored, config)
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_hash_and_recipe_tampering_are_rejected(tmp_path):
    data = tiny_data()
    model = model_for("staeformer", None)
    config = config_for(data)
    optimizer, scheduler = trainer.create_reference_optimizer(model, config, data.train.num_batch)
    path = tmp_path / "last.pt"
    trainer.save_reference_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress={"epoch": 0})
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="byte identity"):
        trainer.load_reference_checkpoint(path)


def test_cpu_or_subset_execution_cannot_be_mislabeled_formal():
    parser = trainer.argument_parser()
    with pytest.raises(ValueError, match="CPU"):
        trainer.validate_arguments(parser.parse_args(["--device", "cpu"]))
    with pytest.raises(ValueError, match="smoke"):
        trainer.validate_arguments(parser.parse_args(["--max-train-batches", "1"]))


def test_microbatch_batchnorm_override_is_recorded_without_changing_native_optimizer():
    recipe = reference_recipe("bist", "native", region="Alameda", epochs=30, microbatch_size=8)
    assert recipe["batch_size"] == 64 and recipe["microbatch_size"] == 8
    assert recipe["batch_normalization_scope"] == "microbatch_override"
    assert recipe["optimizer"]["learning_rate"] == 0.002


def test_calibration_excludes_warmup_times_and_synchronizes_complete_cuda_batches(tmp_path, monkeypatch):
    data = tiny_data()
    model = torch.nn.Linear(1, 1)
    config = config_for(data)
    config["requested_epochs"] = 30
    clock = iter([0, 100, 100, 110, 110, 113, 113, 118,
                  200, 280, 280, 287, 287, 289, 289, 293])
    synchronizations, reset, optimizer_sizes, validation_sizes = [], [], [], []
    monkeypatch.setattr(trainer.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronizations.append(str(device)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: reset.append(str(device)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 456)
    original_tensors = trainer.batch_tensors
    monkeypatch.setattr(trainer, "batch_tensors", lambda batch, data, device: original_tensors(batch, data, torch.device("cpu")))

    def optimization(model, optimizer, scheduler, tensors, config):
        optimizer_sizes.append(tensors["flow_hist"].shape[0])
        return {"valid_count": int(tensors["target_mask"].sum()), "updated": True}

    def validation(model, tensors, microbatch_size):
        validation_sizes.append(tensors["flow_hist"].shape[0])
        return int(tensors["target_mask"].sum())

    monkeypatch.setattr(trainer, "optimize_reference", optimization)
    monkeypatch.setattr(trainer, "_calibration_validation_batch", validation)
    result = trainer.calibrate_reference(model=model, data=data, device=torch.device("cuda:0"),
                                        config=config, warmup_batches=2, timed_batches=2, output=tmp_path)
    assert optimizer_sizes == validation_sizes == [2, 2, 2, 2]
    assert len(synchronizations) == 18
    assert reset == ["cuda:0", "cuda:0"]
    assert result["phases"]["train"]["timed_batch_seconds"] == [3, 5]
    assert result["phases"]["val"]["timed_batch_seconds"] == [2, 4]
    assert result["mean_train_batch_seconds"] == 4
    assert result["mean_val_batch_seconds"] == 3
    assert result["p90_train_batch_seconds"] == pytest.approx(4.8)
    assert result["p90_val_batch_seconds"] == pytest.approx(3.8)
    assert result["estimated_train_val_seconds"] == 30 * (4 * 2 + 3 * 2)
    assert result["train_peak_allocated_bytes"] == result["val_peak_allocated_bytes"] == 123
    assert result["train_peak_reserved_bytes"] == result["val_peak_reserved_bytes"] == 456
    assert result["prediction_error_used_for_configuration_selection"] is False
    assert json.loads((tmp_path / "calibration.json").read_text())["timed_batches"] == 2


def test_cpu_calibration_preserves_full_bist_batchnorm_and_cannot_supply_cuda_budget(tmp_path, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = model_for("bist", bist_checkout)
    config = config_for(data, "bist")
    seen = []

    def capture(module, args):
        seen.append((module.training, args[0].shape[0]))

    handle = model.native.backcast.norm.register_forward_pre_hook(capture)
    result = trainer.calibrate_reference(model=model, data=data, device=torch.device("cpu"), config=config,
                                        warmup_batches=1, timed_batches=2, output=tmp_path)
    handle.remove()
    assert seen == [(True, 2)] * 3 + [(False, 2)] * 3
    assert result["status"] == "calibrated_cpu_engineering_only"
    assert result["cuda_budget_eligible"] is False
    assert result["batch_normalization_scope"] == "effective_batch"
    assert result["phases"]["val"]["skipped_retained_tails"] == 2
    assert result["phases"]["train"]["timed_effective_batch_sizes"] == [2, 2]
    assert result["train_peak_allocated_bytes"] is None
    assert not any("mae" in key for key in result)


def test_calibration_cli_defaults_and_invalid_timing_counts():
    parser = trainer.argument_parser()
    args = parser.parse_args([])
    assert args.warmup_batches == 3 and args.timed_batches == 20
    with pytest.raises(ValueError, match="warmup"):
        trainer.validate_arguments(parser.parse_args(["--warmup-batches", "-1"]))
    with pytest.raises(ValueError, match="timed"):
        trainer.validate_arguments(parser.parse_args(["--timed-batches", "0"]))
    with pytest.raises(ValueError, match="CPU"):
        trainer.validate_arguments(parser.parse_args(["--mode", "calibrate", "--device", "cpu"]))


def test_strong_snapshot_contains_dependencies_and_excludes_unrelated_active_research_files(tmp_path):
    scripts = tmp_path / "scripts"
    package = tmp_path / "ndr_research"
    scripts.mkdir()
    package.mkdir()
    (scripts / "train_ndr_strong_reference.py").write_text("from ndr_research.data import VALUE\n")
    (package / "strong_references.py").write_text("from ndr_research.data import VALUE\n")
    (package / "data.py").write_text("VALUE = 1\n")
    unrelated = package / "transfer_evaluation.py"
    unrelated.write_text("EDIT_IN_PROGRESS = 1\n")
    first = trainer.source_snapshot(tmp_path)
    unrelated.write_text("EDIT_IN_PROGRESS = 2\n")
    second = trainer.source_snapshot(tmp_path)
    assert first["files_digest"] == second["files_digest"]
    assert "ndr_research/data.py" in first["files"]
    assert "ndr_research/transfer_evaluation.py" not in first["files"]
