"""CPU checks for the BiST-NDR plugin arm of the strong-reference trainer."""

from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ndr_research.data import ResearchData, ResearchEventLoader
from ndr_research.strong_references import (
    METHODS,
    PLUGIN_LOSS,
    StrongReference,
    StrongReferenceNDR,
    _native_bist,
    make_strong_reference,
    reference_recipe,
)
from scripts import train_ndr_strong_reference as trainer
from scripts.protocol_v3_event_extension_data import WeeklyClimatology
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_model import BiSTNDR
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_runner import backward_bist_ndr_pair_batch


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
        target = flow + 2 + np.arange(12, dtype=np.float32)
        mask = np.ones((2, 12, 3), dtype=bool)
        mask[0, :index % 4, 0] = False
        calendar = np.stack((np.arange(12, dtype=np.float32) / 288, np.full(12, 3 / 7, dtype=np.float32)), -1)
        return {"x_hist": np.repeat(flow[..., None], 3, -1), "x_hist_mask": np.ones_like(mask),
                "y_true": np.repeat(target[..., None], 3, -1), "y_mask": mask,
                "time_feat": calendar, "sample_start": self.sample_start[index]}


STATISTICS = WeeklyClimatology(baseline=np.full((7, 288, 2), 40, dtype=np.float32),
                               flow_mean=40.0, flow_std=10.0, residual_std=5.0,
                               residual_std_by_node=np.array([5.0, 6.0], dtype=np.float32),
                               valid_training_points=100)


def tiny_data():
    region = TinyRegion()
    loaders = {name: ResearchEventLoader(split=name, region_data=region, indices=np.flatnonzero(region.split == code),
               batch_size=2, flow_mean=40, flow_std=10, sensor_support=None, include_incident=False,
               shuffle=name == "train", drop_last=False, seed=42) for name, code in (("train", 0), ("val", 1))}
    return ResearchData(region, STATISTICS, loaders["train"], loaders["val"],
                        {"cache_bundle_digest": "fixture", "region_cache_digest": "fixture_region", "test_loader_constructed": False})


def plugin_model(checkout, seed=7):
    torch.manual_seed(seed)
    return make_strong_reference(method="bist_ndr", node_count=2, flow_mean=40, flow_std=10, residual_std=5.0,
                                 residual_std_by_node=[5.0, 6.0], bist_checkout=checkout)


def config_for(data, recipe="matched-ndr"):
    value = reference_recipe("bist_ndr", recipe, region="Alameda", epochs=30, batch_size=2)
    return {"epochs": 1, "batch_size": 2, "microbatch_size": 2, "training_recipe": value,
            "max_train_batches": None, "max_val_batches": None, "claim_scope": "smoke_only",
            "source_snapshot": trainer.source_snapshot(), "runtime": trainer.runtime_identity(torch.device("cpu")),
            "training_process_id": os.getpid()}


def deterministic_training_mode(model):
    """Keep the plugin in training mode for its auxiliary losses, without dropout or micro-BN noise."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def test_plugin_wrapper_is_the_verified_pair_adapter_on_the_locked_bist(bist_checkout):
    model = plugin_model(bist_checkout)
    assert "bist_ndr" in METHODS and isinstance(model, StrongReferenceNDR)
    assert isinstance(model.plugin, BiSTNDR)
    assert model.plugin.oracle_sup and model.plugin.dual and not model.plugin.abank
    assert model.upstream["commit"] == "dd94adf7721fcbb9e3feb5d1b44040305199a4cc"
    assert model.upstream["native_engine_created"] is False
    native_parameters = sum(p.numel() for p in model.plugin.native.parameters())
    total_parameters = sum(p.numel() for p in model.parameters())
    assert model.upstream["plugin"]["added_parameters"] == total_parameters - native_parameters > 0
    assert set(inspect.signature(model.forward).parameters) == {"flow_hist", "time_feat", "x_baseline", "y_baseline"}
    assert model.research_spec["output_units"] == "raw_flow"
    assert model.research_spec["target_input"] is False and model.research_spec["incident_input"] is False
    torch.testing.assert_close(model.plugin.sd_node, torch.tensor([5.0, 6.0]))
    assert float(model.plugin.sd_res) == 5.0


def test_plugin_forward_matches_a_directly_built_pair_adapter_exactly(bist_checkout):
    model = plugin_model(bist_checkout).eval()
    direct = BiSTNDR(native=_native_bist(2, bist_checkout)[0], residual_std=5.0,
                     residual_std_by_node=np.array([5.0, 6.0], dtype=np.float32)).eval()
    direct.load_state_dict(model.plugin.state_dict(), strict=True)
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    with torch.no_grad():
        expected = direct(tensors["flow_hist"], tensors["x_baseline"], tensors["y_baseline"], tensors["time_feat"])
        actual = trainer.strong_forward(model, tensors)
    assert actual.shape == (2, 2, 12) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The disturbance head is anchored on the last observation and the normal head on the climatology.
    aux = model.plugin.aux
    torch.testing.assert_close(aux["P"], tensors["flow_hist"][:, :, -1:])
    torch.testing.assert_close(aux["B"], tensors["y_baseline"])


def test_plugin_objective_and_gradients_match_the_verified_pair_runner(bist_checkout):
    reference = plugin_model(bist_checkout)
    candidate = plugin_model(bist_checkout)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    deterministic_training_mode(reference)
    deterministic_training_mode(candidate)
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    tensors["target_mask"][0, 0, :3] = False
    loss_batch = {"y_true": tensors["target"].unsqueeze(-1), "y_mask": tensors["target_mask"].unsqueeze(-1)}
    pair_loss = backward_bist_ndr_pair_batch(
        model=reference.plugin, tensors=tensors, loss_batch=loss_batch,
        loss_args=SimpleNamespace(lambda_gate=1.0, lambda_spec=0.3), device=torch.device("cpu"), max_microbatch_size=2,
    )[0]
    result = trainer.backward_reference_batch(candidate, tensors, microbatch_size=2)
    assert result["updated"] and {"mae", "gate", "spec"} <= set(result["loss_parts"])
    assert result["objective"] == pytest.approx(pair_loss, rel=1e-6)
    for left, right in zip(reference.plugin.parameters(), candidate.plugin.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)


def test_plugin_microbatching_uses_full_effective_batch_normalizers(bist_checkout):
    full = plugin_model(bist_checkout)
    split = plugin_model(bist_checkout)
    split.load_state_dict(full.state_dict(), strict=True)
    deterministic_training_mode(full)
    deterministic_training_mode(split)
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    tensors["target_mask"][1, 1, :5] = False
    whole = trainer.backward_reference_batch(full, tensors, microbatch_size=2)
    pieces = trainer.backward_reference_batch(split, tensors, microbatch_size=1)
    assert whole["objective"] == pytest.approx(pieces["objective"], rel=2e-6)
    for key in ("mae", "gate", "spec"):
        assert whole["loss_parts"][key] == pytest.approx(pieces["loss_parts"][key], rel=2e-6, abs=1e-9)
    for left, right in zip(full.parameters(), split.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-6)


def test_empty_effective_batch_leaves_plugin_untouched(bist_checkout):
    model = plugin_model(bist_checkout).train()
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    tensors["target_mask"][:] = False
    before = copy.deepcopy(model.state_dict())
    result = trainer.backward_reference_batch(model, tensors, microbatch_size=2)
    assert not result["updated"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("epochs", [30, 100])
def test_plugin_native_recipe_shares_the_locked_bist_optimizer_and_declares_the_plugin_loss(epochs):
    plugin = reference_recipe("bist_ndr", "native", region="ContraCosta", epochs=epochs)
    control = reference_recipe("bist", "native", region="ContraCosta", epochs=epochs)
    assert plugin["optimizer"] == control["optimizer"]
    assert plugin["batch_size"] == plugin["microbatch_size"] == control["batch_size"] == 64
    assert plugin["scheduler_step_unit"] == control["scheduler_step_unit"] == "completed_epoch"
    assert plugin["optimizer"]["scheduler"]["milestones"] == [30, 60, 80, 100]
    assert not plugin["early_stopping"] and plugin["requested_epoch_budget"] == epochs
    assert plugin["batch_normalization_scope"] == "effective_batch"
    assert plugin["plugin_loss"]["lambda_gate"] == PLUGIN_LOSS["lambda_gate"] == 1.0
    assert plugin["plugin_loss"]["lambda_specialization"] == PLUGIN_LOSS["lambda_specialization"] == 0.3
    assert len(plugin["plugin_loss"]["loss_source_sha256"]) == 64
    assert control["plugin_loss"] is None
    assert plugin["comparison_role"] == "plugin_candidate" and control["comparison_role"] == "backbone_control"
    assert plugin["optimizer_authority"] == "locked_bist_public_material_profile_shared_by_bist_ndr_plugin"
    matched = reference_recipe("bist_ndr", "matched-ndr", region="ContraCosta", epochs=epochs)
    assert matched["batch_size"] == 16 and matched["optimizer"]["scheduler"]["name"] == "CosineAnnealingLR"
    assert matched["plugin_loss"]["lambda_gate"] == 1.0


def test_plugin_checkpoint_restores_model_optimizer_scheduler_cursor_and_rng_exactly(tmp_path, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = plugin_model(bist_checkout)
    config = config_for(data)
    optimizer, scheduler = trainer.create_reference_optimizer(model, config, data.train.num_batch)
    data.train.start_epoch()
    first = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    trainer.optimize_reference(model, optimizer, scheduler, first, config)
    path = tmp_path / "last.pt"
    progress = {"epoch": 0, "stage": "train", "updates": 1}
    trainer.save_reference_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress)
    checkpoint = trainer.load_reference_checkpoint(path)
    assert checkpoint["model_factory"]["method"] == "bist_ndr"
    assert checkpoint["model_factory"]["residual_std_by_node"] == [5.0, 6.0]
    next_original = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    trainer.optimize_reference(model, optimizer, scheduler, next_original, config)
    expected = copy.deepcopy(model.state_dict())
    restored_data = tiny_data()
    restored = make_strong_reference(**(checkpoint["model_factory"] | {"bist_checkout": bist_checkout, "device": "cpu"}))
    trainer._require_model_identity(restored, checkpoint)
    restored_optimizer, restored_scheduler = trainer.create_reference_optimizer(restored, config, restored_data.train.num_batch)
    assert trainer.restore_reference(checkpoint, restored, restored_optimizer, restored_scheduler, restored_data, config) == progress
    next_restored = trainer.batch_tensors(restored_data.train.next_batch(), restored_data, torch.device("cpu"))
    trainer.optimize_reference(restored, restored_optimizer, restored_scheduler, next_restored, config)
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_plugin_validation_and_calibration_paths_run_on_cpu_without_scores_in_calibration(tmp_path, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = plugin_model(bist_checkout).eval()
    output = trainer.evaluate_reference(model, data, torch.device("cpu"), microbatch_size=1)
    count = sum(data.region.get_sample(int(index))["y_mask"][..., 0].sum() for index in data.val.indices)
    assert output["valid_count"] == count and output["selection"]["all_validation_samples"] is True
    config = config_for(tiny_data())
    result = trainer.calibrate_reference(model=plugin_model(bist_checkout), data=tiny_data(), device=torch.device("cpu"),
                                        config=config, warmup_batches=1, timed_batches=2, output=tmp_path)
    assert result["status"] == "calibrated_cpu_engineering_only" and result["cuda_budget_eligible"] is False
    assert result["phases"]["train"]["timed_effective_batch_sizes"] == [2, 2]
    assert not any("mae" in key for key in result)


def test_plugin_rejects_invalid_scales_and_native_controls_reject_the_plugin_method(bist_checkout):
    with pytest.raises(ValueError, match="native model"):
        StrongReference(method="bist_ndr", node_count=2, flow_mean=40, flow_std=10, bist_checkout=bist_checkout)
    with pytest.raises(ValueError, match="node residual scales"):
        StrongReferenceNDR(method="bist_ndr", node_count=2, flow_mean=40, flow_std=10, residual_std=5.0,
                           residual_std_by_node=[5.0], bist_checkout=bist_checkout)
    with pytest.raises(ValueError, match="positive scales"):
        StrongReferenceNDR(method="bist_ndr", node_count=2, flow_mean=40, flow_std=10, residual_std=0.0,
                           residual_std_by_node=[5.0, 6.0], bist_checkout=bist_checkout)
    with pytest.raises(ValueError, match="bist_ndr"):
        StrongReferenceNDR(method="bist", node_count=2, flow_mean=40, flow_std=10, residual_std=5.0,
                           residual_std_by_node=[5.0, 6.0], bist_checkout=bist_checkout)


def test_cli_accepts_the_plugin_method_and_the_snapshot_declares_its_sources():
    args = trainer.argument_parser().parse_args(["--method", "bist_ndr", "--recipe", "native", "--epochs", "100"])
    assert args.method == "bist_ndr" and args.epochs == 100
    trainer.validate_arguments(args)
    files = trainer.source_snapshot()["files"]
    for relative in ("scripts/reproduction/igstgnn_table4/bist_ndr_pair_model.py",
                     "scripts/train_staeformer_deseason.py", "baselines/GraphWaveNet/model.py",
                     "ndr_research/strong_references.py"):
        assert relative in files
