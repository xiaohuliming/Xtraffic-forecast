from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ndr_research.data import ResearchData, ResearchEventLoader
from ndr_research.dualcast_training import (
    RECIPES, UndefinedAuthorAuxiliary, compute_dualcast_loss, dualcast_recipe,
    load_author_auxiliary, locked_author_identity, make_dualcast_model, regional_adjacency,
)
from scripts import train_dualcast_v3 as trainer
from scripts.protocol_v3_event_extension_data import WeeklyClimatology


@pytest.fixture
def checkout():
    path = Path(os.environ.get("DUALCAST_TEST_CHECKOUT", "/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013/references/DualCast"))
    if not (path / "DualCastS/model/train.py").is_file():
        pytest.skip("locked ignored DualCast reference is not installed")
    return path


def fake_model(checkout, batch=3, *, one_group=False, equal_environment=False, equal_anchors=False):
    intrinsic = torch.tensor([[2.0, 1.0, 0.0], [0.3, 2.0, 1.0], [1.0, 0.0, 2.0]])[:batch].clone().requires_grad_(True)
    environment = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 0.1], [2.0, 0.1, 1.0]])[:batch].clone().requires_grad_(True)
    if equal_environment:
        environment = torch.zeros_like(environment, requires_grad=True)
    anchors = torch.arange(17, dtype=torch.float32).reshape(17, 1, 1, 1).requires_grad_(True)
    if equal_anchors:
        anchors = torch.zeros_like(anchors, requires_grad=True)
    members = [[] for _ in range(17)]
    members[0] = [torch.tensor([[[0.5]]])]
    if not one_group:
        members[1] = [torch.tensor([[[1.5]]])]
    return SimpleNamespace(author_loss=load_author_auxiliary(checkout), last_aux={
        "intrinsic": intrinsic, "environment": environment, "anchors": anchors,
        "prototype_members": members,
    })


def scalar_prediction(batch):
    value = torch.tensor(0.2, requires_grad=True)
    prediction = value.expand(batch, 1, 1)
    return value, prediction, torch.zeros_like(prediction), torch.ones_like(prediction, dtype=torch.bool)


def test_defined_auxiliary_values_match_the_author_reciprocal_equations_exactly(checkout, monkeypatch):
    model = fake_model(checkout)
    permutation = torch.tensor([1, 2, 0])
    monkeypatch.setattr(torch, "randperm", lambda size: permutation)
    _, prediction, target, mask = scalar_prediction(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss, parts = compute_dualcast_loss(model, prediction, target, mask, dualcast_recipe(RECIPES[0], batch_size=3, epochs=30))
    aux = model.last_aux
    filter_kl = F.kl_div(F.log_softmax(aux["intrinsic"], -1), F.softmax(aux["environment"], -1), reduction="batchmean")
    environment_kl = F.kl_div(F.log_softmax(aux["environment"][permutation], -1), F.softmax(aux["environment"], -1), reduction="batchmean")
    expected = torch.tensor(0.2) + 1 / filter_kl + 0.01 / environment_kl + 0.05
    torch.testing.assert_close(loss, expected)
    assert parts["omitted_undefined_auxiliary"] == {}
    assert parts["dbi"] == pytest.approx(0.05)
    assert parts["dbi_requires_grad"] is False


def test_dbi_preserves_frobenius_norm_divided_by_group_size_and_detached_gradient(checkout):
    function = load_author_auxiliary(checkout)
    anchors = torch.tensor([0.0, 3.0]).reshape(2, 1).requires_grad_(True)
    groups = [[torch.tensor([1.0]), torch.tensor([1.0])], [torch.tensor([4.0])]]
    features = torch.zeros(2, 3, requires_grad=True)
    args = SimpleNamespace(l_flt=False, l_env=False, l_dbi=True, gamma=1.0, num_prototypes=2)
    prediction = torch.tensor(2.0, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss, parts = function(prediction * 1, features, features, groups, anchors, args, torch.nn.KLDivLoss(reduction="batchmean"), torch)
    expected = ((2 ** 0.5) / 2 + 1) / 3
    assert float(parts["dbi"]) == pytest.approx(expected)
    assert not parts["dbi"].requires_grad
    loss.backward()
    assert anchors.grad is None
    assert prediction.grad.item() == 1


def test_native_equations_fail_closed_on_singleton_environment_and_single_prototype(checkout):
    model = fake_model(checkout, batch=1, one_group=True)
    _, prediction, target, mask = scalar_prediction(1)
    with pytest.raises(UndefinedAuthorAuxiliary, match="identical_distributions_zero_KL"):
        compute_dualcast_loss(model, prediction, target, mask, dualcast_recipe(RECIPES[0], batch_size=1, epochs=30))


def test_named_compatibility_retains_singleton_main_and_only_defined_auxiliary_terms(checkout):
    model = fake_model(checkout, batch=1, one_group=True)
    value, prediction, target, mask = scalar_prediction(1)
    loss, parts = compute_dualcast_loss(model, prediction, target, mask, dualcast_recipe(RECIPES[1], batch_size=1, epochs=30))
    assert torch.isfinite(loss)
    assert parts["main"] == pytest.approx(0.2)
    assert parts["environment"] == parts["dbi"] == 0
    assert set(parts["omitted_undefined_auxiliary"]) == {"environment", "dbi"}
    loss.backward()
    assert value.grad.item() == 1


def test_identical_environment_distributions_are_not_treated_as_defined_roundoff_KL(checkout):
    model = fake_model(checkout, batch=2, equal_environment=True)
    _, prediction, target, mask = scalar_prediction(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss, parts = compute_dualcast_loss(model, prediction, target, mask, dualcast_recipe(RECIPES[1], batch_size=2, epochs=30))
    assert torch.isfinite(loss)
    assert parts["omitted_undefined_auxiliary"]["environment"] == "identical_distributions_zero_KL"
    assert parts["environment"] == 0


def test_zero_between_anchor_distance_is_explicitly_undefined_not_replaced_by_epsilon(checkout):
    model = fake_model(checkout, equal_anchors=True)
    _, prediction, target, mask = scalar_prediction(3)
    loss, parts = compute_dualcast_loss(model, prediction, target, mask, dualcast_recipe(RECIPES[1], batch_size=3, epochs=30))
    assert torch.isfinite(loss)
    assert parts["omitted_undefined_auxiliary"]["dbi"] == "zero_or_nonfinite_between_anchor_distance"
    assert parts["dbi"] == 0


def test_environment_permutation_consumes_exactly_one_author_rng_draw(checkout):
    model = fake_model(checkout)
    _, prediction, target, mask = scalar_prediction(3)
    torch.manual_seed(17)
    state = torch.get_rng_state()
    expected = torch.randperm(3)
    next_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, parts = compute_dualcast_loss(model, prediction, target, mask, dualcast_recipe(RECIPES[1], batch_size=3, epochs=30))
    assert parts["environment_permutation"] == expected.tolist()
    assert torch.equal(next_rng, torch.get_rng_state())


def test_masked_primary_includes_real_zero_and_ignores_missing_target_sentinels(checkout):
    model = fake_model(checkout, batch=1, one_group=True, equal_environment=True)
    model.last_aux["intrinsic"] = model.last_aux["environment"]
    prediction = torch.tensor([[[2.0], [1000.0]]], requires_grad=True)
    target = torch.tensor([[[0.0], [float("nan")]]])
    mask = torch.tensor([[[True], [False]]])
    loss, parts = compute_dualcast_loss(model, prediction, target, mask, dualcast_recipe(RECIPES[1], batch_size=1, epochs=30))
    assert float(loss) == parts["main"] == 2.0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[1.0], [0.0]]]))


def test_relevant_author_sources_are_hash_locked_before_auxiliary_execution(checkout, tmp_path, monkeypatch):
    import shutil
    import ndr_research.dualcast_training as module
    from ndr_research.dualcast_adapter import UPSTREAM_COMMIT

    for relative in module.AUTHOR_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / relative, destination)
    with (tmp_path / "DualCastS/model/train.py").open("a") as handle:
        handle.write("\n# changed auxiliary source\n")
    monkeypatch.setattr(module.subprocess, "check_output", lambda args, **kwargs: "" if "status" in args else UPSTREAM_COMMIT)
    with pytest.raises(ValueError, match="train.py"):
        locked_author_identity(tmp_path)


class TinyRegion:
    N, T_h, T_p = 8, 12, 12
    region_name = "Alameda"
    include_analysis_labels = False
    train_end, val_end = 80, 150
    split = np.array([0] * 6 + [1] * 3 + [2], dtype=np.int8)
    sample_start = np.array([12, 14, 16, 18, 20, 22, 95, 97, 99, 170], dtype=np.int64)
    edge_index = np.stack((np.arange(8), np.roll(np.arange(8), -1)))

    def get_sample(self, index):
        if self.split[index] == 2:
            raise AssertionError("DualCast engineering attempted a test observation")
        flow = np.arange(96, dtype=np.float32).reshape(8, 12) / 10 + 40 + index
        mask = np.ones((8, 12, 3), dtype=bool)
        mask[0, :index % 5, 0] = False
        calendar = np.stack((np.full(12, 80 / 288, dtype=np.float32), np.full(12, (index % 7) / 7, dtype=np.float32)), -1)
        return {"x_hist": np.repeat(flow[..., None], 3, -1), "x_hist_mask": np.ones_like(mask),
                "y_true": np.repeat((flow + 1)[..., None], 3, -1), "y_mask": mask,
                "time_feat": calendar, "sample_start": self.sample_start[index]}


def tiny_data():
    region = TinyRegion()
    stats = WeeklyClimatology(baseline=np.full((7, 288, 8), 40, dtype=np.float32), flow_mean=40.0, flow_std=10.0,
                              residual_std=5.0, residual_std_by_node=np.full(8, 5, dtype=np.float32), valid_training_points=100)
    loaders = {name: ResearchEventLoader(split=name, region_data=region, indices=np.flatnonzero(region.split == code),
               batch_size=2, flow_mean=40, flow_std=10, sensor_support=None, include_incident=False,
               shuffle=name == "train", drop_last=False, seed=42) for name, code in (("train", 0), ("val", 1))}
    return ResearchData(region, stats, loaders["train"], loaders["val"], {"cache_bundle_digest": "synthetic", "region_cache_digest": "synthetic-region", "test_loader_constructed": False})


def tiny_model(checkout):
    return make_dualcast_model(checkout=checkout, adjacency=regional_adjacency(TinyRegion()), flow_mean=40.0, flow_std=10.0, embed_dim=16, heads=4)


def tiny_config():
    return {"epochs": 1, "requested_epochs": 30, "batch_size": 2, "training_recipe": dualcast_recipe(RECIPES[1], batch_size=2, epochs=30),
            "max_train_batches": None, "max_val_batches": None, "source_snapshot": trainer.source_snapshot(),
            "claim_scope": "smoke_only", "runtime": trainer.runtime_identity(torch.device("cpu")), "training_process_id": os.getpid()}


def test_real_locked_author_forward_has_no_target_or_affected_interface(checkout):
    model = tiny_model(checkout)
    assert set(inspect.signature(model.forward).parameters) == {"flow_hist", "time_feat"}
    assert model.upstream["author_files_digest"] == locked_author_identity(checkout)["author_files_digest"]
    assert model.upstream["adapter_provenance"]["model_config"]["num_prototypes"] == 17


def test_full_effective_author_objective_backpropagates_finite_gradients(checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data, model, config = tiny_data(), tiny_model(checkout), tiny_config()
    optimizer, _ = trainer.create_optimizer(model, config)
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = trainer.optimize_batch(model, optimizer, tensors, config)
    assert result["updated"]
    assert np.isfinite(result["objective"])
    assert result["dbi_requires_grad"] is False
    assert model.base.conv1.weight.grad is not None
    assert torch.isfinite(model.base.conv1.weight.grad).all()


def test_complete_checkpoint_restores_auxiliary_rng_optimizer_loader_and_parameters(checkout, tmp_path):
    trainer.configure_runtime("cpu", 42, 1)
    data, model, config = tiny_data(), tiny_model(checkout), tiny_config()
    optimizer, scheduler = trainer.create_optimizer(model, config)
    data.train.start_epoch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trainer.optimize_batch(model, optimizer, trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu")), config)
    path = tmp_path / "last.pt"
    trainer.save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress={"epoch": 0, "stage": "train"})
    checkpoint = trainer.load_checkpoint(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected_step = trainer.optimize_batch(model, optimizer, trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu")), config)
    expected = copy.deepcopy(model.state_dict())
    restored_data, restored = tiny_data(), tiny_model(checkout)
    restored_optimizer, restored_scheduler = trainer.create_optimizer(restored, config)
    trainer.restore_reference(checkpoint, restored, restored_optimizer, restored_scheduler, restored_data, config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        restored_step = trainer.optimize_batch(restored, restored_optimizer, trainer.batch_tensors(restored_data.train.next_batch(), restored_data, torch.device("cpu")), config)
    assert expected_step["environment_permutation"] == restored_step["environment_permutation"]
    assert expected_step["omitted_undefined_auxiliary"] == restored_step["omitted_undefined_auxiliary"]
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_validation_retains_tail_without_evaluating_batch_undefined_auxiliary(checkout):
    data, model = tiny_data(), tiny_model(checkout)
    result = trainer.evaluate_reference(model, data, torch.device("cpu"), microbatch_size=2)
    assert result["selection"]["samples"] == 3
    assert result["selection"]["all_validation_samples"]
    assert len(result["horizon_valid_counts"]) == 12


def test_cpu_and_subset_cannot_be_classified_as_formal_dualcast_run(checkout):
    with pytest.raises(ValueError, match="CPU"):
        trainer.validate(trainer.parser().parse_args(["--checkout", str(checkout), "--device", "cpu"]))
    with pytest.raises(ValueError, match="smoke"):
        trainer.validate(trainer.parser().parse_args(["--checkout", str(checkout), "--max-train-batches", "1"]))


def test_named_recipes_explicitly_disclaim_exact_native_and_preserve_optimizer():
    for name in RECIPES:
        recipe = dualcast_recipe(name)
        assert recipe["exact_native"] is False
        assert recipe["kl_epsilon_added"] is False
        assert recipe["alternative_auxiliary_loss_substituted"] is False
        assert recipe["optimizer"]["learning_rate"] == 0.001
        assert recipe["optimizer"]["weight_decay"] == 0
        assert recipe["scheduler"] == {"name": "StepLR", "step_size": 10, "gamma": 0.9, "step_unit": "completed_epoch"}
        assert recipe["microbatch_size"] == recipe["batch_size"] == 32


def test_calibration_exercises_complete_auxiliary_objective_and_keeps_cpu_ineligible(checkout, tmp_path):
    trainer.configure_runtime("cpu", 42, 1)
    data, model, config = tiny_data(), tiny_model(checkout), tiny_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = trainer.calibrate(model=model, data=data, device=torch.device("cpu"), config=config,
                                   output=tmp_path, warmup=1, timed=2)
    assert result["status"] == "calibrated_cpu_engineering_only"
    assert result["cuda_budget_eligible"] is False
    assert result["full_effective_batch"] == 2
    assert len(result["phases"]["train"]["seconds"]) == 2
    assert len(result["phases"]["val"]["seconds"]) == 2
    assert result["phases"]["train"]["peak_allocated_bytes"] is None
    assert result["prediction_error_used_for_configuration_selection"] is False
    assert result["exact_native_claim"] is False
