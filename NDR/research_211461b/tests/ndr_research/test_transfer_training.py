from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ndr_research.data import canonical_digest
from ndr_research.graph_loss import (
    build_graph_loss_normalizers,
    compute_graph_loss,
    graph_oracle_teacher,
)
from ndr_research.shared_graph import SharedGraphNDR
from ndr_research.transfer_data import TransferBatchLoader, _array_hash, _json_hash, load_transfer_sources
from scripts import train_ndr_transfer as trainer
from scripts.train_staeformer_deseason import compute_loss


class FixturePreprocessing:
    identity = {"mode": "source_only", "bundle_id": "fixture_source_only",
                "template_sha256": _array_hash(np.ones((7, 288), np.float32) * 40),
                "scale_sha256": _array_hash(np.asarray(5.0, dtype=np.float32)),
                "target_statistics_used": False}
    identity["identity_digest"] = _json_hash(identity)

    def validate_reader(self, reader):
        if reader.role != "source":
            raise AssertionError("target data cannot enter source-only training")

    def to_state(self):
        return {"mode": "source_only", "template": np.ones((7, 288), np.float32) * 40,
                "residual_scale": 5.0, "identity": dict(self.identity)}


class FixtureReader:
    role = "source"
    T_h = T_p = 12

    def __init__(self, region, nodes, train_samples, val_samples):
        self.region, self.N = region, nodes
        self.split = np.asarray([0] * train_samples + [1] * val_samples + [2], dtype=np.int8)
        self.sample_start = np.asarray([20 + 2 * index for index in range(train_samples)]
                                       + [100 + 2 * index for index in range(val_samples)] + [160], dtype=np.int64)
        self.identity = {"region": region, "node_count": nodes,
                         "identity_digest": canonical_digest([region, nodes, train_samples, val_samples])}
        self.observed_source_rows = []

    def indices(self, split):
        if split not in {"train", "val"}:
            raise AssertionError("test metadata/targets are forbidden in this training fixture")
        return np.flatnonzero(self.split == (0 if split == "train" else 1))

    def get_batch(self, indices, preprocessing, *, split, evaluation_gate=None):
        if self.role != "source" or split not in {"train", "val"}:
            raise AssertionError("heldout/test targets were read")
        if np.any(self.split[indices] != (0 if split == "train" else 1)):
            raise AssertionError("split crossed its source-only boundary")
        self.observed_source_rows.extend(indices.tolist())
        batch = len(indices)
        flow = np.arange(self.N * 12, dtype=np.float32).reshape(1, self.N, 12)
        flow = np.repeat(flow / 10 + 30, batch, axis=0) + indices[:, None, None]
        target = np.ones_like(flow) * (45.0 if self.region == "Alameda" else 65.0)
        target[:, :, 1::3] += 1
        observed = np.ones_like(flow, dtype=bool)
        observed[:, 0, -1] = False
        mask = np.ones_like(flow, dtype=bool)
        mask[:, 0, :3] = False
        phase = np.stack((np.arange(12, dtype=np.float32) / 288,
                          np.ones(12, dtype=np.float32) / 7), axis=-1)
        edges = np.stack((np.arange(self.N - 1), np.arange(1, self.N))).astype(np.int64)
        return {"flow_hist": flow.astype(np.float32), "history_mask": observed,
                "x_baseline": np.full_like(flow, 40), "y_baseline": np.full_like(target, 40),
                "time_feat": np.repeat(phase[None], batch, axis=0),
                "node_scale": np.full(self.N, 5.0, dtype=np.float32),
                "target": target.astype(np.float32), "target_mask": mask,
                "sample_start": self.sample_start[indices], "edge_index": edges}


def fixture_data(batch_size=2):
    readers = {"Alameda": FixtureReader("Alameda", 3, 4, 3),
               "ContraCosta": FixtureReader("ContraCosta", 5, 9, 2)}
    preprocessing = FixturePreprocessing()
    train = {region: TransferBatchLoader(reader, preprocessing, split="train", batch_size=batch_size,
                                         seed=42 + number)
             for number, (region, reader) in enumerate(readers.items())}
    validation = {region: TransferBatchLoader(reader, preprocessing, split="val", batch_size=batch_size,
                                              seed=42, shuffle=False)
                  for region, reader in readers.items()}
    return SimpleNamespace(readers=readers, train_loaders=train, val_loaders=validation,
                           preprocessing=preprocessing,
                           identity={"mode": "source_only", "source_regions": list(readers),
                                     "heldout_region": "Orange", "identity_digest": "fixture_data"})


def fixture_model(dropout=0):
    return SharedGraphNDR(hidden_dim=8, road_layers=2, latent_tokens=8, num_heads=2, dropout=dropout)


def fixture_config(data, teacher="original", epochs=2, max_val=None):
    return {"source_regions": list(data.readers), "heldout_region": "Orange",
            "teacher": teacher, "epochs": epochs, "seed": 42, "batch_size": 2,
            "microbatch_size": 1, "learning_rate": 0.001, "weight_decay": 0.0001,
            "gradient_clip": 5.0, "eta_min": 0.00001, "lambda_gate": 1.0,
            "lambda_specialization": 0.3, "max_train_batches": None,
            "max_val_batches": max_val, "log_interval": 100, "threads": 1,
            "claim_scope": "smoke_only", "source_snapshot": trainer.source_snapshot(),
            "runtime": trainer.transfer_runtime_identity(torch.device("cpu"))}


def test_fold_excludes_target_and_plan_covers_every_region_batch():
    trainer.validate_fold(["Alameda", "ContraCosta"], "Orange")
    with pytest.raises(ValueError, match="excluded"):
        trainer.validate_fold(["Alameda", "Orange"], "Orange")
    with pytest.raises(ValueError, match="distinct"):
        trainer.validate_fold(["Alameda", "Alameda"], "Orange")
    counts = {"Alameda": 2, "ContraCosta": 5}
    first = trainer.epoch_region_plan(counts, seed=42, epoch=0)
    assert first == trainer.epoch_region_plan(counts, seed=42, epoch=0)
    assert {region: first.count(region) for region in counts} == counts
    weights = trainer.region_objective_multipliers(counts)
    assert counts["Alameda"] * weights["Alameda"] == counts["ContraCosta"] * weights["ContraCosta"]


def test_original_graph_loss_matches_existing_raw_ndr_loss():
    torch.manual_seed(42)
    shape = (3, 4, 12)
    target = torch.rand(shape) * 50
    baseline = torch.rand(shape) * 50
    persistence = torch.rand((3, 4, 1)) * 50
    mask = torch.rand(shape) > 0.3
    prediction = torch.rand(shape) * 50
    scale = torch.rand(4) + 2
    aux = {"g": torch.rand(shape), "B": baseline, "P": persistence,
           "yn": torch.rand(shape) * 50, "yd": torch.rand(shape) * 50}
    old = SimpleNamespace(aux=aux, oracle_sup=True, dual=True, abank=False, training=True, sd_node=scale)
    expected, parts = compute_loss(old, prediction,
                                   {"y_true": target.unsqueeze(-1), "y_mask": mask.unsqueeze(-1)},
                                   SimpleNamespace(lambda_gate=1.0, lambda_spec=0.3), torch.device("cpu"))
    new = SimpleNamespace(training=True, aux=aux | {"node_scale": scale.view(1, 4, 1).expand(3, -1, -1)})
    actual, graph_parts = compute_graph_loss(new, prediction, target, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for key in parts:
        assert graph_parts[key] == pytest.approx(parts[key], rel=1e-6)


@pytest.mark.parametrize("teacher", ["original", "projected"])
def test_graph_full_effective_batch_and_microbatch_gradients_match(teacher):
    trainer.configure_transfer_runtime("cpu", 42, 1)
    data = fixture_data(batch_size=4)
    batch = data.train_loaders["ContraCosta"].get_next()
    tensors = trainer.graph_batch_tensors(batch, torch.device("cpu"))
    tensors["target_mask"][0] = False
    tensors["target"][0] = 0
    full = fixture_model()
    micro = copy.deepcopy(full)
    full.train()
    micro.train()
    expected = trainer.backward_graph_batch(full, tensors, microbatch_size=4, teacher=teacher)
    actual = trainer.backward_graph_batch(micro, tensors, microbatch_size=1, teacher=teacher)
    assert actual["valid_count"] == expected["valid_count"]
    assert actual["objective"] == pytest.approx(expected["objective"], rel=2e-6)
    for left, right in zip(full.parameters(), micro.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=4e-6)
    assert not any("sd_node" in name or "node_scale" in name for name in full.state_dict())


def test_projected_teacher_is_explicit_and_does_not_change_original_default():
    target = torch.tensor([[[1.0, 3.0, 9.0]]])
    baseline = torch.tensor([[[5.0, 5.0, 5.0]]])
    persistence = torch.tensor([[[1.0]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    scale = torch.ones((1, 1, 1))
    original = graph_oracle_teacher(target, valid, baseline=baseline, persistence=persistence, node_scale=scale)
    projected = graph_oracle_teacher(target, valid, baseline=baseline, persistence=persistence,
                                    node_scale=scale, teacher="projected")
    assert projected["weight"].tolist() == [[[0.0, 0.5, 1.0]]]
    assert original["weight"][0, 0, 2] < 1
    with pytest.raises(ValueError, match="no valid"):
        build_graph_loss_normalizers(target, torch.zeros_like(valid), baseline=baseline,
                                     persistence=persistence, node_scale=scale)


def test_two_source_validation_is_region_mean_not_node_weighted(monkeypatch):
    data = fixture_data()
    model = fixture_model()
    monkeypatch.setattr(trainer, "forward_graph", lambda _, tensors: torch.zeros_like(tensors["target"]))
    metrics = trainer.evaluate_sources(model, data, torch.device("cpu"), microbatch_size=1)
    assert metrics["all_source_validation_samples"]
    assert metrics["selection_split"] == "source_val" and not metrics["target_region_used"]
    rows = list(metrics["regions"].values())
    expected = np.mean([row["global_masked_mae"] for row in rows])
    pooled = sum(row["absolute_error_sum"] for row in rows) / sum(row["valid_count"] for row in rows)
    assert metrics["macro_global_mae"] == expected
    assert expected != pytest.approx(pooled)
    assert metrics["regions"]["Alameda"]["selection"]["samples"] == 3
    assert metrics["regions"]["ContraCosta"]["selection"]["samples"] == 2
    assert all(row["finite_predictions_everywhere"] for row in rows)


def test_batch_whitelist_and_target_reader_fail_closed():
    data = fixture_data()
    batch = data.train_loaders["Alameda"].get_next()
    with pytest.raises(ValueError, match="whitelist"):
        trainer.graph_batch_tensors(batch | {"affected_mask": np.ones(1)}, torch.device("cpu"))
    data.readers["Alameda"].role = "target"
    with pytest.raises(ValueError, match="source-role"):
        trainer.evaluate_source_region(fixture_model(), data, "Alameda", torch.device("cpu"), microbatch_size=1)


def test_nonfinite_predictions_fail_even_on_unobserved_targets(monkeypatch):
    data = fixture_data()
    def bad_forecast(_, tensors):
        prediction = torch.zeros_like(tensors["target"])
        prediction[~tensors["target_mask"]] = float("nan")
        return prediction
    monkeypatch.setattr(trainer, "forward_graph", bad_forecast)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.evaluate_sources(fixture_model(), data, torch.device("cpu"), microbatch_size=1)


def test_interrupted_multi_region_cursor_rng_and_schedule_resume_exactly(tmp_path, monkeypatch):
    device = trainer.configure_transfer_runtime("cpu", 42, 1)
    full_data = fixture_data()
    full_model = fixture_model(dropout=0.1)
    config = fixture_config(full_data)
    full_output = tmp_path / "full"
    full_output.mkdir()
    trainer.train_transfer_run(model=full_model, data=full_data, device=device, config=config, output=full_output)
    trainer.configure_transfer_runtime("cpu", 42, 1)
    interrupted_data = fixture_data()
    interrupted_model = fixture_model(dropout=0.1)
    output = tmp_path / "resumed"
    output.mkdir()
    ticks = iter(range(100))
    with monkeypatch.context() as limited:
        limited.setattr(trainer.time, "monotonic", lambda: float(next(ticks)))
        stopped = trainer.train_transfer_run(model=interrupted_model, data=interrupted_data, device=device,
                                             config=config, output=output, max_wallclock_seconds=3)
    assert stopped["status"] == "stopped_resumable" and stopped["plan_cursor"] == 2
    checkpoint = trainer.load_transfer_checkpoint(output / "last.pt")
    assert set(checkpoint["loaders"]["train"]) == {"Alameda", "ContraCosta"}
    resumed_data, resumed_model = fixture_data(), fixture_model(dropout=0.1)
    completed = trainer.train_transfer_run(model=resumed_model, data=resumed_data, device=device,
                                           config=config, output=output, checkpoint=checkpoint)
    assert completed["completed_epochs"] == 2
    for name, expected in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], expected, rtol=0, atol=0)
    report = trainer.verify_transfer_checkpoint(checkpoint_path=output / "best.pt", data=resumed_data,
                                               device=device, output=output,
                                               current_runtime=trainer.transfer_runtime_identity(device))
    assert report["status"] == "verified_smoke_complete_source_validation"
    assert report["absolute_mae_difference"] == 0
    assert report["all_source_validation_samples"]
    assert report["source_regions"] == ["Alameda", "ContraCosta"]
    assert report["heldout_region"] == "Orange" and not report["target_data_read"]
    bound = trainer.load_transfer_checkpoint(output / "best.pt")
    assert report["selection_metric_digest"] == canonical_digest(bound["selection_metric"])
    assert report["model_config_sha256"] == canonical_digest(bound["model_config"])
    assert bound["source_statistics"]["template"].shape == (7, 288)


def test_resume_rejects_different_source_template_or_region_plan(tmp_path):
    device = trainer.configure_transfer_runtime("cpu", 42, 1)
    data = fixture_data()
    model = fixture_model()
    config = fixture_config(data, epochs=1)
    output = tmp_path / "run"
    output.mkdir()
    trainer.train_transfer_run(model=model, data=data, device=device, config=config, output=output,
                               max_wallclock_seconds=1e-9)
    checkpoint = trainer.load_transfer_checkpoint(output / "last.pt")
    optimizer, scheduler = trainer.create_transfer_optimizer(model, config, 7)
    changed = copy.deepcopy(checkpoint)
    changed["progress"]["plan_cursor"] = 1
    with pytest.raises(ValueError, match="cursor differs"):
        trainer.restore_transfer_training(changed, model, optimizer, scheduler, data, config)
    changed["source_statistics_identity"] = {"identity_digest": "other_source_template"}
    with pytest.raises(ValueError, match="template identity"):
        trainer.restore_transfer_training(changed, model, optimizer, scheduler, data, config)


def test_calibration_times_complete_batches_and_keeps_source_only_labels(tmp_path):
    device = trainer.configure_transfer_runtime("cpu", 42, 1)
    data = fixture_data()
    config = fixture_config(data, epochs=30)
    result = trainer.calibrate_transfer(model=fixture_model(), data=data, device=device,
                                        config=config, output=tmp_path, warmup=1, timed=3)
    assert result["claim_scope"] == "calibration_only"
    assert result["timing_includes_reader"]
    assert result["estimated_train_source_val_seconds"] > 0
    for region in result["regions"].values():
        assert region["train"]["timed_batches"] == region["val"]["timed_batches"] == 3
    assert not (tmp_path / "best.pt").exists()


def test_cli_freezes_teacher_fold_and_rejects_validation_subset_override(monkeypatch):
    parser = trainer.argument_parser()
    args = parser.parse_args([])
    assert args.source_regions == ["Alameda", "ContraCosta"]
    assert args.heldout_region == "Orange" and args.teacher == "original"
    assert args.device == "cuda:0" and args.epochs == 30
    assert args.batch_size == 16 and args.microbatch_size == 2
    args = parser.parse_args(["--mode", "verify", "--checkpoint", "best.pt", "--max-val-batches", "1"])
    with pytest.raises(ValueError, match="frozen source validation"):
        trainer.validate_arguments(args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        trainer.configure_transfer_runtime("cuda:0", 42)
    assert Path(trainer.__file__).name == "train_ndr_transfer.py"
    with pytest.raises(ValueError, match="CPU training is engineering smoke only"):
        trainer.validate_arguments(parser.parse_args(["--device", "cpu", "--max-val-batches", "1"]))
    trainer.validate_arguments(parser.parse_args(["--device", "cpu", "--max-train-batches", "2"]))


def test_verify_uses_checkpoint_fold_for_unspecified_cli_sources(monkeypatch):
    monkeypatch.setattr(trainer, "load_transfer_checkpoint", lambda _: {
        "config": {"source_regions": ["ContraCosta", "Orange"], "heldout_region": "Alameda",
                   "seed": 42, "threads": 1}})
    def stop_after_argument_checks(*_):
        raise RuntimeError("configured runtime after the correct frozen fold")
    monkeypatch.setattr(trainer, "configure_transfer_runtime", stop_after_argument_checks)
    with pytest.raises(RuntimeError, match="correct frozen fold"):
        trainer.main(["--mode", "verify", "--checkpoint", "best.pt", "--heldout-region", "Alameda"])


def test_eager_source_prefix_has_identical_batches_and_graph_gradients(tmp_path):
    helper_path = Path(__file__).with_name("test_transfer_data.py")
    specification = importlib.util.spec_from_file_location("_transfer_fixture_data", helper_path)
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    data_dir, graph_dir, contract = helper.make_fixture(tmp_path)
    lazy = load_transfer_sources(["Alameda", "ContraCosta"], "Orange", data_dir, graph_dir,
                                 batch_size=2, seed=42, contract=contract)
    eager = load_transfer_sources(["Alameda", "ContraCosta"], "Orange", data_dir, graph_dir,
                                  batch_size=2, seed=42, contract=contract)
    report = trainer.apply_source_loading(eager, lazy=False)
    assert report["policy"] == "source_eager_channel_zero_prefix_to_val_end"
    assert report["bytes"] == sum(reader.val_end * reader.N * 5 for reader in eager.readers.values())
    assert not report["heldout_preloaded"]
    assert eager.identity == lazy.identity
    trainer.configure_transfer_runtime("cpu", 42, 1)
    for region in eager.readers:
        left = lazy.train_loaders[region].get_next()
        right = eager.train_loaders[region].get_next()
        for key in left:
            assert left[key].dtype == right[key].dtype
            assert left[key].shape == right[key].shape
            assert left[key].tobytes() == right[key].tobytes()
        left_model = fixture_model()
        right_model = copy.deepcopy(left_model)
        left_result = trainer.backward_graph_batch(left_model, trainer.graph_batch_tensors(left, torch.device("cpu")),
                                                   microbatch_size=1)
        right_result = trainer.backward_graph_batch(right_model, trainer.graph_batch_tensors(right, torch.device("cpu")),
                                                    microbatch_size=1)
        assert left_result == right_result
        for first, second in zip(left_model.parameters(), right_model.parameters()):
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def test_transfer_snapshot_excludes_unimported_reference_code_but_seals_core(monkeypatch):
    unused = "ndr_research/_unimported_reference_fixture.py"
    files = {unused: {"sha256": "unused", "size_bytes": 1}}
    monkeypatch.setattr(trainer, "full_checkout_snapshot", lambda _: {
        "files": files, "files_digest": "unused_full_tree", "git": {"git_sha": "fixture"}})
    before = trainer.transfer_source_snapshot()
    assert unused not in before["files"]
    assert set(before["files"]) == {"ndr_research/shared_graph.py", "ndr_research/transfer_data.py",
                                    "ndr_research/graph_loss.py", "scripts/train_ndr_transfer.py"}
    files[unused]["sha256"] = "unrelated_edit"
    after = trainer.transfer_source_snapshot()
    assert before["files_digest"] == after["files_digest"]
    assert before["scope"] == "candidate_entrypoint_and_imported_python_dependencies"
