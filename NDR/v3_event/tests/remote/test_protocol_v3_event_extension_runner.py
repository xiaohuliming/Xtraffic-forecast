from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from scripts.remote.windows.protocol_v3_event_extension_runner import (
    CANONICAL_MANIFEST,
    SUPPORT_MANIFEST,
    ExtensionInputs,
    build_extension_invocation,
    load_extension_manifest,
    verify_extension_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v3_extension_trainer_accepts_bist_ndr_without_early_stopping() -> None:
    from scripts.train_protocol_v3_event_extension import (
        METHODS,
        validate_training_contract,
    )

    assert "bist_ndr" in METHODS
    args = SimpleNamespace(
        seed=42,
        run_contract="formal30",
        epochs=30,
        patience=20,
        method="bist_ndr",
        variant="v3_event_bist_ndr",
    )
    assert validate_training_contract(args) is None


def test_v3_extension_trainer_rejects_a_method_variant_mismatch() -> None:
    import pytest

    from scripts.train_protocol_v3_event_extension import validate_training_contract

    args = SimpleNamespace(
        seed=42,
        run_contract="formal30",
        epochs=30,
        patience=20,
        method="bist_ndr",
        variant="v3_event_aligned",
    )
    with pytest.raises(ValueError, match="variant"):
        validate_training_contract(args)


def test_v3_extension_builds_the_pinned_bist_ndr_model(tmp_path: Path) -> None:
    import torch

    from scripts.train_protocol_v3_event_extension import FileLogger, _build_model

    statistics = SimpleNamespace(
        flow_mean=20.0,
        flow_std=5.0,
        residual_std=3.0,
        residual_std_by_node=np.ones(3, dtype=np.float32),
    )
    loaders = {"train": SimpleNamespace(num_batch=2)}
    with (tmp_path / "build.log").open("w", encoding="utf-8") as handle:
        model, optimizer, scheduler, provenance, clip, engine = _build_model(
            method="bist_ndr",
            rdata=SimpleNamespace(N=3, region_name="Alameda"),
            loaders=loaders,
            statistics=statistics,
            sensor_support=None,
            support_dir=tmp_path,
            device=torch.device("cpu"),
            seed=42,
            epochs=30,
            logger=FileLogger(handle),
            output_dir=tmp_path,
            igstgnn_data_parallel=False,
        )

    assert type(model).__name__ == "BiSTNDR"
    assert type(optimizer).__name__ == "Adam"
    assert type(scheduler).__name__ == "CosineAnnealingLR"
    assert provenance["upstream_commit"] == (
        "dd94adf7721fcbb9e3feb5d1b44040305199a4cc"
    )
    assert provenance["protocol_adapter"] == "protocol_v3_noleak_dedup_event"
    assert clip == 5
    assert engine is None


def test_v3_extension_evaluates_bist_ndr_in_raw_flow_units(
    tmp_path: Path,
) -> None:
    import torch

    from scripts.train_protocol_v3_event_extension import (
        FileLogger,
        _build_model,
        _evaluate,
    )

    node_count = 3
    statistics = SimpleNamespace(
        baseline=np.full((7, 288, node_count), 20.0, dtype=np.float32),
        flow_mean=20.0,
        flow_std=5.0,
        residual_std=3.0,
        residual_std_by_node=np.ones(node_count, dtype=np.float32),
    )
    with (tmp_path / "evaluate.log").open("w", encoding="utf-8") as handle:
        model, *_ = _build_model(
            method="bist_ndr",
            rdata=SimpleNamespace(N=node_count, region_name="Alameda"),
            loaders={"train": SimpleNamespace(num_batch=1)},
            statistics=statistics,
            sensor_support=None,
            support_dir=tmp_path,
            device=torch.device("cpu"),
            seed=42,
            epochs=30,
            logger=FileLogger(handle),
            output_dir=tmp_path,
            igstgnn_data_parallel=False,
        )
    batch = {
        "x_raw_flow": np.full((2, 12, node_count), 21.0, dtype=np.float32),
        "y_raw_flow": np.full((2, 12, node_count), 22.0, dtype=np.float32),
        "y_mask": np.ones((2, 12, node_count, 1), dtype=np.bool_),
        "time_feat": np.zeros((2, 12, 2), dtype=np.float32),
    }
    loader = SimpleNamespace(get_iterator=lambda: iter((batch,)))

    mae, prediction, actual, mask = _evaluate(
        model=model,
        method="bist_ndr",
        loader=loader,
        statistics=statistics,
        device=torch.device("cpu"),
        capture=True,
    )

    assert np.isfinite(mae)
    assert prediction is not None and prediction.shape == (2, 12, node_count)
    assert actual is not None and actual.shape == prediction.shape
    assert mask is not None and mask.shape == prediction.shape


def test_extension_trainer_bootstraps_project_imports_from_script_path() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/train_protocol_v3_event_extension.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "BiST-NDR" in completed.stdout


def test_extension_manifest_is_exact_seed42_nine_task_matrix() -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    assert len(manifest.tasks) == 9
    assert {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    } == {
        (region, method, 42, "event")
        for region in ("Alameda", "ContraCosta", "Orange")
        for method in ("bist", "igstgnn", "ndr_net")
    }


def test_checked_support_and_invocation_bind_the_v3_adapter(tmp_path: Path) -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            PROJECT_ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=PROJECT_ROOT / "archive/node_order.npy",
    )
    task = next(task for task in manifest.tasks if task.task_id == "v3x002_event_igstgnn_alameda_seed42")
    invocation = build_extension_invocation(
        task,
        manifest,
        ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=True,
            max_microbatch_size=8,
        ),
        tmp_path,
    )
    command = list(invocation.command)
    assert "scripts/train_protocol_v3_event_extension.py" in command
    assert "--igstgnn-data-parallel" in command
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert invocation.required_artifacts == (
        "summary.json",
        "ckpt_best.pt",
        "test_predictions.npz",
        "train.log",
    )
    assert invocation.config["support_manifest_sha256"] == support.manifest_sha256
    assert invocation.config["upstream_curriculum_applied"] is True
    assert command[command.index("--traffic-loading") + 1] == "eager"
    assert invocation.config["traffic_loading"] == "eager"


def test_orange_ndr_invocation_records_its_effective_microbatch(tmp_path: Path) -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            PROJECT_ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=PROJECT_ROOT / "archive/node_order.npy",
    )
    task = next(task for task in manifest.tasks if task.task_id == "v3x009_event_ndr_net_orange_seed42")
    invocation = build_extension_invocation(
        task,
        manifest,
        ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=False,
            max_microbatch_size=0,
        ),
        tmp_path,
    )

    command = list(invocation.command)
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert invocation.config["max_microbatch_size"] == 8
    assert invocation.config["early_stopping_patience"] is None


def test_orange_igstgnn_invocation_records_its_effective_microbatch(
    tmp_path: Path,
) -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            PROJECT_ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=PROJECT_ROOT / "archive/node_order.npy",
    )
    task = next(
        task
        for task in manifest.tasks
        if task.task_id == "v3x008_event_igstgnn_orange_seed42"
    )
    invocation = build_extension_invocation(
        task,
        manifest,
        ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=False,
            max_microbatch_size=0,
        ),
        tmp_path,
    )

    command = list(invocation.command)
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert invocation.config["max_microbatch_size"] == 8
    assert invocation.config["upstream_curriculum_applied"] is True
