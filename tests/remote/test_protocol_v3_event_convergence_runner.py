from __future__ import annotations

from pathlib import Path

from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    CANONICAL_MANIFEST,
    CONVERGENCE_METHODS,
    ConvergenceInputs,
    build_convergence_invocation,
    load_convergence_manifest,
    parse_args,
)
from scripts.remote.linux.protocol_v3_event_multi_gpu import write_shard_manifests
from scripts.remote.windows.protocol_v3_event_extension_runner import ExtensionInputs
from scripts.remote.windows.protocol_v3_runner import ProtocolV3Inputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_convergence_manifest_is_exact_seed42_trainable_39_matrix() -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)

    assert manifest.matrix_id == "protocol_v3_event_convergence_seed42_39_v1"
    assert len(manifest.tasks) == 39
    assert {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    } == {
        (region, method, 42, "event")
        for region in ("Alameda", "ContraCosta", "Orange")
        for method in CONVERGENCE_METHODS
    }
    assert "hl" not in {task.model for task in manifest.tasks}

    dcrnn = manifest.method_contracts["dcrnn"]
    assert dcrnn["max_epochs"] == 100
    assert dcrnn["patience"] == 30
    assert dcrnn["optimizer"]["learning_rate"] == 0.01
    assert dcrnn["optimizer"]["gradient_clip"] == 5.0
    assert dcrnn["optimizer"]["scheduler"]["milestones"] == [10, 50, 90]

    d2stgnn = manifest.method_contracts["d2stgnn"]
    assert d2stgnn["optimizer"]["scheduler"]["milestones"] == [
        1,
        38,
        46,
        54,
        62,
        70,
        80,
    ]

    assert manifest.method_contracts["graphwavenet"]["optimizer"]["scheduler"] == {
        "name": "CosineAnnealingLR",
        "t_max": "max_epochs_times_training_batches",
        "eta_min": "learning_rate_times_0.01",
        "step_unit": "optimizer_step",
    }

    assert manifest.method_contracts["bist"]["patience"] == 20
    assert manifest.method_contracts["igstgnn"]["patience"] == 20
    assert manifest.method_contracts["igstgnn"]["optimizer"]["scheduler"] == {
        "name": "MultiStepLR",
        "milestones": [1, 38, 46, 54, 62, 70, 80],
        "gamma": 0.5,
    }
    assert manifest.method_contracts["ndr_net"]["patience"] is None


def test_convergence_runner_cli_requires_bound_cache_and_support_inputs(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--support-root",
            "support",
            "--node-order",
            "node_order.npy",
            "--cache-bundle-id",
            "bundle-test",
            "--cache-bundle-digest",
            "a" * 64,
            "--code-sha",
            "b" * 40,
        ]
    )

    assert args.manifest.name == "event_convergence_seed42_v1.json"
    assert args.support_root == Path("support")
    assert args.allow_shard is False


def _inputs(tmp_path: Path) -> ConvergenceInputs:
    protocol = ProtocolV3Inputs(
        results_root=tmp_path / "results",
        cache_dir=tmp_path / "cache",
        graph_dir=tmp_path / "graphs",
        cache_bundle_id="bundle-test",
        cache_bundle_digest="a" * 64,
        code_sha="b" * 40,
        python_executable="python",
    )
    extension = ExtensionInputs(
        support_root=tmp_path / "support",
        node_order=tmp_path / "node_order.npy",
        support_manifest_sha256="c" * 64,
        igstgnn_data_parallel=False,
        max_microbatch_size=0,
        cache_dir=protocol.cache_dir,
        graph_dir=protocol.graph_dir,
        python_executable=protocol.python_executable,
    )
    return ConvergenceInputs(protocol=protocol, extension=extension)


def test_dcrnn_invocation_uses_uncompressed_schedule_contract(
    tmp_path: Path,
) -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    task = next(task for task in manifest.tasks if task.task_id.startswith("v3c002_"))

    invocation = build_convergence_invocation(task, manifest, _inputs(tmp_path), tmp_path)
    command = list(invocation.command)

    assert command[command.index("--epochs") + 1] == "100"
    assert command[command.index("--patience") + 1] == "30"
    assert command[command.index("--recipe-profile") + 1] == "convergence100"
    assert invocation.config["method_contract"] == manifest.method_contracts["dcrnn"]
    assert "training_diagnostics.json" in invocation.required_artifacts


def test_orange_ndr_invocation_uses_convergence_contract_and_microbatch(
    tmp_path: Path,
) -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    task = next(task for task in manifest.tasks if task.task_id.startswith("v3c039_"))

    invocation = build_convergence_invocation(task, manifest, _inputs(tmp_path), tmp_path)
    command = list(invocation.command)

    assert command[command.index("--epochs") + 1] == "100"
    assert command[command.index("--patience") + 1] == "20"
    assert command[command.index("--run-contract") + 1] == "convergence100"
    assert command[command.index("--traffic-loading") + 1] == "eager"
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert invocation.config["early_stopping_patience"] is None
    assert invocation.config["method_contract"] == manifest.method_contracts["ndr_net"]
    assert "training_diagnostics.json" in invocation.required_artifacts


def test_convergence_shard_applies_isolated_baseline_microbatch(
    tmp_path: Path,
) -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    task = next(
        task
        for task in manifest.tasks
        if task.region == "Orange" and task.model == "d2stgnn"
    )
    paths = write_shard_manifests(
        CANONICAL_MANIFEST,
        {0: (task,)},
        tmp_path / "manifests",
        results_root_prefix="outputs/convergence/probe",
        microbatch_sizes={task.task_id: 16},
    )

    shard = load_convergence_manifest(paths[0], require_complete_matrix=False)
    invocation = build_convergence_invocation(
        shard.tasks[0], shard, _inputs(tmp_path), tmp_path / "attempt"
    )
    command = list(invocation.command)

    assert command[command.index("--microbatch-size") + 1] == "16"
    assert invocation.config["microbatch_size"] == 16


def test_convergence_shard_applies_isolated_extension_microbatch(
    tmp_path: Path,
) -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    task = next(
        task
        for task in manifest.tasks
        if task.region == "Orange" and task.model == "igstgnn"
    )
    paths = write_shard_manifests(
        CANONICAL_MANIFEST,
        {0: (task,)},
        tmp_path / "manifests",
        results_root_prefix="outputs/convergence/probe",
        microbatch_sizes={task.task_id: 4},
    )

    shard = load_convergence_manifest(paths[0], require_complete_matrix=False)
    invocation = build_convergence_invocation(
        shard.tasks[0], shard, _inputs(tmp_path), tmp_path / "attempt"
    )
    command = list(invocation.command)

    assert command[command.index("--max-microbatch-size") + 1] == "4"
    assert invocation.config["max_microbatch_size"] == 4
