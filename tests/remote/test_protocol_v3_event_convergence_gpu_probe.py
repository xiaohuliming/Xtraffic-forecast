from pathlib import Path

from scripts.remote.linux.protocol_v3_event_convergence_gpu_probe import (
    ProbeInputs,
    build_probe_command,
)
from scripts.remote.linux.protocol_v3_event_convergence_multi_gpu import (
    default_microbatch_sizes,
)
from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    CANONICAL_MANIFEST,
    load_convergence_manifest,
)


def test_convergence_probe_commands_cover_baseline_and_extension_contracts(tmp_path) -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    inputs = ProbeInputs(
        python_executable="python",
        data_dir=tmp_path / "cache",
        graph_dir=tmp_path / "graphs",
        support_root=tmp_path / "support",
        node_order=tmp_path / "node_order.npy",
        output_dir=tmp_path / "probes",
    )
    microbatches = default_microbatch_sizes(manifest.tasks)
    dcrnn = next(task for task in manifest.tasks if task.model == "dcrnn")
    ndr = next(
        task
        for task in manifest.tasks
        if task.model == "ndr_net" and task.region == "Orange"
    )

    baseline_command = build_probe_command(dcrnn, inputs, microbatches)
    extension_command = build_probe_command(ndr, inputs, microbatches)

    assert "protocol_v3_event_gpu_probe" in " ".join(baseline_command)
    assert baseline_command[baseline_command.index("--epochs") + 1] == "100"
    assert baseline_command[
        baseline_command.index("--recipe-profile") + 1
    ] == "convergence100"
    assert "protocol_v3_event_extension_gpu_probe" in " ".join(extension_command)
    assert extension_command[extension_command.index("--epochs") + 1] == "100"
    assert extension_command[
        extension_command.index("--max-microbatch-size") + 1
    ] == "8"
    assert Path(extension_command[extension_command.index("--node-order") + 1]) == inputs.node_order
