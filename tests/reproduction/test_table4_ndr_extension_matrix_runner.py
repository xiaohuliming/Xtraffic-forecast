from pathlib import Path

from scripts.reproduction.igstgnn_table4.ndr_extension_matrix_runner import (
    build_ndr_task_command,
)


def test_ndr_extension_command_binds_both_manifests_and_isolated_result_root():
    command = build_ndr_task_command(
        python_executable=Path("/venv/bin/python"),
        manifest_path=Path("/repo/config/matrix.ndr.json"),
        manifest_sha256="1" * 64,
        base_manifest_path=Path("/repo/config/matrix.base.json"),
        base_manifest_sha256="2" * 64,
        task_id="ndr001_alameda_seed42",
        splits_root=Path("/data/official_splits"),
        results_root=Path("/results/ndr_gpu0"),
        device="cuda:0",
    )

    assert command[:4] == [
        "/venv/bin/python",
        "-X",
        "utf8",
        "-m",
    ]
    assert command[4] == (
        "scripts.reproduction.igstgnn_table4.ndr_extension_runner"
    )
    assert command[command.index("--manifest-sha256") + 1] == "1" * 64
    assert command[command.index("--base-manifest-sha256") + 1] == "2" * 64
    assert command[command.index("--task-id") + 1] == "ndr001_alameda_seed42"
    assert command[command.index("--results-root") + 1] == "/results/ndr_gpu0"
