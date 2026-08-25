from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_linux_launcher_builds_one_isolated_command_per_gpu(tmp_path: Path) -> None:
    from scripts.remote.linux.fdn_benchmark_v1_multi_gpu import build_shard_command

    args = argparse.Namespace(
        python_executable="python",
        manifest=ROOT / "config/fdn_benchmark_v1/go_no_go_pems04_pems08_v1.json",
        repo_root=ROOT,
        data_dir=tmp_path / "data",
        results_root=tmp_path / "results",
        code_sha="a" * 40,
        gpu_count=4,
        recover_completed_attempts=False,
    )

    commands = [build_shard_command(args, index) for index in range(4)]

    assert len({tuple(command) for command in commands}) == 4
    for index, command in enumerate(commands):
        assert command[command.index("--shard-index") + 1] == str(index)
        assert command[command.index("--shard-count") + 1] == "4"
        assert command[command.index("--results-root") + 1] == str(args.results_root)
        assert "--recover-completed-attempts" not in command


def test_linux_launcher_can_enable_completed_attempt_recovery(tmp_path: Path) -> None:
    from scripts.remote.linux.fdn_benchmark_v1_multi_gpu import build_shard_command

    args = argparse.Namespace(
        python_executable="python",
        manifest=ROOT / "config/fdn_benchmark_v1/go_no_go_pems04_pems08_v1.json",
        repo_root=ROOT,
        data_dir=tmp_path / "data",
        results_root=tmp_path / "results",
        code_sha="a" * 40,
        gpu_count=4,
        recover_completed_attempts=True,
    )

    command = build_shard_command(args, 0)

    assert command[-1] == "--recover-completed-attempts"
