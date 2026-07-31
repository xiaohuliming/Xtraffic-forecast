from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    PROJECT_ROOT
    / "scripts"
    / "remote"
    / "windows"
    / "igstgnn_table4_native100_5080.cmd"
)


def test_native100_launcher_runs_all_tasks_without_shutdown_or_old_result_reuse():
    content = LAUNCHER.read_text(encoding="utf-8")
    task_ids = re.findall(r"--task-id (n100\d{3}_[a-z0-9_]+)", content)

    assert len(task_ids) == 18
    assert len(set(task_ids)) == 18
    assert task_ids[0] == "n100001_bist_alameda_seed42"
    assert task_ids[-1] == "n100018_ndr_net_orange_seed2"
    assert "e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246" in content
    assert "D:\\t4\\native100_repo" in content
    assert 'set "RUN_ROOT=D:\\t4\\native100_compare"' in content
    assert 'set "RESULTS_ROOT=%RUN_ROOT%\\results"' in content
    assert "EnableDelayedExpansion" in content
    assert "RUNNER_RC=!ERRORLEVEL!" in content
    assert "shutdown" not in content.lower()
