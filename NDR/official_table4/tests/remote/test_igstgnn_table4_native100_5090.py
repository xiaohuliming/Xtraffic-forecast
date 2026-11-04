from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    PROJECT_ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "igstgnn_table4_native100_5090_orange.sh"
)


def test_native100_5090_launcher_contains_only_paired_orange_tasks():
    content = LAUNCHER.read_text(encoding="utf-8")
    task_ids = re.findall(r"--task-id (n100\d{3}_[a-z0-9_]+)", content)

    assert task_ids == [
        "n100007_bist_orange_seed42",
        "n100008_bist_orange_seed1",
        "n100009_bist_orange_seed2",
        "n100016_ndr_net_orange_seed42",
        "n100017_ndr_net_orange_seed1",
        "n100018_ndr_net_orange_seed2",
    ]
    assert "n100001" not in content
    assert "n100006" not in content
    assert "n100010" not in content
    assert "n100015" not in content
    assert 'REPO="/root/autodl-tmp/native100_repo"' in content
    assert 'RESULTS_ROOT="$RUN_ROOT/results"' in content
    assert 'SPLITS_ROOT="/root/autodl-tmp/native100_data/official_splits"' in content
    assert "shutdown" not in content.lower()
