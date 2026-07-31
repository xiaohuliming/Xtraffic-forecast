from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    PROJECT_ROOT
    / "scripts"
    / "remote"
    / "windows"
    / "igstgnn_table4_bist_ndr_pair_5080.cmd"
)


def test_bist_ndr_pair_launcher_interleaves_all_matched_tasks_without_shutdown():
    content = LAUNCHER.read_text(encoding="utf-8")
    task_ids = re.findall(r"--task-id (pair\d{3}_[a-z0-9_]+)", content)

    assert len(task_ids) == 18
    assert len(set(task_ids)) == 18
    assert task_ids[:6] == [
        "pair001_bist_aligned_alameda_seed42",
        "pair010_bist_ndr_alameda_seed42",
        "pair002_bist_aligned_alameda_seed1",
        "pair011_bist_ndr_alameda_seed1",
        "pair003_bist_aligned_alameda_seed2",
        "pair012_bist_ndr_alameda_seed2",
    ]
    assert "90f5ceb4a0e1306d3f1ff11268b76220a289a37a922c6926665825a56c9cdeab" in content
    assert (
        r"C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages"
        in content
    )
    assert "EnableDelayedExpansion" in content
    assert "RUNNER_RC=!ERRORLEVEL!" in content
    assert "shutdown" not in content.lower()
