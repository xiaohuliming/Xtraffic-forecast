from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_5080_launcher_is_fail_closed_and_uses_isolated_paths() -> None:
    text = (
        ROOT / "scripts/remote/windows/fdn_benchmark_v1_5080.cmd"
    ).read_text(encoding="utf-8")

    assert "D:\\t4\\fdn_benchmark_v1_repo" in text
    assert "D:\\t4\\fdn_benchmark_v1_data" in text
    assert "D:\\t4\\fdn_benchmark_v1\\results" in text
    assert "if exist \"%FDNB_CONTROL%\\runner.exitcode\" exit /b 84" in text
    assert "move /Y" in text
    assert "protocol_v3" not in text.lower()
