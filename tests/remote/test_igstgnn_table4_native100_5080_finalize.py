from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "remote"
    / "windows"
    / "igstgnn_table4_native100_5080_finalize.ps1"
)


def test_native100_5080_finalizer_is_bound_and_fail_closed() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "native100_rtx5080_alameda_contra" in text
    assert "d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e" in text
    assert "POSTFLIGHT_OK tasks=12 files=61" in text
    assert "Python313\\site-packages" in text
    assert "sha256sum -c ../metadata/transfer.sha256" in text
    assert "Get-RemoteReusableBytes" in text
    assert "([long]$capacity + $reusableBytes) -lt $required" in text
    assert "& $Ssh -n" in text
    assert "FINALIZE_BLOCKED" in text
    assert "machine_stays_on" in text
    assert "shutdown" not in text.lower()
