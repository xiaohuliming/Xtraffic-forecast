from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "igstgnn_table4_native100_5090_finalize.sh"
)


def test_native100_finalizer_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_native100_finalizer_is_bound_to_the_locked_orange_shard() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246" in source
    assert "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78" in source
    assert "d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e" in source
    for task_number in ("007", "008", "009", "016", "017", "018"):
        assert f'n100{task_number}_' in source


def test_native100_finalizer_verifies_relay_before_shutdown() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    local_postflight = source.index("NATIVE100_LOCAL_POSTFLIGHT_OK")
    capacity_gate = source.index("relay preparation or capacity gate failed")
    transfer = source.index("native100 result transfer failed")
    remote_verification = source.index("NATIVE100_RELAY_VERIFIED")
    shutdown = source.index('"$SHUTDOWN_BIN" -h now')
    assert local_postflight < capacity_gate < transfer < remote_verification < shutdown
    assert "sha256sum -c" in source
    assert "mv '.$HOST_ID.incoming' '$HOST_ID'" in source
