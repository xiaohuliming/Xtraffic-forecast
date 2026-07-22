from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "protocol_v3_launch_5090_shard.sh"
)
TRANSITION = (
    REPO_ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "protocol_v3_start_5090_shard_after_boundary.sh"
)
FINALIZER = (
    REPO_ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "protocol_v3_finalize_5090_shard.sh"
)
BOUNDARY_TASK = "039_fullwindow_staeformer_base_alameda_seed42"
FORBIDDEN_TASK = "040_fullwindow_graphwavenet_base_contracosta_seed42"


def test_5090_repartition_controls_default_to_an_isolated_ops_root():
    isolated_ops_root = "/root/autodl-tmp/xtraffic_ops_27_18"

    for path in (LAUNCHER, TRANSITION, FINALIZER):
        assert isolated_ops_root in path.read_text(encoding="utf-8")


def test_all_relay_connections_have_established_session_liveness_bounds():
    transition = TRANSITION.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")

    for script in (transition, finalizer):
        assert "ServerAliveInterval=5" in script
        assert "ServerAliveCountMax=3" in script
    assert finalizer.count('"${SSH_OPTIONS[@]}"') >= 4
    assert finalizer.count('-e "$RSYNC_SSH_COMMAND"') >= 5


def test_launcher_verifies_and_runs_only_the_18_task_5090_plan(
    tmp_path: Path,
):
    action_log = tmp_path / "runner_args.json"
    exit_code = tmp_path / "shard18.exitcode"
    fake_ops = tmp_path / "fake_distributed.py"
    fake_ops.write_text(
        """
import json
import os
import sys
from pathlib import Path

Path(os.environ["ACTION_LOG"]).write_text(
    json.dumps(sys.argv[1:]),
    encoding="utf-8",
)
raise SystemExit(17)
""".lstrip(),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "ACTION_LOG": str(action_log),
        "EXIT_CODE": str(exit_code),
        "OPS": str(fake_ops),
        "OPS_ROOT": str(REPO_ROOT),
        "PYTHON": sys.executable,
        "REPO": str(REPO_ROOT),
        "RESULTS": str(tmp_path / "results"),
    }

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 17
    assert exit_code.read_text(encoding="utf-8").strip() == "17"
    arguments = json.loads(action_log.read_text(encoding="utf-8"))
    assert arguments[arguments.index("--shard") + 1] == "5090"
    assert arguments[arguments.index("--results-root") + 1] == str(
        tmp_path / "results"
    )


def test_transition_publishes_ready_only_after_old_processes_are_stopped(
    tmp_path: Path,
):
    results = tmp_path / "results"
    summary = results / "tasks" / BOUNDARY_TASK / "summary.json"
    state = results / "state" / "tasks" / f"{BOUNDARY_TASK}.json"
    summary.parent.mkdir(parents=True)
    state.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"task_id": BOUNDARY_TASK}),
        encoding="utf-8",
    )
    state.write_text(
        json.dumps({"state": "completed", "task_id": BOUNDARY_TASK}),
        encoding="utf-8",
    )
    (results / "queue_status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "total_tasks": 21,
                "completed_tasks": 15,
                "skipped_valid_tasks": 2,
                "current_task": BOUNDARY_TASK,
            }
        ),
        encoding="utf-8",
    )
    boundary_status = tmp_path / "boundary_status.json"
    boundary_status.write_text(
        json.dumps(
            {
                "state": "completed",
                "task_id": BOUNDARY_TASK,
                "pid": 999999,
                "pgid": 999999,
                "stop_result": "terminated",
            }
        ),
        encoding="utf-8",
    )

    action_log = tmp_path / "actions.log"
    old_finalizer_script = (
        tmp_path / "protocol_v3_finalize_5090_shard.sh"
    )
    old_finalizer_script.write_text(
        "#!/usr/bin/env bash\nwhile true; do sleep 1; done\n",
        encoding="utf-8",
    )
    old_finalizer = subprocess.Popen(
        ["bash", str(old_finalizer_script)],
        start_new_session=True,
    )
    old_finalizer_pid_file = tmp_path / "old_finalizer.pid"
    old_finalizer_pid_file.write_text(
        f"{old_finalizer.pid}\n",
        encoding="utf-8",
    )

    new_launcher = tmp_path / "protocol_v3_launch_5090_shard18.sh"
    new_launcher.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'NEW_RUNNER\\n' >> \"$ACTION_LOG\"\n"
        "sleep 1\n"
        "printf '%s\\n' "
        "'{\"state\":\"running\",\"total_tasks\":18,'"
        "'\"completed_tasks\":15,\"skipped_valid_tasks\":15,'"
        "'\"current_task\":null}' "
        "> \"$RESULTS/queue_status.json.tmp\"\n"
        "mv \"$RESULTS/queue_status.json.tmp\" "
        "\"$RESULTS/queue_status.json\"\n"
        "while true; do sleep 1; done\n",
        encoding="utf-8",
    )
    new_finalizer = tmp_path / "protocol_v3_finalize_5090_shard18.sh"
    new_finalizer.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "\"$PYTHON\" - \"$RESULTS/queue_status.json\" <<'PY'\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "queue = json.loads(Path(sys.argv[1]).read_text())\n"
        "assert queue['total_tasks'] == 18\n"
        "assert queue['state'] in {'preflight', 'running'}\n"
        "assert queue['current_task'] in {None, "
        "'043_fullwindow_graphwavenet_base_orange_seed42'}\n"
        "PY\n"
        "printf 'NEW_FINALIZER\\n' >> \"$ACTION_LOG\"\n"
        "while true; do sleep 1; done\n",
        encoding="utf-8",
    )
    fake_setsid = tmp_path / "setsid"
    fake_setsid.write_text(
        "#!/usr/bin/env bash\nexec \"$@\"\n",
        encoding="utf-8",
    )
    fake_setsid.chmod(0o755)
    ready_capture = tmp_path / "ready_on_relay.json"
    ssh_args_log = tmp_path / "ssh_args.log"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$SSH_ARGS_LOG\"\n"
        "cat > \"$READY_CAPTURE\"\n"
        "printf 'PUBLISH_READY\\n' >> \"$ACTION_LOG\"\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    validation = tmp_path / "validate.sh"
    validation.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'VALIDATE_039\\n' >> \"$ACTION_LOG\"\n",
        encoding="utf-8",
    )
    validation.chmod(0o755)
    new_runner_pid_file = tmp_path / "shard18.pid"
    new_finalizer_pid_file = tmp_path / "shard18_finalize.pid"
    environment = {
        **os.environ,
        "ACTION_LOG": str(action_log),
        "BOUNDARY_STATUS": str(boundary_status),
        "BOUNDARY_VALIDATE_CMD": str(validation),
        "FINALIZER": str(new_finalizer),
        "FINALIZER_CONSOLE": str(tmp_path / "finalizer.console"),
        "FINALIZER_PID_FILE": str(new_finalizer_pid_file),
        "LAUNCHER": str(new_launcher),
        "LAUNCH_GRACE_SECONDS": "0.1",
        "OLD_FINALIZER_PID_FILE": str(old_finalizer_pid_file),
        "OLD_RUNNER_PID_FILE": str(tmp_path / "old_runner.pid"),
        "PYTHON": sys.executable,
        "QUEUE_MAX_WAIT_ITERATIONS": "100",
        "QUEUE_POLL_SECONDS": "0.05",
        "READY_CAPTURE": str(ready_capture),
        "READY_MARKER": str(tmp_path / "ready.json"),
        "RESULTS": str(results),
        "SETSID_BIN": str(fake_setsid),
        "SHARD_LOG": str(tmp_path / "runner.log"),
        "SHARD_PID_FILE": str(new_runner_pid_file),
        "SSH_ARGS_LOG": str(ssh_args_log),
        "SSH_BIN": str(fake_ssh),
        "STOP_WAIT_SECONDS": "0.1",
        "TRANSITION_LOG": str(tmp_path / "transition.log"),
    }
    new_pids: list[int] = []
    try:
        completed = subprocess.run(
            ["bash", str(TRANSITION)],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        old_finalizer.wait(timeout=3)
        assert not (results / "tasks" / FORBIDDEN_TASK).exists()
        marker = json.loads(ready_capture.read_text(encoding="utf-8"))
        assert marker["schema_version"] == 1
        assert marker["status"] == "ready"
        assert marker["plan"] == "27_18"
        assert marker["boundary_task"] == BOUNDARY_TASK
        assert marker["forbidden_task"] == FORBIDDEN_TASK
        assert marker["old_finalizer_pid"] == old_finalizer.pid
        assert len(marker["boundary_status_sha256"]) == 64
        assert len(marker["boundary_summary_sha256"]) == 64
        assert len(marker["boundary_state_sha256"]) == 64
        ssh_arguments = ssh_args_log.read_text(encoding="utf-8")
        assert "ServerAliveInterval=5" in ssh_arguments
        assert "ServerAliveCountMax=3" in ssh_arguments
        action_deadline = time.monotonic() + 2
        while time.monotonic() < action_deadline:
            actions = action_log.read_text(encoding="utf-8").splitlines()
            if len(actions) == 4:
                break
            time.sleep(0.05)
        assert actions[:2] == ["VALIDATE_039", "PUBLISH_READY"]
        assert set(actions[2:]) == {"NEW_RUNNER", "NEW_FINALIZER"}
        transition_lines = (
            tmp_path / "transition.log"
        ).read_text(encoding="utf-8").splitlines()
        runner_started = next(
            index
            for index, line in enumerate(transition_lines)
            if "STARTED 5090 shard18 runner" in line
        )
        new_queue_verified = next(
            index
            for index, line in enumerate(transition_lines)
            if "NEW_QUEUE_OK total=18" in line
        )
        finalizer_started = next(
            index
            for index, line in enumerate(transition_lines)
            if "STARTED 5090 shard18 finalizer" in line
        )
        assert runner_started < new_queue_verified < finalizer_started
        new_pids = [
            int(new_runner_pid_file.read_text(encoding="utf-8")),
            int(new_finalizer_pid_file.read_text(encoding="utf-8")),
        ]
    finally:
        if old_finalizer.poll() is None:
            os.killpg(old_finalizer.pid, signal.SIGKILL)
            old_finalizer.wait(timeout=3)
        for pid in new_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if all(
                not _process_exists(pid)
                for pid in new_pids
            ):
                break
            time.sleep(0.05)


def test_transition_refuses_to_publish_or_launch_if_task040_was_promoted(
    tmp_path: Path,
):
    results = tmp_path / "results"
    forbidden_result = results / "tasks" / FORBIDDEN_TASK
    forbidden_result.mkdir(parents=True)
    boundary_status = tmp_path / "boundary_status.json"
    boundary_status.write_text(
        json.dumps(
            {
                "state": "completed",
                "task_id": BOUNDARY_TASK,
                "pid": 999999,
                "pgid": 999999,
                "stop_result": "terminated",
            }
        ),
        encoding="utf-8",
    )
    ready_marker = tmp_path / "ready.json"
    runner_pid_file = tmp_path / "shard18.pid"
    finalizer_pid_file = tmp_path / "shard18_finalize.pid"

    completed = subprocess.run(
        ["bash", str(TRANSITION)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "BOUNDARY_STATUS": str(boundary_status),
            "BOUNDARY_VALIDATE_CMD": "true",
            "FINALIZER_PID_FILE": str(finalizer_pid_file),
            "MAX_WAIT_ITERATIONS": "1",
            "OLD_RUNNER_PID_FILE": str(tmp_path / "old_runner.pid"),
            "PYTHON": sys.executable,
            "READY_MARKER": str(ready_marker),
            "RESULTS": str(results),
            "SHARD_PID_FILE": str(runner_pid_file),
            "TRANSITION_LOG": str(tmp_path / "transition.log"),
        },
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode != 0
    assert "obsolete 5090 queue promoted task040" in completed.stdout
    assert not ready_marker.exists()
    assert not runner_pid_file.exists()
    assert not finalizer_pid_file.exists()


def test_transition_does_not_publish_ready_without_local_shard18_scripts(
    tmp_path: Path,
):
    results = tmp_path / "results"
    summary = results / "tasks" / BOUNDARY_TASK / "summary.json"
    state = results / "state" / "tasks" / f"{BOUNDARY_TASK}.json"
    summary.parent.mkdir(parents=True)
    state.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"task_id": BOUNDARY_TASK}),
        encoding="utf-8",
    )
    state.write_text(
        json.dumps({"state": "completed", "task_id": BOUNDARY_TASK}),
        encoding="utf-8",
    )
    boundary_status = tmp_path / "boundary_status.json"
    boundary_status.write_text(
        json.dumps(
            {
                "state": "completed",
                "task_id": BOUNDARY_TASK,
                "pid": 999999,
                "pgid": 999999,
                "stop_result": "terminated",
            }
        ),
        encoding="utf-8",
    )
    old_finalizer_pid_file = tmp_path / "old_finalizer.pid"
    old_finalizer_pid_file.write_text("999998\n", encoding="utf-8")
    present_finalizer = tmp_path / "protocol_v3_finalize_5090_shard18.sh"
    present_finalizer.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    ready_marker = tmp_path / "ready.json"
    ready_capture = tmp_path / "ready_on_relay.json"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\ncat > \"$READY_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)

    completed = subprocess.run(
        ["bash", str(TRANSITION)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "BOUNDARY_STATUS": str(boundary_status),
            "BOUNDARY_VALIDATE_CMD": "true",
            "FINALIZER": str(present_finalizer),
            "LAUNCHER": str(tmp_path / "missing_shard18_launcher.sh"),
            "MAX_WAIT_ITERATIONS": "1",
            "OLD_FINALIZER_PID_FILE": str(old_finalizer_pid_file),
            "OLD_RUNNER_PID_FILE": str(tmp_path / "old_runner.pid"),
            "PYTHON": sys.executable,
            "READY_CAPTURE": str(ready_capture),
            "READY_MARKER": str(ready_marker),
            "RESULTS": str(results),
            "SSH_BIN": str(fake_ssh),
            "TRANSITION_LOG": str(tmp_path / "transition.log"),
        },
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode != 0
    assert "shard18 launcher is missing" in completed.stdout
    assert not ready_marker.exists()
    assert not ready_capture.exists()


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
