#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
finalizer="$repo_root/scripts/remote/linux/igstgnn_table4_finalize_host.sh"
tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT

bin_root="$tmp_root/bin"
mkdir -p "$bin_root"

cat > "$bin_root/ssh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ssh %s\n' "$*" >> "$ACTION_LOG"
if [[ "$*" == *"df -PB1"* ]]; then
  printf '%s %s %s\n' \
    "${FAKE_REMOTE_FREE:-1000000000}" \
    "${FAKE_REMOTE_INCOMING_BYTES:-0}" \
    "${FAKE_REMOTE_FINAL_EXISTS:-0}"
elif [[ "$*" == *"sha256sum -c"* ]]; then
  exit "${FAKE_VERIFY_RC:-0}"
fi
SH
chmod +x "$bin_root/ssh"

cat > "$bin_root/rsync" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'rsync %s\n' "$*" >> "$ACTION_LOG"
exit "${FAKE_RSYNC_RC:-0}"
SH
chmod +x "$bin_root/rsync"

cat > "$bin_root/shutdown" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'shutdown %s\n' "$*" >> "$ACTION_LOG"
touch "$SHUTDOWN_MARKER"
SH
chmod +x "$bin_root/shutdown"

cat > "$tmp_root/postflight.py" <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RESULTS_BASE"])
report_path = Path(os.environ["POSTFLIGHT_REPORT"])
paths = ["shards/example/queue_status.json", "shards/example/tasks/t/summary.json"]
records = []
for relative in paths:
    path = root / relative
    records.append(
        {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
files_digest = hashlib.sha256(
    json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
report_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "verified",
            "protocol_id": "igstgnn_table4_official_v1",
            "matrix_id": "igstgnn_table4_public_repro_39_v2",
            "plan_id": "igstgnn_table4_4090d_9gpu_v1",
            "plan_sha256": hashlib.sha256(
                Path(os.environ["PLAN"]).read_bytes()
            ).hexdigest(),
            "manifest_sha256": "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78",
            "host_id": os.environ["HOST_ID"],
            "gpu_sku": "NVIDIA GeForce RTX 4090 D",
            "execution_layout": "dual_host_gpu_sku_homogeneous",
            "verified_at": "2026-07-28T12:00:00Z",
            "runtime_host_identity": {"hostname": "test"},
            "validated_shards": 1,
            "validated_tasks": 2,
            "queues": [],
            "tasks": [],
            "files": records,
            "export_paths": paths,
            "file_count": len(records),
            "total_bytes": sum(record["size_bytes"] for record in records),
            "files_digest": files_digest,
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY

make_case() {
  local name=$1
  local root="$tmp_root/$name"
  mkdir -p "$root/results/shards/example/tasks/t"
  printf '%s\n' '{"state":"completed"}' \
    > "$root/results/shards/example/queue_status.json"
  printf '%s\n' '{"status":"completed"}' \
    > "$root/results/shards/example/tasks/t/summary.json"
  printf '%s\n' '{"plan":"test"}' > "$root/plan.json"
  printf '%s\n' '{"manifest":"test"}' > "$root/manifest.json"
}

run_finalizer() {
  local name=$1
  local root="$tmp_root/$name"
  export RESULTS_BASE="$root/results"
  export POSTFLIGHT_REPORT="$root/postflight.json"
  export PLAN="$root/plan.json"
  export MANIFEST="$root/manifest.json"
  export HOST_ID=table4_4090d5
  export ACTION_LOG="$root/actions.log"
  export SHUTDOWN_MARKER="$root/shutdown.called"
  POSTFLIGHT_CMD="python3 '$tmp_root/postflight.py'" \
  FINALIZER_ONCE=1 \
  POSTFLIGHT_REPORT="$POSTFLIGHT_REPORT" \
  TRANSFER_FILE_LIST="$root/transfer.files" \
  TRANSFER_MANIFEST="$root/transfer.sha256" \
  TRANSFER_MARKER="$root/complete.json" \
  SSH_BIN="$bin_root/ssh" \
  RSYNC_BIN="$bin_root/rsync" \
  SHUTDOWN_BIN="$bin_root/shutdown" \
  PYTHON=python3 \
  SPACE_MARGIN_BYTES=1 \
  RELAY=root@relay.test \
  RELAY_BASE=/srv/results/table4/hosts \
  REMOTE_NAME=table4_4090d5 \
  LOG="$root/finalizer.log" \
  bash "$finalizer"
}

make_case success
run_finalizer success
test -f "$tmp_root/success/shutdown.called"
grep -q '^shutdown -h now$' "$tmp_root/success/actions.log"
grep -q 'sha256sum -c' "$tmp_root/success/actions.log"
grep -q 'FINALIZE_SUCCESS' "$tmp_root/success/finalizer.log"
printf '%s\n' "PASS verified transfer shuts the host down"

make_case remote_verification_failure
set +e
FAKE_VERIFY_RC=8 run_finalizer remote_verification_failure
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$tmp_root/remote_verification_failure/shutdown.called"
grep -q 'remote exact tree or hash verification failed' \
  "$tmp_root/remote_verification_failure/finalizer.log"
printf '%s\n' "PASS remote verification failure leaves the host on"

make_case postflight_failure
root="$tmp_root/postflight_failure"
export RESULTS_BASE="$root/results"
export POSTFLIGHT_REPORT="$root/postflight.json"
export PLAN="$root/plan.json"
export MANIFEST="$root/manifest.json"
export HOST_ID=table4_4090d5
export ACTION_LOG="$root/actions.log"
export SHUTDOWN_MARKER="$root/shutdown.called"
set +e
POSTFLIGHT_CMD=false \
FINALIZER_ONCE=1 \
SSH_BIN="$bin_root/ssh" \
RSYNC_BIN="$bin_root/rsync" \
SHUTDOWN_BIN="$bin_root/shutdown" \
PYTHON=python3 \
LOG="$root/finalizer.log" \
bash "$finalizer"
rc=$?
set -e
test "$rc" -ne 0
test ! -e "$root/shutdown.called"
printf '%s\n' "PASS failed postflight leaves the host on"
