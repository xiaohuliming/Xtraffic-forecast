#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
finalizer="$repo_root/scripts/remote/linux/protocol_v3_event_nas_finalize.sh"
tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT

bin_root="$tmp_root/bin"
mkdir -p "$bin_root"

cat > "$bin_root/ssh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ssh %s\n' "$*" >> "$ACTION_LOG"
# Decode the base64 payload so the fake NAS can react to the real remote script.
payload=""
for arg in "$@"; do
  case "$arg" in
    echo*base64*) payload="$(printf '%s' "$arg" | sed -E 's/^echo ([A-Za-z0-9+/=]+) .*/\1/' | base64 -d 2>/dev/null || true)" ;;
  esac
done
if printf '%s' "$payload" | grep -q 'grep -F'; then
  printf 'TREE_VERIFY payload_checked\n' >> "$ACTION_LOG"
  if [ "${FAKE_REMOTE_BOUND:-1}" != "1" ]; then
    exit 1
  fi
fi
if printf '%s' "$payload" | grep -q 'df -Pk'; then
  printf '%s\n' "/dev/x 100000000 1 99999999 1% /nas"
fi
if printf '%s' "$payload" | grep -q 'sha256sum .* awk'; then
  printf 'MISSING\n'
fi
if printf '%s' "$*" | grep -q '/opt/xtraffic/bin/full33-post-publish'; then
  exit "${FAKE_POST_PUBLISH_RC:-0}"
fi
exit "${FAKE_SSH_RC:-0}"
SH
chmod +x "$bin_root/ssh"

cat > "$bin_root/scp" <<'SH'
#!/usr/bin/env bash
printf 'scp %s\n' "$*" >> "$ACTION_LOG"
exit 0
SH
chmod +x "$bin_root/scp"

cat > "$bin_root/shutdown" <<'SH'
#!/usr/bin/env bash
printf 'shutdown %s\n' "$*" >> "$ACTION_LOG"
touch "$SHUTDOWN_MARKER"
SH
chmod +x "$bin_root/shutdown"

fake_postflight="$tmp_root/fake_postflight.py"
cat > "$fake_postflight" <<'PY'
import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--plan")
parser.add_argument("--report")
parser.add_argument("--file-list")
parser.add_argument("--transfer-manifest")
parser.add_argument("--marker")
args = parser.parse_args()
if os.environ.get("FAKE_POSTFLIGHT_RC", "0") != "0":
    sys.exit(int(os.environ["FAKE_POSTFLIGHT_RC"]))
report = {
    "status": "verified",
    "host_id": os.environ.get("FAKE_POSTFLIGHT_HOST", "event_test_host"),
    "validated_tasks": int(os.environ.get("FAKE_POSTFLIGHT_TASKS", "22")),
    "file_count": 1,
    "total_bytes": 10,
    "files": [],
    "files_digest": os.environ.get("FAKE_FILES_DIGEST", "f" * 64),
    "plan_sha256": os.environ.get("FAKE_PLAN_SHA", "e" * 64),
}
open(args.report, "w").write(json.dumps(report))
open(args.file_list, "w").write("")
open(args.transfer_manifest, "w").write("")
open(args.marker, "w").write("{}")
PY

setup_case() {
  local case_dir="$1" state="$2" total="$3" completed="$4" current="$5" failed="$6" rc="$7"
  mkdir -p "$case_dir/control" "$case_dir/results"
  python3 - "$case_dir/results/queue_status.json" \
    "$state" "$total" "$completed" "$current" "$failed" <<'PY'
import json, sys
path, state, total, completed, current, failed = sys.argv[1:7]
json.dump(
    {
        "state": state,
        "total_tasks": int(total),
        "completed_tasks": int(completed),
        "current_task": None if current == "null" else current,
        "failed_count": int(failed),
    },
    open(path, "w"),
)
PY
  printf '%s\n' "$rc" > "$case_dir/control/runner.exitcode"
  printf '{}\n' > "$case_dir/control/nas_backup_plan.json"
}

run_case() {
  local case_dir="$1"
  ACTION_LOG="$case_dir/actions.log" \
  SHUTDOWN_MARKER="$case_dir/shutdown.marker" \
  PATH="$bin_root:$PATH" \
  REPO="$repo_root" \
  CONTROL_ROOT="$case_dir/control" \
  RESULTS_ROOT="$case_dir/results" \
  PYTHON=python3 \
  PLAN="$case_dir/control/nas_backup_plan.json" \
  CANONICAL_MANIFEST="$repo_root/config/protocol_v3/table4_event_baselines_phase1_v1.json" \
  POSTFLIGHT_SCRIPT="$fake_postflight" \
  NAS="user@nas" \
  NAS_BASE="/nas/base" \
  HOST_ID="${CASE_HOST_ID:-event_test_host}" \
  SSH_KEY="$case_dir/control/nas_backup_plan.json" \
  EXPECTED_TASKS=22 \
  POLL_SECONDS=1 \
  SHUTDOWN_ON_SUCCESS="${CASE_SHUTDOWN:-0}" \
  POST_PUBLISH_COMMAND="${CASE_POST_PUBLISH_COMMAND:-}" \
  FAKE_POST_PUBLISH_RC="${FAKE_POST_PUBLISH_RC:-0}" \
  FAKE_SSH_RC="${FAKE_SSH_RC:-0}" \
  FAKE_POSTFLIGHT_RC="${FAKE_POSTFLIGHT_RC:-0}" \
  FAKE_POSTFLIGHT_HOST="${FAKE_POSTFLIGHT_HOST:-event_test_host}" \
  FAKE_POSTFLIGHT_TASKS="${FAKE_POSTFLIGHT_TASKS:-22}" \
  FAKE_REMOTE_BOUND="${FAKE_REMOTE_BOUND:-1}" \
  FAKE_FILES_DIGEST="${FAKE_FILES_DIGEST:-$(printf 'f%.0s' $(seq 64))}" \
    bash "$finalizer" > "$case_dir/stdout.log" 2>&1
}

expect_blocked() {
  local name="$1" case_dir="$2"
  if run_case "$case_dir"; then
    echo "FAIL $name: finalizer exited zero but should have blocked"
    cat "$case_dir/stdout.log"
    exit 1
  fi
  if ! grep -q "FINALIZE_BLOCKED" "$case_dir/stdout.log"; then
    echo "FAIL $name: no FINALIZE_BLOCKED marker"
    cat "$case_dir/stdout.log"
    exit 1
  fi
  if [ -e "$case_dir/shutdown.marker" ]; then
    echo "FAIL $name: blocked run must never shut the host down"
    exit 1
  fi
  echo "ok  $name"
}

case_dir="$tmp_root/failed_queue"
setup_case "$case_dir" failed 22 3 null 1 0
expect_blocked "failed queue blocks" "$case_dir"

case_dir="$tmp_root/short_total"
setup_case "$case_dir" completed 13 13 null 0 0
expect_blocked "wrong total task count blocks" "$case_dir"

case_dir="$tmp_root/short_completed"
setup_case "$case_dir" completed 22 21 null 0 0
expect_blocked "incomplete queue blocks" "$case_dir"

case_dir="$tmp_root/current_task_set"
setup_case "$case_dir" completed 22 22 v3e033_event_d2stgnn_orange_seed42 0 0
expect_blocked "lingering current_task blocks" "$case_dir"

case_dir="$tmp_root/failed_count"
setup_case "$case_dir" completed 22 22 null 1 0
expect_blocked "nonzero failed_count blocks" "$case_dir"

case_dir="$tmp_root/runner_exit"
setup_case "$case_dir" completed 22 22 null 0 1
expect_blocked "nonzero runner exit code blocks" "$case_dir"

case_dir="$tmp_root/postflight_fail"
setup_case "$case_dir" completed 22 22 null 0 0
FAKE_POSTFLIGHT_RC=1 expect_blocked "postflight failure blocks" "$case_dir"

case_dir="$tmp_root/host_mismatch"
setup_case "$case_dir" completed 22 22 null 0 0
FAKE_POSTFLIGHT_HOST=other_host expect_blocked "host_id mismatch blocks" "$case_dir"

case_dir="$tmp_root/task_count_mismatch"
setup_case "$case_dir" completed 22 22 null 0 0
FAKE_POSTFLIGHT_TASKS=21 expect_blocked "postflight task count mismatch blocks" "$case_dir"

case_dir="$tmp_root/already_verified"
setup_case "$case_dir" completed 22 22 null 0 0
if ! run_case "$case_dir"; then
  echo "FAIL already verified tree should succeed"
  cat "$case_dir/stdout.log"
  exit 1
fi
grep -q "FINALIZE_SUCCESS" "$case_dir/stdout.log" || {
  echo "FAIL missing FINALIZE_SUCCESS"; cat "$case_dir/stdout.log"; exit 1; }
grep -q "already_verified_bound_to_this_round" "$case_dir/stdout.log" || {
  echo "FAIL should have short circuited on the bound remote tree"; exit 1; }
grep -q "TREE_VERIFY payload_checked" "$case_dir/actions.log" || {
  echo "FAIL tree verification did not run the digest binding check"; exit 1; }
if grep -q "^scp " "$case_dir/actions.log" 2>/dev/null; then
  echo "FAIL verified tree must not re-upload"
  exit 1
fi
[ ! -e "$case_dir/shutdown.marker" ] || { echo "FAIL default must not shut down"; exit 1; }
echo "ok  verified remote tree short circuits without upload"

case_dir="$tmp_root/shutdown_opt_in"
setup_case "$case_dir" completed 22 22 null 0 0
CASE_SHUTDOWN=1 run_case "$case_dir"
[ -e "$case_dir/shutdown.marker" ] || { echo "FAIL opt-in shutdown did not fire"; exit 1; }
echo "ok  shutdown only fires when explicitly requested"

case_dir="$tmp_root/post_publish_failure"
setup_case "$case_dir" completed 22 22 null 0 0
CASE_POST_PUBLISH_COMMAND=/opt/xtraffic/bin/full33-post-publish \
FAKE_POST_PUBLISH_RC=1 \
  expect_blocked "post-publish composite failure blocks" "$case_dir"

case_dir="$tmp_root/post_publish_success"
setup_case "$case_dir" completed 22 22 null 0 0
CASE_POST_PUBLISH_COMMAND=/opt/xtraffic/bin/full33-post-publish run_case "$case_dir"
grep -q '/opt/xtraffic/bin/full33-post-publish' "$case_dir/actions.log" || {
  echo "FAIL post-publish command did not run"; exit 1; }
grep -q "FINALIZE_SUCCESS" "$case_dir/stdout.log" || {
  echo "FAIL post-publish success did not finalize"; exit 1; }
echo "ok  post-publish composite gates finalizer success"

case_dir="$tmp_root/stale_unbound_tree"
setup_case "$case_dir" completed 22 22 null 0 0
FAKE_REMOTE_BOUND=0 expect_blocked "stale remote tree not bound to this round blocks" "$case_dir"
if grep -q "FINALIZE_SUCCESS" "$case_dir/stdout.log"; then
  echo "FAIL stale tree must never be reported as success"
  exit 1
fi
echo "ok  stale remote tree is never accepted as this round"

if ! grep -q 'verify_remote_tree "$REMOTE_FINAL" "$files_digest" "$plan_sha"' "$finalizer"; then
  echo "FAIL final tree verification is not bound to this round's digests"
  exit 1
fi
if ! grep -q 'verify_remote_tree "$REMOTE_INCOMING" "$files_digest" "$plan_sha"' "$finalizer"; then
  echo "FAIL incoming tree verification is not bound to this round's digests"
  exit 1
fi
echo "ok  every tree verification is bound to this round"

echo "ALL_EVENT_NAS_FINALIZE_TESTS_PASSED"
