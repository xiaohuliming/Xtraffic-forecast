#!/usr/bin/env bash
# Fail-closed source-host postflight and NAS publication for a Linux event shard.
set -Eeuo pipefail

REPO="${REPO:?REPO is required}"
CONTROL_ROOT="${CONTROL_ROOT:?CONTROL_ROOT is required}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
PYTHON="${PYTHON:-python3}"
PLAN="${PLAN:-$CONTROL_ROOT/nas_backup_plan.json}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:-$REPO/config/protocol_v3/table4_event_baselines_phase1_v1.json}"
SUPPORT_MANIFEST="${SUPPORT_MANIFEST:-}"
POSTFLIGHT_SCRIPT="${POSTFLIGHT_SCRIPT:-$REPO/scripts/remote/windows/protocol_v3_event_host_postflight.py}"
NAS="${NAS:?NAS is required}"
NAS_BASE="${NAS_BASE:?NAS_BASE is required}"
HOST_ID="${HOST_ID:?HOST_ID is required}"
SSH_KEY="${SSH_KEY:?SSH_KEY is required}"
EXPECTED_TASKS="${EXPECTED_TASKS:-22}"
RESERVE_BYTES="${RESERVE_BYTES:-134217728}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG="${LOG:-$CONTROL_ROOT/finalize.log}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-0}"
POST_PUBLISH_COMMAND="${POST_PUBLISH_COMMAND:-}"

REPORT="$CONTROL_ROOT/postflight.json"
FILE_LIST="$CONTROL_ROOT/transfer.files"
TRANSFER_MANIFEST="$CONTROL_ROOT/transfer.sha256"
MARKER="$CONTROL_ROOT/_TRANSFER_COMPLETE.json"
QUEUE_STATUS="$RESULTS_ROOT/queue_status.json"
RUNNER_EXITCODE="$CONTROL_ROOT/runner.exitcode"
REMOTE_INCOMING="$NAS_BASE/.$HOST_ID.incoming"
REMOTE_FINAL="$NAS_BASE/$HOST_ID"
PLAN_NAME="$(basename "$PLAN")"
MANIFEST_NAME="$(basename "$CANONICAL_MANIFEST")"
SUPPORT_MANIFEST_NAME=""
if [ -n "$SUPPORT_MANIFEST" ]; then
  SUPPORT_MANIFEST_NAME="$(basename "$SUPPORT_MANIFEST")"
fi

mkdir -p "$CONTROL_ROOT"
: >>"$LOG"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$LOG"
}

blocked() {
  log "FINALIZE_BLOCKED $1"
  exit 1
}

trap 'log "FINALIZE_BLOCKED unexpected error at line $LINENO"' ERR

ssh_run() {
  ssh -n -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=6 "$NAS" "$@"
}

# Remote scripts are shipped base64 encoded so no quoting layer can corrupt them.
ssh_script() {
  local script encoded
  script="$1"
  encoded="$(printf '%s' "$script" | base64 | tr -d '\n')"
  ssh_run "echo $encoded | base64 -d | bash -s"
}

local_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

json_get() {
  "$PYTHON" -c '
import json,sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value.get(key) if isinstance(value, dict) else None
print("" if value is None else value)
' "$1" "$2"
}

assert_safe_file() {
  local path="$1" expected="$2" size="${3:--1}"
  [ -f "$path" ] || blocked "required file is missing: $path"
  [ ! -L "$path" ] || blocked "symbolic link is forbidden: $path"
  if [ "$size" != "-1" ]; then
    local actual
    actual="$(stat -c %s "$path")"
    [ "$actual" = "$size" ] || blocked "file size changed: $path"
  fi
  [ "$(local_sha256 "$path")" = "$expected" ] || blocked "file hash changed: $path"
}

send_verified_file() {
  local local_path="$1" sha="$2" remote_path="$3" size="${4:--1}"
  assert_safe_file "$local_path" "$sha" "$size"
  case "$remote_path" in
    */*) : ;;
    *) blocked "unsafe NAS path: $remote_path" ;;
  esac
  local parent part state
  parent="$(dirname "$remote_path")"
  part="$remote_path.part"
  state="$(ssh_script "set -eu
mkdir -p '$parent'
if [ -f '$remote_path' ] && [ \"\$(sha256sum '$remote_path' | awk '{print \$1}')\" = '$sha' ]; then
  echo MATCH
else
  rm -f '$part' '$remote_path'
  echo UPLOAD
fi" | tail -1 | tr -d '\r')"
  if [ "$state" = "MATCH" ]; then
    log "TRANSFER_SKIP path=$remote_path"
    return 0
  fi
  [ "$state" = "UPLOAD" ] || blocked "unexpected NAS state for $remote_path: $state"
  scp -O -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=6 \
    "$local_path" "$NAS:$part" >/dev/null
  ssh_script "set -eu
test \"\$(sha256sum '$part' | awk '{print \$1}')\" = '$sha'
mv -f '$part' '$remote_path'
sync" >/dev/null
  log "TRANSFER_FILE_OK path=$remote_path"
}

verify_remote_tree() {
  local root="$1" expect_files_digest="$2" expect_plan_sha="$3"
  local expected_metadata
  expected_metadata="'_TRANSFER_COMPLETE.json' '$PLAN_NAME' 'postflight.json' '$MANIFEST_NAME' 'transfer.files' 'transfer.sha256'"
  if [ -n "$SUPPORT_MANIFEST_NAME" ]; then
    expected_metadata="$expected_metadata '$SUPPORT_MANIFEST_NAME'"
  fi
  ssh_script "set -eu
root='$root'
test -d \"\$root/results\"
test -d \"\$root/metadata\"
test -z \"\$(find \"\$root\" -type l -print -quit)\"
cd \"\$root/results\"
sha256sum -c ../metadata/transfer.sha256 >/dev/null
tmp=\$(mktemp -d)
trap 'rm -rf \"\$tmp\"' EXIT
cut -c67- ../metadata/transfer.sha256 | LC_ALL=C sort > \"\$tmp/expected\"
find . -type f | sed 's#^./##' | LC_ALL=C sort > \"\$tmp/actual\"
cmp \"\$tmp/expected\" \"\$tmp/actual\"
printf '%s\n' $expected_metadata | LC_ALL=C sort > \"\$tmp/expected_meta\"
find ../metadata -maxdepth 1 -type f | sed 's#^.*/##' | LC_ALL=C sort > \"\$tmp/actual_meta\"
cmp \"\$tmp/expected_meta\" \"\$tmp/actual_meta\"
grep -F '\"files_digest\": \"$expect_files_digest\"' ../metadata/_TRANSFER_COMPLETE.json >/dev/null
grep -F '\"plan_sha256\": \"$expect_plan_sha\"' ../metadata/_TRANSFER_COMPLETE.json >/dev/null" >/dev/null 2>&1
}

for required in "$PYTHON" "$POSTFLIGHT_SCRIPT" "$PLAN" "$CANONICAL_MANIFEST" "$SSH_KEY"; do
  case "$required" in
    /*) [ -e "$required" ] || blocked "required control file is missing: $required" ;;
    *) command -v "$required" >/dev/null || blocked "required command is missing: $required" ;;
  esac
done
if [ -n "$SUPPORT_MANIFEST" ]; then
  [ -f "$SUPPORT_MANIFEST" ] || blocked "support manifest is missing: $SUPPORT_MANIFEST"
  [ ! -L "$SUPPORT_MANIFEST" ] || blocked "support manifest cannot be a symlink"
fi

while true; do
  if [ ! -f "$QUEUE_STATUS" ]; then
    log "WAIT queue_status_missing"
    sleep "$POLL_SECONDS"
    continue
  fi
  state="$(json_get "$QUEUE_STATUS" state)"
  completed="$(json_get "$QUEUE_STATUS" completed_tasks)"
  total="$(json_get "$QUEUE_STATUS" total_tasks)"
  current="$(json_get "$QUEUE_STATUS" current_task)"
  failed="$(json_get "$QUEUE_STATUS" failed_count)"
  if [ "$state" = "failed" ]; then
    blocked "completion queue entered failed state"
  fi
  if [ "$state" != "completed" ]; then
    log "WAIT queue_state=$state completed=$completed/$EXPECTED_TASKS"
    sleep "$POLL_SECONDS"
    continue
  fi
  if [ "$total" != "$EXPECTED_TASKS" ] || [ "$completed" != "$EXPECTED_TASKS" ] \
     || [ -n "$current" ] || [ "${failed:-0}" != "0" ]; then
    blocked "completed queue identity is not $EXPECTED_TASKS/$EXPECTED_TASKS"
  fi
  if [ ! -f "$RUNNER_EXITCODE" ]; then
    log "WAIT runner_exitcode_missing"
    sleep "$POLL_SECONDS"
    continue
  fi
  runner_code="$(tr -d '[:space:]' <"$RUNNER_EXITCODE")"
  [ "$runner_code" = "0" ] || blocked "runner exit code is $runner_code"
  trainers="$( { pgrep -f 'protocol_v3_baseline_runner|protocol_v3_event_multi_gpu|protocol_v3_event_extension_runner|protocol_v3_event_extension_multi_gpu|train_protocol_v3_event_extension|train_baseline_zoo|train_graphwavenet' 2>/dev/null || true; } | wc -l | tr -d '[:space:]')"
  if [ "${trainers:-0}" != "0" ]; then
    log "WAIT trainer_processes=$trainers"
    sleep "$POLL_SECONDS"
    continue
  fi
  break
done

log "POSTFLIGHT_START"
XTRAFFIC_REPO_ROOT="$REPO" PYTHONPATH="$REPO" "$PYTHON" -X utf8 "$POSTFLIGHT_SCRIPT" \
  --plan "$PLAN" --report "$REPORT" --file-list "$FILE_LIST" \
  --transfer-manifest "$TRANSFER_MANIFEST" --marker "$MARKER" \
  || blocked "source-host postflight failed"

status="$(json_get "$REPORT" status)"
report_host="$(json_get "$REPORT" host_id)"
validated="$(json_get "$REPORT" validated_tasks)"
file_count="$(json_get "$REPORT" file_count)"
total_bytes="$(json_get "$REPORT" total_bytes)"
files_digest="$(json_get "$REPORT" files_digest)"
plan_sha="$(json_get "$REPORT" plan_sha256)"
[ -n "$files_digest" ] || blocked "postflight report has no files_digest"
[ -n "$plan_sha" ] || blocked "postflight report has no plan_sha256"
[ "$status" = "verified" ] || blocked "postflight status is $status"
[ "$report_host" = "$HOST_ID" ] || blocked "postflight host_id mismatch"
[ "$validated" = "$EXPECTED_TASKS" ] || blocked "postflight validated $validated tasks"
log "POSTFLIGHT_OK tasks=$validated files=$file_count bytes=$total_bytes"

finish_success() {
  log "FINALIZE_SUCCESS destination=$REMOTE_FINAL $1"
  if [ "$SHUTDOWN_ON_SUCCESS" = "1" ]; then
    log "SHUTDOWN_REQUESTED"
    shutdown -h now || log "SHUTDOWN_COMMAND_FAILED host_stays_on"
  fi
  exit 0
}

run_post_publish() {
  [ -n "$POST_PUBLISH_COMMAND" ] || return 0
  case "$POST_PUBLISH_COMMAND" in
    /*) : ;;
    *) blocked "POST_PUBLISH_COMMAND must be an absolute path" ;;
  esac
  case "$POST_PUBLISH_COMMAND" in
    *[!A-Za-z0-9_./-]*) blocked "POST_PUBLISH_COMMAND contains unsafe characters" ;;
  esac
  log "POST_PUBLISH_START command=$POST_PUBLISH_COMMAND"
  ssh_run "$POST_PUBLISH_COMMAND '$REMOTE_FINAL'" \
    || blocked "remote post-publish composite failed"
  log "POST_PUBLISH_OK"
}

if verify_remote_tree "$REMOTE_FINAL" "$files_digest" "$plan_sha"; then
  run_post_publish
  finish_success "already_verified_bound_to_this_round"
fi

reusable="$("$PYTHON" - "$REPORT" <<'PY'
import json,sys
print("\n".join(
    f"{item['sha256']}\t{item['size_bytes']}\t{item['path']}\t{item['source_path']}"
    for item in json.load(open(sys.argv[1]))["files"]
))
PY
)"

reusable_bytes=0
while IFS=$'\t' read -r sha size relative source_path; do
  [ -n "$relative" ] || continue
  case "$relative" in
    /*|*..*|*\\*) blocked "unsafe export path: $relative" ;;
  esac
  state="$(ssh_script "set -eu
if [ -f '$REMOTE_INCOMING/results/$relative' ] && [ \"\$(sha256sum '$REMOTE_INCOMING/results/$relative' | awk '{print \$1}')\" = '$sha' ]; then
  echo MATCH
else
  echo MISSING
fi" | tail -1 | tr -d '\r')"
  [ "$state" = "MATCH" ] || [ "$state" = "MISSING" ] \
    || blocked "unexpected NAS state for $relative: $state"
  if [ "$state" = "MATCH" ]; then
    reusable_bytes=$((reusable_bytes + size))
  fi
done <<<"$reusable"

capacity="$(ssh_script "set -eu
mkdir -p '$NAS_BASE'
test ! -e '$REMOTE_FINAL'
mkdir -p '$REMOTE_INCOMING/results' '$REMOTE_INCOMING/metadata'
df -Pk '$NAS_BASE'" | tail -1)"
available_kb="$(printf '%s' "$capacity" | awk '{print $4}')"
case "$available_kb" in
  ''|*[!0-9]*) blocked "NAS capacity response is invalid" ;;
esac
available_bytes=$((available_kb * 1024))
required_bytes=$((total_bytes + RESERVE_BYTES))
if [ $((available_bytes + reusable_bytes)) -lt "$required_bytes" ]; then
  blocked "NAS capacity gate failed available=$available_bytes reusable=$reusable_bytes required=$required_bytes"
fi

log "TRANSFER_START files=$file_count bytes=$total_bytes reusable_bytes=$reusable_bytes"
while IFS=$'\t' read -r sha size relative source_path; do
  [ -n "$relative" ] || continue
  send_verified_file "$source_path" "$sha" "$REMOTE_INCOMING/results/$relative" "$size"
done <<<"$reusable"

for entry in \
  "$REPORT:postflight.json" \
  "$FILE_LIST:transfer.files" \
  "$TRANSFER_MANIFEST:transfer.sha256" \
  "$MARKER:_TRANSFER_COMPLETE.json" \
  "$PLAN:$PLAN_NAME" \
  "$CANONICAL_MANIFEST:$MANIFEST_NAME"; do
  local_path="${entry%%:*}"
  remote_name="${entry##*:}"
  send_verified_file "$local_path" "$(local_sha256 "$local_path")" \
    "$REMOTE_INCOMING/metadata/$remote_name"
done
if [ -n "$SUPPORT_MANIFEST" ]; then
  send_verified_file "$SUPPORT_MANIFEST" "$(local_sha256 "$SUPPORT_MANIFEST")" \
    "$REMOTE_INCOMING/metadata/$SUPPORT_MANIFEST_NAME"
fi

verify_remote_tree "$REMOTE_INCOMING" "$files_digest" "$plan_sha" \
  || blocked "incoming NAS tree failed exact verification"
ssh_script "set -eu
test ! -e '$REMOTE_FINAL'
mv '$REMOTE_INCOMING' '$REMOTE_FINAL'
sync" >/dev/null
verify_remote_tree "$REMOTE_FINAL" "$files_digest" "$plan_sha" \
  || blocked "promoted NAS tree failed exact verification"

run_post_publish
finish_success "published"
