#!/usr/bin/env bash
# Launch the reviewed v3 extension plan and arm strict publication.
set -Eeuo pipefail

SSH_HOST="${SSH_HOST:?SSH_HOST is required}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
REMOTE_BASE="${REMOTE_BASE:-/root/autodl-tmp/v3_event_extensions}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/repo}"
REMOTE_CACHE="${REMOTE_CACHE:-/root/autodl-tmp/region_data_v3_noleak_dedup}"
REMOTE_GRAPHS="${REMOTE_GRAPHS:-/root/autodl-tmp/region_graphs}"
REMOTE_SUPPORT="${REMOTE_SUPPORT:-$REMOTE_BASE/model_support}"
REMOTE_NODE_ORDER="${REMOTE_NODE_ORDER:-$REMOTE_BASE/node_order.npy}"
REMOTE_CONTROL="${REMOTE_CONTROL:-$REMOTE_BASE/control}"
REMOTE_RESULTS="${REMOTE_RESULTS:-$REMOTE_BASE/results}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/miniconda3/bin/python}"
CALIBRATION="${CALIBRATION:-$REMOTE_CONTROL/calibration.json}"
GPU_COUNT="${GPU_COUNT:-4}"
CACHE_BUNDLE_ID="${CACHE_BUNDLE_ID:-bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4}"
CACHE_BUNDLE_DIGEST="${CACHE_BUNDLE_DIGEST:-854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5}"
MATRIX_ID="${MATRIX_ID:-protocol_v3_event_extensions_5090x4_v1}"
HOST_ID="${HOST_ID:-event_rtx5090x4_extensions}"
NAS="${NAS:?NAS is required}"
NAS_BASE="${NAS_BASE:-/var/lib/xtraffic/published/protocol_v3_noleak_dedup/event_extensions/hosts}"
NAS_KEY="${NAS_KEY:-$REMOTE_CONTROL/id_ed25519}"
POST_PUBLISH_COMMAND="${POST_PUBLISH_COMMAND:-/opt/xtraffic/bin/event-extension-post-publish}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-1}"
EXPECTED_PLAN_DIGEST="${EXPECTED_PLAN_DIGEST:?EXPECTED_PLAN_DIGEST is required}"

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"; }
die() { say "EXTENSION_LAUNCH_BLOCKED $1"; exit 1; }

ssh_command=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
if [ -n "$SSH_PASSWORD" ]; then
  command -v sshpass >/dev/null || die "sshpass is required for password authentication"
  ssh_command=(sshpass -p "$SSH_PASSWORD" "${ssh_command[@]}")
fi
rsh() { "${ssh_command[@]}" "$SSH_USER@$SSH_HOST" "$@"; }

CODE_SHA="$(rsh "cd '$REMOTE_REPO' && git rev-parse HEAD" | tr -d '[:space:]')"
[ -n "$CODE_SHA" ] || die "cannot read remote code SHA"
rsh "cd '$REMOTE_REPO' && test -z \"\$(git status --porcelain)\"" \
  || die "remote worktree is dirty"
rsh "test -s '$CALIBRATION'; test -f '$NAS_KEY'; test ! -e '$REMOTE_RESULTS/queue_status.json'" \
  || die "calibration, transfer key, or clean results root gate failed"

rsh "set -eu
if [ -f '$REMOTE_CONTROL/plan.json' ]; then
  mv '$REMOTE_CONTROL/plan.json' '$REMOTE_CONTROL/plan.reviewed.json'
fi
rm -f '$REMOTE_CONTROL/controller.pid' '$REMOTE_CONTROL/runner.exitcode'" \
  || die "cannot archive the reviewed plan"

say "START_EXTENSION_CONTROLLER code_sha=$CODE_SHA digest=$EXPECTED_PLAN_DIGEST"
rsh "set -eu
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' nohup '$REMOTE_PYTHON' -X utf8 \
  -m scripts.remote.linux.protocol_v3_event_extension_multi_gpu \
  --manifest '$REMOTE_REPO/config/protocol_v3/table4_event_extensions_v1.json' \
  --support-manifest '$REMOTE_REPO/config/protocol_v3/event_extension_support_v1.json' \
  --calibration '$CALIBRATION' --repo-root '$REMOTE_REPO' \
  --control-root '$REMOTE_CONTROL' --results-root '$REMOTE_RESULTS' \
  --cache-dir '$REMOTE_CACHE' --graph-dir '$REMOTE_GRAPHS' \
  --support-root '$REMOTE_SUPPORT' --node-order '$REMOTE_NODE_ORDER' \
  --cache-bundle-id '$CACHE_BUNDLE_ID' --cache-bundle-digest '$CACHE_BUNDLE_DIGEST' \
  --code-sha '$CODE_SHA' --matrix-id '$MATRIX_ID' --host-id '$HOST_ID' \
  --gpu-count '$GPU_COUNT' --expect-plan-digest '$EXPECTED_PLAN_DIGEST' \
  --python '$REMOTE_PYTHON' >>'$REMOTE_CONTROL/controller.out' 2>&1 &
echo \$! > '$REMOTE_CONTROL/controller.launch.pid'" \
  || die "extension controller did not start"

sleep 20
rsh "set -eu
test -s '$REMOTE_CONTROL/controller.pid'
kill -0 \$(cat '$REMOTE_CONTROL/controller.pid')
test -s '$REMOTE_CONTROL/plan.json'
test -s '$REMOTE_CONTROL/nas_backup_plan.json'
test -s '$REMOTE_CONTROL/controller_status.json'" \
  || die "extension controller failed its live gate"

LIVE_DIGEST="$(rsh "$REMOTE_PYTHON -c \"import json;print(json.load(open('$REMOTE_CONTROL/plan.json'))['plan_digest'])\"" | tr -d '[:space:]')"
[ "$LIVE_DIGEST" = "$EXPECTED_PLAN_DIGEST" ] || die "live plan digest mismatch"
LIVE_TASKS="$(rsh "$REMOTE_PYTHON -c \"import json;print(json.load(open('$REMOTE_CONTROL/plan.json'))['task_count'])\"" | tr -d '[:space:]')"
[ "$LIVE_TASKS" = "9" ] || die "live plan is not the nine-task matrix"

say "ARM_EXTENSION_FINALIZER destination=$NAS_BASE/$HOST_ID"
rsh "set -eu
cd '$REMOTE_REPO'
REPO='$REMOTE_REPO' CONTROL_ROOT='$REMOTE_CONTROL' RESULTS_ROOT='$REMOTE_RESULTS' \
PYTHON='$REMOTE_PYTHON' PLAN='$REMOTE_CONTROL/nas_backup_plan.json' \
CANONICAL_MANIFEST='$REMOTE_REPO/config/protocol_v3/table4_event_extensions_v1.json' \
SUPPORT_MANIFEST='$REMOTE_REPO/config/protocol_v3/event_extension_support_v1.json' \
NAS='$NAS' NAS_BASE='$NAS_BASE' HOST_ID='$HOST_ID' SSH_KEY='$NAS_KEY' \
EXPECTED_TASKS=9 POST_PUBLISH_COMMAND='$POST_PUBLISH_COMMAND' \
SHUTDOWN_ON_SUCCESS='$SHUTDOWN_ON_SUCCESS' \
nohup bash scripts/remote/linux/protocol_v3_event_nas_finalize.sh \
  >>'$REMOTE_CONTROL/finalize.out' 2>&1 &
echo \$! > '$REMOTE_CONTROL/finalize.pid'
sleep 5
kill -0 \$(cat '$REMOTE_CONTROL/finalize.pid')" \
  || die "extension finalizer did not stay alive"

say "EXTENSION_LAUNCH_OK controller_pid=$(rsh "cat '$REMOTE_CONTROL/controller.pid'" | tr -d '[:space:]') finalizer_pid=$(rsh "cat '$REMOTE_CONTROL/finalize.pid'" | tr -d '[:space:]')"
