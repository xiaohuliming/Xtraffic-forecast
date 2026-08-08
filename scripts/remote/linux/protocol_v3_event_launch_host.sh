#!/usr/bin/env bash
# Launch the deployed v3 event multi-GPU controller and its NAS finalizer.
set -Eeuo pipefail

SSH_HOST="${SSH_HOST:?SSH_HOST is required}"
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
REMOTE_BASE="${REMOTE_BASE:-/root/autodl-tmp}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/v3_event_cc_orange_repo}"
REMOTE_CACHE="${REMOTE_CACHE:-$REMOTE_BASE/region_data_v3_noleak_dedup}"
REMOTE_GRAPHS="${REMOTE_GRAPHS:-$REMOTE_BASE/region_graphs}"
REMOTE_CONTROL="${REMOTE_CONTROL:-$REMOTE_BASE/v3_event_cc_orange/control}"
REMOTE_RESULTS="${REMOTE_RESULTS:-$REMOTE_BASE/v3_event_cc_orange/results}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/miniconda3/bin/python}"
CACHE_BUNDLE_ID="${CACHE_BUNDLE_ID:-bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4}"
CACHE_BUNDLE_DIGEST="${CACHE_BUNDLE_DIGEST:-854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5}"
MATRIX_ID="${MATRIX_ID:-protocol_v3_event_cc_orange_5090x4_v1}"
HOST_ID="${HOST_ID:-event_rtx5090x4_cc_orange}"
REGIONS="${REGIONS:-ContraCosta Orange}"
GPU_COUNT="${GPU_COUNT:-4}"
MICROBATCH_ARGS="${MICROBATCH_ARGS:-}"
ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"
CALIBRATION="${CALIBRATION:-$REMOTE_CONTROL/calibration_measured.json}"
EXPECTED_TASKS="${EXPECTED_TASKS:?EXPECTED_TASKS is required and must match the plan}"
NAS="${NAS:-}"
NAS_BASE="${NAS_BASE:-/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/hosts}"
NAS_KEY="${NAS_KEY:-$REMOTE_CONTROL/id_ed25519}"
SHUTDOWN_ON_SUCCESS="${SHUTDOWN_ON_SUCCESS:-0}"

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"; }
die() { say "LAUNCH_BLOCKED $1"; exit 1; }

rsh() {
  ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    "$SSH_USER@$SSH_HOST" "$@"
}

CODE_SHA="$(rsh "cd '$REMOTE_REPO' && git rev-parse HEAD" | tr -d '[:space:]')"
[ -n "$CODE_SHA" ] || die "cannot read remote code SHA"
rsh "cd '$REMOTE_REPO' && test -z \"\$(git status --porcelain)\"" \
  || die "remote worktree is dirty"
rsh "test -f '$CALIBRATION'" \
  || die "measured calibration is missing at $CALIBRATION"
rsh "test ! -e '$REMOTE_RESULTS/queue_status.json'" \
  || die "results root already carries a queue; refusing to overwrite"

: "${EXPECTED_PLAN_DIGEST:?EXPECTED_PLAN_DIGEST is required; take it from the reviewed plan.json}"

# The deploy step already wrote a plan.json. Archive it so its mere existence
# can never be mistaken for evidence that this launch produced a live plan.
rsh "set -eu
if [ -f '$REMOTE_CONTROL/plan.json' ]; then
  mv '$REMOTE_CONTROL/plan.json' '$REMOTE_CONTROL/plan.reviewed.json'
fi
rm -f '$REMOTE_CONTROL/controller.pid'" \
  || die "cannot archive the reviewed plan"

say "LAUNCH_CONTROLLER code_sha=$CODE_SHA gpus=$GPU_COUNT digest=$EXPECTED_PLAN_DIGEST"
rsh "set -eu
cd '$REMOTE_REPO'
mkdir -p '$REMOTE_CONTROL' '$REMOTE_RESULTS'
PYTHONPATH='$REMOTE_REPO' nohup '$REMOTE_PYTHON' -X utf8 \
  -m scripts.remote.linux.protocol_v3_event_multi_gpu \
  --calibration '$CALIBRATION' \
  --control-root '$REMOTE_CONTROL' --results-root '$REMOTE_RESULTS' \
  --cache-dir '$REMOTE_CACHE' --graph-dir '$REMOTE_GRAPHS' \
  --cache-bundle-id '$CACHE_BUNDLE_ID' \
  --cache-bundle-digest '$CACHE_BUNDLE_DIGEST' \
  --code-sha '$CODE_SHA' --matrix-id '$MATRIX_ID' --host-id '$HOST_ID' \
  --regions $REGIONS --gpu-count $GPU_COUNT $MICROBATCH_ARGS \
  --alloc-conf '$ALLOC_CONF' \
  --expect-plan-digest '$EXPECTED_PLAN_DIGEST' \
  --python '$REMOTE_PYTHON' \
  >>'$REMOTE_CONTROL/controller.out' 2>&1 &
echo \$! > '$REMOTE_CONTROL/controller.launch.pid'" \
  || die "controller did not start"

sleep 30

rsh "set -eu
test -s '$REMOTE_CONTROL/plan.json'
test -s '$REMOTE_CONTROL/controller.pid'
kill -0 \$(cat '$REMOTE_CONTROL/controller.pid')
test -s '$REMOTE_CONTROL/controller_status.json'" \
  || die "controller is not alive with a freshly written plan; inspect $REMOTE_CONTROL/controller.out"

LIVE_DIGEST="$(rsh "$REMOTE_PYTHON -c \"import json;print(json.load(open('$REMOTE_CONTROL/plan.json'))['plan_digest'])\"" | tr -d '[:space:]')"
[ "$LIVE_DIGEST" = "$EXPECTED_PLAN_DIGEST" ] \
  || die "live plan digest $LIVE_DIGEST does not match the reviewed $EXPECTED_PLAN_DIGEST"

LIVE_TASKS="$(rsh "$REMOTE_PYTHON -c \"import json;print(json.load(open('$REMOTE_CONTROL/plan.json'))['task_count'])\"" | tr -d '[:space:]')"
[ "$LIVE_TASKS" = "$EXPECTED_TASKS" ] \
  || die "live plan has $LIVE_TASKS tasks but the finalizer would be armed for $EXPECTED_TASKS"

LIVE_STATE="$(rsh "$REMOTE_PYTHON -c \"import json;print(json.load(open('$REMOTE_CONTROL/controller_status.json'))['state'])\"" | tr -d '[:space:]')"
[ "$LIVE_STATE" = "running" ] || die "controller state is $LIVE_STATE, expected running"

say "CONTROLLER_PID=$(rsh "cat '$REMOTE_CONTROL/controller.pid'" | tr -d '[:space:]') state=$LIVE_STATE"

if [ -z "$NAS" ]; then
  say "LAUNCH_OK nas_finalizer_skipped set NAS to arm automatic backup"
  exit 0
fi

rsh "test -f '$NAS_KEY'" || die "NAS key is missing at $NAS_KEY"
say "ARM_NAS_FINALIZER target=$NAS_BASE/$HOST_ID"
rsh "set -eu
cd '$REMOTE_REPO'
REPO='$REMOTE_REPO' CONTROL_ROOT='$REMOTE_CONTROL' RESULTS_ROOT='$REMOTE_RESULTS' \
PYTHON='$REMOTE_PYTHON' NAS='$NAS' NAS_BASE='$NAS_BASE' HOST_ID='$HOST_ID' \
SSH_KEY='$NAS_KEY' EXPECTED_TASKS='$EXPECTED_TASKS' SHUTDOWN_ON_SUCCESS='$SHUTDOWN_ON_SUCCESS' \
nohup bash scripts/remote/linux/protocol_v3_event_nas_finalize.sh \
  >>'$REMOTE_CONTROL/finalize.out' 2>&1 &
echo \$! > '$REMOTE_CONTROL/finalize.pid'
sleep 5
kill -0 \$(cat '$REMOTE_CONTROL/finalize.pid')" \
  || die "NAS finalizer did not stay alive"

say "LAUNCH_OK finalizer_pid=$(rsh "cat '$REMOTE_CONTROL/finalize.pid'" | tr -d '[:space:]')"
