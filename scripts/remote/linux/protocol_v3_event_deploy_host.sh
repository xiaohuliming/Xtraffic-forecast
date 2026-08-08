#!/usr/bin/env bash
# Idempotent fail-closed deployment of a v3 event multi-GPU shard onto a Linux host.
# Stops before launching training. Launch is a separate authorized step.
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
PROBE_MODELS="${PROBE_MODELS:-dgcrn d2stgnn sttn}"
LOCAL_CACHE="${LOCAL_CACHE:-outputs/dist_net/region_data_v3_noleak_dedup}"
LOCAL_GRAPHS="${LOCAL_GRAPHS:-outputs/region_graphs}"
STAGE="${STAGE:-/tmp/v3_event_deploy}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"; }
die() { say "DEPLOY_BLOCKED $1"; exit 1; }

rsh() {
  ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    "$SSH_USER@$SSH_HOST" "$@"
}

rcp() {
  scp -O -P "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
    "$1" "$SSH_USER@$SSH_HOST:$2"
}

say "PREFLIGHT_LOCAL"
[ -z "$(git status --porcelain)" ] || die "local worktree is dirty"
CODE_SHA="$(git rev-parse HEAD)"
say "code_sha=$CODE_SHA"
python3 -m pytest -q tests/remote tests/training tests/data >/dev/null \
  || die "local regression failed"

mkdir -p "$STAGE"
BUNDLE="$STAGE/xtraffic-v3-event-cc-orange.bundle"
git bundle create "$BUNDLE" HEAD >/dev/null 2>&1 || die "git bundle failed"
BUNDLE_SHA="$(sha256sum "$BUNDLE" | awk '{print $1}')"
say "bundle_sha256=$BUNDLE_SHA"

say "PREFLIGHT_REMOTE"
rsh "set -eu; mkdir -p '$REMOTE_BASE' '$REMOTE_CONTROL' '$REMOTE_RESULTS'; \
  command -v git >/dev/null; test -x '$REMOTE_PYTHON'" \
  || die "remote base environment is not usable"

say "SHIP_REPO"
rcp "$BUNDLE" "$REMOTE_BASE/$(basename "$BUNDLE")"
rsh "set -eu
test \"\$(sha256sum '$REMOTE_BASE/$(basename "$BUNDLE")' | awk '{print \$1}')\" = '$BUNDLE_SHA'
rm -rf '$REMOTE_REPO'
git clone --quiet '$REMOTE_BASE/$(basename "$BUNDLE")' '$REMOTE_REPO'
cd '$REMOTE_REPO'
git checkout --quiet '$CODE_SHA'
test \"\$(git rev-parse HEAD)\" = '$CODE_SHA'
test -z \"\$(git status --porcelain)\"" \
  || die "remote checkout identity failed"

ship_tree() {
  local local_dir="$1" remote_dir="$2" pattern="$3"
  say "SHIP_TREE $local_dir"
  rsh "mkdir -p '$remote_dir'"
  local file base local_sha remote_sha
  while IFS= read -r file; do
    base="$(basename "$file")"
    local_sha="$(sha256sum "$file" | awk '{print $1}')"
    remote_sha="$(rsh "sha256sum '$remote_dir/$base' 2>/dev/null | awk '{print \$1}'" || true)"
    if [ "$local_sha" = "$remote_sha" ]; then
      say "  reuse $base"
      continue
    fi
    say "  upload $base"
    rcp "$file" "$remote_dir/$base.part"
    rsh "set -eu
test \"\$(sha256sum '$remote_dir/$base.part' | awk '{print \$1}')\" = '$local_sha'
mv -f '$remote_dir/$base.part' '$remote_dir/$base'" \
      || die "upload verification failed for $base"
  done < <(find "$local_dir" -maxdepth 1 -type f -name "$pattern" | sort)
}

ship_tree "$LOCAL_CACHE" "$REMOTE_CACHE" '*'
ship_tree "$LOCAL_GRAPHS" "$REMOTE_GRAPHS" '*_sparse_adj.npz'

say "VERIFY_CACHE_READONLY"
rsh "set -eu
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -X utf8 scripts/audit_protocol_v3_cache.py \
  --data-dir '$REMOTE_CACHE' --no-write-manifest" \
  || die "remote strict cache audit failed"

say "REMOTE_GATE_TESTS"
rsh "set -eu
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -m pytest -q tests/remote tests/data" \
  || die "remote gate tests failed"

say "PROBE"
rsh "set -eu
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -X utf8 \
  scripts/remote/linux/protocol_v3_event_gpu_probe.py \
  --region Orange --models $PROBE_MODELS \
  --data-dir '$REMOTE_CACHE' --graph-dir '$REMOTE_GRAPHS' \
  --device cuda:0 --batch-size 32 --microbatch-size 0 \
  --output '$REMOTE_CONTROL/calibration_measured.json'" \
  || die "GPU probe failed; if PROBE_OOM was reported, rerun with a microbatch size and isolate that task"

say "EMIT_PLAN"
rsh "set -eu
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -X utf8 \
  -m scripts.remote.linux.protocol_v3_event_multi_gpu \
  --calibration '$REMOTE_CONTROL/calibration_measured.json' \
  --control-root '$REMOTE_CONTROL' --results-root '$REMOTE_RESULTS' \
  --cache-dir '$REMOTE_CACHE' --graph-dir '$REMOTE_GRAPHS' \
  --cache-bundle-id '$CACHE_BUNDLE_ID' \
  --cache-bundle-digest '$CACHE_BUNDLE_DIGEST' \
  --code-sha '$CODE_SHA' --matrix-id '$MATRIX_ID' --host-id '$HOST_ID' \
  --regions $REGIONS --gpu-count $GPU_COUNT --plan-only" \
  || die "plan emission failed"

say "DEPLOY_READY code_sha=$CODE_SHA host_id=$HOST_ID"
say "review $REMOTE_CONTROL/plan.json, then launch with protocol_v3_event_launch_host.sh"
