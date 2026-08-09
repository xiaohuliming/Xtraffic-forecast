#!/usr/bin/env bash
# Deploy, verify, probe, and plan the nine v3 extension tasks. Does not train.
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
LOCAL_CACHE="${LOCAL_CACHE:-outputs/dist_net/region_data_v3_noleak_dedup}"
LOCAL_GRAPHS="${LOCAL_GRAPHS:-outputs/region_graphs}"
LOCAL_SUPPORT="${LOCAL_SUPPORT:-outputs/reproduction/igstgnn_table4_official_v1/data/official_splits}"
LOCAL_NODE_ORDER="${LOCAL_NODE_ORDER:-archive/node_order.npy}"
LOCAL_UPSTREAM_ROOT="${LOCAL_UPSTREAM_ROOT:-.third_party/igstgnn_table4/pristine}"
GPU_COUNT="${GPU_COUNT:-4}"
CACHE_BUNDLE_ID="${CACHE_BUNDLE_ID:-bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4}"
CACHE_BUNDLE_DIGEST="${CACHE_BUNDLE_DIGEST:-854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5}"
MATRIX_ID="${MATRIX_ID:-protocol_v3_event_extensions_5090_v1}"
HOST_ID="${HOST_ID:-event_extensions_5090}"
STAGE="${STAGE:-/tmp/v3_event_extension_deploy}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1"; }
die() { say "DEPLOY_BLOCKED $1"; exit 1; }

ssh_command=(ssh -n -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
scp_command=(scp -O -P "$SSH_PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
if [ -n "$SSH_PASSWORD" ]; then
  command -v sshpass >/dev/null || die "sshpass is required for password authentication"
  ssh_command=(sshpass -p "$SSH_PASSWORD" "${ssh_command[@]}")
  scp_command=(sshpass -p "$SSH_PASSWORD" "${scp_command[@]}")
fi
rsh() { "${ssh_command[@]}" "$SSH_USER@$SSH_HOST" "$@"; }
rcp() { "${scp_command[@]}" "$1" "$SSH_USER@$SSH_HOST:$2"; }

ship_file() {
  local source="$1" destination="$2"
  local digest remote_digest
  digest="$(sha256sum "$source" | awk '{print $1}')"
  remote_digest="$(rsh "sha256sum '$destination' 2>/dev/null | awk '{print \$1}'" || true)"
  if [ "$digest" = "$remote_digest" ]; then
    return
  fi
  rsh "mkdir -p '$(dirname "$destination")'"
  rcp "$source" "$destination.part"
  rsh "test \"\$(sha256sum '$destination.part' | awk '{print \$1}')\" = '$digest'; mv -f '$destination.part' '$destination'" \
    || die "upload verification failed for $source"
}

say "PREFLIGHT_LOCAL"
git diff --quiet && git diff --cached --quiet \
  || die "tracked or staged files differ from HEAD"
CODE_SHA="$(git rev-parse HEAD)"
/opt/anaconda3/bin/python -m pytest -q \
  tests/data/test_protocol_v3_event_extension_data.py \
  tests/data/test_protocol_v3_event_extension_loading_audit.py \
  tests/remote/test_protocol_v3_event_extension_runner.py \
  tests/remote/test_protocol_v3_event_extension_gpu_probe.py \
  tests/remote/test_protocol_v3_event_extension_multi_gpu.py \
  tests/remote/test_protocol_v3_event_extension_postflight.py \
  tests/remote/test_protocol_v3_event_comparison_report.py \
  tests/reproduction/test_table4_igstgnn_factory.py >/dev/null \
  || die "extension regression failed"

mkdir -p "$STAGE"
BUNDLE="$STAGE/xtraffic-v3-event-extensions.bundle"
git bundle create "$BUNDLE" HEAD >/dev/null 2>&1 || die "git bundle failed"
BUNDLE_SHA="$(sha256sum "$BUNDLE" | awk '{print $1}')"
UPSTREAM_BUNDLE="$STAGE/xtraffic-v3-event-extension-upstreams.tar.gz"
test -d "$LOCAL_UPSTREAM_ROOT/bist_dd94adf" \
  && test -d "$LOCAL_UPSTREAM_ROOT/igstgnn_64c6e24" \
  || die "locked upstream checkouts are missing"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$UPSTREAM_BUNDLE" \
  -C "$LOCAL_UPSTREAM_ROOT" \
  bist_dd94adf igstgnn_64c6e24 \
  || die "locked upstream bundle failed"
UPSTREAM_BUNDLE_SHA="$(sha256sum "$UPSTREAM_BUNDLE" | awk '{print $1}')"

say "PREFLIGHT_REMOTE"
rsh "set -eu; mkdir -p '$REMOTE_BASE' '$REMOTE_CONTROL' '$REMOTE_RESULTS' '$REMOTE_SUPPORT'; command -v git >/dev/null; test -x '$REMOTE_PYTHON'" \
  || die "remote environment is not usable"

say "SHIP_REPO"
ship_file "$BUNDLE" "$REMOTE_BASE/$(basename "$BUNDLE")"
ship_file "$UPSTREAM_BUNDLE" "$REMOTE_BASE/$(basename "$UPSTREAM_BUNDLE")"
rsh "set -eu
test \"\$(sha256sum '$REMOTE_BASE/$(basename "$BUNDLE")' | awk '{print \$1}')\" = '$BUNDLE_SHA'
test \"\$(sha256sum '$REMOTE_BASE/$(basename "$UPSTREAM_BUNDLE")' | awk '{print \$1}')\" = '$UPSTREAM_BUNDLE_SHA'
rm -rf '$REMOTE_REPO'
git clone --quiet '$REMOTE_BASE/$(basename "$BUNDLE")' '$REMOTE_REPO'
cd '$REMOTE_REPO'
git checkout --quiet '$CODE_SHA'
mkdir -p .third_party/igstgnn_table4/pristine
tar --no-same-owner -xzf '$REMOTE_BASE/$(basename "$UPSTREAM_BUNDLE")' \
  -C .third_party/igstgnn_table4/pristine
test \"\$(git rev-parse HEAD)\" = '$CODE_SHA'
test -z \"\$(git status --porcelain)\"" || die "remote checkout identity failed"

say "SHIP_CACHE_AND_GRAPHS"
while IFS= read -r file; do ship_file "$file" "$REMOTE_CACHE/$(basename "$file")"; done \
  < <(find "$LOCAL_CACHE" -maxdepth 1 -type f | sort)
while IFS= read -r file; do ship_file "$file" "$REMOTE_GRAPHS/$(basename "$file")"; done \
  < <(find "$LOCAL_GRAPHS" -maxdepth 1 -type f -name '*_sparse_adj.npz' | sort)

say "SHIP_MODEL_SUPPORT"
for region in Alameda Contra_Costa Orange; do
  for filename in sensors.csv desc_mapping.json type_mapping.json adj_matrix.npy; do
    ship_file "$LOCAL_SUPPORT/$region/$filename" "$REMOTE_SUPPORT/$region/$filename"
  done
done
ship_file "$LOCAL_NODE_ORDER" "$REMOTE_NODE_ORDER"

say "REMOTE_GATES"
rsh "set -eu
cd '$REMOTE_REPO'
mkdir -p archive outputs/reproduction/igstgnn_table4_official_v1/data/official_splits
cp '$REMOTE_NODE_ORDER' archive/node_order.npy
for region in Alameda Contra_Costa Orange; do
  mkdir -p \"outputs/reproduction/igstgnn_table4_official_v1/data/official_splits/\$region\"
  cp '$REMOTE_SUPPORT'/\$region/sensors.csv \
     '$REMOTE_SUPPORT'/\$region/desc_mapping.json \
     '$REMOTE_SUPPORT'/\$region/type_mapping.json \
     '$REMOTE_SUPPORT'/\$region/adj_matrix.npy \
     \"outputs/reproduction/igstgnn_table4_official_v1/data/official_splits/\$region/\"
done
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -X utf8 scripts/audit_protocol_v3_cache.py \
  --data-dir '$REMOTE_CACHE' --graph-dir '$REMOTE_GRAPHS' --no-write-manifest
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -m pytest -q \
  tests/data/test_protocol_v3_event_extension_data.py \
  tests/data/test_protocol_v3_event_extension_loading_audit.py \
  tests/remote/test_protocol_v3_event_extension_runner.py \
  tests/remote/test_protocol_v3_event_extension_gpu_probe.py \
  tests/remote/test_protocol_v3_event_extension_multi_gpu.py \
  tests/remote/test_protocol_v3_event_extension_postflight.py \
  tests/remote/test_protocol_v3_event_comparison_report.py \
  tests/reproduction/test_table4_igstgnn_factory.py
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' - <<'PY'
from scripts.remote.windows.protocol_v3_event_extension_runner import verify_extension_support
value=verify_extension_support(
 support_manifest_path='config/protocol_v3/event_extension_support_v1.json',
 support_root='$REMOTE_SUPPORT', node_order='$REMOTE_NODE_ORDER')
print('SUPPORT_OK',value.manifest_sha256)
PY" || die "remote extension gates failed"

say "EAGER_LAZY_EQUIVALENCE"
rsh "set -euo pipefail
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -X utf8 \
  scripts/audit_protocol_v3_event_extension_loading.py \
  --data-dir '$REMOTE_CACHE' --graph-dir '$REMOTE_GRAPHS' \
  --support-root '$REMOTE_SUPPORT' --node-order '$REMOTE_NODE_ORDER' \
  --batch-limit 20 --output '$REMOTE_CONTROL/loading_equivalence.json'" \
  || die "eager/lazy loading equivalence gate failed"

say "REAL_CUDA_PROBES"
rsh "rm -f '$REMOTE_CONTROL/probes.jsonl' '$REMOTE_CONTROL/calibration.json'"
for region in Alameda ContraCosta Orange; do
  for method in bist igstgnn ndr_net; do
    say "  probe $region $method"
    probe_microbatch=""
    if [ "$region" = "Orange" ] \
      && { [ "$method" = "igstgnn" ] || [ "$method" = "ndr_net" ]; }; then
      probe_microbatch="--max-microbatch-size 8"
    fi
    rsh "set -euo pipefail
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -X utf8 \
  scripts/remote/linux/protocol_v3_event_extension_gpu_probe.py \
  --method '$method' --region '$region' --data-dir '$REMOTE_CACHE' \
  --graph-dir '$REMOTE_GRAPHS' --support-root '$REMOTE_SUPPORT' \
  --node-order '$REMOTE_NODE_ORDER' --device cuda:0 \
  --traffic-loading eager --loader-batches 20 $probe_microbatch | tail -n 1 \
  >> '$REMOTE_CONTROL/probes.jsonl'" || die "CUDA probe failed for $region $method"
  done
done

say "BUILD_CALIBRATION"
rsh "set -eu
cd '$REMOTE_REPO'
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' - <<'PY'
import json
from pathlib import Path
from scripts.remote.windows.protocol_v3_event_extension_runner import load_extension_manifest
probes=[json.loads(line) for line in Path('$REMOTE_CONTROL/probes.jsonl').read_text().splitlines()]
by={(p['region'],p['method']):p for p in probes}
manifest=load_extension_manifest('config/protocol_v3/table4_event_extensions_v1.json')
seconds={}
for task in manifest.tasks:
 p=by[(task.region,task.model)]
 seconds[task.task_id]=p['estimated_task_seconds']
Path('$REMOTE_CONTROL/calibration.json').write_text(json.dumps({'schema_version':1,'task_seconds':seconds},indent=2,sort_keys=True)+'\\n')
PY
PYTHONPATH='$REMOTE_REPO' '$REMOTE_PYTHON' -X utf8 \
  -m scripts.remote.linux.protocol_v3_event_extension_multi_gpu \
  --manifest '$REMOTE_REPO/config/protocol_v3/table4_event_extensions_v1.json' \
  --support-manifest '$REMOTE_REPO/config/protocol_v3/event_extension_support_v1.json' \
  --loading-audit '$REMOTE_CONTROL/loading_equivalence.json' \
  --calibration '$REMOTE_CONTROL/calibration.json' --repo-root '$REMOTE_REPO' \
  --control-root '$REMOTE_CONTROL' --results-root '$REMOTE_RESULTS' \
  --cache-dir '$REMOTE_CACHE' --graph-dir '$REMOTE_GRAPHS' \
  --support-root '$REMOTE_SUPPORT' --node-order '$REMOTE_NODE_ORDER' \
  --cache-bundle-id '$CACHE_BUNDLE_ID' --cache-bundle-digest '$CACHE_BUNDLE_DIGEST' \
  --code-sha '$CODE_SHA' --matrix-id '$MATRIX_ID' --host-id '$HOST_ID' \
  --gpu-count '$GPU_COUNT' --python '$REMOTE_PYTHON' --plan-only" \
  || die "extension plan emission failed"

say "DEPLOY_READY code_sha=$CODE_SHA matrix_id=$MATRIX_ID"
say "review $REMOTE_CONTROL/plan.json before launching training"
