#!/usr/bin/env bash
# Revalidate the published extension tree and opportunistically merge the final table.
set -Eeuo pipefail

PUBLISHED_ROOT="${1:?published host root is required}"
REPO="${REPO:-/opt/xtraffic/v3_event_repo}"
PYTHON="${PYTHON:-/opt/xtraffic/v3env/bin/python}"
STATE_BASE="${STATE_BASE:-/var/lib/xtraffic/protocol_v3_event_extensions}"
MATRIX_ID="${MATRIX_ID:-protocol_v3_event_extensions_5090x4_v1}"
HOST_ID="${HOST_ID:-event_rtx5090x4_extensions}"
EXPECTED_SOURCES="${EXPECTED_SOURCES:-4}"
FINAL_ROOT="$STATE_BASE/$HOST_ID"

fail() { printf 'EVENT_EXTENSION_POST_PUBLISH_FAILED %s\n' "$1" >&2; exit 1; }
case "$PUBLISHED_ROOT" in /*) : ;; *) fail "published root must be absolute" ;; esac
[ "$(basename "$PUBLISHED_ROOT")" = "$HOST_ID" ] || fail "published host mismatch"

if [ "${XTRAFFIC_COMPOSITE_WORKER:-0}" != "1" ]; then
  script="$(readlink -f "$0")"
  unit="xtraffic-v3-extension-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  exec systemd-run --quiet --wait --pipe --collect --unit="$unit" \
    --property=MemoryMax=3221225472 --property=MemorySwapMax=4294967296 \
    --property=TasksMax=128 --property=Nice=10 \
    --setenv=XTRAFFIC_COMPOSITE_WORKER=1 \
    --setenv="REPO=$REPO" --setenv="PYTHON=$PYTHON" \
    --setenv="STATE_BASE=$STATE_BASE" --setenv="MATRIX_ID=$MATRIX_ID" \
    --setenv="HOST_ID=$HOST_ID" --setenv="EXPECTED_SOURCES=$EXPECTED_SOURCES" \
    "$script" "$PUBLISHED_ROOT"
fi

[ -x "$PYTHON" ] || fail "verified Python environment is missing"
[ -d "$REPO/.git" ] || fail "verified repository checkout is missing"
cd "$REPO"
test -z "$(git status --porcelain)" || fail "repository checkout is dirty"
mkdir -p "$STATE_BASE"
attempt="$(mktemp -d "$STATE_BASE/.${HOST_ID}.attempt.XXXXXXXX")"
trap 'rm -rf "$attempt"' EXIT

PYTHONPATH="$REPO" "$PYTHON" -X utf8 \
  -m scripts.remote.linux.protocol_v3_event_extension_composite \
  --published-root "$PUBLISHED_ROOT" --output "$attempt/composite_index.json" \
  --expected-matrix-id "$MATRIX_ID" --expected-host-id "$HOST_ID" \
  --expected-sources "$EXPECTED_SOURCES"

if [ -d "$FINAL_ROOT" ]; then
  "$PYTHON" - "$FINAL_ROOT/composite_index.json" "$attempt/composite_index.json" <<'PY'
import json,sys
old,new=(json.load(open(path)) for path in sys.argv[1:])
old.pop("verified_at",None); new.pop("verified_at",None)
if old != new: raise SystemExit("existing extension composite differs")
PY
  rm -rf "$attempt"
  trap - EXIT
else
  mv "$attempt" "$FINAL_ROOT"
  trap - EXIT
  sync
fi

REPO="$REPO" PYTHON="$PYTHON" EXTENSION_STATE="$FINAL_ROOT" \
  bash "$REPO/scripts/remote/linux/protocol_v3_event_merge_verified.sh"
printf 'EVENT_EXTENSION_POST_PUBLISH_OK state=%s\n' "$FINAL_ROOT"
