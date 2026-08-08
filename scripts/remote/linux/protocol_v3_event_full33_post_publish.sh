#!/usr/bin/env bash
# Revalidate a published full-33 tree and generate the paper-shaped table.
set -Eeuo pipefail

PUBLISHED_ROOT="${1:?published host root is required}"
REPO="${REPO:-/opt/xtraffic/v3_event_repo}"
PYTHON="${PYTHON:-/opt/xtraffic/v3env/bin/python}"
STATE_BASE="${STATE_BASE:-/var/lib/xtraffic/protocol_v3_event_full33}"
MATRIX_ID="protocol_v3_event_full33_5090x4_v1"
HOST_ID="event_rtx5090x4_full33"
FINAL_ROOT="$STATE_BASE/$HOST_ID"

fail() {
  printf 'FULL33_POST_PUBLISH_FAILED %s\n' "$1" >&2
  exit 1
}

case "$PUBLISHED_ROOT" in
  /*) : ;;
  *) fail "published root must be absolute" ;;
esac
[ "$(basename "$PUBLISHED_ROOT")" = "$HOST_ID" ] \
  || fail "published host identity mismatch"

if [ "${XTRAFFIC_COMPOSITE_WORKER:-0}" != "1" ]; then
  script="$(readlink -f "$0")"
  unit="xtraffic-v3-full33-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  exec systemd-run --quiet --wait --pipe --collect \
    --unit="$unit" \
    --property=MemoryMax=3221225472 \
    --property=MemorySwapMax=4294967296 \
    --property=TasksMax=128 \
    --property=Nice=10 \
    --setenv=XTRAFFIC_COMPOSITE_WORKER=1 \
    "$script" "$PUBLISHED_ROOT"
fi

[ -x "$PYTHON" ] || fail "verified Python environment is missing"
[ -d "$REPO/.git" ] || fail "verified repository checkout is missing"
cd "$REPO"
test -z "$(git status --porcelain)" || fail "repository checkout is dirty"

mkdir -p "$STATE_BASE"
attempt="$(mktemp -d "$STATE_BASE/.${HOST_ID}.attempt.XXXXXXXX")"
trap 'rm -rf "$attempt"' EXIT
index="$attempt/composite_index.json"
table="$attempt/table"

PYTHONPATH="$REPO" "$PYTHON" -X utf8 \
  -m scripts.remote.linux.protocol_v3_event_full33_composite \
  --published-root "$PUBLISHED_ROOT" \
  --output "$index" \
  --expected-matrix-id "$MATRIX_ID" \
  --expected-host-id "$HOST_ID"
index_sha="$(sha256sum "$index" | awk '{print $1}')"
PYTHONPATH="$REPO" "$PYTHON" -X utf8 \
  -m scripts.protocol_v3_event_baseline_report \
  --composite-index "$index" \
  --expected-index-sha256 "$index_sha" \
  --published-root "$PUBLISHED_ROOT" \
  --output-dir "$table"

if [ -d "$FINAL_ROOT" ]; then
  "$PYTHON" - "$FINAL_ROOT" "$attempt" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

old_root, new_root = map(Path, sys.argv[1:])
old_index = json.loads((old_root / "composite_index.json").read_text())
new_index = json.loads((new_root / "composite_index.json").read_text())
old_index.pop("verified_at", None)
new_index.pop("verified_at", None)
if old_index != new_index:
    raise SystemExit("existing composite is not bound to the current published tree")

manifest = json.loads((old_root / "table/manifest.json").read_text())
old_index_sha = hashlib.sha256(
    (old_root / "composite_index.json").read_bytes()
).hexdigest()
if manifest.get("composite_index_sha256") != old_index_sha:
    raise SystemExit("existing table manifest is not bound to its composite index")
for name in ("table.csv", "table.md", "table.tex"):
    old = old_root / "table" / name
    new = new_root / "table" / name
    if old.read_bytes() != new.read_bytes():
        raise SystemExit(f"existing table differs from fresh audit: {name}")
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    if manifest.get("outputs", {}).get(name, {}).get("sha256") != digest:
        raise SystemExit(f"existing table hash is invalid: {name}")
PY
  printf 'FULL33_POST_PUBLISH_OK reused_verified=%s\n' "$FINAL_ROOT"
  exit 0
fi

mv "$attempt" "$FINAL_ROOT"
trap - EXIT
sync
printf 'FULL33_POST_PUBLISH_OK published=%s index_sha256=%s\n' \
  "$FINAL_ROOT" "$index_sha"
