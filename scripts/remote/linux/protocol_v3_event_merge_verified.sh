#!/usr/bin/env bash
# Generate the final 42-row table only when both verified indexes exist.
set -Eeuo pipefail

REPO="${REPO:-/opt/xtraffic/v3_event_repo}"
PYTHON="${PYTHON:-/opt/xtraffic/v3env/bin/python}"
BASELINE_STATE="${BASELINE_STATE:-/var/lib/xtraffic/protocol_v3_event_full33/event_rtx5090x4_full33}"
EXTENSION_STATE="${EXTENSION_STATE:-/var/lib/xtraffic/protocol_v3_event_extensions/event_rtx5090x4_extensions}"
FINAL_ROOT="${FINAL_ROOT:-/var/lib/xtraffic/protocol_v3_event_comparison/no_leak_42_v1}"

for index in "$BASELINE_STATE/composite_index.json" "$EXTENSION_STATE/composite_index.json"; do
  if [ ! -f "$index" ]; then
    printf 'EVENT_COMPARISON_PENDING missing=%s\n' "$index"
    exit 0
  fi
done

baseline_root="$($PYTHON -c 'import json,sys;print(json.load(open(sys.argv[1]))["published_root"])' "$BASELINE_STATE/composite_index.json")"
extension_root="$($PYTHON -c 'import json,sys;print(json.load(open(sys.argv[1]))["published_root"])' "$EXTENSION_STATE/composite_index.json")"
baseline_sha="$(sha256sum "$BASELINE_STATE/composite_index.json" | awk '{print $1}')"
extension_sha="$(sha256sum "$EXTENSION_STATE/composite_index.json" | awk '{print $1}')"
mkdir -p "$(dirname "$FINAL_ROOT")"
attempt="$(mktemp -d "$(dirname "$FINAL_ROOT")/.no_leak_42.attempt.XXXXXXXX")"
trap 'rm -rf "$attempt"' EXIT

cd "$REPO"
PYTHONPATH="$REPO" "$PYTHON" -X utf8 \
  -m scripts.protocol_v3_event_comparison_report \
  --baseline-index "$BASELINE_STATE/composite_index.json" \
  --baseline-index-sha256 "$baseline_sha" \
  --baseline-published-root "$baseline_root" \
  --extension-index "$EXTENSION_STATE/composite_index.json" \
  --extension-index-sha256 "$extension_sha" \
  --extension-published-root "$extension_root" \
  --output-dir "$attempt"

if [ -d "$FINAL_ROOT" ]; then
  "$PYTHON" - "$FINAL_ROOT" "$attempt" <<'PY'
from pathlib import Path
import sys
old, new = map(Path, sys.argv[1:])
for name in ("table.csv", "table.md", "table.tex", "manifest.json"):
    if (old / name).read_bytes() != (new / name).read_bytes():
        raise SystemExit(f"existing verified comparison differs: {name}")
PY
  printf 'EVENT_COMPARISON_OK reused=%s\n' "$FINAL_ROOT"
  exit 0
fi

mv "$attempt" "$FINAL_ROOT"
trap - EXIT
sync
printf 'EVENT_COMPARISON_OK published=%s\n' "$FINAL_ROOT"
