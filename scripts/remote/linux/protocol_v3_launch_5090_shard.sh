#!/usr/bin/env bash
set -uo pipefail

REPO=/root/xtraffic_v3_5090_clean
OPS_ROOT=/root/autodl-tmp/xtraffic_ops
OPS="$OPS_ROOT/scripts/remote/distributed/protocol_v3_distributed.py"
PYTHON=/root/miniconda3/bin/python
RESULTS=/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090
EXIT_CODE=/root/autodl-tmp/protocol_v3_5090_shard.exitcode

export PYTHONUTF8=1
export PYTHONPATH="$OPS_ROOT:$REPO"
export XTRAFFIC_REPO_ROOT="$REPO"

rm -f "$EXIT_CODE"
cd "$REPO" || exit 2

"$PYTHON" -X utf8 "$OPS" \
  --shard 5090 \
  --manifest "$REPO/scripts/remote/windows/protocol_v3_manifest.json" \
  --repo-root "$REPO" \
  --results-root "$RESULTS" \
  --cache-dir "$REPO/outputs/dist_net/region_data_v3_noleak_dedup" \
  --graph-dir "$REPO/outputs/region_graphs" \
  --cache-bundle-id "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4" \
  --cache-bundle-digest \
    "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5" \
  --code-sha "3a4952837aaf6fbda1403d50b610b1bf40c6098a" \
  --python "$PYTHON"
runner_rc=$?

printf '%s\n' "$runner_rc" > "$EXIT_CODE"
exit "$runner_rc"
