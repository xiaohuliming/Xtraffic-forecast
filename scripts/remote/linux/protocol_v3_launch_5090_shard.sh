#!/usr/bin/env bash
set -uo pipefail

REPO=${REPO:-/root/xtraffic_v3_5090_clean}
OPS_ROOT=${OPS_ROOT:-/root/autodl-tmp/xtraffic_ops_27_18}
OPS=${OPS:-$OPS_ROOT/scripts/remote/distributed/protocol_v3_distributed.py}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
RESULTS=${RESULTS:-/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090}
EXIT_CODE=${EXIT_CODE:-/root/autodl-tmp/protocol_v3_5090_shard18.exitcode}
MANIFEST=${MANIFEST:-$REPO/scripts/remote/windows/protocol_v3_manifest.json}

export PYTHONUTF8=1
export PYTHONPATH="$OPS_ROOT:$REPO"
export XTRAFFIC_REPO_ROOT="$REPO"

rm -f "$EXIT_CODE"
cd "$REPO" || exit 2

"$PYTHON" -X utf8 - "$OPS_ROOT" "$MANIFEST" <<'PY'
import sys
from pathlib import Path

ops_root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ops_root))

from scripts.remote.distributed import protocol_v3_distributed as distributed

manifest = distributed.runner.load_manifest(
    manifest_path,
    require_production_matrix=True,
)
shards = distributed.build_shard_manifests(manifest)
selected = shards["5090"]
actual_numbers = {
    distributed.task_number(task)
    for task in selected.tasks
}
expected_numbers = {
    5,
    6,
    11,
    12,
    17,
    18,
    *range(31, 40),
    *range(43, 46),
}
expected_inherited = {
    5,
    6,
    11,
    12,
    17,
    18,
    *range(31, 40),
}
if actual_numbers != expected_numbers:
    raise SystemExit(
        "5090 shard contract mismatch: "
        f"expected={sorted(expected_numbers)} actual={sorted(actual_numbers)}"
    )
actual_inherited = set(
    distributed.SHARD_SPECS["5090"].expected_skip_numbers
)
if actual_inherited != expected_inherited:
    raise SystemExit(
        "5090 inherited skip contract mismatch: "
        f"expected={sorted(expected_inherited)} "
        f"actual={sorted(actual_inherited)}"
    )
if 40 in actual_numbers or len(selected.tasks) != 18:
    raise SystemExit("obsolete task040 assignment remains in 5090 shard")
print("SHARD_PLAN_OK shard=5090 tasks=18 inherited=15")
PY
plan_rc=$?
if [[ "$plan_rc" -ne 0 ]]; then
  printf '%s\n' "$plan_rc" > "$EXIT_CODE"
  exit "$plan_rc"
fi

"$PYTHON" -X utf8 "$OPS" \
  --shard 5090 \
  --manifest "$MANIFEST" \
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
