#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$ROOT_DIR/scripts/remote/linux/igstgnn_table4_native100_5090_shutdown_waiter.sh"
TEST_PYTHON=${TEST_PYTHON:-/opt/anaconda3/bin/python}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/run/results" "$TMP/run/control" "$TMP/bin"
cat > "$TMP/run/results/queue_status.json" <<'JSON'
{
  "state": "running",
  "failed_count": 0,
  "verified_task_count": 4
}
JSON
cat > "$TMP/bin/shutdown" <<EOF
#!/usr/bin/env bash
touch "$TMP/shutdown.called"
EOF
chmod +x "$TMP/bin/shutdown"
cat > "$TMP/bin/pgrep" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
chmod +x "$TMP/bin/pgrep"

set +e
RUN_ROOT="$TMP/run" \
WAIT_ONCE=1 \
SHUTDOWN_BIN="$TMP/bin/shutdown" \
PYTHON="$TEST_PYTHON" \
PGREP_BIN="$TMP/bin/pgrep" \
bash "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e

[[ "$rc" != "0" ]]
[[ ! -e "$TMP/shutdown.called" ]]
grep -Fq "NATIVE100_SHUTDOWN_BLOCKED queue is not complete" \
  "$TMP/run/control/shutdown_waiter.log"
echo "verified: incomplete native100 queue keeps the RTX 5090 powered on"

success_root="$TMP/success"
mkdir -p "$success_root/results" "$success_root/control"
"$TEST_PYTHON" - "$ROOT_DIR" "$success_root/results" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

project = Path(sys.argv[1])
results = Path(sys.argv[2])
sys.path.insert(0, str(project))

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    load_native100_compare_manifest,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
)

config = project / "config" / "reproduction" / "igstgnn_table4"
manifest_path = config / "matrix.native100_compare_v1.json"
base_manifest = config / "matrix.public_material_v2.json"
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = load_native100_compare_manifest(
    manifest_path,
    expected_sha256=sha256(manifest_path),
    base_manifest_path=base_manifest,
    expected_base_manifest_sha256=sha256(base_manifest),
)
selected_ids = [
    "n100007_bist_orange_seed42",
    "n100008_bist_orange_seed1",
    "n100009_bist_orange_seed2",
    "n100016_ndr_net_orange_seed42",
    "n100017_ndr_net_orange_seed1",
    "n100018_ndr_net_orange_seed2",
]
checkpoint = results / "checkpoint.pt"
checkpoint.write_bytes(b"checkpoint\n")
train_log = results / "train.log"
train_log.write_text("complete\n", encoding="utf-8")
labels = np.full((2, 12, 3), 10.0, dtype=np.float32)
for task in manifest["tasks"]:
    if task["task_id"] not in selected_ids:
        continue
    write_completed_task_artifacts(
        target_dir=results / task["output_dir"],
        task=task,
        predictions=labels + np.float32(1.0),
        labels=labels,
        checkpoint_source=checkpoint,
        train_log_source=train_log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "training_code_sha256": "3" * 64,
        },
        claim_scope="same_table_native_recipe_100_epoch_supplement",
    )
checkpoint.unlink()
train_log.unlink()
status = {
    "schema_version": 1,
    "state": "completed",
    "protocol_id": "igstgnn_table4_official_v1",
    "matrix_id": "igstgnn_table4_bist_ndr_native100_18_v1",
    "manifest_sha256": sha256(manifest_path),
    "base_manifest_sha256": sha256(base_manifest),
    "selected_task_count": 6,
    "selected_task_ids": selected_ids,
    "verified_task_count": 6,
    "completed_count": 6,
    "completed": selected_ids,
    "skipped_count": 0,
    "skipped": [],
    "failed_count": 0,
    "failed": [],
    "current": None,
    "finished_at": "2026-08-02T00:00:00Z",
}
(results / "queue_status.json").write_text(
    json.dumps(status, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
printf '0\n' > "$success_root/control/runner.exitcode"

RUN_ROOT="$success_root" \
WAIT_ONCE=1 \
SHUTDOWN_BIN="$TMP/bin/shutdown" \
PYTHON="$TEST_PYTHON" \
PGREP_BIN="$TMP/bin/pgrep" \
REPO="$ROOT_DIR" \
bash "$SCRIPT" >/dev/null

[[ -e "$TMP/shutdown.called" ]]
grep -Fq "NATIVE100_SHUTDOWN_SUCCESS shutting_down" \
  "$success_root/control/shutdown_waiter.log"
echo "verified: complete and revalidated native100 queue shuts down the RTX 5090"

failure_root="$TMP/failure"
mkdir -p "$failure_root/control"
cp -R "$success_root/results" "$failure_root/results"
printf '1\n' > "$failure_root/control/runner.exitcode"
cat > "$TMP/bin/failure-shutdown" <<EOF
#!/usr/bin/env bash
touch "$TMP/failure-shutdown.called"
EOF
chmod +x "$TMP/bin/failure-shutdown"

set +e
RUN_ROOT="$failure_root" \
WAIT_ONCE=1 \
SHUTDOWN_BIN="$TMP/bin/failure-shutdown" \
PYTHON="$TEST_PYTHON" \
PGREP_BIN="$TMP/bin/pgrep" \
REPO="$ROOT_DIR" \
bash "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e

[[ "$rc" != "0" ]]
[[ ! -e "$TMP/failure-shutdown.called" ]]
grep -Fq "NATIVE100_SHUTDOWN_BLOCKED completion postflight failed" \
  "$failure_root/control/shutdown_waiter.log"
echo "verified: nonzero native100 runner exit keeps the RTX 5090 powered on"
