#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$ROOT_DIR/scripts/remote/linux/igstgnn_table4_ndr_extension_rebalance_finalize.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

make_case() {
  local name=$1
  local root="$TMP/$name"
  mkdir -p "$root/control" "$root/results" "$root/repo" "$root/bin"
  touch "$root/control/shutdown_guard"
  printf '%s\n' "payload" > "$root/results/prior.txt"
  (
    cd "$root/results"
    sha256sum prior.txt > "$root/control/prior.sha256"
  )
  printf '%s\n' "plan" > "$root/repo/plan.json"
  printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "$root/bin/handoff"
  chmod +x "$root/bin/handoff"
}

make_case incomplete
root="$TMP/incomplete"
printf '%s\n' '{"status":"running"}' > "$root/control/host_status.json"
set +e
HOST_ID=table4_4090d4 \
ROOT="$root" \
REPO="$root/repo" \
RESULTS_BASE="$root/results" \
CONTROL_DIR="$root/control" \
PLAN="$root/repo/plan.json" \
PLAN_SHA256="$(sha256sum "$root/repo/plan.json" | awk '{print $1}')" \
PRIOR_TASK_ARTIFACTS="$root/control/prior.sha256" \
PRIOR_TASK_ARTIFACT_COUNT=1 \
SHUTDOWN_GUARD="$root/control/shutdown_guard" \
HANDOFF_BIN="$root/bin/handoff" \
PYTHON=python3 \
WAIT_ONCE=1 \
bash "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e
[[ "$rc" != "0" ]]
[[ -f "$root/control/shutdown_guard" ]]
grep -Fq "NDR_REBALANCE_FINALIZE_BLOCKED host run is not complete" \
  "$root/control/rebalance_finalize.log"
echo "verified: incomplete rebalance run leaves the host on"

make_case complete
root="$TMP/complete"
plan_sha=$(sha256sum "$root/repo/plan.json" | awk '{print $1}')
python3 - "$root/control/host_status.json" "$plan_sha" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "completed",
            "host_id": "table4_4090d4",
            "plan_sha256": sys.argv[2],
            "shards": [
                {"returncode": 0},
                {"returncode": 0},
                {"returncode": 0},
                {"returncode": 0},
            ],
        }
    ),
    encoding="utf-8",
)
PY
cat > "$root/bin/handoff" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$HOST_ID" == "table4_4090d4" ]]
[[ "$PLAN" == "$EXPECTED_PLAN" ]]
[[ "$SHUTDOWN_DEFER_FILE" == "$EXPECTED_GUARD" ]]
[[ "$RELAY_BASE" == "/srv/ndr-v2/hosts" ]]
touch "$HANDOFF_CALLED"
EOF
chmod +x "$root/bin/handoff"

HOST_ID=table4_4090d4 \
ROOT="$root" \
REPO="$root/repo" \
RESULTS_BASE="$root/results" \
CONTROL_DIR="$root/control" \
PLAN="$root/repo/plan.json" \
PLAN_SHA256="$plan_sha" \
PRIOR_TASK_ARTIFACTS="$root/control/prior.sha256" \
PRIOR_TASK_ARTIFACT_COUNT=1 \
SHUTDOWN_GUARD="$root/control/shutdown_guard" \
WAIT_ONCE=1 \
HANDOFF_BIN="$root/bin/handoff" \
RELAY_BASE=/srv/ndr-v2/hosts \
PYTHON=python3 \
EXPECTED_PLAN="$root/repo/plan.json" \
EXPECTED_GUARD="$root/control/shutdown_guard" \
HANDOFF_CALLED="$root/handoff.called" \
bash "$SCRIPT" >/dev/null

[[ -e "$root/handoff.called" ]]
grep -Fq "NDR_REBALANCE_PRIOR_ARTIFACTS_OK count=1" \
  "$root/control/rebalance_finalize.log"
echo "verified: completed rebalance validates prior artifacts and starts handoff"
