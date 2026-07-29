#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$ROOT_DIR/scripts/remote/linux/igstgnn_table4_ndr_extension_handoff.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p \
  "$TMP/root/ndr_repo/config/reproduction/igstgnn_table4" \
  "$TMP/root/data/official_splits" \
  "$TMP/root/control/table4_4090d4" \
  "$TMP/root/env/bin" \
  "$TMP/bin"
cp "$ROOT_DIR/config/reproduction/igstgnn_table4/distributed_ndr_extension_9gpu_v1.json" \
  "$TMP/root/ndr_repo/config/reproduction/igstgnn_table4/"
cp "$ROOT_DIR/config/reproduction/igstgnn_table4/matrix.ndr_extension_v1.json" \
  "$TMP/root/ndr_repo/config/reproduction/igstgnn_table4/"
cp "$ROOT_DIR/config/reproduction/igstgnn_table4/ndr_extension_profile_v1.json" \
  "$TMP/root/ndr_repo/config/reproduction/igstgnn_table4/"
cp "$ROOT_DIR/config/reproduction/igstgnn_table4/matrix.public_material_v2.json" \
  "$TMP/root/ndr_repo/config/reproduction/igstgnn_table4/"
touch "$TMP/root/control/table4_4090d4/defer_shutdown_for_ndr_extension"
cat > "$TMP/bin/shutdown" <<EOF
#!/usr/bin/env bash
touch "$TMP/shutdown.called"
EOF
chmod +x "$TMP/bin/shutdown"
cp /usr/bin/true "$TMP/root/env/bin/python"

set +e
HOST_ID=table4_4090d4 \
ROOT="$TMP/root" \
PYTHON="$TMP/root/env/bin/python" \
MANIFEST_SHA256="$(sha256sum "$ROOT_DIR/config/reproduction/igstgnn_table4/matrix.ndr_extension_v1.json" | awk '{print $1}')" \
BASE_MANIFEST_SHA256="$(sha256sum "$ROOT_DIR/config/reproduction/igstgnn_table4/matrix.public_material_v2.json" | awk '{print $1}')" \
HANDOFF_WAIT_ONCE=1 \
SHUTDOWN_BIN="$TMP/bin/shutdown" \
REALPATH_BIN=/bin/realpath \
bash "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e

[[ "$rc" != "0" ]]
[[ ! -e "$TMP/shutdown.called" ]]
grep -Fq "NDR_HANDOFF_BLOCKED original host export is not complete" \
  "$TMP/root/ndr_extension/control/table4_4090d4/handoff.log"
echo "verified: incomplete original export blocks NDR launch and shutdown"
