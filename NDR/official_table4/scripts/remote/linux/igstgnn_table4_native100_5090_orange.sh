#!/usr/bin/env bash
set -euo pipefail

REPO="/root/autodl-tmp/native100_repo"
PYTHON="/root/miniconda3/bin/python"
MANIFEST="$REPO/config/reproduction/igstgnn_table4/matrix.native100_compare_v1.json"
MANIFEST_SHA256="e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246"
BASE_MANIFEST="$REPO/config/reproduction/igstgnn_table4/matrix.public_material_v2.json"
BASE_MANIFEST_SHA256="d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
SPLITS_ROOT="/root/autodl-tmp/native100_data/official_splits"
RUN_ROOT="/root/autodl-tmp/native100_orange"
RESULTS_ROOT="$RUN_ROOT/results"
CONTROL_ROOT="$RUN_ROOT/control"
LOG="$CONTROL_ROOT/supervisor.log"
EXIT_CODE="$CONTROL_ROOT/runner.exitcode"

mkdir -p "$CONTROL_ROOT" "$RESULTS_ROOT"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

printf '%s TABLE4_NATIVE100_5090_ORANGE_START\n' "$(date --iso-8601=seconds)" >"$LOG"
set +e
"$PYTHON" -X utf8 -m scripts.reproduction.igstgnn_table4.native100_compare_matrix_runner \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --base-manifest "$BASE_MANIFEST" \
  --base-manifest-sha256 "$BASE_MANIFEST_SHA256" \
  --splits-root "$SPLITS_ROOT" \
  --results-root "$RESULTS_ROOT" \
  --device cuda:0 \
  --python "$PYTHON" \
  --task-id n100007_bist_orange_seed42 \
  --task-id n100008_bist_orange_seed1 \
  --task-id n100009_bist_orange_seed2 \
  --task-id n100016_ndr_net_orange_seed42 \
  --task-id n100017_ndr_net_orange_seed1 \
  --task-id n100018_ndr_net_orange_seed2 >>"$LOG" 2>&1
runner_rc=$?
set -e
printf '%s\n' "$runner_rc" >"$EXIT_CODE.tmp"
mv -f "$EXIT_CODE.tmp" "$EXIT_CODE"
printf '%s TABLE4_NATIVE100_5090_ORANGE_EXIT rc=%s\n' \
  "$(date --iso-8601=seconds)" "$runner_rc" >>"$LOG"
exit "$runner_rc"
