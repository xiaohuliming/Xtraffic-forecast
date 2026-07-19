#!/bin/bash
# Submit the formal NDR event causal-v2 workflow from the HPC login node.
# This is intentionally not a Slurm job: it creates one CPU audit plus three
# independent GPU jobs.  Alameda and ContraCosta may use any allocated GPU;
# Orange explicitly requests an A100 because its 990-node model exceeds the
# safe memory budget of the 24-GB RTX 4090 used in the failed preliminary run.
set -euo pipefail
umask 027

BASE=/home/student/t330026045/traffic_weather
REPO=$BASE/Xtraffic-forecast
HPC_DIR=$REPO/hpc
LOG_DIR=$BASE/ndr_logs
OUT_ROOT=$BASE/outputs/ndrnet_2023_event_causal_v2
AUDIT_SCRIPT=$HPC_DIR/run_ndr_event_causal_v2_cache_audit.sh
REGION_SCRIPT=$HPC_DIR/run_ndr_event_causal_v2_region_gpu.sh

command -v sbatch >/dev/null || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
for SCRIPT in "$AUDIT_SCRIPT" "$REGION_SCRIPT"; do
  [[ -s "$SCRIPT" ]] || { echo "ERROR: missing script: $SCRIPT" >&2; exit 1; }
  bash -n "$SCRIPT"
done

mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Fail before submission if any formal-v2 region contains old or partial
# artifacts.  A fresh versioned root is required for a clean formal run.
for REGION in Alameda ContraCosta Orange; do
  REGION_OUT=$OUT_ROOT/$REGION
  if [[ -d "$REGION_OUT" ]] && \
     [[ -n "$(find "$REGION_OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: non-empty formal output already exists: $REGION_OUT" >&2
    echo "No jobs were submitted. Use a new versioned output root." >&2
    exit 1
  fi
done

AUDIT_JOB=$(sbatch --parsable "$AUDIT_SCRIPT")
AUDIT_JOB=${AUDIT_JOB%%;*}

ALAMEDA_JOB=$(sbatch --parsable \
  --dependency="afterok:$AUDIT_JOB" \
  --job-name=ndr_v2_alameda \
  --gres=gpu:1 \
  --export=ALL,REGION=Alameda \
  "$REGION_SCRIPT")
ALAMEDA_JOB=${ALAMEDA_JOB%%;*}

CC_JOB=$(sbatch --parsable \
  --dependency="afterok:$AUDIT_JOB" \
  --job-name=ndr_v2_cc \
  --gres=gpu:1 \
  --export=ALL,REGION=ContraCosta \
  "$REGION_SCRIPT")
CC_JOB=${CC_JOB%%;*}

ORANGE_JOB=$(sbatch --parsable \
  --dependency="afterok:$AUDIT_JOB" \
  --job-name=ndr_v2_orange \
  --gres=gpu:a100:1 \
  --export=ALL,REGION=Orange \
  "$REGION_SCRIPT")
ORANGE_JOB=${ORANGE_JOB%%;*}

echo "Formal NDR event causal-v2 workflow submitted."
echo "Cache audit: $AUDIT_JOB"
echo "Alameda:     $ALAMEDA_JOB (gpu:1; starts only after audit passes)"
echo "ContraCosta: $CC_JOB (gpu:1; starts only after audit passes)"
echo "Orange:      $ORANGE_JOB (gpu:a100:1; starts only after audit passes)"
echo "All regions run seeds 42, 1, and 2 sequentially, with a 3-day limit."
echo "Logs:        $LOG_DIR"
echo "Results:     $OUT_ROOT"

squeue -j "$AUDIT_JOB,$ALAMEDA_JOB,$CC_JOB,$ORANGE_JOB" \
  -o "%.18i %.10P %.24j %.10T %.12M %.5D %R"
