#!/usr/bin/env bash
# Route B adaptive de-seasonalization: Alameda v0b (control) vs v0c (adaptive alpha),
# seeds 42/1/2, param-matched single GWN. 208-core host + underused GPU (18%/run, 1.7GB/run)
# => launch all 6 concurrently; per-run serial Python loops don't contend much.
set -u
PY=/root/miniconda3/bin/python
cd /root/traffic_fourier || exit 1
mkdir -p outputs/rgdn outputs/ada_logs
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "=== adaptive de-seasonalization PARALLEL start $(date) ==="
pids=()
for seed in 42 1 2; do
  for variant in v0b v0c; do
    log="outputs/ada_logs/${variant}_seed${seed}.log"
    echo "launch $variant seed $seed -> $log"
    $PY scripts/train_rgdn.py --region Alameda --variant "$variant" --seed "$seed" \
        --epochs 30 --num_workers 0 > "$log" 2>&1 &
    pids+=($!)
  done
done
echo "launched ${#pids[@]} runs: ${pids[*]}"
wait
echo "=== all runs done $(date) ==="
