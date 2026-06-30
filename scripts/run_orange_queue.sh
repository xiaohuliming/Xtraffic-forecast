#!/usr/bin/env bash
# Orange v0b vs v0c, seeds 42/1/2 (3rd region confirmation). 6 runs concurrent. ~12h.
set -u
PY=/root/miniconda3/bin/python
cd /root/traffic_fourier || exit 1
mkdir -p outputs/rgdn outputs/ada_logs3
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "=== orange matrix start $(date) ==="
launch(){ $PY scripts/train_rgdn.py --region Orange --variant "$1" --seed "$2" --epochs 30 --num_workers 0 > "outputs/ada_logs3/Orange_$1_seed$2.log" 2>&1 & }
for s in 42 1 2; do launch v0b $s; done
for s in 42 1 2; do launch v0c $s; done
echo "launched 6 ($(date))"; wait; echo "=== done $(date) ==="
