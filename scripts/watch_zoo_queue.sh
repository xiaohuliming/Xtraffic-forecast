#!/usr/bin/env bash
# Fixed-pattern watchdog: finish when every expected summary exists; fall back
# to 5 consecutive zero-process polls. Then tar results, push to the relay
# server, and power off. Run detached on the GPU box:
#   EXPECTED=zoo_expected.txt TAG=zoo_0708 nohup bash scripts/watch_zoo_queue.sh &
# zoo_expected.txt lines: paths to summary.json files the queue must produce.
set -u
cd "$(dirname "$0")/.."
EXPECTED=${EXPECTED:-zoo_expected.txt}
RELAY=${RELAY:-root@8.134.12.5}
TAG=${TAG:-zoo_results}
zero=0
while true; do
  sleep 60
  miss=0
  while read -r f; do [ -n "$f" ] && [ ! -s "$f" ] && miss=$((miss+1)); done < "$EXPECTED"
  np=$(pgrep -fc "train_[b]aseline_zoo" || true)
  echo "$(date +%F_%T) missing=$miss procs=$np"
  if [ "$miss" -eq 0 ]; then echo "all summaries present"; break; fi
  if [ "$np" -eq 0 ]; then zero=$((zero+1)); else zero=0; fi
  if [ "$zero" -ge 5 ]; then echo "stalled: 5 zero-proc polls, $miss missing"; break; fi
done
tar czf "/root/${TAG}.tar.gz" --exclude="*.npz" --exclude="*.pt" outputs/baselines/zoo_* logs 2>/dev/null
for i in 1 2 3; do
  scp -o StrictHostKeyChecking=no "/root/${TAG}.tar.gz" "$RELAY:/root/xtraffic_results/" && break
  sleep 30
done
shutdown -h now
