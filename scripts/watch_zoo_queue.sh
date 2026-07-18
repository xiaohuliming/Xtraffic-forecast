#!/usr/bin/env bash
# Fixed-pattern watchdog: finish when every expected summary exists; fall back
# to 5 consecutive zero-process polls. Then tar results, push to the relay
# server, and power off. Run detached on the GPU box:
#   RELAY=user@relay-host RELAY_DIR=/path/to/results \
#   EXPECTED=zoo_expected.txt TAG=zoo_0708 nohup bash scripts/watch_zoo_queue.sh &
# zoo_expected.txt lines: paths to summary.json files the queue must produce.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
LAB_HOSTS_CONFIG=${LAB_HOSTS_CONFIG:-"$REPO_ROOT/config/lab_hosts.env"}
if [ -r "$LAB_HOSTS_CONFIG" ]; then
  # shellcheck disable=SC1090
  source "$LAB_HOSTS_CONFIG"
fi
EXPECTED=${EXPECTED:-zoo_expected.txt}
RELAY=${RELAY:-${LAB_RELAY_SSH:-}}
: "${RELAY:?Set RELAY or configure LAB_RELAY_SSH in $LAB_HOSTS_CONFIG}"
RELAY_DIR=${RELAY_DIR:-${LAB_RELAY_DIR:-xtraffic_results}}
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
tar czf "/root/${TAG}.tar.gz" --exclude="*.npz" --exclude="*.pt" \
  outputs/protocol_v2_noleak/baselines/zoo_* logs 2>/dev/null
for i in 1 2 3; do
  scp "/root/${TAG}.tar.gz" "$RELAY:$RELAY_DIR/" && break
  sleep 30
done
shutdown -h now
