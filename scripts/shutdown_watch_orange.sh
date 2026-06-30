#!/usr/bin/env bash
cd /root/traffic_fourier || exit 1
LOG=outputs/ada_logs3/shutdown_watcher.log
FLAG=outputs/.OK_TO_SHUTDOWN
rm -f "$FLAG"; echo "watcher start $(date)" >> "$LOG"
sleep 240
while true; do p=$(ps aux|grep train_rgdn|grep -v grep|wc -l); [ "$p" -eq 0 ] && break; sleep 90; done
echo "all procs done $(date); awaiting flag max 30m" >> "$LOG"
for i in $(seq 1 60); do [ -f "$FLAG" ] && { echo "flag seen $(date)">>"$LOG"; break; }; sleep 30; done
echo "powering off $(date)" >> "$LOG"; sync
/usr/bin/shutdown -h now 2>>"$LOG" || /usr/sbin/poweroff 2>>"$LOG" || /usr/sbin/init 0
