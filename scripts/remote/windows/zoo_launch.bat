@echo off
cd /d C:\Users\asus\traffic_fourier
set PYTHONUTF8=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -X utf8 scripts\run_zoo_queue_win.py > zoo_queue_win.log 2>&1
