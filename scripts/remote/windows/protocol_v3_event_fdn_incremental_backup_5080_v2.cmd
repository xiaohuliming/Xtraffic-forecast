@echo off
setlocal EnableExtensions

set "REPO=D:\t4\v3_event_fdn_repo_v2"
set "CONTROL=D:\t4\v3_event_fdn_control_v2"
set "PYTHON=C:\Python313\python.exe"
set "PYTHONPATH=%REPO%;C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages"
set "PYTHONUTF8=1"

"%PYTHON%" -X utf8 "%CONTROL%\incremental_backup.py" ^
  --plan "%CONTROL%\backup_plan.json" ^
  --control-root "%CONTROL%\incremental" ^
  --nas "xhlm@192.168.31.233" ^
  --nas-base "/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_fdn_v2" ^
  --ssh-key "D:\t4\v3_event_nas_control\id_ed25519" ^
  --poll-seconds 60

exit /b %ERRORLEVEL%
