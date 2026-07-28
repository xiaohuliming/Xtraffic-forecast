@echo off
setlocal

set "PROJECT_ROOT=D:\t4\repo"
set "CONTROL_ROOT=D:\t4\control"
set "RESULTS_ROOT=D:\t4\results\igstgnn_table4_public_repro_39_v2"
set "MANIFEST=%PROJECT_ROOT%\config\reproduction\igstgnn_table4\matrix.public_material_v2.json"
set "MANIFEST_SHA256=d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"

cd /d "%PROJECT_ROOT%"
C:\Python313\python.exe "%CONTROL_ROOT%\scheduled_handoff.py" ^
  --project-root "%PROJECT_ROOT%" ^
  --manifest "%MANIFEST%" ^
  --manifest-sha256 "%MANIFEST_SHA256%" ^
  --splits-root D:\t4\data\official_splits ^
  --results-root "%RESULTS_ROOT%" ^
  --device cuda:0 ^
  --pilot-task-id 003_alameda_dcrnn ^
  --pilot-task-id 013_alameda_igstgnn ^
  --status "%CONTROL_ROOT%\scheduled_handoff_status.json" ^
  >> "%CONTROL_ROOT%\scheduled_handoff.log" 2>&1

exit /b %ERRORLEVEL%
