@echo off
setlocal
set "PYTHONUTF8=1"
set "REPO=C:\Users\ASUS\traffic_fourier_v3_final"
set "OPSROOT=C:\Users\ASUS\ops_27_18"
set "PYTHONPATH=%OPSROOT%;%REPO%"
set "XTRAFFIC_REPO_ROOT=%OPSROOT%"
set "OPS=%OPSROOT%\scripts\remote\distributed\protocol_v3_distributed.py"
set "PYTHON=C:\Python313\python.exe"
set "RESULTS=%REPO%\outputs\protocol_v3_noleak_dedup\remote_runs\minimal_45_5080"
set "LOG=C:\Users\ASUS\protocol_v3_5080_shard27.log"
set "EXIT_CODE=C:\Users\ASUS\protocol_v3_5080_shard27.exitcode"
set "PID_FILE=C:\Users\ASUS\protocol_v3_5080_shard27.pid"
rem XTRAFFIC_RECOVER_STAGING_TASK is consumed directly by Python.
rem Never expand its value in cmd.exe.

powershell.exe -NoProfile -Command "$others = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*protocol_v3_finalize_5080_shard.ps1*' }); if ($others.Count -gt 0) { Write-Error 'old 5080 finalizer is still running'; exit 41 }; $self = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $PID); [System.IO.File]::WriteAllText('%PID_FILE%', [string]$self.ParentProcessId)"
if errorlevel 1 exit /b 41

if exist "%EXIT_CODE%" del /f /q "%EXIT_CODE%"
cd /d "%REPO%"
"%PYTHON%" -X utf8 "%OPS%" ^
  --shard 5080 ^
  --manifest "%REPO%\scripts\remote\windows\protocol_v3_manifest.json" ^
  --repo-root "%REPO%" ^
  --results-root "%RESULTS%" ^
  --cache-dir "%REPO%\outputs\dist_net\region_data_v3_noleak_dedup" ^
  --graph-dir "%REPO%\outputs\region_graphs" ^
  --cache-bundle-id "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4" ^
  --cache-bundle-digest "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5" ^
  --code-sha "3a4952837aaf6fbda1403d50b610b1bf40c6098a" ^
  --python "%PYTHON%" > "%LOG%" 2>&1

> "%EXIT_CODE%" echo %ERRORLEVEL%
if exist "%PID_FILE%" del /f /q "%PID_FILE%"
endlocal
