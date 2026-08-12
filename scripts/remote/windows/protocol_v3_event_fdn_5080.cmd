@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO=D:\t4\v3_event_fdn_repo"
set "PYTHON=C:\Python313\python.exe"
set "MANIFEST=%REPO%\config\protocol_v3\table4_event_fdn_v1.json"
set "CACHE=C:\Users\ASUS\traffic_fourier_v3_final\outputs\dist_net\region_data_v3_noleak_dedup"
set "GRAPH=C:\Users\ASUS\traffic_fourier_v3_final\outputs\region_graphs"
set "RUN_ROOT=D:\t4\v3_event_fdn_5080_v1"
set "RESULTS=%RUN_ROOT%\results"
set "CONTROL=%RUN_ROOT%\control"
set "LOG=%CONTROL%\runner.log"
set "EXIT_CODE=%CONTROL%\runner.exitcode"

if not exist "%CONTROL%" mkdir "%CONTROL%"
if not exist "%RESULTS%" mkdir "%RESULTS%"
if exist "%EXIT_CODE%" del /f /q "%EXIT_CODE%"

cd /d "%REPO%"
set "PYTHONPATH=%REPO%;C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages"
set "PYTHONUTF8=1"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=0"

for /f %%i in ('git rev-parse HEAD') do set "CODE_SHA=%%i"
if not defined CODE_SHA exit /b 31

echo %DATE% %TIME% PROTOCOL_V3_EVENT_FDN_START> "%LOG%"
"%PYTHON%" -X utf8 -m scripts.remote.windows.protocol_v3_event_fdn_runner ^
  --manifest "%MANIFEST%" ^
  --repo-root "%REPO%" ^
  --results-root "%RESULTS%" ^
  --cache-dir "%CACHE%" ^
  --graph-dir "%GRAPH%" ^
  --cache-bundle-id "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4" ^
  --cache-bundle-digest "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5" ^
  --code-sha "!CODE_SHA!" ^
  --python "%PYTHON%" >> "%LOG%" 2>&1

set "RUNNER_RC=!ERRORLEVEL!"
> "%EXIT_CODE%.tmp" echo !RUNNER_RC!
move /Y "%EXIT_CODE%.tmp" "%EXIT_CODE%" >nul
echo %DATE% %TIME% PROTOCOL_V3_EVENT_FDN_EXIT rc=!RUNNER_RC!>> "%LOG%"
exit /b !RUNNER_RC!
