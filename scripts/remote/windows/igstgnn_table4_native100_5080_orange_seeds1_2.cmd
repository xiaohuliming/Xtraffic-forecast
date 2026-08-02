@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM REJECTED AFTER LIVE SPEED GATE. AUDIT ARTIFACT ONLY. DO NOT RUN.
echo This launcher is disabled because the RTX 5080 live speed gate failed.
exit /b 90

set "REPO=D:\t4\native100_repo"
set "PYTHON=C:\Python313\python.exe"
set "MANIFEST=%REPO%\config\reproduction\igstgnn_table4\matrix.native100_compare_v1.json"
set "MANIFEST_SHA256=e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246"
set "BASE_MANIFEST=%REPO%\config\reproduction\igstgnn_table4\matrix.public_material_v2.json"
set "BASE_MANIFEST_SHA256=d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
set "SPLITS_ROOT=D:\t4\data\official_splits"
set "RUN_ROOT=D:\t4\native100_orange_5080_recovery"
set "RESULTS_ROOT=%RUN_ROOT%\results"
set "CONTROL_ROOT=%RUN_ROOT%\control"
set "LOG=%CONTROL_ROOT%\supervisor.log"
set "EXIT_CODE=%CONTROL_ROOT%\runner.exitcode"

if not exist "%CONTROL_ROOT%" mkdir "%CONTROL_ROOT%"
if not exist "%RESULTS_ROOT%" mkdir "%RESULTS_ROOT%"

cd /d "%REPO%"
set "PYTHONPATH=%REPO%;C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=0"

echo %DATE% %TIME% TABLE4_NATIVE100_5080_ORANGE_SEEDS1_2_START> "%LOG%"
"%PYTHON%" -X utf8 -m scripts.reproduction.igstgnn_table4.native100_compare_matrix_runner ^
  --manifest "%MANIFEST%" ^
  --manifest-sha256 "%MANIFEST_SHA256%" ^
  --base-manifest "%BASE_MANIFEST%" ^
  --base-manifest-sha256 "%BASE_MANIFEST_SHA256%" ^
  --splits-root "%SPLITS_ROOT%" ^
  --results-root "%RESULTS_ROOT%" ^
  --device cuda:0 ^
  --python "%PYTHON%" ^
  --task-id n100017_ndr_net_orange_seed1 ^
  --task-id n100018_ndr_net_orange_seed2 >> "%LOG%" 2>&1

set "RUNNER_RC=!ERRORLEVEL!"
> "%EXIT_CODE%.tmp" echo !RUNNER_RC!
move /Y "%EXIT_CODE%.tmp" "%EXIT_CODE%" >nul
echo %DATE% %TIME% TABLE4_NATIVE100_5080_ORANGE_SEEDS1_2_EXIT rc=!RUNNER_RC!>> "%LOG%"
exit /b !RUNNER_RC!
