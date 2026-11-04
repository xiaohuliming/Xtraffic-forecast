@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO=D:\t4\bist_ndr_repo"
set "PYTHON=C:\Python313\python.exe"
set "MANIFEST=%REPO%\config\reproduction\igstgnn_table4\matrix.bist_ndr_pair_v1.json"
set "MANIFEST_SHA256=90f5ceb4a0e1306d3f1ff11268b76220a289a37a922c6926665825a56c9cdeab"
set "BASE_MANIFEST=%REPO%\config\reproduction\igstgnn_table4\matrix.public_material_v2.json"
set "BASE_MANIFEST_SHA256=d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
set "SPLITS_ROOT=D:\t4\data\official_splits"
set "RUN_ROOT=D:\t4\bist_ndr_pair"
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

echo %DATE% %TIME% BIST_NDR_PAIR_START> "%LOG%"
"%PYTHON%" -X utf8 -m scripts.reproduction.igstgnn_table4.bist_ndr_pair_matrix_runner ^
  --manifest "%MANIFEST%" ^
  --manifest-sha256 "%MANIFEST_SHA256%" ^
  --base-manifest "%BASE_MANIFEST%" ^
  --base-manifest-sha256 "%BASE_MANIFEST_SHA256%" ^
  --splits-root "%SPLITS_ROOT%" ^
  --results-root "%RESULTS_ROOT%" ^
  --device cuda:0 ^
  --python "%PYTHON%" ^
  --task-id pair001_bist_aligned_alameda_seed42 ^
  --task-id pair010_bist_ndr_alameda_seed42 ^
  --task-id pair002_bist_aligned_alameda_seed1 ^
  --task-id pair011_bist_ndr_alameda_seed1 ^
  --task-id pair003_bist_aligned_alameda_seed2 ^
  --task-id pair012_bist_ndr_alameda_seed2 ^
  --task-id pair004_bist_aligned_contra_costa_seed42 ^
  --task-id pair013_bist_ndr_contra_costa_seed42 ^
  --task-id pair005_bist_aligned_contra_costa_seed1 ^
  --task-id pair014_bist_ndr_contra_costa_seed1 ^
  --task-id pair006_bist_aligned_contra_costa_seed2 ^
  --task-id pair015_bist_ndr_contra_costa_seed2 ^
  --task-id pair007_bist_aligned_orange_seed42 ^
  --task-id pair016_bist_ndr_orange_seed42 ^
  --task-id pair008_bist_aligned_orange_seed1 ^
  --task-id pair017_bist_ndr_orange_seed1 ^
  --task-id pair009_bist_aligned_orange_seed2 ^
  --task-id pair018_bist_ndr_orange_seed2 >> "%LOG%" 2>&1

set "RUNNER_RC=!ERRORLEVEL!"
> "%EXIT_CODE%.tmp" echo !RUNNER_RC!
move /Y "%EXIT_CODE%.tmp" "%EXIT_CODE%" >nul
echo %DATE% %TIME% BIST_NDR_PAIR_EXIT rc=!RUNNER_RC!>> "%LOG%"
exit /b !RUNNER_RC!
