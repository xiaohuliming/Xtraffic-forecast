@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >NUL

if "%~1"=="" exit /b 80
set "FDNB_CODE_SHA=%~1"
set "FDNB_REPO=D:\t4\fdn_benchmark_v1_repo"
set "FDNB_DATA=D:\t4\fdn_benchmark_v1_data"
set "FDNB_RESULTS=D:\t4\fdn_benchmark_v1\results"
set "FDNB_CONTROL=D:\t4\fdn_benchmark_v1\control"
set "FDNB_PYTHON=C:\Python313\python.exe"

if not exist "%FDNB_REPO%\.git" exit /b 81
if not exist "%FDNB_DATA%\PEMS04.npz" exit /b 82
if not exist "%FDNB_DATA%\PEMS08.npz" exit /b 83
if not exist "%FDNB_CONTROL%" mkdir "%FDNB_CONTROL%"
if exist "%FDNB_CONTROL%\runner.exitcode" exit /b 84

cd /d "%FDNB_REPO%"
"%FDNB_PYTHON%" -X utf8 scripts\remote\windows\fdn_benchmark_v1_runner.py ^
  --manifest config\fdn_benchmark_v1\go_no_go_pems04_pems08_v1.json ^
  --repo-root "%FDNB_REPO%" ^
  --data-dir "%FDNB_DATA%" ^
  --results-root "%FDNB_RESULTS%" ^
  --code-sha "%FDNB_CODE_SHA%" ^
  --python "%FDNB_PYTHON%" ^
  1>"%FDNB_CONTROL%\runner.log" 2>&1
set "FDNB_RC=%ERRORLEVEL%"
>"%FDNB_CONTROL%\runner.exitcode.tmp" echo %FDNB_RC%
move /Y "%FDNB_CONTROL%\runner.exitcode.tmp" "%FDNB_CONTROL%\runner.exitcode" >NUL
exit /b %FDNB_RC%
