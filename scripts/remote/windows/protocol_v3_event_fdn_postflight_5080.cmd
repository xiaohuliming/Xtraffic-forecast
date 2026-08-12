@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "REPO=D:\t4\v3_event_fdn_repo"
set "CONTROL=D:\t4\v3_event_fdn_control"
set "RUN_ROOT=D:\t4\v3_event_fdn_5080_v1"
set "RUNNER_EXIT=%RUN_ROOT%\control\runner.exitcode"
set "POSTFLIGHT=%RUN_ROOT%\postflight"
set "PYTHON=C:\Python313\python.exe"
set "XTRAFFIC_REPO_ROOT=%REPO%"
set "PYTHONPATH=%REPO%;C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages"
set "PYTHONUTF8=1"

if not exist "%POSTFLIGHT%" mkdir "%POSTFLIGHT%"

:WAIT_FOR_RUNNER
if not exist "%RUNNER_EXIT%" (
  ping -n 61 127.0.0.1 >nul
  goto WAIT_FOR_RUNNER
)

set /p RUNNER_RC=<"%RUNNER_EXIT%"
if not "!RUNNER_RC!"=="0" (
  > "%POSTFLIGHT%\postflight.exitcode.tmp" echo 41
  move /Y "%POSTFLIGHT%\postflight.exitcode.tmp" "%POSTFLIGHT%\postflight.exitcode" >nul
  exit /b 41
)

"%PYTHON%" -X utf8 "%CONTROL%\protocol_v3_event_host_postflight.py" ^
  --plan "%CONTROL%\backup_plan.json" ^
  --report "%POSTFLIGHT%\source_host_postflight.json" ^
  --file-list "%POSTFLIGHT%\files.list" ^
  --transfer-manifest "%POSTFLIGHT%\files.sha256" ^
  --marker "%POSTFLIGHT%\postflight_complete.json" ^
  > "%POSTFLIGHT%\postflight.log" 2>&1

set "POSTFLIGHT_RC=!ERRORLEVEL!"
> "%POSTFLIGHT%\postflight.exitcode.tmp" echo !POSTFLIGHT_RC!
move /Y "%POSTFLIGHT%\postflight.exitcode.tmp" "%POSTFLIGHT%\postflight.exitcode" >nul
exit /b !POSTFLIGHT_RC!
