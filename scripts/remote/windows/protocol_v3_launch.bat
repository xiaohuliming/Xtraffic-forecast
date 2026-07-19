@echo off
setlocal
cd /d "%~dp0\..\..\.."

if not defined XTRAFFIC_CACHE_BUNDLE_ID (
  echo ERROR: XTRAFFIC_CACHE_BUNDLE_ID is required.
  exit /b 2
)
if not defined XTRAFFIC_CACHE_BUNDLE_DIGEST (
  echo ERROR: XTRAFFIC_CACHE_BUNDLE_DIGEST is required.
  exit /b 2
)
if not defined XTRAFFIC_CODE_SHA (
  echo ERROR: XTRAFFIC_CODE_SHA is required.
  exit /b 2
)
if not defined XTRAFFIC_PYTHON set "XTRAFFIC_PYTHON=python"

set "PYTHONUTF8=1"

"%XTRAFFIC_PYTHON%" -X utf8 scripts\remote\windows\protocol_v3_runner.py ^
  --cache-bundle-id "%XTRAFFIC_CACHE_BUNDLE_ID%" ^
  --cache-bundle-digest "%XTRAFFIC_CACHE_BUNDLE_DIGEST%" ^
  --code-sha "%XTRAFFIC_CODE_SHA%"
set "RUNNER_RC=%ERRORLEVEL%"

if not "%RUNNER_RC%"=="0" (
  echo ERROR: protocol v3 runner failed with exit code %RUNNER_RC%.
)
exit /b %RUNNER_RC%
