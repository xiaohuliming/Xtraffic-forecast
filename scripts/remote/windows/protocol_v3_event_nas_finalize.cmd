@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "D:\t4\v3_event_nas_control\protocol_v3_event_nas_finalize.ps1"
exit /b %ERRORLEVEL%
