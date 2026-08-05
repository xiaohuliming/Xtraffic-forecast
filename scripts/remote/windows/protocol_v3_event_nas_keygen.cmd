@echo off
setlocal
set "KEY=D:\t4\v3_event_nas_control\id_ed25519"
if exist "%KEY%" goto fingerprint
"C:\Windows\System32\OpenSSH\ssh-keygen.exe" -q -t ed25519 -N "" -C xtraffic-event-5080-nas-v1 -f "%KEY%"
if errorlevel 1 exit /b %ERRORLEVEL%
:fingerprint
"C:\Windows\System32\OpenSSH\ssh-keygen.exe" -lf "%KEY%.pub"
exit /b %ERRORLEVEL%
