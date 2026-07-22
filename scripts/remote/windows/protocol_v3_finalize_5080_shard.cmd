@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass ^
  -File C:\Users\ASUS\protocol_v3_finalize_5080_shard.ps1 ^
  >> C:\Users\ASUS\protocol_v3_5080_shard27_finalize_console.log 2>&1
