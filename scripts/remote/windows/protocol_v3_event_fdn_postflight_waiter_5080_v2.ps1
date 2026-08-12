$ErrorActionPreference = "Stop"

$RunnerExit = "D:\t4\v3_event_fdn_5080_v2\control\runner.exitcode"
$PostflightCommand = "D:\t4\v3_event_fdn_control_v2\postflight.cmd"
$ControlRoot = "D:\t4\v3_event_fdn_control_v2"
$WaiterLog = Join-Path $ControlRoot "postflight_waiter.log"

New-Item -ItemType Directory -Force -Path $ControlRoot | Out-Null

function Write-WaiterLog {
    param([string]$Message)
    $timestamp = [DateTimeOffset]::UtcNow.ToString("o")
    Add-Content -LiteralPath $WaiterLog -Encoding UTF8 -Value "$timestamp $Message"
}

Write-WaiterLog "WAIT_START"
while (-not (Test-Path -LiteralPath $RunnerExit -PathType Leaf)) {
    Start-Sleep -Seconds 30
}

$runnerCode = (Get-Content -LiteralPath $RunnerExit -Raw).Trim()
Write-WaiterLog "RUNNER_EXIT_DETECTED rc=$runnerCode"

& "$env:SystemRoot\System32\cmd.exe" /d /c $PostflightCommand
$postflightCode = $LASTEXITCODE
Write-WaiterLog "WAIT_EXIT rc=$postflightCode"
exit $postflightCode
