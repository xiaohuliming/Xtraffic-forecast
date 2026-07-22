[CmdletBinding()]
param(
    [string]$Repo = "C:\Users\ASUS\traffic_fourier_v3_final",
    [string]$ResultsRoot = "C:\Users\ASUS\traffic_fourier_v3_final\outputs\protocol_v3_noleak_dedup\remote_runs\minimal_45_5080",
    [string]$Relay = "root@8.134.12.5",
    [string]$BoundaryMarker = "/root/xtraffic_results/protocol_v3_noleak_dedup/distributed_45_5080_5090/control/repartition_27_18_boundary_ready.json",
    [string]$SshKey = "C:\Users\ASUS\.ssh\xtraffic_relay_ed25519",
    [string]$Ssh = "C:\Windows\System32\OpenSSH\ssh.exe",
    [string]$Python = "C:\Python313\python.exe",
    [string]$MarkerValidator = "C:\Users\ASUS\ops_27_18\scripts\remote\windows\protocol_v3_validate_repartition_marker.py",
    [string]$RunnerLauncher = "C:\Users\ASUS\protocol_v3_launch_5080_shard.cmd",
    [string]$FinalizerLauncher = "C:\Users\ASUS\protocol_v3_finalize_5080_shard.cmd",
    [int]$OldFinalizerPid = 12524,
    [int]$OldFinalizerStopWaitSeconds = 30,
    [int]$NewQueueWaitSeconds = 120,
    [int]$SshWallClockTimeoutSeconds = 30,
    [int]$NativeWallClockTimeoutSeconds = 120,
    [int]$PollSeconds = 30,
    [int]$MaxWaitIterations = 0,
    [string]$Log = "C:\Users\ASUS\protocol_v3_5080_repartition27_waiter.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-WaiterLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0} {1}" -f [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"), $Message
    $parent = Split-Path -Parent $Log
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
    Write-Host $line
}

function Stop-Waiter {
    param([Parameter(Mandatory = $true)][string]$Message)

    throw $Message
}

function Get-DescendantProcessIds {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)

    $snapshot = @(Get-CimInstance Win32_Process)
    $known = [System.Collections.Generic.HashSet[int]]::new()
    [void]$known.Add($RootProcessId)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($processRecord in $snapshot) {
            [int]$processId = $processRecord.ProcessId
            [int]$parentProcessId = $processRecord.ParentProcessId
            if ($known.Contains($parentProcessId) -and $known.Add($processId)) {
                $changed = $true
            }
        }
    }
    return @($known | Where-Object { $_ -ne $RootProcessId })
}

function Stop-ProcessTree {
    param([Parameter(Mandatory = $true)][int]$ProcessId)

    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return
    }
    try {
        $process.Kill($true)
    }
    catch {
        $taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
        $taskkillArguments = @("/PID", ([string]$ProcessId), "/T", "/F")
        $killParameters = @{
            FilePath = $taskkill
            ArgumentList = $taskkillArguments
            WindowStyle = "Hidden"
            Wait = $true
            PassThru = $true
        }
        $kill = Start-Process @killParameters
        if (
            $kill.ExitCode -ne 0 -and
            $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
        ) {
            Stop-Waiter "cannot terminate process tree rooted at PID $ProcessId"
        }
    }
}

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [int]$TimeoutSeconds = $NativeWallClockTimeoutSeconds
    )

    if ($TimeoutSeconds -le 0) {
        Stop-Waiter "native process timeout must be positive"
    }
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    $process = $null
    try {
        $startParameters = @{
            FilePath = $FilePath
            ArgumentList = $ArgumentList
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
            NoNewWindow = $true
            PassThru = $true
        }
        $process = Start-Process @startParameters
        $finished = $process.WaitForExit($TimeoutSeconds * 1000)
        $timedOut = -not $finished
        if ($timedOut) {
            Stop-ProcessTree -ProcessId $process.Id
            [void]$process.WaitForExit(5000)
        }
        else {
            [void]$process.WaitForExit()
            $process.Refresh()
        }
        $output = @()
        if (Test-Path -LiteralPath $stdoutPath -PathType Leaf) {
            $output += @(Get-Content -LiteralPath $stdoutPath -Encoding UTF8)
        }
        if (Test-Path -LiteralPath $stderrPath -PathType Leaf) {
            $output += @(Get-Content -LiteralPath $stderrPath -Encoding UTF8)
        }
        if ($timedOut) {
            $exitCode = 124
        }
        else {
            $exitCode = $process.ExitCode
            if ($null -eq $exitCode) {
                Stop-Waiter "native process exit code is unavailable"
            }
            $exitCode = [int]$exitCode
        }
        return [PSCustomObject]@{
            ExitCode = $exitCode
            TimedOut = $timedOut
            Output = @($output | ForEach-Object { $_.ToString() })
        }
    }
    finally {
        foreach ($temporaryPath in @($stdoutPath, $stderrPath)) {
            if (Test-Path -LiteralPath $temporaryPath) {
                Remove-Item -LiteralPath $temporaryPath -Force
            }
        }
    }
}

function Invoke-SshCapture {
    param([Parameter(Mandatory = $true)][string]$Command)

    $arguments = @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        $Relay,
        $Command
    )
    $invokeParameters = @{
        FilePath = $Ssh
        ArgumentList = $arguments
        TimeoutSeconds = $SshWallClockTimeoutSeconds
    }
    return Invoke-NativeCapture @invokeParameters
}

function Assert-SafeRelayInputs {
    if ($Relay -notmatch "^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$") {
        Stop-Waiter "unsafe relay identity"
    }
    if (
        $BoundaryMarker -notmatch "^/[A-Za-z0-9._/-]+$" -or
        $BoundaryMarker.Contains("//")
    ) {
        Stop-Waiter "unsafe boundary marker path"
    }
    foreach ($part in $BoundaryMarker.Trim('/').Split('/')) {
        if (-not $part -or $part -in @(".", "..")) {
            Stop-Waiter "unsafe boundary marker path"
        }
    }
    foreach (
        $requiredPath in @(
            $SshKey,
            $Ssh,
            $Python,
            $MarkerValidator,
            $RunnerLauncher,
            $FinalizerLauncher
        )
    ) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            Stop-Waiter "required waiter dependency missing: $requiredPath"
        }
    }
}

function Wait-ForBoundaryMarker {
    $iterations = 0
    while ($true) {
        $command = "set -eu; " +
            "if [ ! -e '$BoundaryMarker' ]; then printf '%s\n' BOUNDARY_ABSENT; " +
            "elif [ ! -f '$BoundaryMarker' ] || ! test ! -L '$BoundaryMarker' || [ ! -s '$BoundaryMarker' ]; then printf '%s\n' BOUNDARY_INVALID; " +
            "else printf '%s ' BOUNDARY_READY; base64 < '$BoundaryMarker' | tr -d '\n'; printf '\n'; fi"
        $result = Invoke-SshCapture -Command $command
        if ($result.TimedOut) {
            Write-WaiterLog "SSH_TIMEOUT marker=$BoundaryMarker timeout_seconds=$SshWallClockTimeoutSeconds"
            Write-WaiterLog "WAIT_BOUNDARY_SSH reason=timeout"
            $iterations += 1
            if ($MaxWaitIterations -gt 0 -and $iterations -ge $MaxWaitIterations) {
                Stop-Waiter "boundary marker SSH retry limit reached"
            }
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($result.ExitCode -ne 0) {
            foreach ($line in $result.Output) {
                Write-WaiterLog "SSH_ERROR $line"
            }
            Write-WaiterLog "WAIT_BOUNDARY_SSH reason=exit_code_$($result.ExitCode)"
            $iterations += 1
            if ($MaxWaitIterations -gt 0 -and $iterations -ge $MaxWaitIterations) {
                Stop-Waiter "boundary marker SSH retry limit reached"
            }
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $response = (@($result.Output) -join "").Trim()
        if ($response -eq "BOUNDARY_ABSENT") {
            Write-WaiterLog "WAIT_BOUNDARY marker=$BoundaryMarker"
            $iterations += 1
            if ($MaxWaitIterations -gt 0 -and $iterations -ge $MaxWaitIterations) {
                Stop-Waiter "boundary marker wait limit reached"
            }
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ($response -eq "BOUNDARY_INVALID" -or -not $response.StartsWith("BOUNDARY_READY ")) {
            Stop-Waiter "invalid boundary marker relay response"
        }
        $encoded = $response.Substring("BOUNDARY_READY ".Length)
        try {
            $markerBytes = [Convert]::FromBase64String($encoded)
        }
        catch {
            Stop-Waiter "invalid boundary marker base64"
        }
        $temporaryMarker = [System.IO.Path]::GetTempFileName()
        try {
            [System.IO.File]::WriteAllBytes($temporaryMarker, $markerBytes)
            $validation = Invoke-NativeCapture -FilePath $Python -ArgumentList @(
                "-X", "utf8", $MarkerValidator, "--marker", $temporaryMarker
            )
            foreach ($line in $validation.Output) {
                Write-WaiterLog "MARKER_VALIDATOR $line"
            }
            if (
                $validation.ExitCode -ne 0 -or
                (@($validation.Output) -join "").Trim() -ne "BOUNDARY_MARKER_OK"
            ) {
                Stop-Waiter "invalid boundary marker: BOUNDARY_MARKER_INVALID"
            }
        }
        finally {
            if (Test-Path -LiteralPath $temporaryMarker) {
                Remove-Item -LiteralPath $temporaryMarker -Force
            }
        }
        Write-WaiterLog "BOUNDARY_MARKER_OK marker=$BoundaryMarker"
        return
    }
}

function Assert-LocalRepartitionBoundary {
    $queuePath = Join-Path $ResultsRoot "queue_status.json"
    if (-not (Test-Path -LiteralPath $queuePath -PathType Leaf)) {
        Stop-Waiter "old 5080 queue status is missing: $queuePath"
    }
    try {
        $queue = Get-Content -LiteralPath $queuePath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Stop-Waiter "old 5080 queue status is invalid JSON: $($_.Exception.Message)"
    }
    if (
        $queue.state -ne "completed" -or
        [int]$queue.total_tasks -ne 24 -or
        [int]$queue.completed_tasks -ne 24 -or
        [int]$queue.skipped_valid_tasks -lt 4 -or
        [int]$queue.skipped_valid_tasks -gt 24
    ) {
        Stop-Waiter "old 5080 queue is not the completed 24-task boundary"
    }

    $forbiddenTask = Join-Path $ResultsRoot "tasks\040_fullwindow_graphwavenet_base_contracosta_seed42"
    if (Test-Path -LiteralPath $forbiddenTask) {
        Stop-Waiter "forbidden task040 formal result already exists"
    }
    Write-WaiterLog "LOCAL_BOUNDARY_OK old_queue=24/24 task040_absent=true"
}

function Stop-ExactOldFinalizer {
    $oldPs1 = "C:\Users\ASUS\protocol_v3_finalize_5080_shard.ps1"
    $oldCmd = "C:\Users\ASUS\protocol_v3_finalize_5080_shard.cmd"
    $pidMatches = @(
        Get-CimInstance Win32_Process -Filter "ProcessId = $OldFinalizerPid"
    )
    if ($pidMatches.Count -gt 1) {
        Stop-Waiter "old finalizer PID lookup returned multiple processes"
    }
    if ($pidMatches.Count -eq 1) {
        $candidate = $pidMatches[0]
        $commandLine = [string]$candidate.CommandLine
        $processName = [string]$candidate.Name
        $ps1Match = (
            $processName.Equals("powershell.exe", [System.StringComparison]::OrdinalIgnoreCase) -and
            $commandLine.IndexOf($oldPs1, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
        )
        $cmdMatch = (
            $processName.Equals("cmd.exe", [System.StringComparison]::OrdinalIgnoreCase) -and
            $commandLine.IndexOf($oldCmd, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
        )
        if (-not $ps1Match -and -not $cmdMatch) {
            Stop-Waiter "old finalizer PID command line does not match the exact allowlist"
        }
        $oldTreeProcessIds = @($OldFinalizerPid) + @(
            Get-DescendantProcessIds -RootProcessId $OldFinalizerPid
        )
        Write-WaiterLog "STOP_OLD_FINALIZER pid=$OldFinalizerPid name=$processName"
        Stop-ProcessTree -ProcessId $OldFinalizerPid
    }
    else {
        $oldTreeProcessIds = @($OldFinalizerPid)
        Write-WaiterLog "OLD_FINALIZER_PID_ALREADY_ZERO pid=$OldFinalizerPid"
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($OldFinalizerStopWaitSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $treeAlive = @(
            foreach ($treeProcessId in $oldTreeProcessIds) {
                if ($null -ne (Get-Process -Id $treeProcessId -ErrorAction SilentlyContinue)) {
                    $treeProcessId
                }
            }
        )
        if ($treeAlive.Count -eq 0) {
            break
        }
        Start-Sleep -Milliseconds 250
    }
    foreach ($treeProcessId in $oldTreeProcessIds) {
        if ($null -ne (Get-Process -Id $treeProcessId -ErrorAction SilentlyContinue)) {
            Stop-Waiter "old 5080 finalizer process tree did not stop: $treeProcessId"
        }
    }

    while ($true) {
        $remaining = @(
            Get-CimInstance Win32_Process |
                Where-Object {
                    $_.ProcessId -ne $PID -and
                    $_.CommandLine -and
                    (
                        $_.CommandLine.IndexOf(
                            $oldPs1,
                            [System.StringComparison]::OrdinalIgnoreCase
                        ) -ge 0 -or
                        $_.CommandLine.IndexOf(
                            $oldCmd,
                            [System.StringComparison]::OrdinalIgnoreCase
                        ) -ge 0
                    )
                }
        )
        if ($remaining.Count -eq 0) {
            break
        }
        if ([DateTime]::UtcNow -ge $deadline) {
            $identities = ($remaining | ForEach-Object { [string]$_.ProcessId }) -join ","
            Stop-Waiter "old 5080 finalizer processes remain alive: $identities"
        }
        Start-Sleep -Milliseconds 250
    }
    Write-WaiterLog "OLD_FINALIZER_STOPPED pid=$OldFinalizerPid remaining=0"
}

function Start-CmdDetached {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not $env:ComSpec -or -not (Test-Path -LiteralPath $env:ComSpec -PathType Leaf)) {
        Stop-Waiter "Windows command processor is unavailable"
    }
    $startParameters = @{
        FilePath = $env:ComSpec
        ArgumentList = @("/d", "/c", $Path)
        WindowStyle = "Hidden"
        PassThru = $true
    }
    return Start-Process @startParameters
}

function Wait-ForNewRunnerContract {
    param([Parameter(Mandatory = $true)][object]$RunnerProcess)

    $queuePath = Join-Path $ResultsRoot "queue_status.json"
    $deadline = [DateTime]::UtcNow.AddSeconds($NewQueueWaitSeconds)
    while ([DateTime]::UtcNow -le $deadline) {
        if (Test-Path -LiteralPath $queuePath -PathType Leaf) {
            try {
                $queue = Get-Content -LiteralPath $queuePath -Raw -Encoding UTF8 | ConvertFrom-Json
            }
            catch {
                Stop-Waiter "new 27-task queue status is invalid JSON: $($_.Exception.Message)"
            }
            [int]$queueTotal = $queue.total_tasks
            if ([int]$queue.total_tasks -eq 27) {
                if ($queue.state -eq "failed") {
                    Stop-Waiter "new 27-task runner entered failed state"
                }
                if ($queue.state -in @("preflight", "running", "completed")) {
                    Write-WaiterLog "NEW_QUEUE_OK state=$($queue.state) total=27"
                    return
                }
                Stop-Waiter "new 27-task runner published unexpected state: $($queue.state)"
            }
            if ($queueTotal -ne 24) {
                Stop-Waiter "unexpected queue total while waiting for new runner: $queueTotal"
            }
        }
        $RunnerProcess.Refresh()
        if ($RunnerProcess.HasExited) {
            Stop-Waiter "new 27-task runner exited before publishing its queue"
        }
        Start-Sleep -Seconds 1
    }
    Stop-Waiter "new 27-task runner queue publication timed out"
}

function Start-RepartitionProcesses {
    $runner = Start-CmdDetached -Path $RunnerLauncher
    Write-WaiterLog "RUNNER_STARTED pid=$($runner.Id)"
    Wait-ForNewRunnerContract -RunnerProcess $runner

    $finalizer = Start-CmdDetached -Path $FinalizerLauncher
    Write-WaiterLog "FINALIZER_STARTED pid=$($finalizer.Id)"
    Start-Sleep -Seconds 2
    $finalizer.Refresh()
    if ($finalizer.HasExited) {
        Stop-Waiter "new 27-task finalizer exited during startup"
    }
    Write-WaiterLog "REPARTITION_STARTED runner_pid=$($runner.Id) finalizer_pid=$($finalizer.Id)"
}

try {
    Assert-SafeRelayInputs
    Wait-ForBoundaryMarker
    Assert-LocalRepartitionBoundary
    Stop-ExactOldFinalizer
    Assert-LocalRepartitionBoundary
    Start-RepartitionProcesses
}
catch {
    Write-WaiterLog "REPARTITION_BLOCKED $($_.Exception.Message)"
    exit 1
}
