[CmdletBinding()]
param(
    [string]$Repo = "C:\Users\ASUS\traffic_fourier_v3_final",
    [string]$OpsRoot = "C:\Users\ASUS\ops_27_18",
    [string]$ResultsRoot = "C:\Users\ASUS\traffic_fourier_v3_final\outputs\protocol_v3_noleak_dedup\remote_runs\minimal_45_5080",
    [string]$RunnerExitCode = "C:\Users\ASUS\protocol_v3_5080_shard27.exitcode",
    [string]$Python = "C:\Python313\python.exe",
    [string]$Postflight = "C:\Users\ASUS\ops_27_18\scripts\remote\distributed\protocol_v3_shard_postflight.py",
    [string]$PostflightReport = "C:\Users\ASUS\protocol_v3_5080_shard27_postflight.json",
    [string]$TransferManifest = "C:\Users\ASUS\protocol_v3_5080_shard27_transfer.sha256",
    [string]$TransferFileList = "C:\Users\ASUS\protocol_v3_5080_shard27_files.txt",
    [string]$TransferMarker = "C:\Users\ASUS\protocol_v3_5080_shard27_complete.json",
    [string]$Relay = "root@8.134.12.5",
    [string]$RelayBase = "/root/xtraffic_results/protocol_v3_noleak_dedup/distributed_45_5080_5090/shards",
    [string]$RemoteName = "rtx5080_27",
    [string]$DependencyName = "rtx5090_18",
    [string]$SshKey = "C:\Users\ASUS\.ssh\xtraffic_relay_ed25519",
    [string]$Ssh = "C:\Windows\System32\OpenSSH\ssh.exe",
    [string]$Scp = "C:\Windows\System32\OpenSSH\scp.exe",
    [long]$SpaceMarginBytes = 2147483648,
    [int]$PollSeconds = 60,
    [int]$RunnerExitWaitSeconds = 300,
    [int]$SpaceRetrySeconds = 300,
    [int]$NativeWallClockTimeoutSeconds = 120,
    [int]$SshWallClockTimeoutSeconds = 60,
    [int]$RemoteVerificationWallClockTimeoutSeconds = 3600,
    [int]$ScpWallClockTimeoutSeconds = 7200,
    [int]$PostflightWallClockTimeoutSeconds = 1800,
    [int]$MaxWaitIterations = 0,
    [string]$PidFile = "C:\Users\ASUS\protocol_v3_5080_shard27_finalize.pid",
    [string]$Log = "C:\Users\ASUS\protocol_v3_5080_shard27_finalize.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-FinalizerLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0} {1}" -f [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"), $Message
    $parent = Split-Path -Parent $Log
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
    Write-Host $line
}

function Stop-Finalizer {
    param([Parameter(Mandatory = $true)][string]$Message)

    throw $Message
}

function Read-QueueStatus {
    $queuePath = Join-Path $ResultsRoot "queue_status.json"
    if (-not (Test-Path -LiteralPath $queuePath -PathType Leaf)) {
        Stop-Finalizer "queue status missing: $queuePath"
    }
    try {
        return Get-Content -LiteralPath $queuePath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Stop-Finalizer "queue status is not valid JSON: $($_.Exception.Message)"
    }
}

function Wait-ForCompletedQueue {
    $iterations = 0
    while ($true) {
        $queue = Read-QueueStatus
        $totalTasks = [int]$queue.total_tasks
        $completedTasks = [int]$queue.completed_tasks
        $skippedTasks = [int]$queue.skipped_valid_tasks
        if (
            $queue.State -eq "completed" -and
            $totalTasks -eq 27 -and
            $completedTasks -eq 27 -and
            $skippedTasks -ge 24 -and
            $skippedTasks -le 27
        ) {
            return $queue
        }
        if ($queue.State -in @("failed", "interrupted")) {
            Stop-Finalizer "queue entered terminal state $($queue.State)"
        }
        if ($queue.State -notin @("preflight", "running")) {
            Stop-Finalizer "unexpected queue state $($queue.State) with $completedTasks/$totalTasks"
        }
        Write-FinalizerLog "WAIT queue_state=$($queue.State) completed=$completedTasks/$totalTasks"
        $iterations += 1
        if ($MaxWaitIterations -gt 0 -and $iterations -ge $MaxWaitIterations) {
            Stop-Finalizer "queue wait limit reached"
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Wait-ForRunnerSuccess {
    $queuePath = Join-Path $ResultsRoot "queue_status.json"
    $minimumWriteTime = (Get-Item -LiteralPath $queuePath).LastWriteTimeUtc.AddSeconds(-2)
    $deadline = [DateTime]::UtcNow.AddSeconds($RunnerExitWaitSeconds)
    while ([DateTime]::UtcNow -le $deadline) {
        if (Test-Path -LiteralPath $RunnerExitCode -PathType Leaf) {
            $item = Get-Item -LiteralPath $RunnerExitCode
            $value = (Get-Content -LiteralPath $RunnerExitCode -Raw -Encoding UTF8).Trim()
            if ($item.LastWriteTimeUtc -ge $minimumWriteTime -and $value -match "^-?[0-9]+$") {
                if ([int]$value -ne 0) {
                    Stop-Finalizer "runner exit code is not zero: $value"
                }
                return
            }
        }
        Start-Sleep -Seconds 1
    }
    Stop-Finalizer "runner exit code is missing or stale"
}

function Stop-NativeProcessTree {
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
            Stop-Finalizer "cannot terminate native process tree rooted at PID $ProcessId"
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
        Stop-Finalizer "native process timeout must be positive"
    }
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    $payloadPath = [System.IO.Path]::GetTempFileName()
    $nativeExitCodePath = [System.IO.Path]::GetTempFileName()
    $wrapperSeedPath = [System.IO.Path]::GetTempFileName()
    $wrapperPath = [System.IO.Path]::ChangeExtension(
        $wrapperSeedPath,
        ".ps1"
    )
    Remove-Item -LiteralPath $nativeExitCodePath, $wrapperSeedPath -Force
    $payload = [PSCustomObject]@{
        FilePath = $FilePath
        ArgumentList = @($ArgumentList)
        ExitCodePath = $nativeExitCodePath
    }
    $payload | Export-Clixml -LiteralPath $payloadPath -Encoding UTF8
    $wrapper = @'
param([Parameter(Mandatory = $true)][string]$PayloadPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$payload = Import-Clixml -LiteralPath $PayloadPath
$nativeFile = [string]$payload.FilePath
$nativeArguments = @(
    $payload.ArgumentList | ForEach-Object { [string]$_ }
)
$nativeExitCode = 127
try {
    & $nativeFile @nativeArguments
    $nativeExitCode = [int]$LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine($_.Exception.ToString())
}
finally {
    [System.IO.File]::WriteAllText(
        [string]$payload.ExitCodePath,
        [string]$nativeExitCode,
        [System.Text.UTF8Encoding]::new($false)
    )
}
exit $nativeExitCode
'@
    [System.IO.File]::WriteAllText(
        $wrapperPath,
        $wrapper,
        [System.Text.UTF8Encoding]::new($false)
    )
    $process = $null
    try {
        $startParameters = @{
            FilePath = (Join-Path $PSHOME "powershell.exe")
            ArgumentList = @(
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                $wrapperPath,
                "-PayloadPath",
                $payloadPath
            )
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError = $stderrPath
            NoNewWindow = $true
            PassThru = $true
        }
        $process = Start-Process @startParameters
        $finished = $process.WaitForExit($TimeoutSeconds * 1000)
        $timedOut = -not $finished
        if ($timedOut) {
            Stop-NativeProcessTree -ProcessId $process.Id
            [void]$process.WaitForExit(5000)
        }
        else {
            [void]$process.WaitForExit()
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
            if (-not (Test-Path -LiteralPath $nativeExitCodePath -PathType Leaf)) {
                Stop-Finalizer "native process exit code file is unavailable"
            }
            $exitCodeText = (
                Get-Content -LiteralPath $nativeExitCodePath -Raw -Encoding UTF8
            ).Trim()
            if ($exitCodeText -notmatch "^-?[0-9]+$") {
                Stop-Finalizer "native process exit code is invalid"
            }
            $exitCode = [int]$exitCodeText
        }
        return [PSCustomObject]@{
            ExitCode = $exitCode
            TimedOut = $timedOut
            Output = @($output | ForEach-Object { $_.ToString() })
        }
    }
    finally {
        foreach (
            $temporaryPath in @(
                $stdoutPath,
                $stderrPath,
                $payloadPath,
                $nativeExitCodePath,
                $wrapperSeedPath,
                $wrapperPath
            )
        ) {
            if (Test-Path -LiteralPath $temporaryPath) {
                Remove-Item -LiteralPath $temporaryPath -Force
            }
        }
    }
}

function Invoke-Postflight {
    if (-not (Test-Path -LiteralPath $Postflight -PathType Leaf)) {
        Stop-Finalizer "postflight script missing: $Postflight"
    }
    if (Test-Path -LiteralPath $PostflightReport) {
        Remove-Item -LiteralPath $PostflightReport -Force
    }
    $oldRepoRoot = $env:XTRAFFIC_REPO_ROOT
    $oldPythonPath = $env:PYTHONPATH
    try {
        $env:XTRAFFIC_REPO_ROOT = $OpsRoot
        $env:PYTHONPATH = "$OpsRoot;$Repo"
        $arguments = @(
            "-X", "utf8", $Postflight,
            "--shard", "5080",
            "--manifest", (Join-Path $Repo "scripts\remote\windows\protocol_v3_manifest.json"),
            "--repo-root", $Repo,
            "--results-root", $ResultsRoot,
            "--cache-dir", (Join-Path $Repo "outputs\dist_net\region_data_v3_noleak_dedup"),
            "--graph-dir", (Join-Path $Repo "outputs\region_graphs"),
            "--cache-bundle-id", "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4",
            "--cache-bundle-digest", "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5",
            "--code-sha", "3a4952837aaf6fbda1403d50b610b1bf40c6098a",
            "--python", $Python,
            "--report", $PostflightReport
        )
        Write-FinalizerLog "POSTFLIGHT_START"
        $invokeParameters = @{
            FilePath = $Python
            ArgumentList = $arguments
            TimeoutSeconds = $PostflightWallClockTimeoutSeconds
        }
        $result = Invoke-NativeCapture @invokeParameters
        foreach ($line in $result.Output) {
            Write-FinalizerLog "POSTFLIGHT $line"
        }
        if ($result.TimedOut) {
            Stop-Finalizer "native process timed out during postflight"
        }
        if ($result.ExitCode -ne 0) {
            Stop-Finalizer "postflight validation failed with code $($result.ExitCode)"
        }
    }
    finally {
        $env:XTRAFFIC_REPO_ROOT = $oldRepoRoot
        $env:PYTHONPATH = $oldPythonPath
    }
    if (-not (Test-Path -LiteralPath $PostflightReport -PathType Leaf)) {
        Stop-Finalizer "postflight report is missing"
    }
    if ((Get-Item -LiteralPath $PostflightReport).Length -eq 0) {
        Stop-Finalizer "postflight report is empty"
    }
    Write-FinalizerLog "POSTFLIGHT_OK"
}

function Assert-SafeExportPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)

    if (
        -not $RelativePath -or
        $RelativePath.Contains("\") -or
        $RelativePath.StartsWith("/") -or
        $RelativePath.EndsWith("/")
    ) {
        Stop-Finalizer "unsafe export path: $RelativePath"
    }
    $parts = $RelativePath.Split("/")
    foreach ($part in $parts) {
        if (
            -not $part -or
            $part -in @(".", "..") -or
            $part -notmatch "^[A-Za-z0-9._-]+$"
        ) {
            Stop-Finalizer "unsafe export path: $RelativePath"
        }
    }
}

function Read-VerifiedExportRecords {
    try {
        $report = Get-Content -LiteralPath $PostflightReport -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        Stop-Finalizer "postflight report is not valid JSON: $($_.Exception.Message)"
    }
    if ($report.status -ne "verified" -or $report.shard -ne "5080") {
        Stop-Finalizer "invalid postflight report identity"
    }
    if (
        [int]$report.validated_tasks -ne 27 -or
        [int]$report.expected_skipped_valid_tasks -ne 24 -or
        [int]$report.minimum_skipped_valid_tasks -ne 24
    ) {
        Stop-Finalizer "invalid postflight task counts"
    }
    [int]$actualSkipped = $report.actual_skipped_valid_tasks
    if ($actualSkipped -lt 24 -or $actualSkipped -gt 27) {
        Stop-Finalizer "invalid actual postflight skip count"
    }
    $expectedSkipIds = @($report.expected_skipped_task_ids)
    $actualSkipIds = @($report.actual_skipped_task_ids)
    if ($expectedSkipIds.Count -ne 24 -or $actualSkipIds.Count -ne $actualSkipped) {
        Stop-Finalizer "invalid postflight skipped task identity"
    }
    $actualSkipSet = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal
    )
    foreach ($taskId in $actualSkipIds) {
        if (-not $actualSkipSet.Add([string]$taskId)) {
            Stop-Finalizer "duplicate actual skipped task identity"
        }
    }
    foreach ($taskId in $expectedSkipIds) {
        if (-not $actualSkipSet.Contains([string]$taskId)) {
            Stop-Finalizer "inherited skipped task identity is missing: $taskId"
        }
    }
    $tasks = @($report.tasks)
    if ($tasks.Count -ne 27) {
        Stop-Finalizer "invalid postflight task records"
    }
    [int]$completedStateCount = 0
    [int]$skippedStateCount = 0
    $taskSkipSet = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal
    )
    foreach ($task in $tasks) {
        if ($task.state -eq "completed") {
            $completedStateCount += 1
        }
        elseif ($task.state -eq "skipped_valid") {
            $skippedStateCount += 1
            if (-not $taskSkipSet.Add([string]$task.task_id)) {
                Stop-Finalizer "duplicate skipped task record"
            }
        }
        else {
            Stop-Finalizer "invalid postflight task state: $($task.state)"
        }
    }
    if (
        $skippedStateCount -ne $actualSkipped -or
        $completedStateCount + $skippedStateCount -ne 27 -or
        [int]$report.task_state_counts.completed -ne $completedStateCount -or
        [int]$report.task_state_counts.skipped_valid -ne $skippedStateCount -or
        -not $taskSkipSet.SetEquals($actualSkipSet)
    ) {
        Stop-Finalizer "postflight task state summary is inconsistent"
    }
    if (
        $report.queue_status.state -ne "completed" -or
        [int]$report.queue_status.total_tasks -ne 27 -or
        [int]$report.queue_status.completed_tasks -ne 27 -or
        [int]$report.queue_status.skipped_valid_tasks -ne $actualSkipped -or
        $report.verified_at -ne $report.queue_status.updated_at -or
        $report.source_completed_at -ne $report.queue_status.updated_at
    ) {
        Stop-Finalizer "invalid or unstable postflight queue summary"
    }
    $postflightFilesDigest = ([string]$report.files_digest).ToLowerInvariant()
    if ($postflightFilesDigest -notmatch "^[0-9a-f]{64}$") {
        Stop-Finalizer "invalid postflight files_digest"
    }

    $files = @($report.files)
    $exportPaths = @($report.export_paths)
    if ($files.Count -eq 0 -or $exportPaths.Count -eq 0) {
        Stop-Finalizer "postflight report lacks export records"
    }
    if ([int]$report.file_count -ne $files.Count) {
        Stop-Finalizer "postflight file_count differs from file records"
    }
    $byPath = [System.Collections.Generic.Dictionary[string,object]]::new(
        [System.StringComparer]::Ordinal
    )
    $root = [System.IO.Path]::GetFullPath($ResultsRoot).TrimEnd([char[]]@('\', '/'))
    $rootPrefix = $root + [System.IO.Path]::DirectorySeparatorChar
    [long]$sourceBytes = 0
    foreach ($record in $files) {
        $relative = [string]$record.path
        $digest = ([string]$record.sha256).ToLowerInvariant()
        Assert-SafeExportPath -RelativePath $relative
        if ($digest -notmatch "^[0-9a-f]{64}$") {
            Stop-Finalizer "invalid postflight digest for $relative"
        }
        if ($byPath.ContainsKey($relative)) {
            Stop-Finalizer "duplicate postflight export path: $relative"
        }
        [long]$expectedSize = $record.size_bytes
        if ($expectedSize -lt 0) {
            Stop-Finalizer "invalid postflight size for $relative"
        }
        $nativeRelative = $relative.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        $source = [System.IO.Path]::GetFullPath((Join-Path $root $nativeRelative))
        if (-not $source.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            Stop-Finalizer "export path escapes results root: $relative"
        }
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            Stop-Finalizer "postflight export file is missing: $relative"
        }
        $sourceItem = Get-Item -LiteralPath $source -Force
        if (($sourceItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            Stop-Finalizer "postflight export file is a reparse point: $relative"
        }
        if ([long]$sourceItem.Length -ne $expectedSize) {
            Stop-Finalizer "source file size differs from postflight report: $relative"
        }
        $actualDigest = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualDigest -ne $digest) {
            Stop-Finalizer "source file hash differs from postflight report: $relative"
        }
        $verified = [PSCustomObject]@{
            Path = $relative
            Source = $source
            SizeBytes = $expectedSize
            Sha256 = $digest
        }
        $byPath.Add($relative, $verified)
        $sourceBytes += $expectedSize
    }

    $ordered = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal
    )
    foreach ($exportPathValue in $exportPaths) {
        $exportPath = [string]$exportPathValue
        Assert-SafeExportPath -RelativePath $exportPath
        if (-not $seen.Add($exportPath) -or -not $byPath.ContainsKey($exportPath)) {
            Stop-Finalizer "export path set differs from hashed file records"
        }
        $ordered.Add($byPath[$exportPath])
    }
    if ($seen.Count -ne $byPath.Count) {
        Stop-Finalizer "export path set differs from hashed file records"
    }
    if ([long]$report.total_bytes -ne $sourceBytes) {
        Stop-Finalizer "postflight total_bytes differs from verified exports"
    }

    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    $manifestParent = Split-Path -Parent $TransferManifest
    if ($manifestParent -and -not (Test-Path -LiteralPath $manifestParent)) {
        New-Item -ItemType Directory -Path $manifestParent -Force | Out-Null
    }
    $fileListParent = Split-Path -Parent $TransferFileList
    if ($fileListParent -and -not (Test-Path -LiteralPath $fileListParent)) {
        New-Item -ItemType Directory -Path $fileListParent -Force | Out-Null
    }
    $manifestText = ($ordered | ForEach-Object { "$($_.Sha256)  $($_.Path)`n" }) -join ""
    $fileListText = ($ordered | ForEach-Object { "$($_.Path)`n" }) -join ""
    [System.IO.File]::WriteAllText($TransferManifest, $manifestText, $utf8NoBom)
    [System.IO.File]::WriteAllText($TransferFileList, $fileListText, $utf8NoBom)
    if ((Get-Item -LiteralPath $TransferManifest).Length -eq 0) {
        Stop-Finalizer "transfer manifest is empty"
    }
    $manifestDigest = (
        Get-FileHash -LiteralPath $TransferManifest -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $fileListDigest = (
        Get-FileHash -LiteralPath $TransferFileList -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $reportDigest = (
        Get-FileHash -LiteralPath $PostflightReport -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $markerPayload = [ordered]@{
        actual_skipped_valid_tasks = $actualSkipped
        file_count = $ordered.Count
        file_list_sha256 = $fileListDigest
        manifest_sha256 = $manifestDigest
        minimum_skipped_valid_tasks = 24
        postflight_files_digest = $postflightFilesDigest
        postflight_sha256 = $reportDigest
        schema_version = 1
        shard = "5080"
        source_bytes = $sourceBytes
        status = "verified"
        validated_tasks = 27
    }
    $markerParent = Split-Path -Parent $TransferMarker
    if ($markerParent -and -not (Test-Path -LiteralPath $markerParent)) {
        New-Item -ItemType Directory -Path $markerParent -Force | Out-Null
    }
    [System.IO.File]::WriteAllText(
        $TransferMarker,
        (($markerPayload | ConvertTo-Json -Compress) + "`n"),
        $utf8NoBom
    )
    $markerDigest = (
        Get-FileHash -LiteralPath $TransferMarker -Algorithm SHA256
    ).Hash.ToLowerInvariant()

    return [PSCustomObject]@{
        Records = $ordered.ToArray()
        SourceBytes = $sourceBytes
        Report = $report
        ManifestDigest = $manifestDigest
        FileListDigest = $fileListDigest
        ReportDigest = $reportDigest
        MarkerDigest = $markerDigest
    }
}

function Get-LastNativeOutputLine {
    param([Parameter(Mandatory = $true)][object]$Result)

    $lines = @($Result.Output | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($lines.Count -eq 0) {
        return ""
    }
    return ([string]$lines[-1]).Trim()
}

function Invoke-SshCapture {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [int]$TimeoutSeconds = $SshWallClockTimeoutSeconds
    )

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
        TimeoutSeconds = $TimeoutSeconds
    }
    return Invoke-NativeCapture @invokeParameters
}

function Invoke-SshRequired {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )

    $result = Invoke-SshCapture -Command $Command
    if ($result.TimedOut) {
        Stop-Finalizer "native process timed out during SSH operation"
    }
    if ($result.ExitCode -ne 0) {
        foreach ($line in $result.Output) {
            Write-FinalizerLog "SSH_ERROR $line"
        }
        Stop-Finalizer "$FailureMessage with code $($result.ExitCode)"
    }
    return $result
}

function Test-RemoteVerifiedTree {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$MarkerName,
        [Parameter(Mandatory = $true)][object]$Exports
    )

    $command = "set -eu; root='$Root'; marker_name='$MarkerName'; " +
        "test -d `"`$root`"; test ! -L `"`$root`"; " +
        "test -d `"`$root/results`"; test ! -L `"`$root/results`"; " +
        "test -d `"`$root/metadata`"; test ! -L `"`$root/metadata`"; " +
        "test -f `"`$root/`$marker_name`"; test ! -L `"`$root/`$marker_name`"; " +
        "test `"`$(sha256sum `"`$root/`$marker_name`" | awk '{print `$1}')`" = '$($Exports.MarkerDigest)'; " +
        "test `"`$(sha256sum `"`$root/metadata/transfer.sha256`" | awk '{print `$1}')`" = '$($Exports.ManifestDigest)'; " +
        "test `"`$(sha256sum `"`$root/metadata/transfer.files`" | awk '{print `$1}')`" = '$($Exports.FileListDigest)'; " +
        "test `"`$(sha256sum `"`$root/metadata/postflight.json`" | awk '{print `$1}')`" = '$($Exports.ReportDigest)'; " +
        "test -z `"`$(find `"`$root`" -type l -print -quit)`"; " +
        "test -z `"`$(find `"`$root`" ! -type f ! -type d -print -quit)`"; " +
        "tmp=`$(mktemp -d); " +
        "printf '%s\n' '_TRANSFER_COMPLETE.json' metadata results | LC_ALL=C sort > `"`$tmp/expected.top`"; " +
        "find `"`$root`" -mindepth 1 -maxdepth 1 -printf '%f\n' | sed `"s/^`$marker_name`$/_TRANSFER_COMPLETE.json/`" | LC_ALL=C sort > `"`$tmp/actual.top`"; " +
        "cmp `"`$tmp/actual.top`" `"`$tmp/expected.top`"; " +
        "printf '%s\n' postflight.json transfer.files transfer.sha256 | LC_ALL=C sort > `"`$tmp/expected.metadata`"; " +
        "find `"`$root/metadata`" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort > `"`$tmp/actual.metadata`"; " +
        "cmp `"`$tmp/actual.metadata`" `"`$tmp/expected.metadata`"; " +
        "test -z `"`$(find `"`$root/metadata`" -mindepth 2 -print -quit)`"; " +
        "cd `"`$root/results`"; sha256sum -c ../metadata/transfer.sha256 >/dev/null; " +
        "cut -c67- ../metadata/transfer.sha256 | LC_ALL=C sort > `"`$tmp/manifest.files`"; " +
        "LC_ALL=C sort ../metadata/transfer.files > `"`$tmp/expected.files`"; " +
        "cmp `"`$tmp/manifest.files`" `"`$tmp/expected.files`"; " +
        "find . -type f -printf '%P\n' | LC_ALL=C sort > `"`$tmp/actual.files`"; " +
        "cmp `"`$tmp/actual.files`" `"`$tmp/expected.files`"; " +
        "awk -F/ '{ path=`"`"; for (i=1; i<NF; i++) { path=(path ? path `"/`" : `"`") `$i; print path }}' ../metadata/transfer.files | LC_ALL=C sort -u > `"`$tmp/expected.dirs`"; " +
        "find . -mindepth 1 -type d -printf '%P\n' | LC_ALL=C sort > `"`$tmp/actual.dirs`"; " +
        "cmp `"`$tmp/actual.dirs`" `"`$tmp/expected.dirs`"; rm -rf `"`$tmp`""
    $result = Invoke-SshCapture -Command $command -TimeoutSeconds $RemoteVerificationWallClockTimeoutSeconds
    if ($result.ExitCode -eq 0) {
        return $true
    }
    foreach ($line in $result.Output) {
        Write-FinalizerLog "REMOTE_VERIFY_ERROR $line"
    }
    return $false
}

function Wait-ForDependencyTransfer {
    param([Parameter(Mandatory = $true)][string]$DependencyMarker)

    $iterations = 0
    while ($true) {
        $command = "if [ -s '$DependencyMarker' ]; then printf '%s\n' READY; else printf '%s\n' WAIT; fi"
        $result = Invoke-SshRequired -Command $command -FailureMessage "cannot query 5090 transfer dependency"
        $state = Get-LastNativeOutputLine -Result $result
        if ($state -eq "READY") {
            Write-FinalizerLog "DEPENDENCY_OK marker=$DependencyMarker"
            return
        }
        if ($state -ne "WAIT") {
            Stop-Finalizer "unexpected dependency state: $state"
        }
        Write-FinalizerLog "WAIT_DEPENDENCY marker=$DependencyMarker"
        $iterations += 1
        if ($MaxWaitIterations -gt 0 -and $iterations -ge $MaxWaitIterations) {
            Stop-Finalizer "dependency wait limit reached"
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Wait-ForRelayCapacity {
    param(
        [Parameter(Mandatory = $true)][object]$exports,
        [Parameter(Mandatory = $true)][string]$RemoteIncoming,
        [Parameter(Mandatory = $true)][string]$RemoteFinal
    )

    $iterations = 0
    $requiredCapacity = $exports.SourceBytes + $SpaceMarginBytes
    while ($true) {
        $command = "set -eu; mkdir -p '$RelayBase'; " +
            "free=`$(df -PB1 '$RelayBase' | awk 'NR==2 {print `$4}'); " +
            "incoming=0; if [ -d '$RemoteIncoming' ]; then incoming=`$(du -sb '$RemoteIncoming' | awk '{print `$1}'); fi; " +
            "final_exists=0; if [ -e '$RemoteFinal' ]; then final_exists=1; fi; " +
            "printf '%s %s %s\n' `"`$free`" `"`$incoming`" `"`$final_exists`""
        $result = Invoke-SshRequired -Command $command -FailureMessage "cannot query relay capacity"
        $line = Get-LastNativeOutputLine -Result $result
        $parts = @($line -split "\s+")
        if ($parts.Count -ne 3 -or $parts[0] -notmatch "^[0-9]+$" -or $parts[1] -notmatch "^[0-9]+$") {
            Stop-Finalizer "invalid relay capacity response: $line"
        }
        $finalExists = $parts[2]
        if ($finalExists -eq "1") {
            if (
                Test-RemoteVerifiedTree `
                    -Root $RemoteFinal `
                    -MarkerName "_TRANSFER_COMPLETE.json" `
                    -Exports $exports
            ) {
                return [PSCustomObject]@{ AlreadyVerified = $true }
            }
            Stop-Finalizer "relay destination exists but full verification failed"
        }
        if ($finalExists -ne "0") {
            Stop-Finalizer "invalid relay final destination state: $finalExists"
        }
        [long]$availableCapacity = [long]$parts[0] + [long]$parts[1]
        if ($availableCapacity -ge $requiredCapacity) {
            Write-FinalizerLog "SPACE_OK available=$availableCapacity required=$requiredCapacity"
            return [PSCustomObject]@{ AlreadyVerified = $false }
        }
        Write-FinalizerLog "WAIT_SPACE available=$availableCapacity required=$requiredCapacity"
        $iterations += 1
        if ($MaxWaitIterations -gt 0 -and $iterations -ge $MaxWaitIterations) {
            Stop-Finalizer "relay capacity wait limit reached"
        }
        Start-Sleep -Seconds $SpaceRetrySeconds
    }
}

function Assert-LocalFileDigest {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedDigest,
        [long]$ExpectedSize = -1
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Stop-Finalizer "local transfer file is missing: $Path"
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        Stop-Finalizer "local transfer file is a reparse point: $Path"
    }
    if ($ExpectedSize -ge 0 -and [long]$item.Length -ne $ExpectedSize) {
        Stop-Finalizer "local transfer file changed size: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedDigest) {
        Stop-Finalizer "local transfer file changed hash: $Path"
    }
}

function Upload-RemoteFileByHash {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$ExpectedDigest,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [long]$ExpectedSize = -1
    )

    Assert-LocalFileDigest -Path $LocalPath -ExpectedDigest $ExpectedDigest -ExpectedSize $ExpectedSize
    $remotePart = "$RemotePath.part"
    $remoteParent = $RemotePath.Substring(0, $RemotePath.LastIndexOf('/'))
    $checkCommand = "set -eu; mkdir -p '$remoteParent'; " +
        "if [ -f '$RemotePath' ] && [ `"`$(sha256sum '$RemotePath' | awk '{print `$1}')`" = '$ExpectedDigest' ]; then " +
        "printf '%s\n' MATCH; " +
        "elif [ -f '$remotePart' ] && [ `"`$(sha256sum '$remotePart' | awk '{print `$1}')`" = '$ExpectedDigest' ]; then " +
        "mv -f '$remotePart' '$RemotePath'; printf '%s\n' MATCH; " +
        "else rm -f '$RemotePath' '$remotePart'; printf '%s\n' UPLOAD; fi"
    $check = Invoke-SshRequired -Command $checkCommand -FailureMessage "cannot inspect remote file $RemotePath"
    $state = Get-LastNativeOutputLine -Result $check
    if ($state -eq "MATCH") {
        Write-FinalizerLog "TRANSFER_SKIP sha256=$ExpectedDigest path=$RemotePath"
        return
    }
    if ($state -ne "UPLOAD") {
        Stop-Finalizer "unexpected remote file state for $RemotePath`: $state"
    }

    $scpArguments = @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        $LocalPath,
        "${Relay}:$remotePart"
    )
    $copyParameters = @{
        FilePath = $Scp
        ArgumentList = $scpArguments
        TimeoutSeconds = $ScpWallClockTimeoutSeconds
    }
    $copy = Invoke-NativeCapture @copyParameters
    if ($copy.TimedOut) {
        Stop-Finalizer "native process timed out during SCP upload for $RemotePath"
    }
    if ($copy.ExitCode -ne 0) {
        foreach ($line in $copy.Output) {
            Write-FinalizerLog "SCP_ERROR $line"
        }
        Stop-Finalizer "file upload failed for $RemotePath with code $($copy.ExitCode)"
    }
    $promoteCommand = "set -eu; test -f '$remotePart'; " +
        "test `"`$(sha256sum '$remotePart' | awk '{print `$1}')`" = '$ExpectedDigest'; " +
        "mv -f '$remotePart' '$RemotePath'; sync"
    Invoke-SshRequired -Command $promoteCommand -FailureMessage "remote file verification failed for $RemotePath" | Out-Null
    Write-FinalizerLog "TRANSFER_FILE_OK sha256=$ExpectedDigest path=$RemotePath"
}

function Transfer-VerifiedExports {
    param(
        [Parameter(Mandatory = $true)][object]$exports,
        [Parameter(Mandatory = $true)][string]$RemoteIncoming,
        [Parameter(Mandatory = $true)][string]$RemoteFinal
    )

    Write-FinalizerLog "TRANSFER_START bytes=$($exports.SourceBytes) files=$($exports.Records.Count)"
    foreach ($record in $exports.Records) {
        Upload-RemoteFileByHash `
            -LocalPath $record.Source `
            -ExpectedDigest $record.Sha256 `
            -RemotePath "$RemoteIncoming/results/$($record.Path)" `
            -ExpectedSize $record.SizeBytes
    }
    Upload-RemoteFileByHash `
        -LocalPath $TransferManifest `
        -ExpectedDigest $exports.ManifestDigest `
        -RemotePath "$RemoteIncoming/metadata/transfer.sha256"
    Upload-RemoteFileByHash `
        -LocalPath $TransferFileList `
        -ExpectedDigest $exports.FileListDigest `
        -RemotePath "$RemoteIncoming/metadata/transfer.files"
    Upload-RemoteFileByHash `
        -LocalPath $PostflightReport `
        -ExpectedDigest $exports.ReportDigest `
        -RemotePath "$RemoteIncoming/metadata/postflight.json"
    Upload-RemoteFileByHash `
        -LocalPath $TransferMarker `
        -ExpectedDigest $exports.MarkerDigest `
        -RemotePath "$RemoteIncoming/_TRANSFER_COMPLETE.json.tmp"

    if (
        -not (
            Test-RemoteVerifiedTree `
                -Root $RemoteIncoming `
                -MarkerName "_TRANSFER_COMPLETE.json.tmp" `
                -Exports $exports
        )
    ) {
        Stop-Finalizer "remote exact tree or hash verification failed"
    }
    $promoteCommand = "set -eu; " +
        "test ! -e '$RemoteFinal'; " +
        "mv '$RemoteIncoming/_TRANSFER_COMPLETE.json.tmp' '$RemoteIncoming/_TRANSFER_COMPLETE.json'; " +
        "sync; mv '$RemoteIncoming' '$RemoteFinal'; sync"
    Invoke-SshRequired `
        -Command $promoteCommand `
        -FailureMessage "remote atomic promotion failed" | Out-Null
    if (
        -not (
            Test-RemoteVerifiedTree `
                -Root $RemoteFinal `
                -MarkerName "_TRANSFER_COMPLETE.json" `
                -Exports $exports
        )
    ) {
        Stop-Finalizer "promoted remote tree failed full verification"
    }
    Write-FinalizerLog "REMOTE_VERIFY_OK destination=${Relay}:$RemoteFinal"
}

try {
    $pidParent = Split-Path -Parent $PidFile
    if ($pidParent -and -not (Test-Path -LiteralPath $pidParent)) {
        New-Item -ItemType Directory -Path $pidParent -Force | Out-Null
    }
    [System.IO.File]::WriteAllText(
        $PidFile,
        (([string]$PID) + [Environment]::NewLine),
        [System.Text.UTF8Encoding]::new($false)
    )
    if (-not (Test-Path -LiteralPath $ResultsRoot -PathType Container)) {
        Stop-Finalizer "results root missing: $ResultsRoot"
    }
    $completedQueue = Wait-ForCompletedQueue
    Wait-ForRunnerSuccess
    Write-FinalizerLog "QUEUE_OK completed=27/27 skipped=$($completedQueue.skipped_valid_tasks) runner_exit=0"
    Invoke-Postflight
    $exports = Read-VerifiedExportRecords
    Write-FinalizerLog "EXPORTS_OK files=$($exports.Records.Count) bytes=$($exports.SourceBytes)"
    if (
        $RelayBase -notmatch "^/[A-Za-z0-9._/-]+$" -or
        $RelayBase.Contains("//") -or
        $RemoteName -notmatch "^[A-Za-z0-9._-]+$" -or
        $DependencyName -notmatch "^[A-Za-z0-9._-]+$"
    ) {
        Stop-Finalizer "unsafe relay destination"
    }
    foreach ($remotePart in $RelayBase.Trim('/').Split('/')) {
        if (-not $remotePart -or $remotePart -in @(".", "..")) {
            Stop-Finalizer "unsafe relay destination"
        }
    }
    if ($Relay -notmatch "^[A-Za-z0-9._-]+@[A-Za-z0-9._:-]+$") {
        Stop-Finalizer "unsafe relay identity"
    }
    foreach ($requiredLocalPath in @($SshKey, $Ssh, $Scp)) {
        if (-not (Test-Path -LiteralPath $requiredLocalPath -PathType Leaf)) {
            Stop-Finalizer "required transfer executable or key missing: $requiredLocalPath"
        }
    }
    $RemoteIncoming = "$RelayBase/.${RemoteName}.incoming"
    $RemoteFinal = "$RelayBase/$RemoteName"
    $DependencyMarker = "$RelayBase/$DependencyName/_TRANSFER_COMPLETE.json"
    Wait-ForDependencyTransfer -DependencyMarker $DependencyMarker
    $relayState = Wait-ForRelayCapacity `
        -exports $exports `
        -RemoteIncoming $RemoteIncoming `
        -RemoteFinal $RemoteFinal
    if ($relayState.AlreadyVerified) {
        Write-FinalizerLog "REMOTE_ALREADY_VERIFIED destination=${Relay}:$RemoteFinal"
    }
    else {
        Transfer-VerifiedExports `
            -exports $exports `
            -RemoteIncoming $RemoteIncoming `
            -RemoteFinal $RemoteFinal
    }
    Write-FinalizerLog "FINALIZE_SUCCESS machine_stays_on"
    exit 0
}
catch {
    Write-FinalizerLog "FINALIZE_BLOCKED $($_.Exception.Message)"
    exit 1
}
finally {
    if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
        Remove-Item -LiteralPath $PidFile -Force
    }
}
