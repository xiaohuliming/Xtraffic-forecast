[CmdletBinding()]
param(
    [string]$Repo = "D:\t4\dgfix",
    [string]$ControlRoot = "D:\t4\v3_event_nas_control",
    [string]$Python = "C:\Python313\python.exe",
    [string]$PostflightScript = "D:\t4\v3_event_nas_control\protocol_v3_event_host_postflight.py",
    [string]$Plan = "D:\t4\v3_event_nas_control\event_baseline_5080_nas_backup_v1.json",
    [string]$CanonicalManifest = "D:\t4\dgfix\config\protocol_v3\table4_event_baselines_phase1_v1.json",
    [string]$Nas = "xhlm@192.168.31.233",
    [string]$NasBase = "/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/hosts",
    [string]$HostId = "event_rtx5080_alameda_contra",
    [string]$SshKey = "D:\t4\v3_event_nas_control\id_ed25519",
    [string]$Ssh = "C:\Windows\System32\OpenSSH\ssh.exe",
    [string]$Scp = "C:\Windows\System32\OpenSSH\scp.exe",
    [long]$ReserveBytes = 134217728,
    [int]$PollSeconds = 60,
    [string]$Log = "D:\t4\v3_event_nas_control\finalize.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Report = Join-Path $ControlRoot "postflight.json"
$FileList = Join-Path $ControlRoot "transfer.files"
$TransferManifest = Join-Path $ControlRoot "transfer.sha256"
$Marker = Join-Path $ControlRoot "_TRANSFER_COMPLETE.json"
$RunnerExitCode = "D:\t4\v3_event_dgfix_v1\control\runner.exitcode"
$QueueStatus = "D:\t4\v3_event_dgfix_v1\results\queue_status.json"
$RemoteIncoming = "$NasBase/.${HostId}.incoming"
$RemoteFinal = "$NasBase/$HostId"

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = "{0} {1}" -f [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"), $Message
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
    Write-Output $line
}

function Stop-Finalizer {
    param([Parameter(Mandatory = $true)][string]$Message)
    throw $Message
}

function Invoke-Ssh {
    param([Parameter(Mandatory = $true)][string]$Command)
    & $Ssh -n -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 `
        -o ServerAliveInterval=10 -o ServerAliveCountMax=6 $Nas $Command
    if ($LASTEXITCODE -ne 0) {
        Stop-Finalizer "SSH command failed with code $LASTEXITCODE"
    }
}

function Get-LocalSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-SafeFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [long]$Size = -1
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Stop-Finalizer "required file is missing: $Path"
    }
    $item = Get-Item -LiteralPath $Path -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        Stop-Finalizer "reparse point is forbidden: $Path"
    }
    if ($Size -ge 0 -and [long]$item.Length -ne $Size) {
        Stop-Finalizer "file size changed: $Path"
    }
    if ((Get-LocalSha256 -Path $Path) -ne $Sha256) {
        Stop-Finalizer "file hash changed: $Path"
    }
}

function Get-RemoteReusableBytes {
    param([Parameter(Mandatory = $true)][object[]]$Files)
    [long]$reusable = 0
    foreach ($record in $Files) {
        $relative = [string]$record.path
        $remotePath = "$RemoteIncoming/results/$relative"
        $sha256 = [string]$record.sha256
        $output = @(& $Ssh -n -i $SshKey -o IdentitiesOnly=yes `
            -o BatchMode=yes -o StrictHostKeyChecking=accept-new `
            -o ConnectTimeout=15 $Nas `
            "set -eu; if [ -f '$remotePath' ] && [ `"`$(sha256sum '$remotePath' | awk '{print `$1}')`" = '$sha256' ]; then echo MATCH; else echo MISSING; fi")
        if ($LASTEXITCODE -ne 0 -or $output.Count -eq 0) {
            Stop-Finalizer "cannot inspect resumable NAS path: $remotePath"
        }
        $state = ([string]$output[-1]).Trim()
        if ($state -eq "MATCH") {
            $reusable += [long]$record.size_bytes
        }
        elseif ($state -ne "MISSING") {
            Stop-Finalizer "unexpected NAS state for $remotePath`: $state"
        }
    }
    return $reusable
}

function Send-VerifiedFile {
    param(
        [Parameter(Mandatory = $true)][string]$LocalPath,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [long]$Size = -1
    )
    Assert-SafeFile -Path $LocalPath -Sha256 $Sha256 -Size $Size
    $slash = $RemotePath.LastIndexOf('/')
    if ($slash -le 0) {
        Stop-Finalizer "unsafe NAS path: $RemotePath"
    }
    $parent = $RemotePath.Substring(0, $slash)
    $part = "$RemotePath.part"
    $output = @(& $Ssh -n -i $SshKey -o IdentitiesOnly=yes `
        -o BatchMode=yes -o StrictHostKeyChecking=accept-new `
        -o ConnectTimeout=15 $Nas `
        "set -eu; mkdir -p '$parent'; if [ -f '$RemotePath' ] && [ `"`$(sha256sum '$RemotePath' | awk '{print `$1}')`" = '$Sha256' ]; then echo MATCH; else rm -f '$part' '$RemotePath'; echo UPLOAD; fi")
    if ($LASTEXITCODE -ne 0 -or $output.Count -eq 0) {
        Stop-Finalizer "cannot inspect NAS path: $RemotePath"
    }
    $state = ([string]$output[-1]).Trim()
    if ($state -eq "MATCH") {
        Write-Log "TRANSFER_SKIP path=$RemotePath"
        return
    }
    if ($state -ne "UPLOAD") {
        Stop-Finalizer "unexpected NAS state for $RemotePath`: $state"
    }
    & $Scp -O -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 `
        -o ServerAliveInterval=10 -o ServerAliveCountMax=6 `
        $LocalPath "${Nas}:$part"
    if ($LASTEXITCODE -ne 0) {
        Stop-Finalizer "SCP failed for $RemotePath with code $LASTEXITCODE"
    }
    Invoke-Ssh -Command "set -eu; test `"`$(sha256sum '$part' | awk '{print `$1}')`" = '$Sha256'; mv -f '$part' '$RemotePath'; sync"
    Write-Log "TRANSFER_FILE_OK path=$RemotePath"
}

function Test-RemoteTree {
    param([Parameter(Mandatory = $true)][string]$Root)
    $command = "set -eu; root='$Root'; test -d `"`$root/results`"; test -d `"`$root/metadata`"; " +
        "test -z `"`$(find `"`$root`" -type l -print -quit)`"; cd `"`$root/results`"; " +
        "sha256sum -c ../metadata/transfer.sha256 >/dev/null; " +
        "tmp=`$(mktemp -d); trap 'rm -rf `"`$tmp`"' EXIT; " +
        "cut -c67- ../metadata/transfer.sha256 | LC_ALL=C sort > `"`$tmp/expected`"; " +
        "find . -type f | sed 's#^./##' | LC_ALL=C sort > `"`$tmp/actual`"; cmp `"`$tmp/expected`" `"`$tmp/actual`"; " +
        "printf '%s\n' '_TRANSFER_COMPLETE.json' 'event_baseline_5080_nas_backup_v1.json' 'postflight.json' 'table4_event_baselines_phase1_v1.json' 'transfer.files' 'transfer.sha256' | LC_ALL=C sort > `"`$tmp/expected_meta`"; " +
        "find ../metadata -maxdepth 1 -type f | sed 's#^.*/##' | LC_ALL=C sort > `"`$tmp/actual_meta`"; cmp `"`$tmp/expected_meta`" `"`$tmp/actual_meta`""
    & $Ssh -n -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $Nas $command
    return ($LASTEXITCODE -eq 0)
}

try {
    New-Item -ItemType Directory -Path $ControlRoot -Force | Out-Null
    if (-not (Test-Path -LiteralPath $Log -PathType Leaf)) {
        [IO.File]::WriteAllText($Log, "", [Text.UTF8Encoding]::new($false))
    }
    foreach ($path in @($Python, $PostflightScript, $Plan, $CanonicalManifest, $SshKey, $Ssh, $Scp)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            Stop-Finalizer "required control file is missing: $path"
        }
    }
    while ($true) {
        if (-not (Test-Path -LiteralPath $QueueStatus -PathType Leaf)) {
            Write-Log "WAIT queue_status_missing"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $queue = Get-Content -LiteralPath $QueueStatus -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($queue.state -eq "failed") {
            Stop-Finalizer "completion queue entered failed state"
        }
        if ($queue.state -ne "completed") {
            Write-Log "WAIT queue_state=$($queue.state) completed=$($queue.completed_tasks)/13"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        if ([int]$queue.total_tasks -ne 13 -or [int]$queue.completed_tasks -ne 13 -or $null -ne $queue.current_task) {
            Stop-Finalizer "completed queue identity is not 13/13"
        }
        if (-not (Test-Path -LiteralPath $RunnerExitCode -PathType Leaf)) {
            Write-Log "WAIT runner.exitcode_missing"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $runnerCode = [int](Get-Content -LiteralPath $RunnerExitCode -Raw -Encoding UTF8).Trim()
        if ($runnerCode -ne 0) {
            Stop-Finalizer "runner exit code is $runnerCode"
        }
        $trainers = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match "^python" -and $_.CommandLine -match (
                "protocol_v3_baseline_runner|train_baseline_zoo|" +
                "train_graphwavenet"
            )
        })
        if ($trainers.Count -ne 0) {
            Write-Log "WAIT trainer_processes=$($trainers.Count)"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        break
    }

    $env:XTRAFFIC_REPO_ROOT = $Repo
    $env:PYTHONPATH = "$Repo;C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages"
    Write-Log "POSTFLIGHT_START"
    & $Python -X utf8 $PostflightScript --plan $Plan --report $Report `
        --file-list $FileList --transfer-manifest $TransferManifest `
        --marker $Marker
    if ($LASTEXITCODE -ne 0) {
        Stop-Finalizer "source-host postflight failed with code $LASTEXITCODE"
    }
    $postflight = Get-Content -LiteralPath $Report -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($postflight.status -ne "verified" -or $postflight.host_id -ne $HostId -or [int]$postflight.validated_tasks -ne 22) {
        Stop-Finalizer "postflight identity or task count mismatch"
    }
    Write-Log "POSTFLIGHT_OK tasks=22 files=$($postflight.file_count) bytes=$($postflight.total_bytes)"

    if (Test-RemoteTree -Root $RemoteFinal) {
        Write-Log "FINALIZE_SUCCESS destination=$RemoteFinal already_verified machine_stays_on"
        exit 0
    }
    $reusableBytes = Get-RemoteReusableBytes -Files @($postflight.files)
    $capacityOutput = @(& $Ssh -n -i $SshKey -o IdentitiesOnly=yes `
        -o BatchMode=yes -o StrictHostKeyChecking=accept-new `
        -o ConnectTimeout=15 $Nas `
        "set -eu; mkdir -p '$NasBase'; test ! -e '$RemoteFinal'; mkdir -p '$RemoteIncoming/results' '$RemoteIncoming/metadata'; df -Pk '$NasBase'")
    if ($LASTEXITCODE -ne 0 -or $capacityOutput.Count -eq 0) {
        Stop-Finalizer "NAS capacity query failed"
    }
    $parts = @((([string]$capacityOutput[-1]).Trim()) -split "\s+")
    if ($parts.Count -lt 4 -or $parts[3] -notmatch "^[0-9]+$") {
        Stop-Finalizer "NAS capacity response is invalid"
    }
    [long]$availableBytes = [long]$parts[3] * 1024
    [long]$requiredBytes = [long]$postflight.total_bytes + $ReserveBytes
    if (($availableBytes + $reusableBytes) -lt $requiredBytes) {
        Stop-Finalizer "NAS capacity gate failed"
    }
    Write-Log "TRANSFER_START files=$($postflight.file_count) bytes=$($postflight.total_bytes) reusable_bytes=$reusableBytes"
    foreach ($record in @($postflight.files)) {
        $relative = [string]$record.path
        if ($relative.Contains("\") -or $relative.Contains("..") -or $relative.StartsWith("/")) {
            Stop-Finalizer "unsafe export path: $relative"
        }
        Send-VerifiedFile -LocalPath ([string]$record.source_path) `
            -Sha256 ([string]$record.sha256) `
            -RemotePath "$RemoteIncoming/results/$relative" `
            -Size ([long]$record.size_bytes)
    }
    $metadata = @(
        [PSCustomObject]@{ Local = $Report; Name = "postflight.json" },
        [PSCustomObject]@{ Local = $FileList; Name = "transfer.files" },
        [PSCustomObject]@{ Local = $TransferManifest; Name = "transfer.sha256" },
        [PSCustomObject]@{ Local = $Marker; Name = "_TRANSFER_COMPLETE.json" },
        [PSCustomObject]@{ Local = $Plan; Name = "event_baseline_5080_nas_backup_v1.json" },
        [PSCustomObject]@{ Local = $CanonicalManifest; Name = "table4_event_baselines_phase1_v1.json" }
    )
    foreach ($entry in $metadata) {
        $digest = Get-LocalSha256 -Path ([string]$entry.Local)
        Send-VerifiedFile -LocalPath ([string]$entry.Local) -Sha256 $digest `
            -RemotePath "$RemoteIncoming/metadata/$([string]$entry.Name)"
    }
    if (-not (Test-RemoteTree -Root $RemoteIncoming)) {
        Stop-Finalizer "incoming NAS tree failed exact verification"
    }
    Invoke-Ssh -Command "set -eu; test ! -e '$RemoteFinal'; mv '$RemoteIncoming' '$RemoteFinal'; sync"
    if (-not (Test-RemoteTree -Root $RemoteFinal)) {
        Stop-Finalizer "promoted NAS tree failed exact verification"
    }
    Write-Log "FINALIZE_SUCCESS destination=$RemoteFinal machine_stays_on"
    exit 0
}
catch {
    Write-Log "FINALIZE_BLOCKED $($_.Exception.Message)"
    Write-Log "FINALIZE_STACK $($_.ScriptStackTrace)"
    exit 1
}
