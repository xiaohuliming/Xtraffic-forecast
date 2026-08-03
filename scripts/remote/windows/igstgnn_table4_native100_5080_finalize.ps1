[CmdletBinding()]
param(
    [string]$Repo = "D:\t4\native100_repo",
    [string]$ResultsRoot = "D:\t4\native100_compare\results",
    [string]$ControlRoot = "D:\t4\native100_control\finalize_5080",
    [string]$Python = "C:\Python313\python.exe",
    [string]$PostflightScript = "D:\t4\native100_control\native100_host_postflight.py",
    [string]$Plan = "D:\t4\native100_control\distributed_native100_5080_5090_v1.json",
    [string]$Manifest = "D:\t4\native100_repo\config\reproduction\igstgnn_table4\matrix.native100_compare_v1.json",
    [string]$BaseManifest = "D:\t4\native100_repo\config\reproduction\igstgnn_table4\matrix.public_material_v2.json",
    [string]$Relay = "root@8.134.12.5",
    [string]$RelayBase = "/root/xtraffic_results/igstgnn_table4_official_v1/native100_5080_5090_v1/hosts",
    [string]$HostId = "native100_rtx5080_alameda_contra",
    [string]$SshKey = "C:\Users\ASUS\.ssh\xtraffic_relay_ed25519",
    [string]$Ssh = "C:\Windows\System32\OpenSSH\ssh.exe",
    [string]$Scp = "C:\Windows\System32\OpenSSH\scp.exe",
    [long]$ReserveBytes = 134217728,
    [string]$Log = "D:\t4\native100_control\finalize_5080\finalize.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$MatrixSha256 = "e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246"
$BaseSha256 = "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
$PlanSha256 = "d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e"
$Report = Join-Path $ControlRoot "postflight.json"
$FileList = Join-Path $ControlRoot "transfer.files"
$TransferManifest = Join-Path $ControlRoot "transfer.sha256"
$Marker = Join-Path $ControlRoot "_TRANSFER_COMPLETE.json"
$RemoteIncoming = "$RelayBase/.${HostId}.incoming"
$RemoteFinal = "$RelayBase/$HostId"

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
        -o ServerAliveInterval=10 -o ServerAliveCountMax=6 $Relay $Command
    if ($LASTEXITCODE -ne 0) {
        Stop-Finalizer "SSH command failed with code $LASTEXITCODE"
    }
}

function Get-LocalSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RemoteReusableBytes {
    param([Parameter(Mandatory = $true)][object[]]$Files)
    [long]$reusable = 0
    foreach ($record in $Files) {
        $relative = [string]$record.path
        $remotePath = "$RemoteIncoming/results/$relative"
        $sha256 = [string]$record.sha256
        $stateOutput = @(& $Ssh -n -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
            -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $Relay `
            "set -eu; if [ -f '$remotePath' ] && [ `"`$(sha256sum '$remotePath' | awk '{print `$1}')`" = '$sha256' ]; then echo MATCH; else echo MISSING; fi")
        if ($LASTEXITCODE -ne 0 -or $stateOutput.Count -eq 0) {
            Stop-Finalizer "cannot inspect resumable remote path: $remotePath"
        }
        $state = ([string]$stateOutput[-1]).Trim()
        if ($state -eq "MATCH") {
            $reusable += [long]$record.size_bytes
        }
        elseif ($state -ne "MISSING") {
            Stop-Finalizer "unexpected resumable state for $remotePath`: $state"
        }
    }
    return $reusable
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
        Stop-Finalizer "unsafe remote path: $RemotePath"
    }
    $remoteParent = $RemotePath.Substring(0, $slash)
    $remotePart = "$RemotePath.part"
    $stateOutput = @(& $Ssh -n -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $Relay `
        "set -eu; mkdir -p '$remoteParent'; if [ -f '$RemotePath' ] && [ `"`$(sha256sum '$RemotePath' | awk '{print `$1}')`" = '$Sha256' ]; then echo MATCH; else rm -f '$remotePart' '$RemotePath'; echo UPLOAD; fi")
    if ($LASTEXITCODE -ne 0) {
        Stop-Finalizer "cannot inspect remote path: $RemotePath"
    }
    if ($stateOutput.Count -eq 0) {
        Stop-Finalizer "remote path inspection returned no state: $RemotePath"
    }
    $state = ([string]$stateOutput[-1]).Trim()
    if ($state -eq "MATCH") {
        Write-Log "TRANSFER_SKIP path=$RemotePath"
        return
    }
    if ($state -ne "UPLOAD") {
        Stop-Finalizer "unexpected remote state for $RemotePath`: $state"
    }
    & $Scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 `
        -o ServerAliveInterval=10 -o ServerAliveCountMax=6 `
        $LocalPath "${Relay}:$remotePart"
    if ($LASTEXITCODE -ne 0) {
        Stop-Finalizer "SCP failed for $RemotePath with code $LASTEXITCODE"
    }
    Invoke-Ssh -Command "set -eu; test `"`$(sha256sum '$remotePart' | awk '{print `$1}')`" = '$Sha256'; mv -f '$remotePart' '$RemotePath'; sync"
    Write-Log "TRANSFER_FILE_OK path=$RemotePath"
}

function Test-RemoteTree {
    param([Parameter(Mandatory = $true)][string]$Root)
    $command = "set -eu; root='$Root'; test -d `"`$root/results`"; test -d `"`$root/metadata`"; " +
        "test -z `"`$(find `"`$root`" -type l -print -quit)`"; cd `"`$root/results`"; " +
        "sha256sum -c ../metadata/transfer.sha256 >/dev/null; " +
        "tmp=`$(mktemp -d); trap 'rm -rf `"`$tmp`"' EXIT; " +
        "cut -c67- ../metadata/transfer.sha256 | LC_ALL=C sort > `"`$tmp/expected`"; " +
        "find . -type f -printf '%P\n' | LC_ALL=C sort > `"`$tmp/actual`"; cmp `"`$tmp/expected`" `"`$tmp/actual`"; " +
        "printf '%s\n' '_TRANSFER_COMPLETE.json' 'distributed_native100_5080_5090_v1.json' 'matrix.native100_compare_v1.json' 'matrix.public_material_v2.json' 'postflight.json' 'transfer.files' 'transfer.sha256' | LC_ALL=C sort > `"`$tmp/expected_meta`"; " +
        "find ../metadata -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort > `"`$tmp/actual_meta`"; cmp `"`$tmp/expected_meta`" `"`$tmp/actual_meta`"; " +
        "test `"`$(sha256sum ../metadata/distributed_native100_5080_5090_v1.json | awk '{print `$1}')`" = '$PlanSha256'; " +
        "test `"`$(sha256sum ../metadata/matrix.native100_compare_v1.json | awk '{print `$1}')`" = '$MatrixSha256'; " +
        "test `"`$(sha256sum ../metadata/matrix.public_material_v2.json | awk '{print `$1}')`" = '$BaseSha256'"
    & $Ssh -n -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $Relay $command
    return ($LASTEXITCODE -eq 0)
}

try {
    New-Item -ItemType Directory -Path $ControlRoot -Force | Out-Null
    [IO.File]::WriteAllText($Log, "", [Text.UTF8Encoding]::new($false))
    foreach ($path in @($Python, $PostflightScript, $Plan, $Manifest, $BaseManifest, $SshKey, $Ssh, $Scp)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            Stop-Finalizer "required control file is missing: $path"
        }
    }
    $queue = Get-Content -LiteralPath (Join-Path $ResultsRoot "queue_status.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($queue.state -ne "completed" -or [int]$queue.verified_task_count -ne 12 -or [int]$queue.failed_count -ne 0) {
        Stop-Finalizer "5080 queue is not completed 12/12"
    }
    $trainers = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "^python" -and $_.CommandLine -match "native100_compare_(matrix_)?runner"
    })
    if ($trainers.Count -ne 0) {
        Stop-Finalizer "native100 training process is still running"
    }
    $env:PYTHONPATH = "$Repo;C:\Users\ASUS\AppData\Roaming\Python\Python313\site-packages"
    Write-Log "POSTFLIGHT_START"
    & $Python -X utf8 $PostflightScript `
        --plan $Plan --plan-sha256 $PlanSha256 `
        --manifest $Manifest --manifest-sha256 $MatrixSha256 `
        --base-manifest $BaseManifest --base-manifest-sha256 $BaseSha256 `
        --host-id $HostId --results-root $ResultsRoot `
        --report $Report --file-list $FileList `
        --transfer-manifest $TransferManifest --marker $Marker
    if ($LASTEXITCODE -ne 0) {
        Stop-Finalizer "source-host postflight failed with code $LASTEXITCODE"
    }
    $postflight = Get-Content -LiteralPath $Report -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($postflight.status -ne "verified" -or [int]$postflight.validated_tasks -ne 12 -or [int]$postflight.file_count -ne 61) {
        Stop-Finalizer "postflight identity or count mismatch"
    }
    Write-Log "POSTFLIGHT_OK tasks=12 files=61 bytes=$($postflight.total_bytes)"

    if (Test-RemoteTree -Root $RemoteFinal) {
        Write-Log "REMOTE_ALREADY_VERIFIED destination=$RemoteFinal"
        exit 0
    }
    $required = [long]$postflight.total_bytes + $ReserveBytes
    $reusableBytes = Get-RemoteReusableBytes -Files @($postflight.files)
    $capacityOutput = @(& $Ssh -n -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
        -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 $Relay `
        "set -eu; mkdir -p '$RelayBase'; test ! -e '$RemoteFinal'; mkdir -p '$RemoteIncoming/results' '$RemoteIncoming/metadata'; df -PB1 '$RelayBase'")
    if ($LASTEXITCODE -ne 0 -or $capacityOutput.Count -eq 0) {
        Stop-Finalizer "relay capacity query failed rc=$LASTEXITCODE output=$($capacityOutput -join '|')"
    }
    $capacityParts = @((([string]$capacityOutput[-1]).Trim()) -split "\s+")
    if ($capacityParts.Count -lt 4) {
        Stop-Finalizer "relay capacity response is invalid: $($capacityOutput -join '|')"
    }
    $capacity = $capacityParts[3]
    if ($capacity -notmatch "^[0-9]+$" -or ([long]$capacity + $reusableBytes) -lt $required) {
        Stop-Finalizer "relay capacity or destination gate failed"
    }
    Write-Log "TRANSFER_START files=61 bytes=$($postflight.total_bytes) reusable_bytes=$reusableBytes"
    foreach ($record in @($postflight.files)) {
        $relative = [string]$record.path
        if ($relative.Contains("\") -or $relative.Contains("..") -or $relative.StartsWith("/")) {
            Stop-Finalizer "unsafe export path: $relative"
        }
        $source = Join-Path $ResultsRoot ($relative -replace "/", "\")
        Send-VerifiedFile -LocalPath $source -Sha256 ([string]$record.sha256) `
            -RemotePath "$RemoteIncoming/results/$relative" -Size ([long]$record.size_bytes)
    }
    $metadata = @(
        [PSCustomObject]@{ Local = $Report; Name = "postflight.json" },
        [PSCustomObject]@{ Local = $FileList; Name = "transfer.files" },
        [PSCustomObject]@{ Local = $TransferManifest; Name = "transfer.sha256" },
        [PSCustomObject]@{ Local = $Marker; Name = "_TRANSFER_COMPLETE.json" },
        [PSCustomObject]@{ Local = $Plan; Name = "distributed_native100_5080_5090_v1.json" },
        [PSCustomObject]@{ Local = $Manifest; Name = "matrix.native100_compare_v1.json" },
        [PSCustomObject]@{ Local = $BaseManifest; Name = "matrix.public_material_v2.json" }
    )
    foreach ($entry in $metadata) {
        $digest = Get-LocalSha256 -Path ([string]$entry.Local)
        Send-VerifiedFile -LocalPath ([string]$entry.Local) -Sha256 $digest `
            -RemotePath "$RemoteIncoming/metadata/$([string]$entry.Name)"
    }
    if (-not (Test-RemoteTree -Root $RemoteIncoming)) {
        Stop-Finalizer "incoming relay tree failed exact verification"
    }
    Invoke-Ssh -Command "set -eu; test ! -e '$RemoteFinal'; mv '$RemoteIncoming' '$RemoteFinal'; sync"
    if (-not (Test-RemoteTree -Root $RemoteFinal)) {
        Stop-Finalizer "promoted relay tree failed exact verification"
    }
    Write-Log "FINALIZE_SUCCESS destination=$RemoteFinal machine_stays_on"
    exit 0
}
catch {
    Write-Log "FINALIZE_BLOCKED $($_.Exception.Message)"
    Write-Log "FINALIZE_STACK $($_.ScriptStackTrace)"
    exit 1
}
