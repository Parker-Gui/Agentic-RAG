param(
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot '.local'
$binaryRoot = Join-Path $runtimeRoot 'postgres-env\Library\bin'
$clusterPath = Join-Path $runtimeRoot 'postgres-data'
$pidFile = Join-Path $clusterPath 'postmaster.pid'

if (-not (Test-Path -LiteralPath (Join-Path $binaryRoot 'postgres.exe'))) {
    throw 'Local PostgreSQL runtime is missing. See docs/zh/crawler-local-validation.md.'
}

function Test-LocalCluster {
    if (-not (Test-Path -LiteralPath $pidFile)) { return $false }
    $serverPid = [int](Get-Content -LiteralPath $pidFile -TotalCount 1)
    $serverProcess = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
    return ($null -ne $serverProcess -and $serverProcess.ProcessName -eq 'postgres')
}

if ($Action -eq 'stop') {
    if (Test-LocalCluster) {
        & (Join-Path $binaryRoot 'pg_ctl.exe') -D $clusterPath -m fast -w stop
        if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL shutdown failed.' }
    }
    Write-Output 'Local PostgreSQL stopped.'
    exit 0
}

if ($Action -eq 'start' -and -not (Test-LocalCluster)) {
    if (-not (Test-Path -LiteralPath (Join-Path $clusterPath 'PG_VERSION'))) {
        throw 'The local database cluster has not been initialized.'
    }
    $arguments = '-D "{0}" -h 127.0.0.1 -p 5432' -f $clusterPath
    Start-Process -FilePath (Join-Path $binaryRoot 'postgres.exe') -ArgumentList $arguments `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtimeRoot 'postgres.stdout.log') `
        -RedirectStandardError (Join-Path $runtimeRoot 'postgres.stderr.log') | Out-Null
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        & (Join-Path $binaryRoot 'pg_isready.exe') -h 127.0.0.1 -p 5432 -q
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Milliseconds 250
    }
}

& (Join-Path $binaryRoot 'pg_ctl.exe') -D $clusterPath status
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $binaryRoot 'pg_isready.exe') -h 127.0.0.1 -p 5432
exit $LASTEXITCODE
