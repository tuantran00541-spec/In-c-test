param(
    [string]$ModelDir = ".\packed\kimi-vl-a3b-q8",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Binary = ".\build\Release\kvl_generate.exe",
    [int]$RamMiB = 4096,
    [int]$MaxNew = 64,
    [int]$ThreadCount = 1,
    [string]$CacheList = "512,768,1024,1280,1536,2048",
    [string]$Prompt = "Xin chào. Hãy trả lời đúng một câu ngắn."
)

$ErrorActionPreference = 'Stop'
foreach ($path in @($Python, $Binary)) {
    if (-not (Test-Path $path)) { throw "Missing required path: $path" }
}
if (-not (Test-Path $ModelDir)) { throw "Missing model directory: $ModelDir" }
if ($ThreadCount -lt 1) { throw "-ThreadCount must be positive" }

$CacheValues = @()
foreach ($part in $CacheList.Split(',')) {
    $value = 0
    if (-not [int]::TryParse($part.Trim(), [ref]$value) -or $value -lt 1) {
        throw "Invalid -CacheList entry: '$part'"
    }
    $CacheValues += $value
}
if ($CacheValues.Count -eq 0) { throw "-CacheList must contain at least one positive integer" }

$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NESTED = 'FALSE'
$env:OMP_NUM_THREADS = [string]$ThreadCount

$timingRx = [regex]'\[kvl\] timing first_token=([0-9.]+)s avg_next=([0-9.]+)s total=([0-9.]+)s generated=([0-9]+)'
$idsRx = [regex]'\[kvl\] generated ids:\s*(.*)$'
$planRx = [regex]'\[kvl\] prompt tokens=([0-9]+).*?cache=([0-9]+) MiB trunk_cache=([0-9]+) MiB.*?RAM plan=([0-9.]+)/([0-9]+) MiB'
$cacheRx = [regex]'kvl_cache: .*?reads=([0-9]+) bytes=([0-9.]+) MiB .*?failures=([0-9]+)'
$trunkRx = [regex]'kvl_trunk_cache: .*?reads=([0-9.]+) MiB'

function Quote-ProcessArg([string]$Value) {
    if ($null -eq $Value) { return '""' }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-CacheCase([int]$CacheMiB) {
    Write-Host ""
    Write-Host "=== cache=$CacheMiB MiB thread=$ThreadCount ==="

    $argsList = @(
        '.\tools\kvl_chat.py',
        $ModelDir,
        $Prompt,
        '--binary', $Binary,
        '--cache-mib', [string]$CacheMiB,
        '--ram-mib', [string]$RamMiB,
        '--max-new', [string]$MaxNew,
        '--temperature', '0',
        '--seed', '1',
        '--show-tokens'
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Resolve-Path $Python).Path
    $psi.WorkingDirectory = (Get-Location).Path
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.Arguments = (($argsList | ForEach-Object { Quote-ProcessArg ([string]$_) }) -join ' ')

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { throw "Failed to start $Python" }
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    $exitCode = $proc.ExitCode
    $proc.Dispose()

    $lines = @(($stdout + "`n" + $stderr) -split "`r?`n")
    if ($exitCode -ne 0) {
        $lines | ForEach-Object { if ($_ -ne '') { Write-Host $_ } }
        throw "cache=$CacheMiB failed with exit code $exitCode"
    }

    $timing = $null; $ids = $null; $plan = $null; $cache = $null; $trunk = $null
    foreach ($line in $lines) {
        $m = $timingRx.Match($line); if ($m.Success) { $timing = $m }
        $m = $idsRx.Match($line); if ($m.Success) { $ids = $m.Groups[1].Value.Trim() }
        $m = $planRx.Match($line); if ($m.Success) { $plan = $m }
        $m = $cacheRx.Match($line); if ($m.Success) { $cache = $m }
        $m = $trunkRx.Match($line); if ($m.Success) { $trunk = $m }
    }
    if (-not $timing -or $null -eq $ids -or -not $plan -or -not $cache -or -not $trunk) {
        $lines | ForEach-Object { if ($_ -ne '') { Write-Host $_ } }
        throw "Missing one or more metrics for cache=$CacheMiB"
    }

    $expertMiB = [double]$cache.Groups[2].Value
    $trunkReadMiB = [double]$trunk.Groups[1].Value
    [pscustomobject]@{
        cache_mib = $CacheMiB
        trunk_cache_mib = [int]$plan.Groups[3].Value
        ram_plan_mib = [double]$plan.Groups[4].Value
        first_token_s = [double]$timing.Groups[1].Value
        avg_next_s = [double]$timing.Groups[2].Value
        total_s = [double]$timing.Groups[3].Value
        generated = [int]$timing.Groups[4].Value
        expert_reads = [uint64]$cache.Groups[1].Value
        expert_mib = $expertMiB
        trunk_read_mib = $trunkReadMiB
        measured_io_mib = $expertMiB + $trunkReadMiB
        failures = [uint64]$cache.Groups[3].Value
        ids = $ids
    }
}

$rows = @()
foreach ($cacheMiB in $CacheValues) { $rows += Invoke-CacheCase $cacheMiB }

$reference = $rows[0].ids
$bad = @($rows | Where-Object { $_.ids -ne $reference -or $_.failures -ne 0 })

Write-Host ""
Write-Host "=== HARD-CAP CACHE/TRUNK TRADEOFF ==="
$rows | Select-Object cache_mib,trunk_cache_mib,first_token_s,avg_next_s,expert_reads,expert_mib,trunk_read_mib,measured_io_mib,generated | Format-Table -AutoSize

$bestDecode = $rows | Sort-Object avg_next_s | Select-Object -First 1
$bestIo = $rows | Sort-Object measured_io_mib | Select-Object -First 1
Write-Host ""
Write-Host ("fastest_decode cache={0} trunk={1} avg_next={2:N3}s" -f $bestDecode.cache_mib,$bestDecode.trunk_cache_mib,$bestDecode.avg_next_s)
Write-Host ("lowest_measured_io cache={0} trunk={1} io={2:N1}MiB" -f $bestIo.cache_mib,$bestIo.trunk_cache_mib,$bestIo.measured_io_mib)

if ($bad.Count -gt 0) {
    Write-Host "FAIL: token IDs changed or I/O failures occurred in $($bad.Count) case(s)."
    exit 3
}
Write-Host "TOKEN_IDS_EXACT_PASS"
Write-Host "The sweep preserves the same total RAM cap; trunk_cache=auto absorbs each expert-cache change."
