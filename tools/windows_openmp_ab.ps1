param(
    [string]$ModelDir = ".\packed\kimi-vl-a3b-q8",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$BaselineBinary = ".\build\Release\kvl_generate.exe",
    [string]$PilotBinary = ".\build-omp\Release\kvl_generate.exe",
    [int]$CacheMiB = 512,
    [int]$RamMiB = 4096,
    [int]$MaxNew = 64,
    [int[]]$Threads = @(1,2,4,6),
    [string]$Prompt = "Xin chào. Hãy trả lời đúng một câu ngắn."
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

foreach ($path in @($Python, $BaselineBinary, $PilotBinary)) {
    if (-not (Test-Path $path)) { throw "Missing required path: $path" }
}
if (-not (Test-Path $ModelDir)) { throw "Missing model directory: $ModelDir" }

$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NESTED = 'FALSE'

$timingRx = [regex]'\[kvl\] timing first_token=([0-9.]+)s avg_next=([0-9.]+)s total=([0-9.]+)s generated=([0-9]+)'
$idsRx = [regex]'\[kvl\] generated ids:\s*(.*)$'

function Invoke-KvlCase {
    param(
        [string]$Label,
        [string]$Binary,
        [int]$ThreadCount
    )

    $env:OMP_NUM_THREADS = [string]$ThreadCount
    Write-Host ""
    Write-Host "=== $Label threads=$ThreadCount ==="

    $lines = & $Python .\tools\kvl_chat.py `
        $ModelDir `
        $Prompt `
        --binary $Binary `
        --cache-mib $CacheMiB `
        --ram-mib $RamMiB `
        --max-new $MaxNew `
        --temperature 0 `
        --seed 1 `
        --show-tokens 2>&1 | ForEach-Object { "$_" }

    if ($LASTEXITCODE -ne 0) {
        $lines | ForEach-Object { Write-Host $_ }
        throw "$Label threads=$ThreadCount failed with exit code $LASTEXITCODE"
    }

    $timing = $null
    $ids = $null
    foreach ($line in $lines) {
        $m = $timingRx.Match($line)
        if ($m.Success) { $timing = $m }
        $m2 = $idsRx.Match($line)
        if ($m2.Success) { $ids = $m2.Groups[1].Value.Trim() }
    }
    if (-not $timing) {
        $lines | ForEach-Object { Write-Host $_ }
        throw "No timing line found for $Label threads=$ThreadCount"
    }
    if ($null -eq $ids) {
        $lines | ForEach-Object { Write-Host $_ }
        throw "No generated-id line found for $Label threads=$ThreadCount"
    }

    [pscustomobject]@{
        label = $Label
        threads = $ThreadCount
        first_token_s = [double]$timing.Groups[1].Value
        avg_next_s = [double]$timing.Groups[2].Value
        total_s = [double]$timing.Groups[3].Value
        generated = [int]$timing.Groups[4].Value
        ids = $ids
    }
}

$rows = @()
foreach ($t in $Threads) {
    $rows += Invoke-KvlCase -Label 'baseline' -Binary $BaselineBinary -ThreadCount $t
    $rows += Invoke-KvlCase -Label 'openmp' -Binary $PilotBinary -ThreadCount $t
}

$reference = $rows[0].ids
$bad = @($rows | Where-Object { $_.ids -ne $reference })

Write-Host ""
Write-Host "=== CONTROLLED A/B SUMMARY ==="
$rows | Select-Object label,threads,first_token_s,avg_next_s,total_s,generated | Format-Table -AutoSize

Write-Host ""
Write-Host "=== OPENMP VS BASELINE AT SAME THREAD COUNT ==="
foreach ($t in $Threads) {
    $b = $rows | Where-Object { $_.label -eq 'baseline' -and $_.threads -eq $t } | Select-Object -First 1
    $o = $rows | Where-Object { $_.label -eq 'openmp' -and $_.threads -eq $t } | Select-Object -First 1
    $ttftRatio = if ($o.first_token_s -gt 0) { $b.first_token_s / $o.first_token_s } else { 0 }
    $decodeRatio = if ($o.avg_next_s -gt 0) { $b.avg_next_s / $o.avg_next_s } else { 0 }
    Write-Host ("threads={0} ttft={1:N3}x decode={2:N3}x baseline_next={3:N3}s openmp_next={4:N3}s" -f `
        $t,$ttftRatio,$decodeRatio,$b.avg_next_s,$o.avg_next_s)
}

if ($bad.Count -gt 0) {
    Write-Host ""
    Write-Host "FAIL: generated token IDs changed in $($bad.Count) case(s)."
    foreach ($row in $bad) { Write-Host "  $($row.label) threads=$($row.threads)" }
    exit 3
}

Write-Host ""
Write-Host "TOKEN_IDS_EXACT_PASS"
Write-Host "Use the fastest stable row only as a Dell-specific candidate; do not infer speed from hosted CI."
