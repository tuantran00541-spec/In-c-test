param(
    [string]$ModelDir = ".\packed\kimi-vl-a3b-q8",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$BaselineBinary = ".\build-perf-research\core\Release\kvl_generate.exe",
    [string]$CandidateBinary = ".\build-perf-research\Release\kvl_generate_combined_exact.exe",
    [int]$CacheMiB = 512,
    [int]$RamMiB = 4096,
    [string]$ThreadList = "2,4,6",
    [switch]$SkipLong
)

$ErrorActionPreference = 'Stop'

foreach ($path in @($Python, $BaselineBinary, $CandidateBinary)) {
    if (-not (Test-Path $path)) { throw "Missing required path: $path" }
}
if (-not (Test-Path $ModelDir)) { throw "Missing model directory: $ModelDir" }

$Threads = @()
foreach ($part in $ThreadList.Split(',')) {
    $value = 0
    if (-not [int]::TryParse($part.Trim(), [ref]$value) -or $value -lt 1) {
        throw "Invalid -ThreadList entry: '$part'"
    }
    $Threads += $value
}
if ($Threads.Count -eq 0) { throw "-ThreadList must contain at least one positive integer" }

$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NESTED = 'FALSE'

# Keep this .ps1 strictly ASCII so Windows PowerShell 5.1 does not depend on a
# UTF-8 BOM. Decode the exact Vietnamese prompts from UTF-8 Base64 at runtime.
$ShortPromptB64 = 'WGluIGNow6BvLiBIw6N5IHRy4bqjIGzhu51pIMSRw7puZyBt4buZdCBjw6J1IG5n4bqvbi4='
$LongPromptB64 = 'SMOjeSB2aeG6v3QgbeG7mXQgYsOgaSBraG/huqNuZyA3MDDigJMxMDAwIHThu6sgYuG6sW5nIHRp4bq/bmcgVmnhu4d0IHbhu4EgY2jhu6cgxJHhu4E6IE3hu5l0IGNoaeG6v2MgbGFwdG9wIHBo4buVIHRow7RuZyBjw7MgdGjhu4MgY2jhuqF5IG3DtCBow6xuaCBBSSBs4bubbiBob8OgbiB0b8OgbiBi4bqxbmcgQ1BVIG5oxrAgdGjhur8gbsOgbz8gR2nhuqNpIHRow61jaCB0aGVvIGPDoWNoIGThu4UgaGnhu4N1IG5oxrBuZyB24bqrbiBjw7MgY2hpIHRp4bq/dCBr4bu5IHRodeG6rXQuIELDoGkgdmnhur90IGPhuqduIGPDsyBt4bufIMSR4bqndSwgcGjhuqduIGdp4bqjaSB0aMOtY2ggduG7gSBSQU0sIFNTRC9OVk1lLCBxdWFudGl6YXRpb24sIE1vRSwgY2FjaGUsIHThu5FjIMSR4buZIHRva2VuLCBnaeG7m2kgaOG6oW4gdGjhu7FjIHThur8sIHbDoCBwaOG6p24ga+G6v3QgbHXhuq1uLiBLaMO0bmcgZMO5bmcgZGFuaCBzw6FjaCBn4bqhY2ggxJHhuqd1IGTDsm5nIHF1w6Egbmhp4buBdTsgxrB1IHRpw6puIHZp4bq/dCB0aMOgbmggY8OhYyDEkW/huqFuIHbEg24gbGnhu4FuIG3huqFjaC4gSMOjeSBkdXkgdHLDrCBt4bqhY2ggdsSDbiB04buxIG5oacOqbiwga2jDtG5nIGzhurdwIMO9LCBraMO0bmcgZOG7q25nIGdp4buvYSBjaOG7q25nLCB2w6AgdGnhur9wIHThu6VjIGNobyDEkeG6v24ga2hpIGhvw6BuIHRow6BuaCB0b8OgbiBi4buZIGLDoGkgdmnhur90Lg=='
$ShortPrompt = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($ShortPromptB64))
$LongPrompt = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($LongPromptB64))

$timingRx = [regex]'\[kvl\] timing first_token=([0-9.]+)s avg_next=([0-9.]+)s total=([0-9.]+)s generated=([0-9]+)'
$idsRx = [regex]'\[kvl\] generated ids:\s*(.*)$'

function Quote-ProcessArg([string]$Value) {
    if ($null -eq $Value) { return '""' }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-CapturedKvl {
    param(
        [string]$Label,
        [string]$Binary,
        [int]$ThreadCount
    )

    $env:OMP_NUM_THREADS = [string]$ThreadCount
    Write-Host ""
    Write-Host "=== SHORT EXACT GATE: $Label threads=$ThreadCount ==="

    $argsList = @(
        '.\tools\kvl_chat.py',
        $ModelDir,
        $ShortPrompt,
        '--binary', $Binary,
        '--cache-mib', [string]$CacheMiB,
        '--ram-mib', [string]$RamMiB,
        '--max-new', '64',
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

    $combined = $stdout + "`n" + $stderr
    $lines = @($combined -split "`r?`n")
    if ($exitCode -ne 0) {
        $lines | ForEach-Object { if ($_ -ne '') { Write-Host $_ } }
        throw "$Label threads=$ThreadCount failed with exit code $exitCode"
    }

    $timing = $null
    $ids = $null
    foreach ($line in $lines) {
        $m = $timingRx.Match($line)
        if ($m.Success) { $timing = $m }
        $m2 = $idsRx.Match($line)
        if ($m2.Success) { $ids = $m2.Groups[1].Value.Trim() }
    }
    if (-not $timing -or $null -eq $ids) {
        $lines | ForEach-Object { if ($_ -ne '') { Write-Host $_ } }
        throw "Could not parse timing/token IDs for $Label threads=$ThreadCount"
    }

    $row = [pscustomobject]@{
        label = $Label
        threads = $ThreadCount
        first_token_s = [double]$timing.Groups[1].Value
        avg_next_s = [double]$timing.Groups[2].Value
        total_s = [double]$timing.Groups[3].Value
        generated = [int]$timing.Groups[4].Value
        ids = $ids
    }
    Write-Host ("first={0:N3}s next={1:N3}s total={2:N3}s generated={3}" -f `
        $row.first_token_s,$row.avg_next_s,$row.total_s,$row.generated)
    return $row
}

$baselineThread = $Threads[0]
$baseline = Invoke-CapturedKvl -Label 'baseline' -Binary $BaselineBinary -ThreadCount $baselineThread
$candidates = @()
foreach ($t in $Threads) {
    $candidates += Invoke-CapturedKvl -Label 'combined-exact' -Binary $CandidateBinary -ThreadCount $t
}

$bad = @($candidates | Where-Object { $_.ids -ne $baseline.ids })
Write-Host ""
Write-Host "=== SHORT EXACT-GATE SUMMARY ==="
@($baseline) + $candidates | Select-Object label,threads,first_token_s,avg_next_s,total_s,generated | Format-Table -AutoSize

if ($bad.Count -gt 0) {
    Write-Host ""
    Write-Host "FAIL: combined candidate changed generated token IDs in $($bad.Count) thread setting(s)."
    foreach ($row in $bad) { Write-Host "  threads=$($row.threads)" }
    Write-Host "Long prompt was NOT started."
    exit 3
}

Write-Host ""
Write-Host "COMBINED_SHORT_TOKEN_IDS_EXACT_PASS"
$winner = $candidates | Sort-Object total_s,avg_next_s,first_token_s | Select-Object -First 1
Write-Host ("selected_threads={0} short_total={1:N3}s short_first={2:N3}s short_next={3:N3}s" -f `
    $winner.threads,$winner.total_s,$winner.first_token_s,$winner.avg_next_s)

Write-Host ""
Write-Host "=== SAVED LONG BASELINE: SAME EXACT PROMPT ==="
Write-Host "prompt_tokens=306"
Write-Host "first_token=172.500s"
Write-Host "avg_next=2.256s/token"
Write-Host "total=2481.093s"
Write-Host "generated=1024"
Write-Host "expert_reads=161059"
Write-Host "expert_bytes=1331882.43 MiB"
Write-Host "expert_evictions=160998"
Write-Host "trunk_reads=2985.67 MiB"
Write-Host "This is the previously captured Dell baseline; it is not rerun here."

if ($SkipLong) {
    Write-Host ""
    Write-Host "SKIP_LONG requested; exact gate completed without starting the expensive run."
    exit 0
}

$env:OMP_NUM_THREADS = [string]$winner.threads
Write-Host ""
Write-Host "=== START LONG COMBINED-EXACT RUN ==="
Write-Host "threads=$($winner.threads) cache=$CacheMiB MiB ram=$RamMiB MiB max_new=1024 temp=0.2 seed=1"
Write-Host "Token timings will stream directly below."
Write-Host ""

# Inherit the console for the expensive run so token timings remain visible live.
# This is intentionally one candidate run only; the historical baseline above is
# reused instead of burning another full-model baseline pass.
& $Python .\tools\kvl_chat.py `
    $ModelDir `
    $LongPrompt `
    --binary $CandidateBinary `
    --cache-mib $CacheMiB `
    --ram-mib $RamMiB `
    --max-new 1024 `
    --temperature 0.2 `
    --seed 1 `
    --show-tokens
if ($LASTEXITCODE -ne 0) {
    throw "Long combined-exact generation failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "LONG_COMBINED_EXACT_RUN_FINISHED"
Write-Host "Compare final timing/cache stats plus RAM, CPU and disk observations against the saved baseline above."
