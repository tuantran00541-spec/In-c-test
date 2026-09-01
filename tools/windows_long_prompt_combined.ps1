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

$ShortPrompt = "Xin chào. Hãy trả lời đúng một câu ngắn."
$LongPrompt = "Hãy viết một bài khoảng 700–1000 từ bằng tiếng Việt về chủ đề: Một chiếc laptop phổ thông có thể chạy mô hình AI lớn hoàn toàn bằng CPU như thế nào? Giải thích theo cách dễ hiểu nhưng vẫn có chi tiết kỹ thuật. Bài viết cần có mở đầu, phần giải thích về RAM, SSD/NVMe, quantization, MoE, cache, tốc độ token, giới hạn thực tế, và phần kết luận. Không dùng danh sách gạch đầu dòng quá nhiều; ưu tiên viết thành các đoạn văn liền mạch. Hãy duy trì mạch văn tự nhiên, không lặp ý, không dừng giữa chừng, và tiếp tục cho đến khi hoàn thành toàn bộ bài viết."

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
# This is intentionally one candidate run only; the 41-minute historical baseline
# above is reused instead of burning another full-model baseline pass.
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
Write-Host "Compare the final [kvl] timing, kvl_cache, kvl_trunk_cache, RAM peak and CPU/disk observations against the saved baseline above."
