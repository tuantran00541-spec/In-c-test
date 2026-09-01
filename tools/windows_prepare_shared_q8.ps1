param(
    [string]$ModelDir = ".\packed\kimi-vl-a3b-q8",
    [string]$SidecarDir = ".\packed\kimi-vl-a3b-q8-shared-q8",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [int]$SamplesPerLayer = 2,
    [double]$MaxRelativeRms = 0.05,
    [int]$BaselineExpertCacheMiB = 512
)

$ErrorActionPreference = 'Stop'

foreach ($p in @($Python, (Join-Path $ModelDir 'trunk.bin'), (Join-Path $ModelDir 'trunk.idx'))) {
    if (-not (Test-Path $p)) { throw "Missing required path: $p" }
}

Write-Host "=== PACK SHARED Q8 SIDECAR ==="
& $Python .\research\kimi_perf\pack_shared_q8_from_trunk.py $ModelDir $SidecarDir
if ($LASTEXITCODE -ne 0) { throw "shared-Q8 pack failed" }

foreach ($p in @((Join-Path $SidecarDir 'shared_q8.bin'), (Join-Path $SidecarDir 'shared_q8.idx'))) {
    if (-not (Test-Path $p)) { throw "Missing sidecar output: $p" }
}

Write-Host ""
Write-Host "=== REAL WEIGHT NUMERICAL GATE ==="
& $Python .\research\kimi_perf\validate_shared_q8_real.py `
    $ModelDir $SidecarDir `
    --samples $SamplesPerLayer `
    --max-rel $MaxRelativeRms
if ($LASTEXITCODE -ne 0) { throw "shared-Q8 real-weight numerical gate failed" }

Write-Host ""
Write-Host "=== CACHE RESIDENCY REBALANCE ==="
& $Python .\research\kimi_perf\shared_q8_memory_plan.py `
    $ModelDir `
    --sidecar-dir $SidecarDir `
    --baseline-expert-cache-mib $BaselineExpertCacheMiB
if ($LASTEXITCODE -ne 0) { throw "shared-Q8 memory rebalance planner failed" }

Write-Host ""
Write-Host "SHARED_Q8_REAL_ASSET_GATE_PASS"
Write-Host "Sidecar is isolated; trunk.bin/trunk.idx were read-only inputs."
Write-Host "Do not treat this numerical gate as model-quality PASS; generation A/B is still required."
