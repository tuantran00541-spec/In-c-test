param(
    [string]$BuildDir = "build-perf-research"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

cmake -S .\research\kimi_perf -B $BuildDir -A x64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

cmake --build $BuildDir --config Release --target `
    kvl_generate `
    kvl_generate_profile `
    kvl_generate_expert_parallel `
    kvl_generate_fused_prefill `
    kvl_q8_expert_parallel_probe `
    kvl_mla_fused_prefill_probe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$baseline = Join-Path $BuildDir 'core\Release\kvl_generate.exe'
$profile = Join-Path $BuildDir 'Release\kvl_generate_profile.exe'
$parallel = Join-Path $BuildDir 'Release\kvl_generate_expert_parallel.exe'
$fused = Join-Path $BuildDir 'Release\kvl_generate_fused_prefill.exe'
$q8Probe = Join-Path $BuildDir 'Release\kvl_q8_expert_parallel_probe.exe'
$fusedProbe = Join-Path $BuildDir 'Release\kvl_mla_fused_prefill_probe.exe'
foreach ($p in @($baseline,$profile,$parallel,$fused,$q8Probe,$fusedProbe)) {
    if (-not (Test-Path $p)) { throw "Missing expected binary: $p" }
}

$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NESTED = 'FALSE'
$env:OMP_NUM_THREADS = '6'
& $q8Probe
if ($LASTEXITCODE -ne 0) { throw "Q8 expert-parallel bit-exact probe failed" }
& $fusedProbe
if ($LASTEXITCODE -ne 0) { throw "MLA fused-prefill bit-exact probe failed" }

Write-Host "PASS: isolated performance research binaries built"
Write-Host "  row-parallel baseline : $baseline"
Write-Host "  phase profiler         : $profile"
Write-Host "  expert-parallel pilot  : $parallel"
Write-Host "  fused-prefill pilot    : $fused"
Write-Host "  Q8 exactness probe     : $q8Probe"
Write-Host "  MLA exactness probe    : $fusedProbe"
Write-Host ""
Write-Host "No main build directory or packed weights were modified."
