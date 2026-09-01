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
    kvl_generate_token_parallel_prefill `
    kvl_generate_decode_reuse `
    kvl_generate_shared_q8 `
    kvl_q8_expert_parallel_probe `
    kvl_mla_fused_prefill_probe `
    kvl_mla_token_parallel_prefill_probe `
    kvl_mla_decode_reuse_probe `
    kvl_router_stack_probe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$baseline = Join-Path $BuildDir 'core\Release\kvl_generate.exe'
$profile = Join-Path $BuildDir 'Release\kvl_generate_profile.exe'
$parallel = Join-Path $BuildDir 'Release\kvl_generate_expert_parallel.exe'
$fused = Join-Path $BuildDir 'Release\kvl_generate_fused_prefill.exe'
$tokenParallel = Join-Path $BuildDir 'Release\kvl_generate_token_parallel_prefill.exe'
$decodeReuse = Join-Path $BuildDir 'Release\kvl_generate_decode_reuse.exe'
$sharedQ8 = Join-Path $BuildDir 'Release\kvl_generate_shared_q8.exe'
$q8Probe = Join-Path $BuildDir 'Release\kvl_q8_expert_parallel_probe.exe'
$fusedProbe = Join-Path $BuildDir 'Release\kvl_mla_fused_prefill_probe.exe'
$tokenParallelProbe = Join-Path $BuildDir 'Release\kvl_mla_token_parallel_prefill_probe.exe'
$decodeProbe = Join-Path $BuildDir 'Release\kvl_mla_decode_reuse_probe.exe'
$routerProbe = Join-Path $BuildDir 'Release\kvl_router_stack_probe.exe'
foreach ($p in @($baseline,$profile,$parallel,$fused,$tokenParallel,$decodeReuse,$sharedQ8,$q8Probe,$fusedProbe,$tokenParallelProbe,$decodeProbe,$routerProbe)) {
    if (-not (Test-Path $p)) { throw "Missing expected binary: $p" }
}

$env:OMP_DYNAMIC = 'FALSE'
$env:OMP_NESTED = 'FALSE'
$env:OMP_NUM_THREADS = '6'
& $q8Probe
if ($LASTEXITCODE -ne 0) { throw "Q8 expert-parallel full-MoE bit-exact probe failed" }
& $fusedProbe
if ($LASTEXITCODE -ne 0) { throw "MLA fused-prefill bit-exact probe failed" }
& $tokenParallelProbe
if ($LASTEXITCODE -ne 0) { throw "MLA token-parallel prefill bit-exact probe failed" }
& $decodeProbe
if ($LASTEXITCODE -ne 0) { throw "MLA decode-reuse bit-exact probe failed" }
& $routerProbe
if ($LASTEXITCODE -ne 0) { throw "stack router bit-exact probe failed" }

Write-Host "PASS: isolated performance research binaries built"
Write-Host "  row-parallel baseline : $baseline"
Write-Host "  phase profiler         : $profile"
Write-Host "  expert-parallel pilot  : $parallel"
Write-Host "  fused-prefill pilot    : $fused"
Write-Host "  token-parallel prefill : $tokenParallel"
Write-Host "  decode-reuse pilot     : $decodeReuse"
Write-Host "  shared-Q8 pilot        : $sharedQ8"
Write-Host "  Q8 exactness probe     : $q8Probe"
Write-Host "  fused prefill exact    : $fusedProbe"
Write-Host "  token prefill exact    : $tokenParallelProbe"
Write-Host "  decode exactness       : $decodeProbe"
Write-Host "  router stack exact     : $routerProbe"
Write-Host ""
Write-Host "No main build directory or packed weights were modified."
