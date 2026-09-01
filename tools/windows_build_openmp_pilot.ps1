param(
    [string]$BuildDir = "build-omp"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

cmake -S . -B $BuildDir -A x64 `
    -DKVL_USE_AVX2=ON `
    -DKVL_MSVC_PACKED_Q8_PARALLEL_KERNELS=ON

cmake --build $BuildDir --config Release --target kvl_generate kvl_generate_vl

$textBin = Join-Path $BuildDir 'Release\kvl_generate.exe'
$vlBin = Join-Path $BuildDir 'Release\kvl_generate_vl.exe'
if (-not (Test-Path $textBin)) { throw "Missing $textBin" }
if (-not (Test-Path $vlBin)) { throw "Missing $vlBin" }

Write-Host "PASS: packed-Q8 OpenMP pilot binaries built"
Write-Host "  text: $textBin"
Write-Host "  vision: $vlBin"
Write-Host ""
Write-Host "Baseline remains under build\Release; pilot is isolated in $BuildDir\Release."
Write-Host "Suggested first target-laptop sweep: OMP_NUM_THREADS=1,2,4,6"
