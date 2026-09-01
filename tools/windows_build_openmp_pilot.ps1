param(
    [string]$BuildDir = "build-omp"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$files = @(
    'src/ops.c',
    'src/q8_ops.c',
    'src/q5_ops.c'
)

$original = @{}
foreach ($file in $files) {
    $original[$file] = Get-Content $file -Raw
}

$old = @"
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64)
#endif
    for (int o = 0; o < out; ++o) {
"@
$new = @"
    int o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(out > 64)
#endif
    for (o = 0; o < out; ++o) {
"@

try {
    foreach ($file in $files) {
        $text = $original[$file]
        if (-not $text.Contains($old)) {
            throw "Expected OpenMP loop not found in $file"
        }
        [System.IO.File]::WriteAllText(
            (Join-Path (Get-Location) $file),
            $text.Replace($old, $new),
            [System.Text.UTF8Encoding]::new($false)
        )
    }

    cmake -S . -B $BuildDir -A x64 `
        -DKVL_USE_AVX2=ON `
        -DCMAKE_C_FLAGS="/openmp"

    cmake --build $BuildDir --config Release --target kvl_generate kvl_generate_vl

    $textBin = Join-Path $BuildDir 'Release\kvl_generate.exe'
    $vlBin = Join-Path $BuildDir 'Release\kvl_generate_vl.exe'
    if (-not (Test-Path $textBin)) { throw "Missing $textBin" }
    if (-not (Test-Path $vlBin)) { throw "Missing $vlBin" }

    Write-Host "PASS: OpenMP pilot binaries built"
    Write-Host "  text: $textBin"
    Write-Host "  vision: $vlBin"
    Write-Host ""
    Write-Host "Suggested first target-laptop sweep: OMP_NUM_THREADS=1,2,4,6"
}
finally {
    foreach ($file in $files) {
        if ($original.ContainsKey($file)) {
            [System.IO.File]::WriteAllText(
                (Join-Path (Get-Location) $file),
                $original[$file],
                [System.Text.UTF8Encoding]::new($false)
            )
        }
    }
}
