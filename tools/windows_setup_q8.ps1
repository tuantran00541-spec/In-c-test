param(
    [string]$WorkDir = "checkpoints\kimi-vl-work",
    [string]$RuntimeDir = "packed\kimi-vl-a3b-q8",
    [int]$BuildJobs = 2,
    [switch]$KeepSourceShards
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. $Hint"
    }
}

Require-Command "python" "Install 64-bit Python 3.11+ and add it to PATH."
Require-Command "cmake" "Install CMake and add it to PATH."

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$Venv = Join-Path $RepoRoot ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "[1/5] Creating Python virtual environment at $Venv"
    python -m venv $Venv
}

Write-Host "[2/5] Installing Python dependencies"
& $Py -m pip install --upgrade pip
& $Py -m pip install --index-url https://download.pytorch.org/whl/cpu torch
& $Py -m pip install -r requirements-user.txt

Write-Host "[3/5] Configuring native C runtime"
cmake -S . -B build -DKVL_USE_AVX2=ON

Write-Host "[4/5] Building Release binaries"
cmake --build build --config Release --parallel $BuildJobs

Write-Host "[5/5] Downloading and packing pinned Kimi-VL Q8 runtime"
$PrepareArgs = @(
    "tools/prepare_kimi_vl_q8.py",
    $WorkDir,
    $RuntimeDir
)
if ($KeepSourceShards) {
    $PrepareArgs += "--keep-source-shards"
}
& $Py @PrepareArgs

$RuntimeAbs = (Resolve-Path $RuntimeDir).Path
Write-Host ""
Write-Host "READY: $RuntimeAbs"
Write-Host "Example:"
Write-Host ".\.venv\Scripts\python.exe tools\kvl_vl_chat.py `"$RuntimeAbs`" `"C:\path\image.jpg`" `"Describe this image in one short sentence.`" --cache-mib 512 --ram-mib 4096 --max-new 16 --temperature 0 --show-tokens"
