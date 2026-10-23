# Builds the native Windows app: UI bundle + PyInstaller one-folder exe.
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\build_exe.ps1
#   powershell -ExecutionPolicy Bypass -File tools\build_exe.ps1 -SkipUi
param(
    [switch]$SkipUi
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Spec = Join-Path $Root "packaging\airec.spec"
$Ui = Join-Path $Root "ui"
$Exe = Join-Path $Root "dist\AI Recogniser\AI Recogniser.exe"

if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python. Create it and install '.[dev,shell,build]'."
}
if (-not (Test-Path $Spec)) {
    throw "PyInstaller spec not found: $Spec"
}

if (-not $SkipUi) {
    Push-Location $Ui
    try {
        if (-not (Test-Path (Join-Path $Ui "node_modules"))) {
            npm.cmd ci
            if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
        }
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "UI build failed" }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path (Join-Path $Ui "dist\index.html"))) {
    throw "UI build is missing: ui\dist\index.html. Run without -SkipUi first."
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $Root "build\pyinstaller") `
    $Spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $Exe)) {
    throw "Build finished but exe is missing: $Exe"
}

$SizeMb = [math]::Round((Get-Item $Exe).Length / 1MB, 1)
Write-Host ""
Write-Host "Built: $Exe"
Write-Host "Exe size: $SizeMb MB (plus the _internal folder)"
Write-Host "Run it directly; no Python installation is required on the target PC."
