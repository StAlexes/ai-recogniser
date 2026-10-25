# Builds the native Windows app: UI bundle + PyInstaller one-folder exe.
# Usage:
#   build.cmd
#   build.cmd -SkipUi
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
    throw "Python venv not found: $Python. Create it and install '.[dev,shell,asr,build]'."
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

# The previous build may still be running from dist and hold
# _internal\...\ClrLoader.dll, which makes PyInstaller fail with PermissionError.
$Running = Get-Process -Name "AI Recogniser" -ErrorAction SilentlyContinue
if ($Running) {
    $Pids = ($Running | Select-Object -ExpandProperty Id) -join ", "
    throw "Close AI Recogniser before building (running PIDs: $Pids)."
}

$BuildDir = Join-Path $Root "build"
$BuildLog = Join-Path $BuildDir "pyinstaller.log"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# PyInstaller writes progress and harmless warnings to stderr. Windows
# PowerShell 5.1 turns native stderr into a NativeCommandError when
# $ErrorActionPreference is Stop, so temporarily switch to Continue, capture
# everything in a log and then check the real exit code.
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $BuildDir "pyinstaller") `
    $Spec 2>&1 | Out-File -FilePath $BuildLog -Encoding utf8
$BuildCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
Get-Content $BuildLog -Tail 80 | Write-Host
if ($BuildCode -ne 0) {
    Write-Host "Full build log: $BuildLog"
    throw "PyInstaller failed with exit code $BuildCode"
}
if (-not (Test-Path $Exe)) {
    throw "Build finished but exe is missing: $Exe"
}

$SizeMb = [math]::Round((Get-Item $Exe).Length / 1MB, 1)
Write-Host ""
Write-Host "Built: $Exe"
Write-Host "Exe size: $SizeMb MB (plus the _internal folder)"
Write-Host "Run it directly; no Python installation is required on the target PC."

# Generate version.json alongside the exe for build traceability.
$VersionPy = Join-Path $Root "app\airec\version.py"
$VersionJson = Join-Path $Root "dist\AI Recogniser\version.json"
$Version = "0.1.0.dev0"
if (Test-Path $VersionPy) {
    $VersionContent = Get-Content $VersionPy -Raw
    $VersionLine = $VersionContent | Select-String '^__version__\s*='
    if ($VersionLine) {
        $Version = ($VersionLine.ToString() -split '=')[-1].Trim().Trim('"').Trim("'")
    }
}
$GitHash = ""
$GitStatus = "unknown"
try {
    $GitHash = (git -C $Root rev-parse --short HEAD 2>$null).Trim()
    $Status = (git -C $Root status --porcelain 2>$null)
    if ($Status) {
        $GitStatus = "dirty"
    } else {
        $GitStatus = "clean"
    }
} catch {
    # Git may not be available in all environments.
}
$VersionData = @{
    version = $Version
    built_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz")
    git_commit = $GitHash
    git_status = $GitStatus
    exe_name = "AI Recogniser.exe"
    exe_size_mb = $SizeMb
} | ConvertTo-Json -Depth 3
$VersionData | Out-File -Encoding utf8 -FilePath $VersionJson
Write-Host ""
Write-Host "Build manifest: $VersionJson"
