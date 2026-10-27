@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [AI Recogniser] Python venv not found: %~dp0.venv\Scripts\python.exe
  echo Create it and install dependencies:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e ".[dev,shell,asr,build]"
  exit /b 1
)

echo [AI Recogniser] Building the native exe into dist...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_exe.ps1" %*
set "BUILD_CODE=%ERRORLEVEL%"
if not "%BUILD_CODE%"=="0" (
  echo [AI Recogniser] Build failed with exit code %BUILD_CODE%.
  exit /b %BUILD_CODE%
)

echo [AI Recogniser] Done. The build is in a versioned folder under dist\.
exit /b 0
