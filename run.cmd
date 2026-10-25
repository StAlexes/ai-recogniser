@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [AI Recogniser] Python venv not found: %PYTHON%
  echo Create it and install dependencies:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -e ".[dev,shell,asr]"
  exit /b 1
)

if not exist "ui\dist\index.html" (
  echo [AI Recogniser] UI build is missing, building it...
  pushd "ui"
  if not exist "node_modules" (
    call npm.cmd ci
    if errorlevel 1 goto :ui_failed
  )
  call npm.cmd run build
  if errorlevel 1 goto :ui_failed
  popd
)

echo [AI Recogniser] Starting from source...
"%PYTHON%" "shell\main.py" %*
exit /b %errorlevel%

:ui_failed
popd
echo [AI Recogniser] UI build failed. Fix the error above and try again.
exit /b 1
