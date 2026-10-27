@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "ROOT=%CD%"

rem Регистрирует контекстное меню Windows "Распознать в AI Recogniser"
rem для поддерживаемых аудио/видео форматов. Запись в HKCU — не требует прав админа.
rem Использование: register_context_menu.cmd

rem Builds live in versioned folders (dist\AI Recogniser 0.2.1\).  Pick the
rem newest one, with a fallback to the legacy non-versioned layout.
set "EXE=%ROOT%\dist\AI Recogniser\AI Recogniser.exe"
if exist "%EXE%" goto :exe_found

for /f "delims=" %%D in ('dir /b /ad /o-d "%ROOT%\dist\AI Recogniser *" 2^>nul') do (
  if exist "%ROOT%\dist\%%D\AI Recogniser.exe" (
    set "EXE=%ROOT%\dist\%%D\AI Recogniser.exe"
    goto :exe_found
  )
)

echo [AI Recogniser] exe не найден в dist. Сначала выполните build.cmd.
exit /b 1

:exe_found
if not exist "%EXE%" (
  echo [AI Recogniser] exe не найден: %EXE%
  echo Сначала выполните build.cmd для сборки.
  exit /b 1
)

rem Превращаем относительный путь в абсолютный
for %%F in ("%EXE%") do set "EXE=%%~fF"

set "EXTENSIONS=.mp3 .mp4 .m4a .wav .flac .ogg .opus .mkv .mov .webm .avi"

echo [AI Recogniser] Регистрация контекстного меню в HKCU...

for %%E in (%EXTENSIONS%) do (
  rem Удалим старую запись, если была
  reg delete "HKCU\Software\Classes\%%E\shell\AIRecogniser" /f >nul 2>&1

  reg add "HKCU\Software\Classes\%%E\shell\AIRecogniser" /ve /d "Распознать в AI Recogniser" /f >nul 2>&1
  reg add "HKCU\Software\Classes\%%E\shell\AIRecogniser" /v "Icon" /d "\"%EXE%\",0" /f >nul 2>&1
  reg add "HKCU\Software\Classes\%%E\shell\AIRecogniser\command" /ve /d "\"%EXE%\" %%*" /f >nul 2>&1
  if errorlevel 1 (
    echo [AI Recogniser] Ошибка регистрации %%E
  ) else (
    echo [AI Recogniser] %%E -> OK
  )
)

echo.
echo [AI Recogniser] Готово. Перезапустите Проводник, чтобы меню появилось.
echo Удаление: unregister_context_menu.cmd
pause
