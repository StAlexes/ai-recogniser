@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

rem Регистрирует контекстное меню Windows "Распознать в AI Recogniser"
rem для поддерживаемых аудио/видео форматов. Запись в HKCU — не требует прав админа.
rem Использование: register_context_menu.cmd

set "EXE=%~dp0..\dist\AI Recogniser\AI Recogniser.exe"
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

  reg add "HKCU\Software\Classes\%%E\shell\AIRecogniser" /v "" /d "Распознать в AI Recogniser" /f >nul 2>&1
  reg add "HKCU\Software\Classes\%%E\shell\AIRecogniser" /v "Icon" /d "%EXE%,0" /f >nul 2>&1
  reg add "HKCU\Software\Classes\%%E\shell\AIRecogniser\command" /v "" /d "%EXE% "%%1"" /f >nul 2>&1
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
