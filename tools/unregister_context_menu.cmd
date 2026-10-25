@echo off
setlocal EnableExtensions

rem Удаляет контекстное меню "Распознать в AI Recogniser" из HKCU.

set "EXTENSIONS=.mp3 .mp4 .m4a .wav .flac .ogg .opus .mkv .mov .webm .avi"

echo [AI Recogniser] Удаление контекстного меню из HKCU...

for %%E in (%EXTENSIONS%) do (
  reg delete "HKCU\Software\Classes\%%E\shell\AIRecogniser" /f >nul 2>&1
  if errorlevel 1 (
    echo [AI Recogniser] %%E -> не найдено
  ) else (
    echo [AI Recogniser] %%E -> удалено
  )
)

echo.
echo [AI Recogniser] Готово. Перезапустите Проводник.
pause
