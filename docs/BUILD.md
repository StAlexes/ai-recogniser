# Сборка нативного exe

Документ описывает M0-сборку десктопного приложения. Установщик Inno Setup, подпись кода и
офлайн-бандл моделей появятся на этапе M6; здесь — рабочая one-folder сборка, которую можно
запускать без Python.

## Что получается

```
dist\AI Recogniser\
  AI Recogniser.exe      # оболочка + точка входа ядра, без консоли
  _internal\             # CPython, библиотеки, UI, иконки, .NET-сборки WebView2
```

`AI Recogniser.exe` — обычный Windows GUI-процесс. При старте он поднимает ядро тем же exe с
внутренним флагом `--core-worker`, ждёт `session.json` и `/healthz`, затем открывает окно
WebView2. Ядро работает в дочернем процессе, поэтому падение или зависание UI не блокирует
HTTP-сервер; при выходе shell останавливает ядро и удаляет сессию.

One-folder выбран осознанно: при one-file PyInstaller распаковывает рантайм при каждом запуске,
а дочерний процесс ядра распаковывал бы его второй раз. Для архитектуры «shell + core worker»
это заметно увеличивает старт и расход диска. Готовый установщик сможет упаковать эту папку.

## Требования к сборочной машине

- Windows 10/11 x64.
- Python 3.12+ (проверено на 3.13) и venv `.venv`.
- Node.js 20.19+ и npm — для сборки UI.
- Зависимости: `pip install -e ".[dev,shell,build]"`.
- WebView2 Runtime на машине разработчика (на Windows 11 есть из коробки).

## Команды

```powershell
# из корня репозитория
.\.venv\Scripts\pip install -e ".[dev,shell,build]"

# UI + PyInstaller
powershell -ExecutionPolicy Bypass -File tools\build_exe.ps1

# только exe, если ui\dist уже собран
powershell -ExecutionPolicy Bypass -File tools\build_exe.ps1 -SkipUi
```

Скрипт собирает UI (`npm.cmd run build`), затем запускает PyInstaller со
[`packaging/airec.spec`](../packaging/airec.spec). Артефакты: `dist\` и `build\pyinstaller\`,
оба каталога исключены из git.

PyInstaller — только сборочная зависимость (extra `build`), в рантайм приложения не входит.

## Что попадает в сборку

| Путь в `_internal` | Источник | Зачем |
|---|---|---|
| `ui\dist\` | `ui/dist` | собранный React/Fluent UI |
| `assets\` | `shell/assets` | иконка окна и трея |
| `webview\lib\` | пакет pywebview | .NET-сборки WebView2, interop DLL |
| `clr_loader`, `pythonnet` | пакеты | загрузка .NET в pywebview |
| `pystray` | пакет | трей, backend `_win32` |
| `uvicorn`/`anyio` | пакеты | динамические импорты ASGI-сервера |

Метаданные exe (версия файла, продукт, copyright) заданы в
[`packaging/version_info.txt`](../packaging/version_info.txt). При смене версии приложения
обновите и этот файл; версия в `pyproject.toml`/`airec.version` остаётся источником истины для
API и UI.

## Пути в frozen-режиме

- `resource_root()` (`sys._MEIPASS`) — только чтение: UI, иконки, будущие встроенные бинарники.
- `app_root()` — каталог exe: `portable.flag`, portable-данные, `bin`.
- `data_root()` — `%LOCALAPPDATA%\AIRecogniser` или `<exe>\data` в portable-режиме.
- Логи: `<data>\logs\shell.log`, `app.log`, `core.out.log` (stdout ядра), `crash.log`.

`--core-worker` — внутренний флаг, не для пользователя. В dev-режиме shell по-прежнему запускает
ядро командой `python -m airec`.

## Проверка сборки

1. **Smoke-тест** (окно не открывается, ядро и UI проверяются по HTTP):

   ```powershell
   $env:AIREC_DATA_DIR = "$env:TEMP\airec-exe-smoke"
   .\dist\AI Recogniser\AI Recogniser.exe --smoke-test
   $LASTEXITCODE   # ожидается 0
   ```

   Exe собран без консоли, поэтому JSON не печатается; результат и причина ошибки — в
   `<data>\logs\shell.log`.

2. **GUI**: запустить exe, проверить очередь/базу знаний/модели/настройки, трей и закрытие.
   При `close_to_tray=false` закрытие окна завершает процесс и ядро; `session.json` удаляется.

3. **Portable**: скопировать папку в другое место, создать рядом с exe `portable.flag`,
   запустить. Данные должны появиться в `<папка>\data`, в `%LOCALAPPDATA%` — ничего нового.

4. **Single instance**: второй запуск exe должен завершиться сразу, оставив одно окно и одно ядро.

5. **Автономность**: проверить папку на машине без Python (чистая Windows VM или Windows Sandbox).

Для отладки UI внутри WebView2 можно задать `AIREC_DEBUG_PORT=9222`; тогда pywebview включает
CDP только на loopback. В обычном запуске порт не открывается.

## Ограничения M0-сборки

- Нет ASR/ffmpeg/LLM: это каркас M0, распознавание появится в M1. Поэтому exe пока небольшой
  (~11 МБ exe + ~67 МБ `_internal`).
- Нет установщика, обновлений и ассоциаций файлов — этап M6.
- Exe не подписан; SmartScreen может предупредить. Подпись появится при наличии сертификата.
- На Windows 10 нужен WebView2 Runtime; установщик позже добавит bootstrapper/проверку.
- One-folder нужно распространять целиком; один exe без `_internal` не работает.
