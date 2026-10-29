# Сборка нативного exe

Документ описывает M1-сборку десктопного приложения: one-folder exe с ядром, UI и ASR-стеком.
Установщик Inno Setup, подпись кода и офлайн-бандл моделей появятся на этапе M6; здесь — рабочая
сборка, которую можно запускать без Python.

## Что получается

```
dist\
  AI Recogniser <версия>\
    AI Recogniser.exe    # оболочка + точка входа ядра, без консоли
    _internal\           # CPython, библиотеки, UI, иконки, ASR, .NET-сборки WebView2
    version.json         # версия, дата сборки, git-коммит и статус дерева
```

Каждая сборка попадает в отдельную папку `dist\AI Recogniser <версия>\`. Версия берётся из
`app/airec/version.py`; старые папки не перезаписываются.

`AI Recogniser.exe` — обычный Windows GUI-процесс. При старте он поднимает ядро тем же exe с
внутренним флагом `--core-worker`, ждёт `session.json` и `/healthz`, затем открывает окно
WebView2. Ядро работает в дочернем процессе, поэтому падение или зависание UI не блокирует
HTTP-сервер; при выходе shell останавливает ядро и удаляет сессию.

One-folder выбран осознанно: при one-file PyInstaller распаковывает рантайм при каждом запуске,
а дочерний процесс ядра распаковывал бы его второй раз. Для архитектуры «shell + core worker»
это заметно увеличивает старт и расход диска. Готовый установщик сможет упаковать эту папку.

Текущий размер сборки (2026-10-23): **`AI Recogniser.exe` ~14.6 МБ**, `_internal` **~210.8 МБ**,
769 файлов. Модель GigaAM (~215 МБ) в поставку не входит и скачивается при первом распознавании.

## Требования к сборочной машине

- Windows 10/11 x64.
- Python 3.12+ (проверено на 3.13) и venv `.venv`.
- Node.js 20.19+ и npm — для сборки UI.
- Зависимости: `pip install -e ".[dev,shell,asr,build]"`.
- WebView2 Runtime на машине разработчика (на Windows 11 есть из коробки).
- Сеть для первой загрузки модели GigaAM; сам exe собирается офлайн, если модель уже в кэше.

## Команды

```bat
REM из корня репозитория
REM запуск из исходников «на лету»: ядро + окно + трей
run.cmd

REM сборка нативного exe в dist: UI + PyInstaller
build.cmd

REM только exe, если ui\dist уже собран
build.cmd -SkipUi
```

`run.cmd` проверяет `.venv`, при отсутствии `ui\dist\index.html` собирает UI и запускает
`shell\main.py`; остальные аргументы передаются в оболочку. `build.cmd` — тонкая обёртка над
`tools\build_exe.ps1`: передаёт аргументы (`-SkipUi`), возвращает его код выхода и печатает путь
к готовому exe. Если приложение запущено из `dist`, скрипт остановится с понятной ошибкой:
процесс держит `_internal\...\ClrLoader.dll` и PyInstaller не сможет перезаписать папку.

Скрипт собирает UI (`npm.cmd run build`), затем запускает PyInstaller со
[`packaging/airec.spec`](../packaging/airec.spec). Артефакты: `dist\` и `build\pyinstaller\`,
оба каталога исключены из git. Полный лог PyInstaller пишется в `build\pyinstaller.log`; скрипт
проверяет реальный exit code и не маскирует ошибки stderr.

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
| `onnxruntime\` | пакет onnxruntime | CPU-инференс ASR (`onnxruntime.dll` ~18.4 МБ) |
| `onnx_asr\` | пакет onnx-asr | загрузка/запуск GigaAM |
| `numpy\`, `yaml\`, `huggingface_hub\` | зависимости ASR | массивы, конфиг, загрузка модели |
| `multipart\`, `python_multipart\` | python-multipart | приём загруженных файлов |

FFmpeg в поставку не входит: LGPL-сборка скачивается по требованию в папку данных
(`data\runtime\ffmpeg\`) при первом не-WAV файле и проверяется по размеру и SHA-256.

Спецификация использует `collect_all` для `onnx_asr`, `onnxruntime` и
`python-multipart`, потому что часть импортов и бинарников создаётся динамически и не видна
статическому анализатору PyInstaller. Предупреждение `onnxruntime.quantization requires onnx`
безвредно: квантование приложению не нужно, ASR-модель уже в int8.

Silero VAD (~2 МБ) в поставку не входит: `onnx-asr` скачивает его из
`istupakov/silero-vad-onnx` при первом длинном файле и кэширует рядом с GigaAM. После этого
нарезка работает офлайн. Если VAD-модель недоступна, ядро переходит на фиксированные окна по
60 секунд, поэтому длинный файл всё равно будет распознан.

Метаданные exe (версия файла, продукт, copyright) заданы в
[`packaging/version_info.txt`](../packaging/version_info.txt). При смене версии приложения
обновите и этот файл; версия в `pyproject.toml`/`airec.version` остаётся источником истины для
API и UI.

## Пути в frozen-режиме

- `resource_root()` (`sys._MEIPASS`) — только чтение: UI, иконки, встроенные бинарники.
- `app_root()` — каталог exe: `portable.flag`, portable-данные, `bin`.
- `data_root()` — `%LOCALAPPDATA%\AIRecogniser` или `<exe>\data` в portable-режиме.
- Логи: `<data>\logs\shell.log`, `app.log`, `core.out.log` (stdout ядра), `crash.log`.

`--core-worker` — внутренний флаг, не для пользователя. В dev-режиме shell по-прежнему запускает
ядро командой `python -m airec`.

## Проверка сборки

### 1. Smoke-тест

Окно не открывается; shell поднимает ядро и проверяет health/UI по HTTP.

```powershell
$smoke = Join-Path $env:TEMP "airec-exe-smoke"
Remove-Item -Recurse -Force $smoke -ErrorAction SilentlyContinue
$env:AIREC_DATA_DIR = $smoke
$out = Join-Path $env:TEMP "airec-smoke-out.txt"
$exe = ".\dist\AI Recogniser 0.2.1\AI Recogniser.exe"
$p = Start-Process -FilePath $exe `
    -ArgumentList "--smoke-test" -Wait -PassThru `
    -RedirectStandardOutput $out
$p.ExitCode     # ожидается 0
Get-Content $out -Encoding utf8
```

В текущей сборке JSON печатается в stdout, например
`{"ok": true, "port": 12120, "version": "0.2.1", "ui_built": true}`; тот же результат
пишется в `<data>\logs\shell.log`. Используйте `Start-Process -Wait`, потому что GUI-exe,
запущенный через `&`, не всегда дожидается завершения в PowerShell. Подставьте свою версию:
папка сборки называется `dist\AI Recogniser <версия>`.

### 2. Распознавание в frozen-ядре

Проверяет, что внутри exe действительно есть ffmpeg, ONNX Runtime и onnx-asr. Запустите ядро
отдельно (для теста без токена) и отправьте mp3:

```powershell
$data = Join-Path $env:TEMP "airec-exe-run"
Remove-Item -Recurse -Force $data -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $data | Out-Null
$env:AIREC_DATA_DIR = $data
$dist = ".\dist\AI Recogniser 0.2.1"
$core = Start-Process -FilePath "$dist\AI Recogniser.exe" `
    -ArgumentList "--core-worker", "--port", "8766", "--no-auth" `
    -WorkingDirectory $dist -PassThru

Invoke-RestMethod http://127.0.0.1:8766/api/v1/health

$job = curl.exe -s -F "file=@data/poc/gigaam-example.mp3;type=audio/mpeg" `
    http://127.0.0.1:8766/api/v1/jobs | ConvertFrom-Json
Invoke-RestMethod "http://127.0.0.1:8766/api/v1/jobs/$($job.id)"
Get-Content "$data\results\$($job.id)\transcript.txt" -Encoding utf8

Stop-Process -Id $core.Id
```

Ожидается статус `done`, корректный русский текст и файл `transcript.txt`. Первый запуск
включает загрузку модели и поэтому медленнее; затем RTFx заметно выше.

### 2a. Длинный файл и VAD

Проверяет, что запись длиннее прежнего лимита 300 с не падает с ошибкой ONNX, а режется на
сегменты. Можно склеить тестовый файл из `data/poc/gigaam-example.mp3`:

```powershell
$ff = (Resolve-Path ".\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-*.exe").Path
& $ff -nostdin -hide_banner -loglevel error -y -stream_loop 30 `
    -i "data\poc\gigaam-example.mp3" -t 678 -vn -ac 1 -ar 16000 -c:a pcm_s16le `
    "$env:TEMP\airec-long-678s.wav"

$job = curl.exe -s -F "file=@$env:TEMP\airec-long-678s.wav;type=audio/wav" `
    http://127.0.0.1:8766/api/v1/jobs | ConvertFrom-Json
Invoke-RestMethod "http://127.0.0.1:8766/api/v1/jobs/$($job.id)"
```

Ожидается `status: done`, `duration_s` около 678, несколько десятков сегментов
(`segments`), непустой `text` и корректный `transcript.txt`. В `job.json` фазы проходят через
«Нарезка по сегментам» и «Распознавание сегментов». Первый длинный файл дополнительно скачивает
Silero VAD (~2 МБ); после кэширования VAD работает офлайн.

### 3. RAG в frozen-ядре

Нужен отдельно запущенный `ragkb` на `http://127.0.0.1:8000`.

```powershell
Invoke-RestMethod http://127.0.0.1:8766/api/v1/rag/status
Invoke-RestMethod "http://127.0.0.1:8766/api/v1/rag/search?q=SSL&k=2"
Invoke-RestMethod -Method Post -ContentType "application/json" `
    -Body '{"question":"Что такое SSL?","k":2}' `
    http://127.0.0.1:8766/api/v1/rag/ask
```

Если `ragkb` не запущен, status должен вернуть `reachable: false` с понятной ошибкой, а
распознавание — продолжать работать.

### 4. GUI и оболочка

Запустить exe, проверить очередь/базу знаний/модели/настройки, трей и закрытие.
При `close_to_tray=false` закрытие окна завершает процесс и ядро; `session.json` удаляется.

### 5. Portable и single instance

Скопировать папку в другое место, создать рядом с exe `portable.flag`, запустить. Данные должны
появиться в `<папка>\data`, в `%LOCALAPPDATA%` — ничего нового. Второй запуск exe должен
завершиться сразу, оставив одно окно и одно ядро.

### 6. Автономность

Проверить папку на машине без Python (чистая Windows VM или Windows Sandbox). Для первого
распознавания нужна сеть (скачивание GigaAM); после кэширования модели — офлайн.

Для отладки UI внутри WebView2 можно задать `AIREC_DEBUG_PORT=9222`; тогда pywebview включает
CDP только на loopback. В обычном запуске порт не открывается.

## Ограничения текущей сборки

- **ASR:** GigaAM v3 e2e CTC int8 на CPU, Silero VAD для длинных записей. Таймкодов в
  `transcript.txt` и диаризации пока нет; сегменты сохраняются в `job.json` для будущего экспорта.
- **Очередь:** in-memory, один worker; состояние задач в SQLite WAL
  (`data/airec_jobs.sqlite`), выживает при перезапуске. Отмена и повтор доступны через
  API `POST /api/v1/jobs/{id}/cancel` и `POST /api/v1/jobs/{id}/retry`. Исходный файл
  хранится в `results/<job_id>/source<ext>`. Незавершённая задача после перезапуска
  помечается как прерванная.
- **Менеджер моделей:** каталог и скачивание моделей с проверкой SHA-256
  (`app/airec/models/`, API `GET /api/v1/models`, `POST /api/v1/models/download/{id}`).
- **Оповещения:** Windows toast-уведомления через иконку трея при завершении и ошибке задач.
- **Экспорт:** только `.txt` в `transcript.txt`; через API `GET /api/v1/jobs/{id}/export`
  доступны SRT, VTT, Markdown, JSON с тайм-кодами сегментов VAD.
- **Результаты:** только `.txt`; экспорт SRT/VTT/Markdown — позже.
- **RAG:** внешний `ragkb`; индекс и модель не входят в exe. Без сервиса доступность RAG
  показывается как ошибка, остальное приложение работает.
- **Модель:** GigaAM скачивается при первом запуске; офлайн-бандл — M6.
- **ffmpeg:** используется GPLv3-бинарник из `imageio-ffmpeg 0.6.0`; до релиза нужна LGPL-сборка
  или GPL-совместимая поставка (см. `docs/LICENSES.md`).
- **Установка:** нет установщика, обновлений, ассоциаций файлов и подписи; SmartScreen может
  предупредить. One-folder нужно распространять целиком; один exe без `_internal` не работает.
- **Windows 10:** нужен WebView2 Runtime; установщик позже добавит bootstrapper/проверку.
