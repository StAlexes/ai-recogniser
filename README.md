# AI Recogniser

[![CI](https://github.com/StAlexes/ai-recogniser/actions/workflows/ci.yml/badge.svg)](https://github.com/StAlexes/ai-recogniser/actions/workflows/ci.yml)

Локальное Windows-приложение для распознавания русской речи из mp3/mp4: очередь задач в трее,
GigaAM v3, локальные саммари, база знаний и глоссарий. Без облака по умолчанию.

**Статус:** M0 — каркас готов (ядро + UI-прототип + оболочка). Распознавание файлов появится в M1.
План: [`docs/PLAN.md`](docs/PLAN.md). История версий: [`CHANGELOG.md`](CHANGELOG.md).
Замеры GigaAM: [`docs/M0-NOTES.md`](docs/M0-NOTES.md).

## Что уже работает (M0)

- Ядро на `127.0.0.1` со случайным портом и сессионным токеном: health, настройки, профиль ПК, каталог моделей.
- Окно WebView2 с Fluent UI: очередь (drag&drop файлов), база знаний, модели, настройки, светлая/тёмная тема.
- Трей, закрытие в трей, уведомления, single-instance, автозапуск.
- GigaAM v3 e2e CTC int8: **RTFx 56.8** на Ryzen 7 8845H, **352 МБ** RSS — CPU быстрее iGPU Radeon 780M.

## Требования для разработки

- Windows 10 1809+ / Windows 11, WebView2 Runtime (на Windows 11 уже установлен).
- Python 3.12+ (проверено на 3.13).
- Node.js 20.19+ и npm — только для сборки UI.

## Стек

- **Ядро:** Python 3.12+, FastAPI, SQLite (M1), ONNX Runtime (M1) на `127.0.0.1` со случайным портом и токеном.
- **UI:** React + TypeScript + Fluent UI v9, собирается Vite в `ui/dist`.
- **Оболочка:** WebView2 (pywebview), трей, single-instance, автозапуск.
- **Модели:** GigaAM v3 e2e CTC int8 (MIT) + опционально Multilingual/Whisper/LLM (M1+).

## Разработка

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev,shell]"

# UI
cd ui
npm.cmd install
npm.cmd run build
cd ..

# ядро отдельно (dev, без токена)
.\.venv\Scripts\python -m airec --port 8765 --no-auth

# приложение целиком (ядро + окно + трей)
.\.venv\Scripts\python shell\main.py
```

UI в dev-режиме: `npm.cmd run dev` (Vite проксирует `/api` на `127.0.0.1:8765`).

## Проверки

```powershell
.\.venv\Scripts\ruff check app shell tools
.\.venv\Scripts\pytest
```

## Данные и приватность

- Установленный режим: `%LOCALAPPDATA%\AIRecogniser` (модели, кэш, логи, результаты, `config.json`).
- Portable: файл `portable.flag` рядом с exe → данные в `.\data`, реестр не трогаем.
- API доступен только на loopback и требует токен из `session.json`; наружу ничего не уходит,
  кроме явных загрузок моделей/обновлений.
- Секреты и токены не пишутся в логи.

## Структура

```
app/airec/      ядро (api, core, jobs, media, asr, llm, rag — по этапам)
app/tests/      unit + e2e
ui/             React + Fluent UI v9
shell/          окно, трей, single-instance, автозапуск
docs/           план, лицензии, заметки
tools/          вспомогательные скрипты (иконка, POC)
```
