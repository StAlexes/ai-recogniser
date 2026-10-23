# Changelog

Все заметные изменения проекта AI Recogniser.
Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), версионирование — [SemVer 2.0.0](https://semver.org/lang/ru/).

Правила ведения:
- Новая версия добавляется **сверху**, история не переписывается.
- Незавершённые изменения копятся в `[Unreleased]`.
- Дата релиза — ISO `YYYY-MM-DD`, версия в `pyproject.toml` должна совпадать с тегом `vX.Y.Z`.
- Раздел «Модели» фиксирует версии ASR/LLM/эмбеддингов и промптов, попавшие в релиз.

## [Unreleased]

### Added
- План разработки `docs/PLAN.md` (v1.2): архитектура, модели, этапы, риски, критерии приёмки.
- Заведены версионирование (SemVer) и накопительный changelog.
- Зафиксированы решения 2026-10-23: стек Python + WebView2 + Fluent UI; локальная LLM с опцией внешних API; диаризация в v1.0 (beta).
- Рассмотрены GigaAM Multilingual и GigaChat Audio (публикация Сбера, 14.07.2026): Multilingual добавлен как опциональный ASR-движок (v1.0), GigaChat Audio — как post-1.0 эксперимент на GPU/внешнем API (ADR-009).
- **M0, ядро:** FastAPI-приложение с loopback-биндингом, случайным портом и сессионным токеном;
  `/healthz` (без авторизации), `/api/v1/health`, `/api/v1/settings` (GET/PUT с атомарной записью),
  `/api/v1/system/profile` (кэш `profile.json`), `/api/v1/models` (static-каталог), заглушка `POST /api/v1/jobs` (501).
- **M0, оболочка:** `shell/main.py` — окно WebView2, трей (pystray), закрытие в трей, single-instance
  через именованный mutex, управление жизненным циклом ядра, `--smoke-test`, `--autostart on/off/status`
  (HKCU Run; в portable-режиме реестр не трогается), иконка приложения (10 размеров).
- **M0, UI:** прототип на React 18 + TypeScript + Fluent UI v9 (Vite): очередь с drag&drop,
  база знаний, каталог моделей с профилем ПК, настройки (чтение/запись через API), светлая/тёмная тема,
  русская локализация, доступность с клавиатуры.
- **M0, измерения:** POC GigaAM v3 e2e CTC int8 на референсном ноуте (Ryzen 7 8845H):
  RTFx **56.8**, загрузка 1.7 с, peak RSS **352 МБ**, корректная пунктуация. DirectML (отдельный
  runtime, ORT 1.24.4) — RTFx 49.3 и 622 МБ RSS: iGPU Radeon 780M проигрывает CPU, дефолт CPU.
  Подробности: `docs/M0-NOTES.md`, воспроизводимый скрипт `tools/poc_gigaam.py`.
- CI: отдельные job'ы для ядра (ruff + pytest) и UI (typecheck + build); UI-job запускается,
  когда в репозитории есть `ui/package.json`.

### Changed
- —

### Fixed
- Полноэкранные оверлеи тултипов Fluent UI: `FluentProvider` копировал стили корня на mount-узлы
  порталов, из-за чего страницы с подсказками накрывались непрозрачным слоем. Отключено через
  `applyStylesToPortals={false}`; порталы по-прежнему получают переменные темы.
- Профиль ПК показывает физический GPU первым: виртуальные адаптеры (Parsec, VMware, Remote и т.п.)
  помечаются и уходят в конец списка.
- `config.json` с UTF-8 BOM (после сохранения в Блокноте) читается корректно, настройки не сбрасываются.

### Security
- —

### Модели
- Планируется: GigaAM v3 e2e CTC int8 (MIT), Silero VAD, multilingual-e5, T-lite-it-2.1 8B / Qwen3-4B (GGUF).

---

<!-- Шаблон для следующего релиза:

## [0.1.0] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Security
- ...

### Модели
- ASR: gigaam-v3-e2e-ctc@1.0.0 (int8)
- LLM: ...
- Промпты: summary@1

-->
