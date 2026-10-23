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
- План разработки `docs/PLAN.md` (v1.0): архитектура, модели, этапы, риски, критерии приёмки.
- Заведены версионирование (SemVer) и накопительный changelog.
- Зафиксированы решения 2026-10-23: стек Python + WebView2 + Fluent UI; локальная LLM с опцией внешних API; диаризация в v1.0 (beta).
- Рассмотрены GigaAM Multilingual и GigaChat Audio (публикация Сбера, 14.07.2026): Multilingual добавлен как опциональный ASR-движок (v1.0), GigaChat Audio — как post-1.0 эксперимент на GPU/внешнем API (ADR-009).

### Changed
- —

### Fixed
- —

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
