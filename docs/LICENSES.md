# Third-party components and licenses

Черновик для M0. Перед релизом фиксируем точные версии, ссылки и тексты лицензий.

## Runtime / libraries

| Компонент | Назначение | Лицензия | Примечание |
|---|---|---|---|
| FastAPI | HTTP API ядра | MIT | |
| Uvicorn | ASGI-сервер | BSD-3-Clause | |
| Pydantic | валидация настроек/схем | MIT | |
| platformdirs | пути данных | MIT | |
| psutil | телеметрия железа | BSD-3-Clause | |
| pywebview | окно WebView2 | BSD-3-Clause | |
| pystray | иконка в трее | LGPL-3.0 | динамическая линковка, исходники доступны |
| Pillow | иконки | HPND | |
| React / Fluent UI v9 | интерфейс | MIT | |
| Vite | сборка UI | MIT | |

## Models (weights)

| Модель | Лицензия | Использование |
|---|---|---|
| GigaAM v3 e2e CTC (ai-sage / salute-developers) | MIT | основная ASR-модель, 214 МБ int8 |
| GigaAM Multilingual (220M/600M) | MIT | опциональный ASR для KZ/KG/UZ/EN |
| Silero VAD | MIT | нарезка аудио |
| pyannote segmentation-3.0 + CAM++ (3D-Speaker) | MIT / Apache-2.0 | диаризация (beta) |
| multilingual-e5 | MIT | эмбеддинги базы знаний |
| Qwen3-4B-Instruct | Apache-2.0 | локальная LLM (лёгкий профиль) |
| T-lite-it-2.1 8B | **проверить до релиза** | локальная LLM (RU) |
| YandexGPT-5-Lite-8B | **проверить до релиза** | альтернатива |
| GigaChat Audio 10B-A1.8B | MIT | эксперимент после 1.0, не в поставке |

## Tools

| Компонент | Лицензия | Примечание |
|---|---|---|
| FFmpeg / ffprobe | LGPL-2.1+ (сборка без GPL-модулей) | поставляем отдельными exe, предложение исходников |
| llama.cpp (`llama-server`) | MIT | |
| onnx-asr | MIT | |
| Inno Setup | фриварная лицензия | установщик |

## Правила

- Не включаем в поставку GigaAM v1 (NC-лицензия).
- Для ffmpeg используем LGPL-сборку; в «О программе» — ссылка на исходники и тексты лицензий.
- Лицензии моделей проверяются перед каждым релизом; изменения фиксируются в CHANGELOG (раздел «Модели»).
