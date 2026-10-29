# Third-party components and licenses

Актуально для M1 (2026-10-23). Перед публичным релизом фиксируем точные версии, ссылки и тексты
лицензий; спорные компоненты отмечены явно.

## Проект

AI Recogniser распространяется по лицензии MIT (см. `LICENSE`).

## Runtime / libraries

| Компонент | Назначение | Лицензия | Примечание |
|---|---|---|---|
| FastAPI | HTTP API ядра | MIT | |
| Uvicorn | ASGI-сервер | BSD-3-Clause | |
| Pydantic | валидация настроек/схем | MIT | |
| platformdirs | пути данных | MIT | |
| psutil | телеметрия железа | BSD-3-Clause | |
| httpx | HTTP-клиент RAG | BSD-3-Clause | |
| python-multipart | multipart-загрузка файлов | Apache-2.0 | |
| pywebview | окно WebView2 | BSD-3-Clause | |
| pythonnet / clr-loader | загрузка .NET-сборок WebView2 | MIT | |
| WebView2 Runtime | движок окна | Microsoft Software License Terms | отдельный компонент Windows, не входит в exe |
| pystray | иконка в трее | LGPL-3.0 | динамическая линковка, исходники доступны |
| Pillow | иконки | HPND | |
| React / Fluent UI v9 | интерфейс | MIT | |
| Vite | сборка UI | MIT | |
| onnx-asr | загрузка и запуск ASR-моделей | MIT | |
| ONNX Runtime | инференс ASR на CPU | MIT | |
| NumPy | массивы/аудио | BSD-3-Clause | |
| huggingface-hub | загрузка моделей GigaAM и Silero VAD | Apache-2.0 | |
| PyYAML | конфигурация onnx-asr | MIT | |
| certifi | CA-сертификаты httpx | MPL-2.0 | |

## Models (weights)

| Модель | Лицензия | Использование |
|---|---|---|
| GigaAM v3 e2e CTC (ai-sage / salute-developers) | MIT | основная ASR-модель, 214 МБ int8, скачивается при первом запуске |
| GigaAM Multilingual (220M/600M) | MIT | опциональный ASR для KZ/KG/UZ/EN (план) |
| Silero VAD v5 (ONNX, `istupakov/silero-vad-onnx`) | MIT | нарезка длинных записей; ~2 МБ, скачивается при первом длинном файле |
| PyAnnote Segmentation 3.0 (`onnx-community/pyannote-segmentation-3.0`, ONNX) | MIT | диаризация по умолчанию: end-to-end разметка до 3 спикеров; ~6 МБ, скачивается при первом запуске |
| WeSpeaker VoxCeleb (ResNet34 / CAM++ / ECAPA-TDNN512 / ResNet293-LM), ONNX | Apache-2.0 | альтернативная диаризация: эмбеддинги спикеров + cosine-кластеризация, скачиваются по выбору |
| multilingual-e5 | MIT | эмбеддинги базы знаний (план) |
| Qwen3-4B-Instruct | Apache-2.0 | локальная LLM, лёгкий профиль (план) |
| T-lite-it-2.1 8B | **проверить до релиза** | локальная LLM для RU (план) |
| YandexGPT-5-Lite-8B | **проверить до релиза** | альтернатива (план) |
| GigaChat Audio 10B-A1.8B | MIT | эксперимент после 1.0, не в поставке |
| gemma3:4b (через ragkb/Ollama) | Gemma Terms of Use | внешний локальный RAG-сервис, не входит в поставку |

## Tools

| Компонент | Лицензия | Примечание |
|---|---|---|
| FFmpeg / ffprobe | **LGPLv3** | не входит в поставку: LGPL-сборка BtbN (n8.1.3, shared, без `--enable-gpl`) скачивается по требованию при первом не-WAV файле |
| llama.cpp (`llama-server`) | MIT | LLM-рантайм; скачивается по требованию |
| Python | PSF License | рантайм внутри PyInstaller-сборки |
| PyInstaller | GPL-2.0-or-later с исключением для bootloader | только сборка exe; скомпилированный продукт не становится GPL |
| Inno Setup | фриварная лицензия | будущий установщик |

## FFmpeg: лицензии

FFmpeg **не входит в поставку**. Ранее собиравшийся `imageio-ffmpeg 0.6.0`
(`ffmpeg-win-x86_64-v7.1.exe` от gyan.dev) был собран с `--enable-gpl
--enable-version3`, то есть был **GPLv3-бинарником**, и его бандлинг делал всю
поставку предметом GPLv3. Этот бинарник и зависимость `imageio-ffmpeg` удалены.

Вместо этого приложение скачивает по требованию **LGPL-сборку** FFmpeg
(BtbN FFmpeg-Builds, неизменяемый тег `autobuild-2026-10-28-19-47`,
`ffmpeg-n8.1.3-25-g611cec792a-win64-lgpl-shared-8.1.zip`, лицензия LGPLv3,
без `--enable-gpl` и без x264/x265). Архив проверяется по размеру и SHA-256 и
распаковывается в папку данных. Проверка `ffmpeg -buildconf` подтверждает
отсутствие `--enable-gpl`. PCM WAV обрабатывается без FFmpeg.

## Правила

- Не включаем в поставку GigaAM v1 (NC-лицензия).
- Для ffmpeg цель — LGPL-сборка; в «О программе» — ссылка на исходники и тексты лицензий.
- `ragkb` и его индекс не входят в поставку; пользователь запускает сервис отдельно и отвечает
  за лицензии его моделей и документов.
- Лицензии моделей проверяются перед каждым релизом; изменения фиксируются в CHANGELOG (раздел
  «Модели»).
