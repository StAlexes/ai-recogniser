## GigaAM POC (M0)

**Дата:** 2026-10-23
**Машина:** Windows 11, AMD Ryzen 7 8845H (8 ядер / 16 потоков), 31.31 ГБ RAM,
iGPU AMD Radeon 780M (драйвер 32.0.31041.1004; в замере не участвовал).
**Окружение:** Python 3.13.15 (venv), onnx-asr 0.12.0, onnxruntime 1.31.0 (CPU-сборка), numpy 2.5.4.
**Модель:** `gigaam-v3-e2e-ctc`, `quantization="int8"` — 224.9 МБ, HF-репозиторий
`istupakov/gigaam-v3-onnx` (лицензия MIT), кэш `%USERPROFILE%\.cache\huggingface\hub\models--istupakov--gigaam-v3-onnx`.
**Сэмпл:** `data/samples/gigaam-example.wav` — официальный короткий пример GigaAM
(`https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav`, ссылка из colab/README GigaAM),
11.29 с, 16 кГц, моно, PCM16, 361 324 байта,
SHA-256 `D8AAAA18A5098D7C6DE0595AE7AC1E64CACD0D4022AF3595213BDAF23BE77E69`.

| Провайдер | Load, с | Infer, с (mean) | RTFx | Peak RSS, МБ | Длина текста |
|---|---|---|---|---|---|
| CPU (`CPUExecutionProvider`), int8 | 1.674 | 0.199 (прогоны 0.197 / 0.201) | 56.8 | 352.5 | 160 симв. / 26 слов |
| DirectML (`DmlExecutionProvider`), int8, отдельный venv | 2.596 | 0.229 | 49.3 | 621.5 | 161 симв. / 26 слов |

Warmup: 0.224 с (не входит в замер). Cold load: 18.562 с, включая скачивание 214.9 МБ
(3 файла: ONNX int8, yaml, vocab); загрузка с прогретого кэша — 1.674 с.
Peak RSS процесса за всё время жизни — 359.4 МБ (включая скачивание), после загрузки модели — 325.6 МБ.

**Команды:**

```powershell
# 1) прогрев кэша модели (скачивание ~215 МБ, разово)
.\.venv\Scripts\python.exe tools\poc_gigaam.py --wav data\samples\gigaam-example.wav `
  --download-only --json data\poc\gigaam-cold.json

# 2) замер CPU int8 (warmup + 2 тайминговых прогона)
.\.venv\Scripts\python.exe tools\poc_gigaam.py --wav data\samples\gigaam-example.wav `
  --provider cpu --runs 2 --json data\poc\gigaam-result.json --transcript-out data\poc\transcript.txt

# 3) проверка доступных EP (DirectML)
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
# -> ['AzureExecutionProvider', 'CPUExecutionProvider']

# 4) DirectML в отдельном venv (не заменяет CPU-runtime в основном)
python -m venv .venv-dml
.\.venv-dml\Scripts\python.exe -m pip install onnx-asr onnxruntime-directml huggingface-hub numpy psutil
.\.venv-dml\Scripts\python.exe tools\poc_gigaam.py --wav data\samples\gigaam-example.wav `
  --provider dml --runs 3 --json data\poc\gigaam-dml.json
```

**Транскрипт (первые 300 символов, файл — весь):**

> Ничьих, не требуя похвал, счастлив уж я надеждой сладкой, Что дева с трепетом любви посмотрит, может быть украдкой На песни грешные мои. У лукоморья дуб зелёный

Пунктуация и регистр присутствуют «из коробки» (e2e CTC); текст распознан как связная поэтическая речь.

**Оговорки:**

- **DirectML проверен отдельно:** в основном venv стоит CPU-сборка onnxruntime 1.31.0 без `DmlExecutionProvider`.
  В отдельном `.venv-dml` с `onnxruntime-directml` 1.24.4 EP доступен и работает: RTFx **49.3** против **56.8** на CPU,
  peak RSS **621 МБ** против **352 МБ**, загрузка 2.6 с против 1.7 с. Вывод: на Radeon 780M iGPU-ускорение GigaAM v3 int8
  **не даёт выигрыша**; дефолт — CPU. `onnxruntime-directml` в поставку не берём (плюс ~200 МБ и хуже latency/RAM).
  Оговорка: версии ORT разные (1.24.4 vs 1.31.0), но разрыв по памяти и скорости устойчивый.
- **Короткий сэмпл:** 11.3 с, без VAD, чанков и longform-пайплайна. RTFx 56.8 на коротком файле
  близок к паспортным ≈59 (9800X3D) и выше ожидаемых 25–40 из PLAN §5.1; на длинных файлах
  с VAD/нарезкой и постобработкой цифра будет другой — линейно не экстраполировать.
- **Статистика:** 2 тайминговых прогона, один процесс, без закрепления потоков и фоновой нагрузки;
  разброс между прогонами ~2%.
- **Лицензия сэмпла:** код и веса GigaAM — MIT, но отдельная лицензия самого demo-WAV на CDN Sber
  явно не указана. Для поставки и golden-корпуса использовать собственные/лицензированные записи,
  этот файл в дистрибутив не включать.
- **API:** `onnx_asr.load_model(model, quantization=..., providers=[...])`; `model.recognize(path)`
  принимает путь к WAV и возвращает `str` (для списка — `list[str]`). `--provider auto` выбирает DML,
  если EP доступен, иначе CPU.
- **Кодировка консоли:** JSON и transcript пишутся в UTF-8; консоль Windows в cp866 показывает
  кириллицу как «кракозябры» — смотреть файлы или `chcp 65001`.
- **Python:** в venv 3.13.15, тогда как PLAN/README заявляют 3.12+; на результат не влияет.
