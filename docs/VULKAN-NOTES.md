# Vulkan и iGPU: что проверено и что решено

**Дата:** 2026-10-23  
**Машина:** Ryzen 7 8845H, 32 ГБ RAM, Radeon 780M (iGPU, общая память)  
**Вопрос:** имеет ли смысл переводить модели на Vulkan, чтобы ускорить приложение?

## Коротко

| Направление | Решение | Почему |
|---|---|---|
| GigaAM v3 ASR | **CPU int8, оставить** | официального Vulkan EP у ONNX Runtime нет; DirectML на 780M медленнее CPU; WebGPU EP на Windows идёт через D3D12, а не Vulkan |
| Локальная LLM | **Vulkan — опция для `llama.cpp`**, не базовая зависимость | Vulkan-бэкенд `llama.cpp` реален и работает на iGPU, но выигрыш зависит от модели, контекста и охлаждения; нужен отдельный бенчмарк |
| Whisper | **не добавлять сейчас** | `whisper.cpp` умеет Vulkan, но для 780M есть сообщения о падениях на beam search; качество русского хуже GigaAM, а вес больше |
| ONNX Runtime WebGPU | **эксперимент, не путь к Vulkan** | плагинный EP существует, но на Windows использует Dawn/D3D12; покрытие операторов и поддержка GigaAM не гарантированы |

## 1. GigaAM и ONNX Runtime

ONNX Runtime не предоставляет официальный Vulkan Execution Provider. В списке
[Execution Providers](https://onnxruntime.ai/docs/execution-providers/) есть CPU, CUDA,
TensorRT, DirectML, OpenVINO, ROCm, WebGPU и другие, но не Vulkan.

Для Windows практические варианты:

- **CPUExecutionProvider** — текущий режим.
- **DmlExecutionProvider (DirectML)** — работает через D3D12, не Vulkan.
- **onnxruntime-ep-webgpu** — отдельный плагин; на Windows WebGPU реализуется через
  Dawn/D3D12, то есть это тоже не нативный Vulkan. Кроме того, нужно проверять покрытие
  операторов конкретной модели.

Замер M0 на референсном ноутбуке (GigaAM v3 e2e CTC int8, клип 11,29 с):

| Провайдер | Инференс | RTFx | Peak RSS |
|---|---:|---:|---:|
| CPU | 0,199 с | **56,8** | **352 МБ** |
| DirectML | 0,229 с | 49,3 | 622 МБ |

Вывод: **для GigaAM CPU остаётся быстрее и заметно экономнее по памяти**. Переход на
DirectML/WebGPU не даёт выигрыша на Radeon 780M и усложняет поставку.

## 2. Локальная LLM

`llama.cpp` имеет Vulkan-бэкенд и является реалистичным способом задействовать iGPU для
GGUF-моделей. Это не то же самое, что ONNX Runtime: для LLM нужен отдельный `llama-server`
или встроенная библиотека, а не ASR-рантайм.

Что учитывать:

- Radeon 780M использует общую системную память; прирост зависит от размера модели и контекста.
- На 780M результаты Vulkan и ROCm заметно различаются; ROCm может быть быстрее, но хуже
  подходит для универсальной portable-поставки.
- Для русского языка кандидаты — T-lite-it-2.1 8B и Qwen3-4B в GGUF; перед включением
  нужен замер prompt processing / token generation и RSS на этой же машине.
- Vulkan-сборка `llama.cpp` увеличивает размер поставки и требует отдельных лицензионных
  текстов; её нельзя делать обязательной для всех пользователей.

**Решение:** в v1.0 LLM-рантайм проектируется с возможностью выбрать CPU или Vulkan, но
Vulkan-бэкенд включается только после отдельного бенчмарка и не блокирует ASR-сценарий.

## 3. Whisper

`whisper.cpp` поддерживает Vulkan, и в 2026 году появились сборки с заметным ускорением на
GPU. Однако:

- для GigaAM v3 русское качество и размер модели лучше;
- на Radeon 780M есть известные проблемы с Vulkan и большим beam size
  ([whisper.cpp #3723](https://github.com/ggml-org/whisper.cpp/issues/3723));
- Whisper large-v3-turbo весит ~1,6 ГБ против 214 МБ у GigaAM int8.

**Решение:** Whisper остаётся опциональным мультиязычным fallback на будущее, но не
добавляется в текущую сборку и не является причиной для Vulkan-интеграции.

## 4. Что делать дальше

1. Оставить GigaAM на CPU int8 и не тратить время на Vulkan EP для ASR.
2. Если понадобится ускорить LLM на iGPU — сделать отдельный спайк `llama.cpp` Vulkan:
   замерить T-lite 8B q4 и Qwen3-4B q4, prompt processing, token/s, RSS и температуру.
3. WebGPU EP проверять только как эксперимент на копии модели; не включать в релиз без
   подтверждённого покрытия операторов и стабильности.
4. Перед включением Vulkan в поставку — проверить лицензии `llama.cpp`, драйверов и
   сопутствующих библиотек.

## Ссылки

- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [onnxruntime-ep-webgpu на PyPI](https://pypi.org/project/onnxruntime-ep-webgpu/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Замеры GigaAM M0](M0-NOTES.md)
