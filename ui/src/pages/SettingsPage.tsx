import { useEffect, useRef, useState } from 'react';
import {
  Badge,
  Button,
  Dropdown,
  Input,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  Option,
  Switch,
} from '@fluentui/react-components';
import {
  ArrowClockwiseRegular,
  DismissRegular,
  FolderRegular,
  SettingsRegular,
} from '@fluentui/react-icons';
import {
  ApiError,
  AuthError,
  getHealth,
  getModels,
  getSettings,
  putSettings,
} from '../api';
import type {
  HealthResponse,
  ModelItem,
  Settings,
  ThemeMode,
} from '../api';
import { ErrorState, LoadingState } from '../components/PageState';
import { PageHeader } from '../components/PageHeader';

const THEME_OPTIONS: ReadonlyArray<{ value: ThemeMode; label: string }> = [
  { value: 'system', label: 'Системная' },
  { value: 'light', label: 'Светлая' },
  { value: 'dark', label: 'Тёмная' },
];

function isThemeMode(value: string | undefined): value is ThemeMode {
  return value === 'system' || value === 'light' || value === 'dark';
}

function describeError(reason: unknown): string {
  return reason instanceof ApiError
    ? reason.message
    : 'Не удалось загрузить настройки.';
}

interface SaveResult {
  intent: 'success' | 'error';
  text: string;
}

interface SettingsPageProps {
  onAuthError: () => void;
  onThemeModeChange: (mode: ThemeMode) => void;
  onManageModels: () => void;
}

export function SettingsPage({
  onAuthError,
  onThemeModeChange,
  onManageModels,
}: SettingsPageProps) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [asrModels, setAsrModels] = useState<ModelItem[]>([]);
  const [allModels, setAllModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveResult, setSaveResult] = useState<SaveResult | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [autosave, setAutosave] = useState<'idle' | 'saving' | 'saved' | 'error'>(
    'idle'
  );
  const savedPayload = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async (): Promise<void> => {
      setLoading(true);
      setLoadError(null);

      const [settingsResult, healthResult, modelsResult] = await Promise.allSettled([
        getSettings(),
        getHealth(),
        getModels(),
      ]);

      if (cancelled) {
        return;
      }

      if (settingsResult.status === 'rejected') {
        setLoading(false);
        if (settingsResult.reason instanceof AuthError) {
          onAuthError();
          return;
        }
        setLoadError(describeError(settingsResult.reason));
        return;
      }

      setSettings(settingsResult.value);
      savedPayload.current = JSON.stringify(settingsResult.value);
      setAutosave('idle');
      onThemeModeChange(settingsResult.value.theme);

      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value);
      } else if (healthResult.reason instanceof AuthError) {
        onAuthError();
      }

      if (modelsResult.status === 'fulfilled') {
        setAsrModels(modelsResult.value.items.filter((model) => model.kind === 'asr'));
        setAllModels(modelsResult.value.items);
      } else if (modelsResult.reason instanceof AuthError) {
        onAuthError();
      }

      setLoading(false);
    };

    void load();

    return () => {
      cancelled = true;
    };
  }, [reloadKey, onAuthError, onThemeModeChange]);

  const update = (patch: Partial<Settings>): void => {
    setSettings((current) => (current === null ? current : { ...current, ...patch }));
    setSaveResult(null);
  };

  const handleThemeChange = (value: string | undefined): void => {
    if (!isThemeMode(value)) {
      return;
    }
    update({ theme: value });
    onThemeModeChange(value);
  };

  // Settings are applied immediately: debounce changes and persist them
  // automatically.  The explicit "Сохранить" button was removed on purpose.
  useEffect(() => {
    if (settings === null) {
      return;
    }
    const payload = JSON.stringify(settings);
    if (payload === savedPayload.current) {
      return;
    }

    setAutosave('saving');
    const timer = setTimeout(() => {
      void (async () => {
        try {
          await putSettings(settings);
          savedPayload.current = payload;
          setAutosave('saved');
          setSaveResult(null);
        } catch (reason: unknown) {
          if (reason instanceof AuthError) {
            onAuthError();
            return;
          }
          setAutosave('error');
          setSaveResult({
            intent: 'error',
            text:
              reason instanceof ApiError
                ? reason.message
                : 'Не удалось сохранить настройки.',
          });
        }
      })();
    }, 400);

    return () => {
      clearTimeout(timer);
    };
  }, [settings, onAuthError]);

  // Model download progress polling

  const asrOptions: Array<{ value: string; label: string }> = asrModels.map((model) => ({
    value: model.id,
    label: model.title,
  }));
  const currentAsr = settings?.asr_model ?? '';
  if (currentAsr !== '' && !asrOptions.some((option) => option.value === currentAsr)) {
    asrOptions.unshift({ value: currentAsr, label: currentAsr });
  }
  const asrLabel =
    asrOptions.find((option) => option.value === currentAsr)?.label ?? currentAsr;

  // LLM models: locally downloaded GGUF files plus models already installed
  // in the user's Ollama instance.
  const ollamaModels = allModels.filter((model) => model.kind === 'ollama');
  const localLlmModels = allModels.filter((model) => model.kind === 'llm-gguf');
  const relevantLlmModels =
    settings?.llm_provider === 'ollama' ? ollamaModels : localLlmModels;
  const llmOptions: Array<{ value: string; label: string }> = [
    { value: 'auto', label: 'Авто (по умолчанию)' },
    ...relevantLlmModels.map((model) => ({
      value: model.kind === 'ollama' ? model.title : model.filename || model.title,
      label:
        model.kind === 'ollama'
          ? `${model.title} · Ollama`
          : `${model.title}${model.installed ? '' : ' · не скачана'}`,
    })),
  ];
  const currentLlm = settings?.llm_model ?? 'auto';
  if (!llmOptions.some((option) => option.value === currentLlm)) {
    llmOptions.unshift({ value: currentLlm, label: currentLlm });
  }
  const llmLabel =
    llmOptions.find((option) => option.value === currentLlm)?.label ?? currentLlm;
  const isExternalLlm = settings?.llm_provider === 'external';
  const usesHttpLlm =
    settings?.llm_provider === 'external' || settings?.llm_provider === 'ollama';

  // Diarization models come from the manifest (kind: "diarization").  The
  // "custom" entry keeps the old local-folder workflow for advanced users.
  const diarizationModels = allModels.filter((model) => model.kind === 'diarization');
  const diarizationOptions: Array<{ value: string; label: string }> = [
    ...diarizationModels.map((model) => ({
      value: model.id,
      label: `${model.title}${model.installed ? '' : ' · не скачана'}`,
    })),
    { value: 'custom', label: 'Своя модель (локальная папка)' },
  ];
  const currentDiarization = settings?.diarize_model ?? '';
  const diarizationValue =
    currentDiarization !== '' ? currentDiarization : diarizationModels[0]?.id ?? 'custom';
  if (!diarizationOptions.some((option) => option.value === diarizationValue)) {
    diarizationOptions.unshift({ value: diarizationValue, label: diarizationValue });
  }
  const diarizationLabel =
    diarizationOptions.find((option) => option.value === diarizationValue)?.label ??
    diarizationValue;
  const selectedDiarizationModel = diarizationModels.find(
    (model) => model.id === diarizationValue,
  );
  const diarizationHint =
    diarizationValue === 'custom'
      ? 'Папка с одной моделью WeSpeaker в формате ONNX (файл .onnx).'
      : selectedDiarizationModel
        ? [
            selectedDiarizationModel.note,
            selectedDiarizationModel.license,
            selectedDiarizationModel.size_mb
              ? `${selectedDiarizationModel.size_mb} МБ`
              : null,
            selectedDiarizationModel.installed
              ? 'скачана'
              : 'загрузится при первом запуске',
          ]
            .filter(Boolean)
            .join(' · ')
        : 'Модель загрузится автоматически при первом запуске.';

  const themeLabel =
    THEME_OPTIONS.find((option) => option.value === settings?.theme)?.label ?? 'Системная';
  const localModelCount = allModels.filter((model) => model.local).length;
  const cachedModelCount = allModels.filter(
    (model) => !model.local && model.location === 'huggingface-cache',
  ).length;

  return (
    <>
      <PageHeader
        title="Настройки"
        subtitle="Изменения применяются и сохраняются сразу"
        actions={
          <span
            className={`settings-autosave settings-autosave-${autosave}`}
            role="status"
            aria-live="polite"
          >
            {autosave === 'saving'
              ? 'Сохранение…'
              : autosave === 'saved'
                ? 'Сохранено'
                : autosave === 'error'
                  ? 'Ошибка сохранения'
                  : ''}
          </span>
        }
      />

      {loading ? (
        <LoadingState label="Загружаем настройки…" />
      ) : loadError !== null || settings === null ? (
        <ErrorState
          title="Настройки недоступны"
          description={loadError ?? 'Ядро не вернуло настройки.'}
          action={
            <Button
              icon={<ArrowClockwiseRegular />}
              onClick={() => {
                setReloadKey((value) => value + 1);
              }}
            >
              Повторить
            </Button>
          }
        />
      ) : (
        <div className="settings-stack">
          {saveResult !== null ? (
            <MessageBar intent={saveResult.intent} className="settings-message">
              <MessageBarBody>{saveResult.text}</MessageBarBody>
              <MessageBarActions
                containerAction={
                  <Button
                    appearance="transparent"
                    icon={<DismissRegular />}
                    aria-label="Скрыть сообщение"
                    onClick={() => {
                      setSaveResult(null);
                    }}
                  />
                }
              />
            </MessageBar>
          ) : null}

          <section className="settings-card surface" aria-labelledby="settings-appearance">
            <h2 id="settings-appearance" className="settings-card-title">
              Внешний вид
            </h2>
            <p className="settings-card-sub">Тема применяется сразу во всём окне.</p>

            <div className="settings-rows">
              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Тема</span>
                  <span className="settings-hint">
                    «Системная» следует настройкам Windows.
                  </span>
                </div>
                <div className="settings-control settings-control-theme">
                  <Dropdown
                    aria-label="Тема"
                    value={themeLabel}
                    selectedOptions={[settings.theme]}
                    onOptionSelect={(_event, data) => {
                      handleThemeChange(data.optionValue);
                    }}
                  >
                    {THEME_OPTIONS.map((option) => (
                      <Option key={option.value} value={option.value}>
                        {option.label}
                      </Option>
                    ))}
                  </Dropdown>
                </div>
              </div>
            </div>
          </section>

          <section className="settings-card surface" aria-labelledby="settings-behaviour">
            <h2 id="settings-behaviour" className="settings-card-title">
              Поведение
            </h2>
            <p className="settings-card-sub">Как приложение ведёт себя при закрытии и завершении задач.</p>

            <div className="settings-rows">
              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Сворачивать в трей при закрытии</span>
                  <span className="settings-hint">Приложение продолжит работу в фоне.</span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Сворачивать в трей при закрытии"
                    checked={settings.close_to_tray}
                    onChange={(_event, data) => {
                      update({ close_to_tray: data.checked });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Уведомления</span>
                  <span className="settings-hint">
                    Нативные уведомления Windows с кнопками «Открыть результат» и
                    «Открыть папку».
                  </span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Уведомления"
                    checked={settings.notifications}
                    onChange={(_event, data) => {
                      update({ notifications: data.checked });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">
                    Распознавать файл немедленно при добавлении
                  </span>
                  <span className="settings-hint">
                    При отключении файлы добавляются в очередь, но не начинаются
                    до нажатия «Распознать».
                  </span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Распознавать файл немедленно при добавлении"
                    checked={settings.recognize_immediately}
                    onChange={(_event, data) => {
                      update({ recognize_immediately: data.checked });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">
                    Пункт «Распознать в AI Recogniser» в контекстном меню
                  </span>
                  <span className="settings-hint">
                    Добавляет команду для аудио и видео в контекстное меню
                    Проводника. Изменение применяется сразу, без перезапуска.
                  </span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Пункт в контекстном меню Проводника"
                    checked={settings.context_menu_enabled}
                    onChange={(_event, data) => {
                      update({ context_menu_enabled: data.checked });
                    }}
                  />
                </div>
              </div>
            </div>
          </section>

          <section className="settings-card surface" aria-labelledby="settings-recognition">
            <h2 id="settings-recognition" className="settings-card-title">
              Распознавание
            </h2>
            <p className="settings-card-sub">Модель по умолчанию для новых задач.</p>

            <div className="settings-rows">
              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Модель распознавания</span>
                  <span className="settings-hint">
                    Список берётся из локального каталога моделей.
                  </span>
                </div>
                <div className="settings-control settings-control-model">
                  <Dropdown
                    aria-label="Модель распознавания"
                    value={asrLabel}
                    selectedOptions={[settings.asr_model]}
                    onOptionSelect={(_event, data) => {
                      if (data.optionValue !== undefined) {
                        update({ asr_model: data.optionValue });
                      }
                    }}
                  >
                    {asrOptions.map((option) => (
                      <Option key={option.value} value={option.value}>
                        {option.label}
                      </Option>
                    ))}
                  </Dropdown>
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Нарезка по сегментам (VAD)</span>
                  <span className="settings-hint">
                    Длинные записи делятся по паузам речи перед распознаванием.
                  </span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Нарезка по сегментам VAD"
                    checked={settings.vad_enabled}
                    onChange={(_event, data) => {
                      update({ vad_enabled: data.checked });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Максимальная длина сегмента</span>
                  <span className="settings-hint">
                    От 5 до 120 секунд. Слишком длинный сегмент не влезет в модель.
                  </span>
                </div>
                <div className="settings-control settings-control-rag-k">
                  <Input
                    type="number"
                    min={5}
                    max={120}
                    aria-label="Максимальная длина сегмента, секунды"
                    value={String(settings.vad_max_segment_s)}
                    disabled={!settings.vad_enabled}
                    onChange={(_event, data) => {
                      const parsed = Number.parseInt(data.value, 10);
                      if (Number.isFinite(parsed)) {
                        update({ vad_max_segment_s: Math.min(120, Math.max(5, parsed)) });
                      }
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Минимальная пауза</span>
                  <span className="settings-hint">
                    Паузы короче этого значения не разделяют сегменты, мс.
                  </span>
                </div>
                <div className="settings-control settings-control-rag-k">
                  <Input
                    type="number"
                    min={50}
                    max={5000}
                    step={50}
                    aria-label="Минимальная пауза, миллисекунды"
                    value={String(settings.vad_min_silence_ms)}
                    disabled={!settings.vad_enabled}
                    onChange={(_event, data) => {
                      const parsed = Number.parseInt(data.value, 10);
                      if (Number.isFinite(parsed)) {
                        update({ vad_min_silence_ms: Math.min(5000, Math.max(50, parsed)) });
                      }
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Отступ вокруг речи</span>
                  <span className="settings-hint">
                    Сколько миллисекунд речи оставлять до и после сегмента.
                  </span>
                </div>
                <div className="settings-control settings-control-rag-k">
                  <Input
                    type="number"
                    min={0}
                    max={1000}
                    step={10}
                    aria-label="Отступ вокруг речи, миллисекунды"
                    value={String(settings.vad_speech_pad_ms)}
                    disabled={!settings.vad_enabled}
                    onChange={(_event, data) => {
                      const parsed = Number.parseInt(data.value, 10);
                      if (Number.isFinite(parsed)) {
                        update({ vad_speech_pad_ms: Math.min(1000, Math.max(0, parsed)) });
                      }
                    }}
                  />
                </div>
              </div>
            </div>
          </section>

          <section className="settings-card surface" aria-labelledby="settings-diarization">
            <h2 id="settings-diarization" className="settings-card-title">
              Диаризация (кто говорит)
            </h2>
            <p className="settings-card-sub">
              Определяет участников разговора по голосу и помечает каждый сегмент
              именем спикера. Модель эмбеддингов (25–115 МБ) можно выбрать ниже;
              она скачивается автоматически при первом использовании.
            </p>

            <div className="settings-rows">
              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Разбор по участникам</span>
                  <span className="settings-hint">
                    Помечать сегменты именами спикеров (Спикер 1, Спикер 2…).
                  </span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Диаризация спикеров"
                    checked={settings.diarize_enabled}
                    onChange={(_event, data) => {
                      update({ diarize_enabled: data.checked });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Порог сходства</span>
                  <span className="settings-hint">
                    От 0.1 до 0.99. Выше — строже группировка по одному спикеру.
                  </span>
                </div>
                <div className="settings-control settings-control-rag-k">
                  <Input
                    type="number"
                    min={0.1}
                    max={0.99}
                    step={0.05}
                    aria-label="Порог сходства диаризации"
                    value={String(settings.diarize_threshold)}
                    disabled={!settings.diarize_enabled}
                    onChange={(_event, data) => {
                      const parsed = Number.parseFloat(data.value);
                      if (Number.isFinite(parsed)) {
                        update({ diarize_threshold: Math.min(0.99, Math.max(0.1, parsed)) });
                      }
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Модель диаризации</span>
                  <span className="settings-hint">{diarizationHint}</span>
                </div>
                <div className="settings-control settings-control-model">
                  <Dropdown
                    aria-label="Модель диаризации"
                    value={diarizationLabel}
                    selectedOptions={[diarizationValue]}
                    disabled={!settings.diarize_enabled}
                    onOptionSelect={(_event, data) => {
                      if (data.optionValue === undefined) {
                        return;
                      }
                      update({
                        diarize_model: data.optionValue,
                        ...(data.optionValue === 'custom'
                          ? {}
                          : { diarize_model_path: '' }),
                      });
                    }}
                  >
                    {diarizationOptions.map((option) => (
                      <Option key={option.value} value={option.value}>
                        {option.label}
                      </Option>
                    ))}
                  </Dropdown>
                </div>
              </div>

              {diarizationValue === 'custom' ? (
                <div className="settings-row">
                  <div className="settings-row-text">
                    <span className="settings-label">Папка с моделью</span>
                    <span className="settings-hint">
                      Локальная папка с одним ONNX-файлом WeSpeaker. Используется
                      вместо модели из каталога.
                    </span>
                  </div>
                  <div className="settings-control settings-control-rag">
                    <Input
                      className="settings-input-wide"
                      aria-label="Папка с моделью диаризации"
                      value={settings.diarize_model_path}
                      disabled={!settings.diarize_enabled}
                      placeholder="Например, D:\models\wespeaker"
                      onChange={(_event, data) => {
                        update({ diarize_model_path: data.value });
                      }}
                    />
                  </div>
                </div>
              ) : null}
            </div>
          </section>

          <section className="settings-card surface" aria-labelledby="settings-llm">
            <h2 id="settings-llm" className="settings-card-title">
              LLM-саммари
            </h2>
            <p className="settings-card-sub">
              Локальная LLM через llama-server (llama.cpp) или внешний OpenAI-совместимый API.
              Саммари и ответы на вопросы по транскрипту.
            </p>

            <div className="settings-rows">
              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Провайдер LLM</span>
                  <span className="settings-hint">
                    «llamacpp-local» — локальный llama-server на этом ПК;
                    «external» — внешний API (только по явному включению).
                  </span>
                </div>
                <div className="settings-control settings-control-model">
                  <Dropdown
                    aria-label="Провайдер LLM"
                    value={settings.llm_provider}
                    selectedOptions={[settings.llm_provider]}
                    onOptionSelect={(_event, data) => {
                      if (data.optionValue !== undefined) {
                        update({ llm_provider: data.optionValue });
                      }
                    }}
                  >
                    <Option key="llamacpp-local" value="llamacpp-local">
                      llama.cpp (локально)
                    </Option>
                    <Option key="ollama" value="ollama">
                      Ollama (локально)
                    </Option>
                    <Option key="external" value="external">
                      Внешний OpenAI-совместимый API
                    </Option>
                  </Dropdown>
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Модель LLM</span>
                  <span className="settings-hint">
                    {settings.llm_provider === 'ollama'
                      ? 'Модели, уже установленные в Ollama.'
                      : 'GGUF-модель для llama-server или модель внешнего API.'}
                  </span>
                </div>
                <div className="settings-control settings-control-model">
                  <Dropdown
                    aria-label="Модель LLM"
                    value={llmLabel}
                    selectedOptions={[currentLlm]}
                    onOptionSelect={(_event, data) => {
                      if (data.optionValue !== undefined) {
                        update({ llm_model: data.optionValue });
                      }
                    }}
                  >
                    {llmOptions.map((option) => (
                      <Option key={option.value} value={option.value}>
                        {option.label}
                      </Option>
                    ))}
                  </Dropdown>
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Составлять саммари автоматически</span>
                  <span className="settings-hint">
                    После успешного распознавания сразу запускать LLM-саммари.
                    Требуется выбранный провайдер и установленная модель.
                  </span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Автоматическая саммаризация"
                    checked={settings.summarize_enabled}
                    onChange={(_event, data) => {
                      update({ summarize_enabled: data.checked });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Адрес внешнего API</span>
                  <span className="settings-hint">
                    Для ollama / OpenAI-совместимого API. Например http://127.0.0.1:11434.
                  </span>
                </div>
                <div className="settings-control settings-control-rag">
                  <Input
                    className="settings-input-wide"
                    aria-label="Адрес внешнего LLM API"
                    value={settings.external_llm_base_url}
                    disabled={!usesHttpLlm}
                    placeholder="http://127.0.0.1:11434"
                    onChange={(_event, data) => {
                      update({ external_llm_base_url: data.value });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Ключ API</span>
                  <span className="settings-hint">
                    Для внешнего API (если требуется авторизация).
                  </span>
                </div>
                <div className="settings-control settings-control-rag">
                  <Input
                    type="password"
                    aria-label="Ключ API LLM"
                    value={settings.external_llm_api_key}
                    disabled={!isExternalLlm}
                    placeholder="••••••••"
                    onChange={(_event, data) => {
                      update({ external_llm_api_key: data.value });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Размер контекста</span>
                  <span className="settings-hint">
                    Контекстное окно модели в токенах (512–131072).
                  </span>
                </div>
                <div className="settings-control settings-control-rag-k">
                  <Input
                    type="number"
                    min={512}
                    max={131072}
                    step={512}
                    aria-label="Размер контекста LLM"
                    value={String(settings.llm_ctx_size)}
                    disabled={usesHttpLlm}
                    onChange={(_event, data) => {
                      const parsed = Number.parseInt(data.value, 10);
                      if (Number.isFinite(parsed)) {
                        update({ llm_ctx_size: Math.min(131072, Math.max(512, parsed)) });
                      }
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Потоки CPU</span>
                  <span className="settings-hint">
                    Количество потоков для llama-server (0 = автоматически).
                  </span>
                </div>
                <div className="settings-control settings-control-rag-k">
                  <Input
                    type="number"
                    min={0}
                    max={256}
                    step={1}
                    aria-label="Потоки CPU для LLM"
                    value={String(settings.llm_threads)}
                    disabled={usesHttpLlm}
                    onChange={(_event, data) => {
                      const parsed = Number.parseInt(data.value, 10);
                      if (Number.isFinite(parsed)) {
                        update({ llm_threads: Math.min(256, Math.max(0, parsed)) });
                      }
                    }}
                  />
                </div>
              </div>
            </div>
          </section>

          <section className="settings-card surface" aria-labelledby="settings-rag">
            <h2 id="settings-rag" className="settings-card-title">
              База знаний
            </h2>
            <p className="settings-card-sub">
              Подключение к локальному ragkb. Интеграцию можно выключить в любой момент.
            </p>

            <div className="settings-rows">
              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Использовать RAG</span>
                  <span className="settings-hint">
                    Поиск и ответы по документам в уже запущенном ragkb.
                  </span>
                </div>
                <div className="settings-control">
                  <Switch
                    aria-label="Использовать RAG"
                    checked={settings.rag_enabled}
                    onChange={(_event, data) => {
                      update({ rag_enabled: data.checked });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Адрес ragkb</span>
                  <span className="settings-hint">
                    Обычно http://127.0.0.1:8000. Сервис должен быть запущен отдельно.
                  </span>
                </div>
                <div className="settings-control settings-control-rag">
                  <Input
                    className="settings-input-wide"
                    aria-label="Адрес ragkb"
                    value={settings.rag_url}
                    disabled={!settings.rag_enabled}
                    placeholder="http://127.0.0.1:8000"
                    onChange={(_event, data) => {
                      update({ rag_url: data.value });
                    }}
                  />
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-text">
                  <span className="settings-label">Фрагментов в ответе</span>
                  <span className="settings-hint">
                    Сколько источников запрашивать у ragkb: от 1 до 20.
                  </span>
                </div>
                <div className="settings-control settings-control-rag-k">
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    aria-label="Количество фрагментов RAG"
                    value={String(settings.rag_k)}
                    disabled={!settings.rag_enabled}
                    onChange={(_event, data) => {
                      const parsed = Number.parseInt(data.value, 10);
                      if (Number.isFinite(parsed)) {
                        update({ rag_k: Math.min(20, Math.max(1, parsed)) });
                      }
                    }}
                  />
                </div>
              </div>
            </div>
          </section>

          <section className="settings-card surface" aria-labelledby="settings-data">
            <div className="settings-card-head">
              <h2 id="settings-data" className="settings-card-title">
                Данные
              </h2>
              {health?.portable === true ? (
                <Badge appearance="tint" color="informative" size="small">
                  Portable
                </Badge>
              ) : null}
            </div>
            <p className="settings-card-sub">
              Модели, кэш, логи и результаты. Путь задаётся при запуске приложения.
            </p>

            <label className="settings-label data-dir-label" htmlFor="data-dir">
              Папка данных
            </label>
            <Input
              id="data-dir"
              className="data-dir-input"
              readOnly
              value={health?.data_dir ?? 'Недоступно'}
              title={health?.data_dir}
              contentBefore={<FolderRegular />}
              aria-label="Папка данных"
            />
          </section>

          <section className="settings-card surface" aria-labelledby="settings-models">
            <div className="settings-card-head">
              <h2 id="settings-models" className="settings-card-title">
                Модели и ИИ
              </h2>
              <Badge appearance="tint" color="informative" size="small">
                {localModelCount} / {allModels.length}
              </Badge>
            </div>
            <p className="settings-card-sub">
              Установка и проверка моделей, прогресс скачивания и реальные файлы
              на диске.
            </p>

            <div className="settings-manage-models">
              <div className="settings-row-text">
                <span className="settings-label">Менеджер ИИ-моделей</span>
                <span className="settings-hint">
                  Локально установлено: {localModelCount}. В кэше Hugging Face: {cachedModelCount}.
                  Здесь же можно скачать модель в папку данных и открыть её расположение.
                </span>
              </div>
              <Button
                appearance="primary"
                icon={<SettingsRegular />}
                onClick={onManageModels}
              >
                Менеджер ИИ-моделей
              </Button>
            </div>
          </section>

          <p className="settings-footnote">
            Настройки хранятся локально и применяются сразу после изменения.
          </p>
        </div>
      )}
    </>
  );
}
