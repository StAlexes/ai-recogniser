import { useEffect, useState } from 'react';
import {
  Badge,
  Button,
  Dropdown,
  Input,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  Option,
  Spinner,
  Switch,
} from '@fluentui/react-components';
import {
  ArrowClockwiseRegular,
  DismissRegular,
  FolderRegular,
  SaveRegular,
} from '@fluentui/react-icons';
import {
  ApiError,
  AuthError,
  getHealth,
  getModels,
  getSettings,
  putSettings,
} from '../api';
import type { HealthResponse, ModelInfo, Settings, ThemeMode } from '../api';
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
}

export function SettingsPage({ onAuthError, onThemeModeChange }: SettingsPageProps) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [asrModels, setAsrModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<SaveResult | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

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
      onThemeModeChange(settingsResult.value.theme);

      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value);
      } else if (healthResult.reason instanceof AuthError) {
        onAuthError();
      }

      if (modelsResult.status === 'fulfilled') {
        setAsrModels(modelsResult.value.items.filter((model) => model.kind === 'asr'));
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

  const save = async (): Promise<void> => {
    if (settings === null || saving) {
      return;
    }

    setSaving(true);
    setSaveResult(null);

    try {
      const updated = await putSettings(settings);
      setSettings(updated);
      onThemeModeChange(updated.theme);
      setSaveResult({ intent: 'success', text: 'Настройки сохранены.' });
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setSaveResult({
        intent: 'error',
        text:
          reason instanceof ApiError
            ? reason.message
            : 'Не удалось сохранить настройки.',
      });
    } finally {
      setSaving(false);
    }
  };

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
  const themeLabel =
    THEME_OPTIONS.find((option) => option.value === settings?.theme)?.label ?? 'Системная';

  return (
    <>
      <PageHeader
        title="Настройки"
        subtitle="Поведение приложения, модель распознавания и папка данных"
        actions={
          <Button
            appearance="primary"
            icon={saving ? <Spinner size="tiny" /> : <SaveRegular />}
            disabled={settings === null || loading || saving}
            onClick={() => {
              void save();
            }}
          >
            {saving ? 'Сохранение…' : 'Сохранить'}
          </Button>
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
                    Показывать уведомление о завершении задачи.
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
              именем спикера. Требует загрузки модели эмбеддингов (~50 МБ).
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
                  <span className="settings-label">Путь к модели диаризации</span>
                  <span className="settings-hint">
                    Папка с моделью Wespeaker ONNX (оставьте пустым для автоматической загрузки).
                  </span>
                </div>
                <div className="settings-control settings-control-rag">
                  <Input
                    className="settings-input-wide"
                    aria-label="Путь к модели диаризации"
                    value={settings.diarize_model_path}
                    disabled={!settings.diarize_enabled}
                    placeholder="Пусто — автозагрузка из Hugging Face"
                    onChange={(_event, data) => {
                      update({ diarize_model_path: data.value });
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

          <p className="settings-footnote">
            Настройки хранятся локально и применяются после нажатия «Сохранить».
          </p>
        </div>
      )}
    </>
  );
}
