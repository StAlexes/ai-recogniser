import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Badge,
  Button,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  ProgressBar,
  Spinner,
} from '@fluentui/react-components';
import {
  ArrowClockwiseRegular,
  ArrowLeftRegular,
  ArrowDownloadRegular,
  DismissRegular,
  FolderOpenRegular,
} from '@fluentui/react-icons';
import {
  ApiError,
  AuthError,
  downloadModel,
  getModels,
  revealModelsFolder,
} from '../api';
import type { ModelItem } from '../api';
import { PageHeader } from '../components/PageHeader';
import { ErrorState, LoadingState } from '../components/PageState';

interface ModelsPageProps {
  onBack: () => void;
  onAuthError: () => void;
}

const KIND_ORDER = ['asr', 'diarization', 'runtime', 'llm-gguf', 'vad'] as const;

const KIND_LABELS: Record<string, string> = {
  asr: 'Распознавание речи',
  diarization: 'Диаризация (спикеры)',
  runtime: 'Среда llama.cpp (llama-server)',
  'llm-gguf': 'Языковые модели для саммари',
  vad: 'Голосовая активность (VAD)',
  ollama: 'Ollama (локальные модели)',
};

function kindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind;
}

function formatSize(model: ModelItem): string {
  const mb =
    model.size_mb ??
    (model.size_bytes !== undefined ? model.size_bytes / (1024 * 1024) : undefined);
  if (mb === undefined || !Number.isFinite(mb) || mb <= 0) {
    return '—';
  }
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1).replace('.', ',')} ГБ`;
  }
  return `${mb < 10 ? mb.toFixed(1).replace('.', ',') : Math.round(mb)} МБ`;
}

function downloadPhase(model: ModelItem): { value: number; phase: string; failed: boolean } | null {
  const progress = model.download_progress;
  if (!progress) {
    return null;
  }
  return {
    value: Math.max(0, Math.min(progress.value, 1)),
    phase: progress.phase,
    failed: progress.status === 'error' || progress.phase.toLowerCase().startsWith('ошибка'),
  };
}

function installedLabel(model: ModelItem): { text: string; color: 'success' | 'informative' | 'subtle' } {
  if (model.kind === 'runtime' && model.installed) {
    return { text: 'llama-server установлен', color: 'success' };
  }
  if (model.local) {
    return { text: 'Установлена в папке данных', color: 'success' };
  }
  if (model.location === 'huggingface-cache') {
    return { text: 'В кэше Hugging Face', color: 'informative' };
  }
  if (model.kind === 'ollama' && model.installed) {
    return { text: 'Доступна в Ollama', color: 'success' };
  }
  return { text: 'Не установлена', color: 'subtle' };
}

export function ModelsPage({ onBack, onAuthError }: ModelsPageProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [activeIds, setActiveIds] = useState<Set<string>>(() => new Set());

  const loadModels = useCallback(
    async (initial = false): Promise<void> => {
      try {
        const response = await getModels();
        setModels(response.items);
        setError(null);
        setActiveIds((current) => {
          if (current.size === 0) {
            return current;
          }
          const next = new Set<string>();
          for (const id of current) {
            const item = response.items.find((candidate) => candidate.id === id);
            if (!item || item.local) {
              continue;
            }
            const progress = item.download_progress;
            if (progress?.status === 'error' || progress?.status === 'done') {
              continue;
            }
            next.add(id);
          }
          return next;
        });
      } catch (reason: unknown) {
        if (reason instanceof AuthError) {
          onAuthError();
          return;
        }
        setError(
          reason instanceof ApiError
            ? reason.message
            : 'Не удалось получить список моделей.',
        );
      } finally {
        if (initial) {
          setLoading(false);
        }
      }
    },
    [onAuthError],
  );

  useEffect(() => {
    void loadModels(true);
    const timer = window.setInterval(() => {
      void loadModels(false);
    }, 2000);
    return () => {
      window.clearInterval(timer);
    };
  }, [loadModels]);

  const grouped = useMemo(() => {
    const groups = new Map<string, ModelItem[]>();
    for (const model of models) {
      const list = groups.get(model.kind) ?? [];
      list.push(model);
      groups.set(model.kind, list);
    }
    const order = [...KIND_ORDER];
    for (const kind of groups.keys()) {
      if (!order.includes(kind as (typeof KIND_ORDER)[number])) {
        order.push(kind as (typeof KIND_ORDER)[number]);
      }
    }
    return order
      .filter((kind) => groups.has(kind))
      .map((kind) => ({ kind, items: groups.get(kind) ?? [] }));
  }, [models]);

  const installedCount = models.filter((model) => model.local).length;
  const cachedCount = models.filter(
    (model) => !model.local && model.location === 'huggingface-cache',
  ).length;

  const refresh = async (): Promise<void> => {
    setRefreshing(true);
    await loadModels(false);
    setRefreshing(false);
  };

  const openFolder = async (): Promise<void> => {
    try {
      await revealModelsFolder();
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(
        reason instanceof ApiError
          ? reason.message
          : 'Не удалось открыть папку моделей.',
      );
    }
  };

  const install = async (model: ModelItem): Promise<void> => {
    setError(null);
    setNotice(null);
    setActiveIds((current) => new Set(current).add(model.id));
    try {
      await downloadModel(model.id);
      setNotice(`Скачивание «${model.title}» началось.`);
      window.setTimeout(() => {
        void loadModels(false);
      }, 700);
    } catch (reason: unknown) {
      setActiveIds((current) => {
        const next = new Set(current);
        next.delete(model.id);
        return next;
      });
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(
        reason instanceof ApiError
          ? reason.message
          : `Не удалось начать скачивание «${model.title}».`,
      );
    }
  };

  return (
    <>
      <PageHeader
        title="Менеджер ИИ-моделей"
        subtitle="Реальное скачивание и проверка локальных файлов моделей"
        badge={
          <Badge appearance="tint" color="informative">
            {installedCount} локально
          </Badge>
        }
        actions={
          <div className="queue-actions">
            <Button icon={<ArrowLeftRegular />} onClick={onBack}>
              Настройки
            </Button>
            <Button
              icon={refreshing ? <Spinner size="tiny" /> : <ArrowClockwiseRegular />}
              disabled={refreshing}
              onClick={() => {
                void refresh();
              }}
            >
              Проверить
            </Button>
            <Button icon={<FolderOpenRegular />} onClick={() => void openFolder()}>
              Папка моделей
            </Button>
          </div>
        }
      />

      <div className="models-intro surface">
        <p>
          Модели хранятся в папке данных приложения. Скачивание идёт напрямую с
          Hugging Face с докачкой при обрыве связи; после завершения файл
          проверяется и становится доступен приложению.
        </p>
        <p className="models-intro-meta">
          Локально: {installedCount} · в кэше Hugging Face: {cachedCount} · всего в
          каталоге: {models.length}
        </p>
      </div>

      {notice !== null ? (
        <MessageBar intent="success" className="queue-notice">
          <MessageBarBody>{notice}</MessageBarBody>
          <MessageBarActions
            containerAction={
              <Button
                appearance="transparent"
                icon={<DismissRegular />}
                aria-label="Скрыть сообщение"
                onClick={() => {
                  setNotice(null);
                }}
              />
            }
          />
        </MessageBar>
      ) : null}

      {error !== null ? (
        <MessageBar intent="error" className="queue-notice">
          <MessageBarBody>{error}</MessageBarBody>
          <MessageBarActions
            containerAction={
              <Button
                appearance="transparent"
                icon={<DismissRegular />}
                aria-label="Скрыть ошибку"
                onClick={() => {
                  setError(null);
                }}
              />
            }
          />
        </MessageBar>
      ) : null}

      {loading ? (
        <LoadingState label="Загрузка каталога моделей…" />
      ) : models.length === 0 ? (
        <ErrorState
          title="Каталог пуст"
          description="В сборке отсутствует manifest.json с описанием моделей."
        />
      ) : (
        <div className="models-groups">
          {grouped.map((group) => (
            <section
              key={group.kind}
              className="models-group surface"
              aria-labelledby={`models-group-${group.kind}`}
            >
              <div className="models-group-head">
                <h2 id={`models-group-${group.kind}`} className="models-group-title">
                  {kindLabel(group.kind)}
                </h2>
                <Badge appearance="tint" color="informative" size="small">
                  {group.items.length}
                </Badge>
              </div>
              <ul className="models-list">
                {group.items.map((model) => {
                  const state = installedLabel(model);
                  const progress = downloadPhase(model);
                  const downloading =
                    !model.local &&
                    (activeIds.has(model.id) ||
                      progress?.phase === 'Скачивание модели' ||
                      progress?.phase === 'Подготовка' ||
                      progress?.phase === 'Проверка контрольной суммы');
                  const canInstall =
                    !model.local &&
                    model.kind !== 'ollama' &&
                    typeof model.url === 'string' &&
                    model.url !== '';
                  return (
                    <li key={model.id} className="model-row">
                      <div className="model-main">
                        <div className="model-title-row">
                          <span className="model-title">{model.title}</span>
                          <Badge
                            appearance="tint"
                            color={state.color}
                            size="small"
                          >
                            {state.text}
                          </Badge>
                        </div>
                        <p className="model-meta">
                          {formatSize(model)}
                          {model.purpose ? ` · ${model.purpose}` : ''}
                          {model.license ? ` · ${model.license}` : ''}
                        </p>
                        {model.note ? <p className="model-note">{model.note}</p> : null}
                        {model.location === 'huggingface-cache' ? (
                          <p className="model-note">
                            Модель уже используется из общего кэша Hugging Face.
                            Кнопка ниже скачает её в папку данных приложения.
                          </p>
                        ) : null}
                        {progress !== null && !model.local ? (
                          <div
                            className="model-progress"
                            data-failed={progress.failed ? 'true' : 'false'}
                          >
                            <ProgressBar
                              value={progress.failed ? 1 : progress.value}
                              thickness="medium"
                            />
                            <span className="model-progress-text">
                              {progress.failed
                                ? progress.phase
                                : `${progress.phase} · ${Math.round(progress.value * 100)}%`}
                            </span>
                          </div>
                        ) : null}
                      </div>
                      <div className="model-actions">
                        {model.local || (model.kind === 'ollama' && model.installed) ? (
                          <Button size="small" appearance="subtle" disabled>
                            {model.kind === 'ollama' ? 'В Ollama' : 'Установлена'}
                          </Button>
                        ) : (
                          <Button
                            size="small"
                            appearance="primary"
                            icon={
                              downloading ? <Spinner size="tiny" /> : <ArrowDownloadRegular />
                            }
                            disabled={downloading || !canInstall}
                            onClick={() => {
                              void install(model);
                            }}
                          >
                            {downloading
                              ? 'Скачивание…'
                              : !canInstall
                                ? 'Нет ссылки'
                                : progress?.failed
                                  ? 'Повторить'
                                  : model.location === 'huggingface-cache'
                                    ? 'Скачать локально'
                                    : 'Установить'}
                          </Button>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))}
        </div>
      )}
    </>
  );
}
