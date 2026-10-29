import { useCallback, useEffect, useState } from 'react';
import {
  Badge,
  Button,
  Dropdown,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  Option,
  ProgressBar,
  Spinner,
  Textarea,
} from '@fluentui/react-components';
import {
  ArrowClockwiseRegular,
  ArrowLeftRegular,
  CopyRegular,
  DismissRegular,
  FolderOpenRegular,
  HistoryRegular,
  PlayRegular,
  SaveRegular,
  SparkleRegular,
  StopRegular,
} from '@fluentui/react-icons';
import {
  AuthError,
  cancelJob,
  getJob,
  retryJob,
  revealJobFolder,
  stageAction,
  startJob,
  token,
} from '../api';
import type { JobDetail, StageAction, StageKey } from '../api';
import { PageHeader } from '../components/PageHeader';
import { openResultFolder, saveResultAs } from '../desktop';
import { formatBytes } from '../format';
import {
  describeError,
  formatDate,
  formatDuration,
  formatSeconds,
  isActive,
  jobElapsedSeconds,
  STAGE_LABELS,
  stageStatusText,
  statusColor,
  statusLabel,
  transcriptText,
} from '../jobUtils';

type ExportFormat = 'txt' | 'srt' | 'vtt' | 'md' | 'json';

const EXPORT_EXTENSIONS: Record<ExportFormat, string> = {
  txt: '.txt',
  srt: '.srt',
  vtt: '.vtt',
  md: '.md',
  json: '.json',
};

const SUMMARY_MAX_POINTS = 8;

interface JobPageProps {
  jobId: string;
  onBack: () => void;
  onAuthError: () => void;
}

function summaryList(value: unknown): any[] {
  return Array.isArray(value) ? value.slice(0, SUMMARY_MAX_POINTS) : [];
}

function SummaryPanel({ summary }: { summary: Record<string, any> }) {
  const topics = summaryList(summary.topics);
  const decisions = summaryList(summary.decisions);
  const actionItems = summaryList(summary.action_items);
  const risks = summaryList(summary.risks);
  return (
    <section className="summary-panel surface" aria-labelledby="job-summary-title">
      <h2 id="job-summary-title" className="summary-heading">
        Саммари
      </h2>
      <div className="summary-tldr">{summary.tl_dr || 'Нет краткого содержания.'}</div>
      {topics.length > 0 ? (
        <>
          <div className="summary-subtitle">Темы</div>
          <ul className="summary-list">
            {topics.map((topic, index) => (
              <li key={index}>{String(topic)}</li>
            ))}
          </ul>
        </>
      ) : null}
      {decisions.length > 0 ? (
        <>
          <div className="summary-subtitle">Решения</div>
          <ul className="summary-list">
            {decisions.map((decision, index) => (
              <li key={index}>
                {String(decision?.formulation ?? decision)}
                {decision?.timecode ? (
                  <span className="segment-timer">[{String(decision.timecode)}]</span>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}
      {actionItems.length > 0 ? (
        <>
          <div className="summary-subtitle">Задачи</div>
          <ul className="summary-list">
            {actionItems.map((item, index) => (
              <li key={index}>
                {String(item?.formulation ?? item)}
                {item?.owner ? (
                  <span className="summary-owner">({String(item.owner)})</span>
                ) : null}
                {item?.timecode ? (
                  <span className="segment-timer">[{String(item.timecode)}]</span>
                ) : null}
              </li>
            ))}
          </ul>
        </>
      ) : null}
      {risks.length > 0 ? (
        <>
          <div className="summary-subtitle">Риски</div>
          <ul className="summary-list">
            {risks.map((risk, index) => (
              <li key={index}>{String(risk)}</li>
            ))}
          </ul>
        </>
      ) : null}
    </section>
  );
}

function StageControl({
  name,
  stage,
  canStart,
  busy,
  onAction,
}: {
  name: StageKey;
  stage: JobDetail['stages'][StageKey];
  canStart: boolean;
  busy: boolean;
  onAction: (action: StageAction) => void;
}) {
  const running = stage.status === 'running';
  return (
    <div className="job-stage-control" data-status={stage.status}>
      <div className="job-stage-control-head">
        <span className="job-stage-control-name">{STAGE_LABELS[name]}</span>
        <span className="job-stage-control-status">{stageStatusText(stage)}</span>
      </div>
      <ProgressBar
        value={stage.progress}
        thickness="medium"
        aria-label={`${STAGE_LABELS[name]}: ${Math.round(stage.progress * 100)}%`}
      />
      <span className="job-stage-control-phase" title={stage.phase}>
        {stage.phase || 'Ожидает'}
      </span>
      <div className="job-stage-control-actions">
        <Button
          size="small"
          appearance="subtle"
          icon={<PlayRegular />}
          disabled={busy || running || !canStart || stage.status === 'done'}
          onClick={() => onAction('start')}
        >
          Запустить
        </Button>
        <Button
          size="small"
          appearance="subtle"
          icon={<StopRegular />}
          disabled={busy || !running}
          onClick={() => onAction('stop')}
        >
          Стоп
        </Button>
        <Button
          size="small"
          appearance="subtle"
          icon={<ArrowClockwiseRegular />}
          disabled={busy || !canStart}
          onClick={() => onAction('restart')}
        >
          Перезапустить
        </Button>
      </div>
    </div>
  );
}

export function JobPage({ jobId, onBack, onAuthError }: JobPageProps) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [stageBusy, setStageBusy] = useState<StageKey | null>(null);
  const [format, setFormat] = useState<ExportFormat>('txt');
  const [now, setNow] = useState(() => Date.now());

  const loadJob = useCallback(
    async (silent = false): Promise<void> => {
      try {
        const fresh = await getJob(jobId);
        setJob(fresh);
        if (!silent) {
          setError(null);
        }
      } catch (reason: unknown) {
        if (reason instanceof AuthError) {
          onAuthError();
          return;
        }
        if (!silent) {
          setError(describeError(reason));
        }
      }
    },
    [jobId, onAuthError],
  );

  useEffect(() => {
    setJob(null);
    setError(null);
    setNotice(null);
    setStageBusy(null);
    void loadJob();
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  const active =
    job !== null &&
    (isActive(job.status) ||
      Object.values(job.stages ?? {}).some((stage) => stage.status === 'running'));

  useEffect(() => {
    if (!active) {
      return;
    }
    setNow(Date.now());
    const tick = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => {
      window.clearInterval(tick);
    };
  }, [active]);

  useEffect(() => {
    if (!active) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadJob(true);
    }, 1200);
    return () => {
      window.clearInterval(timer);
    };
  }, [active, loadJob]);

  const displayText = job === null ? '' : transcriptText(job);
  const speakerCount = new Set(
    (job?.segments ?? []).map((segment) => segment.speaker).filter(Boolean),
  ).size;

  const handleOpenFolder = async (): Promise<void> => {
    if (job === null) {
      return;
    }
    try {
      const bridgeResult = await openResultFolder(job.id);
      if (bridgeResult !== null) {
        if (!bridgeResult.ok) {
          setError(bridgeResult.error ?? 'Не удалось открыть папку результата.');
        }
        return;
      }
      await revealJobFolder(job.id);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
  };

  const handleCopy = async (): Promise<void> => {
    if (displayText === '') {
      return;
    }
    try {
      await navigator.clipboard.writeText(displayText);
      setNotice('Текст скопирован в буфер обмена.');
    } catch {
      setError('Не удалось скопировать текст. Выделите его в поле вручную.');
    }
  };

  const handleSave = async (): Promise<void> => {
    if (job === null || job.status !== 'done') {
      return;
    }
    const base = job.filename.replace(/\.[^.]+$/, '') || 'transcript';
    const extension = EXPORT_EXTENSIONS[format];
    setSaving(true);
    setError(null);
    try {
      const bridgeResult = await saveResultAs(job.id, `${base}${extension}`, format);
      if (bridgeResult !== null) {
        if (bridgeResult.cancelled) {
          return;
        }
        if (!bridgeResult.ok) {
          setError(bridgeResult.error ?? 'Не удалось сохранить результат.');
          return;
        }
        setNotice(`Результат сохранён: ${bridgeResult.path ?? ''}`);
        return;
      }
      // Browser fallback: download the selected format.
      const response = await fetch(
        `/api/v1/jobs/${encodeURIComponent(job.id)}/export?format=${format}`,
        {
          method: 'GET',
          credentials: 'same-origin',
          headers: {
            Accept: '*/*',
            ...(token !== null ? { Authorization: `Bearer ${token}` } : {}),
          },
          cache: 'no-store',
        },
      );
      if (!response.ok) {
        setError(`Сохранение не удалось (HTTP ${response.status}).`);
        return;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${base}${extension}`;
      link.click();
      URL.revokeObjectURL(url);
      setNotice('Результат сохранён в папку загрузок браузера.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = async (): Promise<void> => {
    if (job === null) {
      return;
    }
    try {
      const updated = await cancelJob(job.id);
      setJob(updated);
      setNotice('Задача отменена.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
  };

  const handleStart = async (): Promise<void> => {
    if (job === null) {
      return;
    }
    try {
      const updated = await startJob(job.id);
      setJob(updated);
      setNotice('Задача запущена.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
  };

  const handleRetry = async (): Promise<void> => {
    if (job === null) {
      return;
    }
    try {
      const updated = await retryJob(job.id);
      setJob(updated);
      setNotice('Задача перезапущена.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
  };

  const handleStageAction = async (
    stage: StageKey,
    action: StageAction,
  ): Promise<void> => {
    if (job === null || stageBusy !== null) {
      return;
    }
    setStageBusy(stage);
    setError(null);
    try {
      const updated = await stageAction(job.id, stage, action);
      setJob(updated);
      const label = STAGE_LABELS[stage];
      if (action === 'start') {
        setNotice(`Этап «${label}» запущен.`);
      } else if (action === 'stop') {
        setNotice(`Этап «${label}» остановлен.`);
      } else {
        setNotice(`Этап «${label}» перезапущен.`);
      }
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
      await loadJob(true);
    } finally {
      setStageBusy(null);
    }
  };

  const summaryStage = job?.stages?.summary;
  const summaryRunning = summaryStage?.status === 'running';
  const recognitionDone = job?.stages?.recognition?.status === 'done';
  const canStartDiarization =
    job !== null && job.status === 'done' && recognitionDone && job.segments.length > 0;
  const canStartSummary =
    job !== null && job.status === 'done' && job.text.trim() !== '';

  const subtitle =
    job === null
      ? 'Загрузка задачи…'
      : [
          statusLabel(job.status),
          formatDate(job.created_at),
          job.audio_format.toUpperCase(),
          formatBytes(job.size_bytes),
          job.duration_s !== null ? `аудио ${formatSeconds(job.duration_s)}` : null,
          job.status !== 'queued'
            ? `время ${formatDuration(jobElapsedSeconds(job, now))}`
            : null,
          speakerCount > 0 ? `спикеров: ${speakerCount}` : null,
          job.rtf !== null ? `RTFx ${job.rtf.toFixed(1).replace('.', ',')}` : null,
        ]
          .filter((item): item is string => item !== null)
          .join(' · ');

  return (
    <div className="job-page">
      <PageHeader
        title={job?.filename ?? 'Задача'}
        subtitle={subtitle}
        badge={
          job !== null ? (
            <Badge appearance="tint" color={statusColor(job.status)} size="medium">
              {statusLabel(job.status)}
            </Badge>
          ) : null
        }
        actions={
          <Button icon={<ArrowLeftRegular />} onClick={onBack}>
            К очереди
          </Button>
        }
      />

      {notice !== null ? (
        <MessageBar intent="info" className="queue-notice">
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

      <section className="job-toolbar surface" aria-label="Действия с задачей">
        <div className="job-toolbar-actions">
          <Button
            size="small"
            icon={<FolderOpenRegular />}
            onClick={() => {
              void handleOpenFolder();
            }}
            disabled={job === null}
          >
            Папка
          </Button>
          <Dropdown
            aria-label="Формат сохранения"
            value={format.toUpperCase()}
            selectedOptions={[format]}
            onOptionSelect={(_event, data) => {
              if (data.optionValue !== undefined) {
                setFormat(data.optionValue as ExportFormat);
              }
            }}
            disabled={job === null || job.status !== 'done'}
          >
            <Option value="txt">TXT</Option>
            <Option value="srt">SRT</Option>
            <Option value="vtt">VTT</Option>
            <Option value="md">Markdown</Option>
            <Option value="json">JSON</Option>
          </Dropdown>
          <Button
            size="small"
            icon={saving ? <Spinner size="tiny" /> : <SaveRegular />}
            disabled={job === null || job.status !== 'done' || saving || displayText === ''}
            onClick={() => {
              void handleSave();
            }}
          >
            Сохранить как…
          </Button>
          <Button
            size="small"
            icon={<CopyRegular />}
            disabled={displayText === ''}
            onClick={() => {
              void handleCopy();
            }}
          >
            Копировать
          </Button>
          {job !== null && isActive(job.status) ? (
            <Button
              appearance="subtle"
              size="small"
              icon={<DismissRegular />}
              onClick={() => {
                void handleCancel();
              }}
            >
              Отменить
            </Button>
          ) : null}
          {job !== null && job.status === 'queued' && job.paused ? (
            <Button
              appearance="subtle"
              size="small"
              icon={<PlayRegular />}
              onClick={() => {
                void handleStart();
              }}
            >
              Запустить
            </Button>
          ) : null}
        </div>

        {job !== null ? (
          <div className="job-stage-controls" aria-label="Управление этапами">
            <StageControl
              name="recognition"
              stage={job.stages.recognition}
              canStart={
                job.status !== 'done' && (!isActive(job.status) || job.paused === true)
              }
              busy={stageBusy !== null}
              onAction={(action) => {
                void handleStageAction('recognition', action);
              }}
            />
            <StageControl
              name="diarization"
              stage={job.stages.diarization}
              canStart={canStartDiarization}
              busy={stageBusy !== null}
              onAction={(action) => {
                void handleStageAction('diarization', action);
              }}
            />
            <StageControl
              name="summary"
              stage={job.stages.summary}
              canStart={canStartSummary}
              busy={stageBusy !== null}
              onAction={(action) => {
                void handleStageAction('summary', action);
              }}
            />
          </div>
        ) : null}
      </section>

      {job !== null && isActive(job.status) ? (
        <div className="transcript-progress job-live-progress" role="status" aria-live="polite">
          <ProgressBar value={job.progress} thickness="medium" />
          <span>{job.phase}</span>
        </div>
      ) : null}

      {job !== null && (job.status === 'error' || job.status === 'cancelled') ? (
        <MessageBar intent="error" className="transcript-error">
          <MessageBarBody>
            {job.status === 'cancelled'
              ? 'Задача отменена.'
              : job.error ?? 'Не удалось распознать файл.'}
          </MessageBarBody>
          <MessageBarActions
            containerAction={
              <Button
                appearance="subtle"
                icon={<HistoryRegular />}
                onClick={() => {
                  void handleRetry();
                }}
              >
                Повторить
              </Button>
            }
          />
        </MessageBar>
      ) : null}

      {job !== null && job.status === 'done' ? (
        <section
          className="transcript-panel job-transcript-panel surface"
          aria-labelledby="job-transcript-title"
        >
          <div className="transcript-head">
            <div>
              <h2 id="job-transcript-title" className="files-title">
                Расшифровка
              </h2>
              <p className="job-meta">
                {job.segments.length > 0 ? `фрагментов: ${job.segments.length}` : 'текст целиком'}
                {speakerCount > 0 ? ` · спикеров: ${speakerCount}` : ''}
              </p>
            </div>
          </div>
          <Textarea
            className="job-transcript"
            readOnly
            resize="none"
            value={displayText === '' ? 'Речь не распознана.' : displayText}
            aria-label="Результат распознавания"
          />
        </section>
      ) : null}

      {job !== null && job.status === 'done' ? (
        job.summary !== null ? (
          <SummaryPanel summary={job.summary} />
        ) : summaryStage?.status === 'error' ? (
          <section className="summary-panel surface" aria-labelledby="job-summary-state">
            <h2 id="job-summary-state" className="summary-heading">
              Саммари
            </h2>
            <MessageBar intent="warning">
              <MessageBarBody>
                {summaryStage.phase || 'Не удалось составить саммари.'}
              </MessageBarBody>
              <MessageBarActions
                containerAction={
                  <Button
                    appearance="subtle"
                    icon={<SparkleRegular />}
                    disabled={stageBusy !== null}
                    onClick={() => {
                      void handleStageAction('summary', 'restart');
                    }}
                  >
                    Повторить
                  </Button>
                }
              />
            </MessageBar>
          </section>
        ) : summaryRunning ? (
          <section className="summary-panel surface" aria-labelledby="job-summary-state">
            <h2 id="job-summary-state" className="summary-heading">
              Саммари
            </h2>
            <div className="summary-loading">
              <Spinner size="small" />
              <span>{summaryStage?.phase || 'Составляем саммари…'}</span>
            </div>
          </section>
        ) : null
      ) : null}
    </div>
  );
}
