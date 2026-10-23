import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import {
  Badge,
  Button,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  ProgressBar,
  Spinner,
  Textarea,
} from '@fluentui/react-components';
import {
  AddRegular,
  ArrowDownloadRegular,
  CopyRegular,
  DismissRegular,
  InfoRegular,
  MusicNote2Regular,
  ShieldCheckmarkRegular,
  SparkleRegular,
  VideoRegular,
} from '@fluentui/react-icons';
import {
  ApiError,
  AuthError,
  getJob,
  getJobs,
  uploadJob,
} from '../api';
import type { JobDetail, JobStatus, JobSummary } from '../api';
import { Dropzone } from '../components/Dropzone';
import { InfoCard } from '../components/InfoCard';
import { PageHeader } from '../components/PageHeader';
import { formatBytes, pluralFiles } from '../format';

const ACCEPTED_EXTENSIONS = new Set([
  'mp3',
  'mp4',
  'm4a',
  'wav',
  'flac',
  'ogg',
  'opus',
  'mkv',
  'mov',
  'webm',
  'avi',
]);

const VIDEO_EXTENSIONS = new Set(['mp4', 'mkv', 'mov', 'webm', 'avi']);

const ACCEPT_ATTRIBUTE = Array.from(ACCEPTED_EXTENSIONS)
  .map((extension) => `.${extension}`)
  .join(',');

type BadgeColor = 'brand' | 'danger' | 'informative' | 'success';

interface SelectedFile {
  id: string;
  file: File;
  name: string;
  size: number;
  kind: 'audio' | 'video';
}

interface QueuePageProps {
  onAuthError: () => void;
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.');
  if (dot < 1 || dot === name.length - 1) {
    return '';
  }
  return name.slice(dot + 1).toLowerCase();
}

function isActive(status: JobStatus): boolean {
  return status === 'queued' || status === 'running';
}

function statusLabel(status: JobStatus): string {
  switch (status) {
    case 'queued':
      return 'В очереди';
    case 'running':
      return 'Распознавание';
    case 'done':
      return 'Готово';
    case 'error':
      return 'Ошибка';
  }
}

function statusColor(status: JobStatus): BadgeColor {
  switch (status) {
    case 'queued':
      return 'informative';
    case 'running':
      return 'brand';
    case 'done':
      return 'success';
    case 'error':
      return 'danger';
  }
}

function describeError(reason: unknown): string {
  if (reason instanceof ApiError) {
    return reason.message;
  }
  return 'Не удалось выполнить запрос.';
}

function formatSeconds(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value < 0) {
    return '—';
  }
  return `${value.toFixed(1).replace('.', ',')} с`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString('ru-RU', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function QueuePage({ onAuthError }: QueuePageProps) {
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rejectedCount, setRejectedCount] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const nextId = useRef(0);

  const openPicker = (): void => {
    inputRef.current?.click();
  };

  const addFiles = (incoming: FileList): void => {
    const accepted: SelectedFile[] = [];
    let rejected = 0;

    for (const file of Array.from(incoming)) {
      const extension = extensionOf(file.name);
      if (!ACCEPTED_EXTENSIONS.has(extension)) {
        rejected += 1;
        continue;
      }
      nextId.current += 1;
      accepted.push({
        id: `file-${nextId.current}`,
        file,
        name: file.name,
        size: file.size,
        kind: VIDEO_EXTENSIONS.has(extension) ? 'video' : 'audio',
      });
    }

    if (accepted.length > 0) {
      setFiles((current) => [...current, ...accepted]);
      setNotice(null);
      setError(null);
    }
    setRejectedCount(rejected);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const selected = event.currentTarget.files;
    if (selected !== null) {
      addFiles(selected);
    }
    event.currentTarget.value = '';
  };

  const removeFile = (id: string): void => {
    setFiles((current) => current.filter((file) => file.id !== id));
  };

  const clearFiles = (): void => {
    setFiles([]);
    setRejectedCount(0);
  };

  useEffect(() => {
    let cancelled = false;
    getJobs()
      .then((response) => {
        if (!cancelled) {
          setJobs(response.items);
        }
      })
      .catch((reason: unknown) => {
        if (cancelled) {
          return;
        }
        if (reason instanceof AuthError) {
          onAuthError();
        }
      });
    return () => {
      cancelled = true;
    };
  }, [onAuthError]);

  const hasActive = jobs.some((job) => isActive(job.status))
    || (detail !== null && isActive(detail.status));

  useEffect(() => {
    if (!hasActive) {
      return;
    }
    let cancelled = false;
    const tick = async (): Promise<void> => {
      try {
        const response = await getJobs();
        if (cancelled) {
          return;
        }
        setJobs(response.items);
        if (detail !== null && isActive(detail.status)) {
          const fresh = await getJob(detail.id);
          if (!cancelled) {
            setDetail(fresh);
          }
        }
      } catch (reason: unknown) {
        if (reason instanceof AuthError) {
          onAuthError();
        }
      }
    };
    void tick();
    const timer = window.setInterval(() => {
      void tick();
    }, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [hasActive, detail?.id, detail?.status, onAuthError]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectJob = async (jobId: string): Promise<void> => {
    try {
      setDetail(await getJob(jobId));
      setError(null);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
  };

  const startRecognition = async (): Promise<void> => {
    if (files.length === 0 || starting) {
      return;
    }
    setStarting(true);
    setError(null);
    setNotice(null);
    const uploadedIds = new Set<string>();
    let firstJob: JobDetail | null = null;
    let firstError: string | null = null;

    for (const file of files) {
      try {
        const job = await uploadJob(file.file);
        uploadedIds.add(file.id);
        if (firstJob === null) {
          firstJob = job;
        }
      } catch (reason: unknown) {
        if (reason instanceof AuthError) {
          onAuthError();
          setStarting(false);
          return;
        }
        if (firstError === null) {
          firstError = describeError(reason);
        }
      }
    }

    setFiles((current) => current.filter((file) => !uploadedIds.has(file.id)));
    setStarting(false);
    if (firstJob !== null) {
      setDetail(firstJob);
      setNotice(
        `Файлов в очереди: ${uploadedIds.size}. Текст появится здесь после обработки.`,
      );
    }
    if (firstError !== null) {
      setError(firstError);
    }
    try {
      const response = await getJobs();
      setJobs(response.items);
    } catch {
      // The detail page and the next poll will retry.
    }
  };

  const copyTranscript = async (): Promise<void> => {
    if (detail === null || detail.text === '') {
      return;
    }
    try {
      await navigator.clipboard.writeText(detail.text);
      setNotice('Текст скопирован в буфер обмена.');
    } catch {
      setError('Не удалось скопировать текст. Выделите его в поле вручную.');
    }
  };

  const downloadTranscript = (): void => {
    if (detail === null || detail.text === '') {
      return;
    }
    const blob = new Blob([detail.text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const base = detail.filename.replace(/\.[^.]+$/, '');
    link.href = url;
    link.download = `${base || 'transcript'}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <PageHeader
        title="Очередь"
        subtitle="Локальное распознавание аудио и видео"
        actions={
          <div className="queue-actions">
            <Button icon={<AddRegular />} onClick={openPicker} disabled={starting}>
              Добавить файлы
            </Button>
            <Button
              appearance="primary"
              icon={starting ? <Spinner size="tiny" /> : <SparkleRegular />}
              disabled={files.length === 0 || starting}
              onClick={() => {
                void startRecognition();
              }}
            >
              {starting ? 'Загрузка…' : 'Распознать'}
            </Button>
          </div>
        }
      />

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        multiple
        tabIndex={-1}
        aria-hidden="true"
        onChange={handleInputChange}
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

      <Dropzone onFiles={addFiles} onBrowse={openPicker} />

      {rejectedCount > 0 ? (
        <p className="queue-hint" role="status">
          Пропущено файлов: {rejectedCount}. Поддерживаются mp3, mp4, m4a, wav, flac, ogg, opus,
          mkv, mov, webm и avi.
        </p>
      ) : null}

      {files.length === 0 ? (
        <div className="empty-note">
          <InfoRegular aria-hidden="true" />
          <span>
            {jobs.length === 0
              ? 'Очередь пуста. Добавьте mp3 или mp4, затем нажмите «Распознать». Файлы обрабатываются на этом компьютере.'
              : 'Добавьте следующий файл или откройте готовый результат в списке задач ниже.'}
          </span>
        </div>
      ) : (
        <section className="files-panel surface" aria-labelledby="selected-files-title">
          <div className="files-head">
            <h2 id="selected-files-title" className="files-title">
              К отправке: {pluralFiles(files.length)}
            </h2>
            <Button appearance="subtle" size="small" onClick={clearFiles} disabled={starting}>
              Очистить
            </Button>
          </div>
          <ul className="files-list">
            {files.map((file) => (
              <li key={file.id} className="file-row">
                <span className="file-icon" data-kind={file.kind} aria-hidden="true">
                  {file.kind === 'video' ? (
                    <VideoRegular fontSize={20} />
                  ) : (
                    <MusicNote2Regular fontSize={20} />
                  )}
                </span>
                <span className="file-name" title={file.name}>
                  {file.name}
                </span>
                <span className="file-size">{formatBytes(file.size)}</span>
                <Badge appearance="tint" color="informative" size="small">
                  {file.kind === 'video' ? 'видео' : 'аудио'}
                </Badge>
                <Button
                  appearance="transparent"
                  size="small"
                  icon={<DismissRegular />}
                  aria-label={`Убрать ${file.name}`}
                  disabled={starting}
                  onClick={() => {
                    removeFile(file.id);
                  }}
                />
              </li>
            ))}
          </ul>
        </section>
      )}

      {jobs.length > 0 ? (
        <section className="jobs-panel surface" aria-labelledby="jobs-title">
          <div className="files-head">
            <h2 id="jobs-title" className="files-title">
              Задачи
            </h2>
            <Badge appearance="tint" color="informative" size="small">
              {jobs.length}
            </Badge>
          </div>
          <ul className="jobs-list">
            {jobs.map((job) => (
              <li key={job.id}>
                <button
                  type="button"
                  className="job-row"
                  data-selected={detail?.id === job.id}
                  onClick={() => {
                    void selectJob(job.id);
                  }}
                >
                  <span className="job-main">
                    <span className="file-name" title={job.filename}>
                      {job.filename}
                    </span>
                    <span className="job-meta">
                      {formatDate(job.created_at)} · {job.audio_format.toUpperCase()}
                    </span>
                  </span>
                  <Badge appearance="tint" color={statusColor(job.status)} size="small">
                    {statusLabel(job.status)}
                  </Badge>
                </button>
                {isActive(job.status) ? (
                  <ProgressBar
                    className="job-progress"
                    value={job.progress}
                    thickness="medium"
                    aria-label={`Прогресс ${job.filename}`}
                  />
                ) : null}
                {job.status === 'error' && job.error !== null ? (
                  <p className="job-error">{job.error}</p>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {detail !== null ? (
        <section className="transcript-panel surface" aria-labelledby="transcript-title">
          <div className="transcript-head">
            <div>
              <h2 id="transcript-title" className="files-title">
                {detail.filename}
              </h2>
              <p className="job-meta">
                {statusLabel(detail.status)}
                {detail.duration_s !== null
                  ? ` · аудио ${formatSeconds(detail.duration_s)}`
                  : ''}
                {detail.elapsed_s !== null ? ` · обработка ${formatSeconds(detail.elapsed_s)}` : ''}
                {detail.rtf !== null
                  ? ` · RTFx ${detail.rtf.toFixed(1).replace('.', ',')}`
                  : ''}
              </p>
            </div>
            <div className="transcript-actions">
              <Button
                size="small"
                icon={<CopyRegular />}
                disabled={detail.text === ''}
                onClick={() => {
                  void copyTranscript();
                }}
              >
                Копировать
              </Button>
              <Button
                size="small"
                icon={<ArrowDownloadRegular />}
                disabled={detail.text === ''}
                onClick={downloadTranscript}
              >
                Скачать .txt
              </Button>
              <Button
                appearance="subtle"
                size="small"
                icon={<DismissRegular />}
                aria-label="Закрыть результат"
                onClick={() => {
                  setDetail(null);
                }}
              />
            </div>
          </div>

          {isActive(detail.status) ? (
            <div className="transcript-progress" role="status" aria-live="polite">
              <ProgressBar value={detail.progress} thickness="medium" />
              <span>{detail.phase}</span>
            </div>
          ) : null}

          {detail.status === 'error' ? (
            <MessageBar intent="error" className="transcript-error">
              <MessageBarBody>{detail.error ?? 'Не удалось распознать файл.'}</MessageBarBody>
            </MessageBar>
          ) : null}

          {detail.status === 'done' ? (
            <Textarea
              className="transcript-box"
              readOnly
              resize="vertical"
              value={detail.text === '' ? 'Речь не распознана.' : detail.text}
              aria-label="Результат распознавания"
            />
          ) : null}
        </section>
      ) : null}

      <div className="info-grid">
        <InfoCard tone="brand" icon={<SparkleRegular fontSize={18} />} title="GigaAM v3 · 214 МБ">
          Русский язык, пунктуация и нормализация из коробки. Модель загружается один раз.
        </InfoCard>
        <InfoCard
          tone="teal"
          icon={<InfoRegular fontSize={18} />}
          title="ffmpeg внутри"
        >
          mp3, mp4, m4a и видео автоматически приводятся к 16 кГц mono WAV.
        </InfoCard>
        <InfoCard
          tone="purple"
          icon={<ShieldCheckmarkRegular fontSize={18} />}
          title="Приватность"
        >
          Аудио не покидает компьютер. Сеть нужна только для первой загрузки модели.
        </InfoCard>
      </div>
    </>
  );
}
