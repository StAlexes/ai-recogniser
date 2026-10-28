import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import {
  Badge,
  Button,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  Spinner,
} from '@fluentui/react-components';
import {
  AddRegular,
  DismissRegular,
  FolderOpenRegular,
  HistoryRegular,
  InfoRegular,
  MusicNote2Regular,
  PlayRegular,
  ShieldCheckmarkRegular,
  SparkleRegular,
  VideoRegular,
} from '@fluentui/react-icons';
import {
  AuthError,
  cancelJob,
  getJobs,
  getSettings,
  retryJob,
  revealJobFolder,
  startJob,
  uploadJob,
} from '../api';
import type { JobSummary } from '../api';
import { Dropzone } from '../components/Dropzone';
import { InfoCard } from '../components/InfoCard';
import { PageHeader } from '../components/PageHeader';
import { openResultFolder } from '../desktop';
import { formatBytes, pluralFiles } from '../format';
import {
  describeError,
  extensionOf,
  formatDate,
  formatDuration,
  isActive,
  jobElapsedSeconds,
  StageProgress,
  statusColor,
  statusLabel,
} from '../jobUtils';

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

interface SelectedFile {
  id: string;
  file: File;
  name: string;
  size: number;
  kind: 'audio' | 'video';
}

interface QueuePageProps {
  onAuthError: () => void;
  onOpenJob: (jobId: string) => void;
}

export function QueuePage({ onAuthError, onOpenJob }: QueuePageProps) {
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rejectedCount, setRejectedCount] = useState(0);
  const [recognizeImmediately, setRecognizeImmediately] = useState(true);
  const [now, setNow] = useState(() => Date.now());
  const inputRef = useRef<HTMLInputElement>(null);
  const nextId = useRef(0);
  const uploadingRef = useRef(false);

  const refreshJobs = async (): Promise<void> => {
    try {
      const response = await getJobs();
      setJobs(response.items);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
      }
    }
  };

  useEffect(() => {
    let cancelled = false;
    getSettings()
      .then((settings) => {
        if (!cancelled) {
          setRecognizeImmediately(settings.recognize_immediately);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled && reason instanceof AuthError) {
          onAuthError();
        }
      });
    return () => {
      cancelled = true;
    };
  }, [onAuthError]);

  useEffect(() => {
    void refreshJobs();
    return undefined;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const hasActive = jobs.some(
    (job) =>
      isActive(job.status) ||
      job.stages?.summary?.status === 'running' ||
      job.stages?.recognition?.status === 'running' ||
      job.stages?.diarization?.status === 'running',
  );

  useEffect(() => {
    if (!hasActive) {
      return;
    }
    setNow(Date.now());
    const timer = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => {
      window.clearInterval(timer);
    };
  }, [hasActive]);

  useEffect(() => {
    if (!hasActive) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshJobs();
    }, 1200);
    return () => {
      window.clearInterval(timer);
    };
  }, [hasActive]); // eslint-disable-line react-hooks/exhaustive-deps

  const openPicker = (): void => {
    inputRef.current?.click();
  };

  const uploadBatch = async (
    batch: SelectedFile[],
    startAfterUpload: boolean,
  ): Promise<void> => {
    if (batch.length === 0 || uploadingRef.current) {
      return;
    }
    uploadingRef.current = true;
    setUploading(true);
    setError(null);
    setNotice(`Загрузка файлов: ${batch.length}…`);
    const uploadedIds = new Set<string>();
    const jobIds: string[] = [];
    let firstJobId: string | null = null;
    let firstError: string | null = null;

    for (const file of batch) {
      try {
        const job = await uploadJob(file.file);
        uploadedIds.add(file.id);
        jobIds.push(job.id);
        if (firstJobId === null) {
          firstJobId = job.id;
        }
      } catch (reason: unknown) {
        if (reason instanceof AuthError) {
          onAuthError();
          uploadingRef.current = false;
          setUploading(false);
          return;
        }
        if (firstError === null) {
          firstError = describeError(reason);
        }
      }
    }

    if (startAfterUpload) {
      for (const jobId of jobIds) {
        try {
          await startJob(jobId);
        } catch (reason: unknown) {
          if (reason instanceof AuthError) {
            onAuthError();
            uploadingRef.current = false;
            setUploading(false);
            return;
          }
          if (firstError === null) {
            firstError = describeError(reason);
          }
        }
      }
    }

    setFiles((current) => current.filter((file) => !uploadedIds.has(file.id)));
    uploadingRef.current = false;
    setUploading(false);
    if (firstError !== null) {
      setError(firstError);
    } else if (firstJobId !== null) {
      setNotice(
        startAfterUpload
          ? 'Файлы добавлены. Обработка запущена.'
          : 'Файлы добавлены. Обработка начинается автоматически.',
      );
      await refreshJobs();
      onOpenJob(firstJobId);
      return;
    }
    await refreshJobs();
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
      setNotice(null);
      setError(null);
      if (recognizeImmediately && !uploadingRef.current) {
        void uploadBatch(accepted, false);
      } else {
        setFiles((current) => [...current, ...accepted]);
      }
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

  const handleOpenFolder = async (jobId: string): Promise<void> => {
    try {
      const bridgeResult = await openResultFolder(jobId);
      if (bridgeResult !== null) {
        if (!bridgeResult.ok) {
          setError(bridgeResult.error ?? 'Не удалось открыть папку результата.');
        }
        return;
      }
      await revealJobFolder(jobId);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
  };

  const handleCancel = async (jobId: string): Promise<void> => {
    try {
      await cancelJob(jobId);
      setNotice('Задача отменена.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
    await refreshJobs();
  };

  const handleRetry = async (jobId: string): Promise<void> => {
    try {
      await retryJob(jobId);
      setNotice('Задача перезапущена.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
    await refreshJobs();
  };

  const handleStart = async (jobId: string): Promise<void> => {
    try {
      await startJob(jobId);
      setNotice('Задача запущена.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    }
    await refreshJobs();
  };

  const startRecognition = async (): Promise<void> => {
    if (files.length === 0 || uploading) {
      return;
    }
    const batch = files;
    await uploadBatch(batch, true);
  };

  return (
    <>
      <PageHeader
        title="Очередь"
        subtitle="Локальное распознавание аудио и видео"
        actions={
          <div className="queue-actions">
            <Button icon={<AddRegular />} onClick={openPicker} disabled={uploading}>
              Добавить файлы
            </Button>
            {!recognizeImmediately || files.length > 0 ? (
              <Button
                appearance="primary"
                icon={uploading ? <Spinner size="tiny" /> : <SparkleRegular />}
                disabled={files.length === 0 || uploading}
                onClick={() => {
                  void startRecognition();
                }}
              >
                {uploading ? 'Загрузка…' : 'Распознать'}
              </Button>
            ) : null}
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
          Пропущено файлов: {rejectedCount}. Поддерживаются mp3, mp4, m4a, wav, flac, ogg,
          opus, mkv, mov, webm и avi.
        </p>
      ) : null}

      {files.length === 0 ? (
        <div className="empty-note">
          <InfoRegular aria-hidden="true" />
          <span>
            {jobs.length === 0
              ? recognizeImmediately
                ? 'Очередь пуста. Добавьте mp3 или mp4 — обработка начнётся сразу. Файлы не покидают этот компьютер.'
                : 'Очередь пуста. Добавьте mp3 или mp4, затем нажмите «Распознать». Файлы обрабатываются на этом компьютере.'
              : 'Добавьте следующий файл или откройте готовый результат в списке задач ниже.'}
          </span>
        </div>
      ) : (
        <section className="files-panel surface" aria-labelledby="selected-files-title">
          <div className="files-head">
            <h2 id="selected-files-title" className="files-title">
              К отправке: {pluralFiles(files.length)}
            </h2>
            <Button appearance="subtle" size="small" onClick={clearFiles} disabled={uploading}>
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
                  disabled={uploading}
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
            {jobs.map((job) => {
              const elapsed = formatDuration(jobElapsedSeconds(job, now));
              const percent = Math.round(Math.max(0, Math.min(job.progress, 1)) * 100);
              return (
                <li key={job.id}>
                  <button
                    type="button"
                    className="job-row"
                    onClick={() => {
                      onOpenJob(job.id);
                    }}
                  >
                    <span className="job-main">
                      <span className="file-name" title={job.filename}>
                        {job.filename}
                      </span>
                      <span className="job-meta">
                        {formatDate(job.created_at)} · {job.audio_format.toUpperCase()} ·{' '}
                        {formatBytes(job.size_bytes)}
                        {job.status !== 'queued' ? ` · время: ${elapsed}` : ''}
                        {job.status !== 'queued' ? ` · ${percent}%` : ''}
                      </span>
                    </span>
                    <Badge appearance="tint" color={statusColor(job.status)} size="small">
                      {statusLabel(job.status)}
                    </Badge>
                  </button>

                  <div className="job-stages" aria-label={`Этапы задачи ${job.filename}`}>
                    <StageProgress name="recognition" stage={job.stages.recognition} />
                    <StageProgress name="diarization" stage={job.stages.diarization} />
                    <StageProgress name="summary" stage={job.stages.summary} />
                  </div>

                  <div className="job-actions">
                    <Button
                      appearance="subtle"
                      size="small"
                      icon={<FolderOpenRegular />}
                      onClick={() => {
                        void handleOpenFolder(job.id);
                      }}
                    >
                      Папка
                    </Button>
                    {isActive(job.status) ? (
                      <Button
                        appearance="subtle"
                        size="small"
                        icon={<DismissRegular />}
                        onClick={() => {
                          void handleCancel(job.id);
                        }}
                      >
                        Отменить
                      </Button>
                    ) : null}
                    {job.status === 'queued' && job.paused ? (
                      <Button
                        appearance="subtle"
                        size="small"
                        icon={<PlayRegular />}
                        onClick={() => {
                          void handleStart(job.id);
                        }}
                      >
                        Запустить
                      </Button>
                    ) : null}
                    {job.status === 'error' || job.status === 'cancelled' ? (
                      <Button
                        appearance="subtle"
                        size="small"
                        icon={<HistoryRegular />}
                        onClick={() => {
                          void handleRetry(job.id);
                        }}
                      >
                        Повторить
                      </Button>
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      <div className="info-grid">
        <InfoCard tone="brand" icon={<SparkleRegular fontSize={18} />} title="GigaAM v3 · 214 МБ">
          Русский язык, пунктуация и нормализация из коробки. Длинные записи режутся
          Silero VAD на сегменты по паузам.
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
