import { ProgressBar } from '@fluentui/react-components';
import type { JobStage, JobStatus, JobSummary, StageKey } from './api';
import { ApiError } from './api';

export type BadgeColor = 'brand' | 'danger' | 'informative' | 'success';

export function describeError(reason: unknown): string {
  if (reason instanceof ApiError) {
    return reason.message;
  }
  return 'Не удалось выполнить запрос.';
}

export const STAGE_LABELS: Record<StageKey, string> = {
  recognition: 'Распознавание',
  diarization: 'Диаризация',
  summary: 'Саммаризация',
};

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.');
  if (dot < 1 || dot === name.length - 1) {
    return '';
  }
  return name.slice(dot + 1).toLowerCase();
}

export function isActive(status: JobStatus): boolean {
  return status === 'queued' || status === 'running';
}

export function statusLabel(status: JobStatus): string {
  switch (status) {
    case 'queued':
      return 'В очереди';
    case 'running':
      return 'Распознавание';
    case 'done':
      return 'Готово';
    case 'error':
      return 'Ошибка';
    case 'cancelled':
      return 'Отменено';
  }
}

export function statusColor(status: JobStatus): BadgeColor {
  switch (status) {
    case 'queued':
      return 'informative';
    case 'running':
      return 'brand';
    case 'done':
      return 'success';
    case 'error':
      return 'danger';
    case 'cancelled':
      return 'danger';
  }
}

export function formatSeconds(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value < 0) {
    return '—';
  }
  return `${value.toFixed(1).replace('.', ',')} с`;
}

export function formatDate(value: string): string {
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

/** Human-readable duration for job cards (seconds → mm:ss / h:mm:ss). */
export function formatDuration(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value < 0) {
    return '—';
  }
  const total = Math.round(value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  if (minutes > 0) {
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
  }
  return `${seconds} с`;
}

export function jobElapsedSeconds(job: JobSummary, now: number): number | null {
  if (job.elapsed_s !== null) {
    return job.elapsed_s;
  }
  if (isActive(job.status) && job.started_at !== null) {
    const started = new Date(job.started_at).getTime();
    if (Number.isFinite(started)) {
      return Math.max((now - started) / 1000, 0);
    }
  }
  return null;
}

export function stageStatusText(stage: JobStage): string {
  switch (stage.status) {
    case 'pending':
      return 'ожидает';
    case 'running':
      return stage.phase || 'выполняется';
    case 'done':
      return 'готово';
    case 'error':
      return 'ошибка';
    case 'skipped':
      return stage.phase || 'пропущено';
  }
}

export function StageProgress({ name, stage }: { name: StageKey; stage: JobStage }) {
  const percent = Math.round(Math.max(0, Math.min(stage.progress, 1)) * 100);
  return (
    <div className="job-stage" data-status={stage.status}>
      <span className="job-stage-label">{STAGE_LABELS[name]}</span>
      <ProgressBar
        className="job-stage-bar"
        value={stage.progress}
        thickness="medium"
        aria-label={`${STAGE_LABELS[name]}: ${percent}%`}
      />
      <span className="job-stage-value" title={stageStatusText(stage)}>
        {stage.status === 'skipped' ? '—' : `${percent}%`}
      </span>
    </div>
  );
}

/** Format a transcript for display and copy: speaker-aware when available. */
export function transcriptText(job: {
  text: string;
  segments: { start_s: number; text: string; speaker: string | null }[];
}): string {
  const segments = job.segments.filter((segment) => segment.text.trim() !== '');
  if (!segments.some((segment) => segment.speaker)) {
    return job.text.trim();
  }
  return segments
    .map((segment) => {
      const total = maxClock(segment.start_s);
      const speaker = segment.speaker ? `${segment.speaker}: ` : '';
      return `[${total}] ${speaker}${segment.text.trim()}`;
    })
    .join('\n');
}

function maxClock(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return '00:00';
  }
  const total = Math.round(value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}
