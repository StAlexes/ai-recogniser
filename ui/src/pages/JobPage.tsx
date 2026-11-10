import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
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
  DocumentArrowRightRegular,
  FolderOpenRegular,
  HistoryRegular,
  PlayRegular,
  SaveRegular,
  StopRegular,
} from '@fluentui/react-icons';
import {
  AuthError,
  ApiError,
  cancelJob,
  fetchJobAudio,
  getDiarizationProfiles,
  getJob,
  getJobReview,
  getSettings,
  putJobReview,
  retryJob,
  revealJobFolder,
  revealJobSource,
  setJobDiarizeProfile,
  stageAction,
  startJob,
  token,
} from '../api';
import type {
  DiarizationProfilesResponse,
  JobDetail,
  ReviewCorrection,
  StageAction,
  StageKey,
} from '../api';
import { PageHeader } from '../components/PageHeader';
import { openResultFolder, revealSourceFile, saveResultAs } from '../desktop';
import { formatBytes, formatClock } from '../format';
import {
  describeError,
  formatDate,
  formatDuration,
  isActive,
  jobElapsedSeconds,
  STAGE_LABELS,
  stageElapsedSeconds,
  stageStatusText,
  statusColor,
  statusLabel,
} from '../jobUtils';

const SUMMARY_MAX_POINTS = 8;

/** The saved results a finished job can show: one per processing stage. */
type ResultView = 'recognition' | 'diarization' | 'summary';

const RESULT_LABELS: Record<ResultView, string> = {
  recognition: 'Расшифровка',
  diarization: 'Расшифровка с диаризацией',
  summary: 'Саммари',
};

/** Форматы, которые WebView2 (Chromium) воспроизводит без конвертации. */
const PLAYABLE_AUDIO = new Set([
  'mp3',
  'm4a',
  'mp4',
  'wav',
  'ogg',
  'opus',
  'flac',
  'webm',
  'mov',
]);

interface JobPageProps {
  jobId: string;
  onBack: () => void;
  onAuthError: () => void;
}

function summaryList(value: unknown): any[] {
  return Array.isArray(value) ? value.slice(0, SUMMARY_MAX_POINTS) : [];
}

/** Formatted summary content, rendered inside the shared result panel. */
function SummaryBody({ summary }: { summary: Record<string, any> }) {
  // Кастомный промпт возвращает свободный текст (custom: true) — показываем
  // его как есть: пользователь сам задаёт оформление в настройках.
  if (summary.custom === true && typeof summary.tl_dr === 'string') {
    return (
      <div className="summary-body">
        <div className="summary-tldr">{summary.tl_dr || 'Пустой ответ модели.'}</div>
      </div>
    );
  }
  const topics = summaryList(summary.topics);
  const decisions = summaryList(summary.decisions);
  const actionItems = summaryList(summary.action_items);
  const risks = summaryList(summary.risks);
  return (
    <div className="summary-body">
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
    </div>
  );
}

function StageControl({
  name,
  stage,
  canStart,
  busy,
  now,
  onAction,
  children,
}: {
  name: StageKey;
  stage: JobDetail['stages'][StageKey];
  canStart: boolean;
  busy: boolean;
  now: number;
  onAction: (action: StageAction) => void;
  children?: ReactNode;
}) {
  const running = stage.status === 'running';
  // Одна кнопка вместо трёх: запущено — «Стоп», завершено или упало —
  // «Перезапустить», иначе — «Запустить».
  let action: StageAction;
  let icon: ReactNode;
  let label: string;
  if (running) {
    action = 'stop';
    icon = <StopRegular />;
    label = 'Стоп';
  } else if (stage.status === 'done' || stage.status === 'error') {
    action = 'restart';
    icon = <ArrowClockwiseRegular />;
    label = 'Перезапустить';
  } else {
    action = 'start';
    icon = <PlayRegular />;
    label = 'Запустить';
  }
  const disabled = busy || (action !== 'stop' && !canStart);
  // Живой счётчик: пока этап выполняется, время растёт от started_at; для
  // завершённых показываем сохранённое elapsed_s.
  const elapsed = stageElapsedSeconds(stage, now);
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
        {elapsed !== null ? ` · ${formatDuration(elapsed)}` : ''}
      </span>
      {children}
      <div className="job-stage-control-actions">
        <Button
          size="small"
          appearance="subtle"
          icon={icon}
          disabled={disabled}
          onClick={() => onAction(action)}
        >
          {label}
        </Button>
      </div>
    </div>
  );
}

export function JobPage({ jobId, onBack, onAuthError }: JobPageProps) {
  const [job, setJob] = useState<JobDetail | null>(null);
  // Единая верхняя плашка: и сообщения, и ошибки.  Всё, что раньше показывалось
  // плашкой внизу страницы, выводится здесь.
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // Чем можно ответить на ошибку прямо из верхней плашки: перезапустить
  // упавший этап или повторить задачу целиком.
  const [errorRetry, setErrorRetry] = useState<
    { kind: 'job' } | { kind: 'stage'; stage: StageKey } | null
  >(null);
  // Ошибка обработчика действия: показывается в верхней плашке без кнопки
  // повтора (само действие пользователь может вызвать снова вручную).
  const showError = useCallback((message: string): void => {
    setErrorRetry(null);
    setError(message);
  }, []);
  // Ошибка этапа: рядом с текстом предлагаем перезапуск этого этапа.
  const showStageError = useCallback((stage: StageKey, message: string): void => {
    setErrorRetry({ kind: 'stage', stage });
    setError(message);
  }, []);
  // Ошибка задачи целиком (например, сбой распознавания): предлагаем «Повторить».
  const showJobError = useCallback((message: string): void => {
    setErrorRetry({ kind: 'job' });
    setError(message);
  }, []);
  const [saving, setSaving] = useState(false);
  const [stageBusy, setStageBusy] = useState<StageKey | null>(null);
  const [now, setNow] = useState(() => Date.now());
  // Группы гиперпараметров из YAML-конфига и глобальные настройки; селект
  // ниже показывает локальный выбор профиля для этой задачи.
  const [profiles, setProfiles] = useState<DiarizationProfilesResponse | null>(null);
  const [globalProfile, setGlobalProfile] = useState('meeting');
  const [globalModel, setGlobalModel] = useState('pyannote-segmentation-3.0');
  const [profileDraft, setProfileDraft] = useState<string | null>(null);
  // null while the newest available result is followed automatically;
  // a manual pick makes the selection sticky for this visit.
  const [view, setView] = useState<ResultView | null>(null);
  const [viewPicked, setViewPicked] = useState(false);
  // Режим проверки спикеров: пользователь слушает фрагменты и указывает
  // правильную метку; отчёт хранится отдельно от разметки задачи.
  const [reviewMode, setReviewMode] = useState(false);
  const [reviewBusy, setReviewBusy] = useState(false);
  // Локальные правки: позиция фрагмента (в списке фрагментов с текстом,
  // совпадает с original_labels отчёта) → правильный спикер.  До сохранения
  // они живут только здесь.
  const [reviewEdits, setReviewEdits] = useState<Map<number, string>>(new Map());
  const [reviewComments, setReviewComments] = useState<Map<number, string>>(
    new Map(),
  );
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioFormat = job?.audio_format ?? '';
  // Какие ошибки этапов уже показаны верхним уведомлением — чтобы опрос не
  // дублировал одну и ту же плашку каждые пару секунд.
  const announcedStages = useRef(new Set<string>());

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
          showError(describeError(reason));
        }
      }
    },
    [jobId, onAuthError, showError],
  );

  useEffect(() => {
    setJob(null);
    setError(null);
    setErrorRetry(null);
    setNotice(null);
    setStageBusy(null);
    setView(null);
    setViewPicked(false);
    setProfileDraft(null);
    setReviewMode(false);
    setReviewEdits(new Map());
    setReviewComments(new Map());
    setAudioUrl(null);
    announcedStages.current = new Set();
    void loadJob();
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Список групп и глобальные настройки грузим один раз на страницу задачи;
  // правки YAML подхватятся при следующем открытии страницы.
  useEffect(() => {
    let cancelled = false;
    const load = async (): Promise<void> => {
      const [profilesResult, settingsResult] = await Promise.allSettled([
        getDiarizationProfiles(),
        getSettings(),
      ]);
      if (cancelled) {
        return;
      }
      if (settingsResult.status === 'fulfilled') {
        setGlobalProfile(settingsResult.value.diarize_profile || 'meeting');
        setGlobalModel(settingsResult.value.diarize_model);
      }
      if (profilesResult.status === 'fulfilled') {
        setProfiles(profilesResult.value);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

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

  // Фоновые ошибки приходят только с опросом — их не ловит ни один обработчик
  // клика.  Показываем их в верхней плашке один раз на ошибку, чтобы опрос
  // каждые 1.2 с не заваливал экран одинаковыми сообщениями.
  useEffect(() => {
    if (job === null) {
      return;
    }
    // Забываем прошлые ошибки, когда состояние ушло из «ошибочного»: если
    // задача/этап упадёт снова (например, после «Повторить»), плашка вернётся.
    if (job.status !== 'error' && job.status !== 'cancelled') {
      for (const key of announcedStages.current) {
        if (key.startsWith('job:')) {
          announcedStages.current.delete(key);
        }
      }
    }
    for (const [name, stage] of Object.entries(job.stages ?? {})) {
      if (stage.status === 'error') {
        continue;
      }
      for (const key of announcedStages.current) {
        if (key.startsWith(`${name}:`)) {
          announcedStages.current.delete(key);
        }
      }
    }

    if (job.status === 'error') {
      const message = (job.error ?? 'Не удалось распознать файл.').trim();
      const key = `job:${message}`;
      if (!announcedStages.current.has(key)) {
        announcedStages.current.add(key);
        showJobError(message);
      }
      return;
    }

    if (job.status === 'cancelled') {
      if (!announcedStages.current.has('job:cancelled')) {
        announcedStages.current.add('job:cancelled');
        setNotice('Задача отменена.');
      }
      return;
    }

    for (const [name, stage] of Object.entries(job.stages ?? {})) {
      if (stage.status !== 'error') {
        continue;
      }
      const key = `${name}:${stage.phase}`;
      if (announcedStages.current.has(key)) {
        continue;
      }
      announcedStages.current.add(key);
      const label = STAGE_LABELS[name as StageKey] ?? name;
      const detail = stage.phase.trim();
      showStageError(
        name as StageKey,
        detail === '' || detail === 'Ошибка'
          ? `Этап «${label}» завершился с ошибкой.`
          : `${label}: ${detail}`,
      );
    }
  }, [job, showJobError, showStageError]);

  const recognitionText = job !== null ? (job.text_plain ?? job.text ?? '').trim() : '';
  const speakersText = (job?.text_speakers ?? '').trim();
  const summaryText = (job?.summary_text ?? '').trim();
  const hasSummary = job !== null && (job.summary !== null || summaryText !== '');

  /** Which stage results exist and which one is the newest. */
  const resultViews = useMemo((): { views: ResultView[]; latest: ResultView } | null => {
    if (job === null || job.status !== 'done') {
      return null;
    }
    const views: ResultView[] = ['recognition'];
    if (speakersText !== '') {
      views.push('diarization');
    }
    if (hasSummary) {
      views.push('summary');
    }
    return {
      views,
      latest: hasSummary ? 'summary' : speakersText !== '' ? 'diarization' : 'recognition',
    };
  }, [job, speakersText, hasSummary]);

  // Always show the newest result (саммари → диаризация → расшифровка)
  // until the user picks one explicitly.
  useEffect(() => {
    if (resultViews === null) {
      return;
    }
    if (view !== null && resultViews.views.includes(view)) {
      if (!viewPicked && view !== resultViews.latest) {
        setView(resultViews.latest);
      }
      return;
    }
    // Nothing selected yet, or the selected result disappeared (restart).
    setView(resultViews.latest);
    if (view !== null) {
      setViewPicked(false);
    }
  }, [resultViews, viewPicked, view]);

  const viewText =
    view === 'diarization'
      ? speakersText
      : view === 'summary'
        ? summaryText
        : recognitionText;
  const speakerCount = new Set(
    (job?.segments ?? []).map((segment) => segment.speaker).filter(Boolean),
  ).size;

  // Фрагменты с текстом — тот же список, что сервер кладёт в отчёт проверки
  // (original_labels), поэтому позиции исправлений совпадают.
  const reviewFragments = useMemo(() => {
    if (job === null) {
      return [];
    }
    return job.segments
      .map((segment, index) => ({ ...segment, index }))
      .filter((segment) => segment.text.trim() !== '');
  }, [job]);

  // Метки для выпадающих списков: исходные + уже введённые исправления +
  // «следующий новый» (правило совпадает с серверной валидацией).
  const reviewSpeakerOptions = useMemo(() => {
    const labels = new Set<string>();
    const numbers: number[] = [];
    for (const fragment of reviewFragments) {
      if (fragment.speaker !== null && fragment.speaker !== '') {
        labels.add(fragment.speaker);
      }
    }
    for (const value of reviewEdits.values()) {
      labels.add(value);
    }
    for (const label of labels) {
      const match = /Спикер (\d+)/.exec(label);
      if (match !== null) {
        numbers.push(Number(match[1]));
      }
    }
    const sorted = [...labels].sort((a, b) => {
      const na = /Спикер (\d+)/.exec(a);
      const nb = /Спикер (\d+)/.exec(b);
      const va = na === null ? Number.MAX_SAFE_INTEGER : Number(na[1]);
      const vb = nb === null ? Number.MAX_SAFE_INTEGER : Number(nb[1]);
      return va - vb || a.localeCompare(b);
    });
    sorted.push(`Спикер ${Math.max(0, ...numbers) + 1}`);
    return sorted;
  }, [reviewFragments, reviewEdits]);

  // Аудио для прослушивания: Blob с токеном в заголовке → objectURL.
  // При выходе из режима или смене задачи ссылка освобождается.
  useEffect(() => {
    if (!reviewMode || jobId === '') {
      setAudioUrl(null);
      return;
    }
    if (!PLAYABLE_AUDIO.has(audioFormat.toLowerCase())) {
      setAudioUrl(null);
      return;
    }
    let revoked = false;
    let objectUrl: string | null = null;
    const load = async (): Promise<void> => {
      try {
        const blob = await fetchJobAudio(jobId);
        if (revoked) {
          return;
        }
        objectUrl = URL.createObjectURL(blob);
        setAudioUrl(objectUrl);
      } catch (reason: unknown) {
        if (reason instanceof AuthError) {
          onAuthError();
          return;
        }
        // Плеер необязателен: разметка по тексту и времени работает и без него.
        setAudioUrl(null);
      }
    };
    void load();
    return () => {
      revoked = true;
      if (objectUrl !== null) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [reviewMode, jobId, audioFormat, onAuthError]);

  const enterReview = useCallback(async (): Promise<void> => {
    if (job === null) {
      return;
    }
    setError(null);
    // 404 — отчёта ещё нет: начинаем с пустых исправлений.
    try {
      const report = await getJobReview(job.id);
      setReviewEdits(
        new Map(report.corrections.map((item) => [item.index, item.correct])),
      );
      setReviewComments(
        new Map(
          report.corrections
            .filter((item) => item.comment !== '')
            .map((item) => [item.index, item.comment]),
        ),
      );
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      if (!(reason instanceof ApiError && reason.status === 404)) {
        showError(describeError(reason));
        return;
      }
      setReviewEdits(new Map());
      setReviewComments(new Map());
    }
    setReviewMode(true);
  }, [job, onAuthError, showError]);

  const saveReview = useCallback(async (): Promise<void> => {
    if (job === null) {
      return;
    }
    setReviewBusy(true);
    setError(null);
    try {
      const corrections: ReviewCorrection[] = [...reviewEdits.entries()].map(
        ([index, correct]) => ({
          index,
          correct,
          comment: reviewComments.get(index) ?? '',
        }),
      );
      const report = await putJobReview(job.id, corrections);
      setNotice(
        `Отчёт проверки сохранён: исправлений — ${report.stats.corrected}.`,
      );
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      showError(describeError(reason));
    } finally {
      setReviewBusy(false);
    }
  }, [job, reviewEdits, reviewComments, onAuthError, showError]);

  const resetReview = (): void => {
    setReviewEdits(new Map());
    setReviewComments(new Map());
  };

  const playFragment = (startS: number): void => {
    const audio = audioRef.current;
    if (audio === null) {
      return;
    }
    // Небольшой отступ влево: услышать начало слова, а не его середину.
    audio.currentTime = Math.max(0, startS - 0.2);
    void audio.play().catch(() => undefined);
  };

  const handleOpenFolder = async (): Promise<void> => {
    if (job === null) {
      return;
    }
    try {
      const bridgeResult = await openResultFolder(job.id);
      if (bridgeResult !== null) {
        if (!bridgeResult.ok) {
          showError(bridgeResult.error ?? 'Не удалось открыть папку результата.');
        }
        return;
      }
      await revealJobFolder(job.id);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      showError(describeError(reason));
    }
  };

  const handleRevealSource = async (): Promise<void> => {
    if (job === null) {
      return;
    }
    setError(null);
    try {
      const bridgeResult = await revealSourceFile(job.id);
      if (bridgeResult !== null) {
        if (!bridgeResult.ok) {
          showError(bridgeResult.error ?? 'Не удалось открыть расположение файла.');
        }
        return;
      }
      // Browser fallback (dev mode): same endpoint through the core API.
      await revealJobSource(job.id);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      showError(describeError(reason));
    }
  };

  const handleCopy = async (): Promise<void> => {
    if (viewText === '') {
      return;
    }
    try {
      await navigator.clipboard.writeText(viewText);
      setNotice('Текст скопирован в буфер обмена.');
    } catch {
      showError('Не удалось скопировать текст. Выделите его в поле вручную.');
    }
  };

  const handleSave = async (): Promise<void> => {
    if (job === null || job.status !== 'done' || view === null) {
      return;
    }
    const base = job.filename.replace(/\.[^.]+$/, '') || 'transcript';
    setSaving(true);
    setError(null);
    try {
      const bridgeResult = await saveResultAs(job.id, base, view);
      if (bridgeResult !== null) {
        if (bridgeResult.cancelled) {
          return;
        }
        if (!bridgeResult.ok) {
          showError(bridgeResult.error ?? 'Не удалось сохранить результат.');
          return;
        }
        setNotice(`Результат сохранён: ${bridgeResult.path ?? ''}`);
        return;
      }
      // Browser fallback: download the default format of the active view.
      const response = await fetch(
        `/api/v1/jobs/${encodeURIComponent(job.id)}/export?view=${view}`,
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
        showError(`Сохранение не удалось (HTTP ${response.status}).`);
        return;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${base}.txt`;
      link.click();
      URL.revokeObjectURL(url);
      setNotice('Результат сохранён в папку загрузок браузера.');
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      showError(describeError(reason));
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
      showError(describeError(reason));
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
      showError(describeError(reason));
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
      showError(describeError(reason));
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
    setErrorRetry(null);
    try {
      // Перезапуск диаризации уносит с собой выбранный на странице профиль:
      // "по настройкам" сбрасывает переопределение, группа — сохраняет его.
      const options =
        stage === 'diarization' &&
        action !== 'stop' &&
        job.diarize_profile !== undefined
          ? { diarizeProfile: job.diarize_profile }
          : undefined;
      const updated = await stageAction(job.id, stage, action, options);
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
      showError(describeError(reason));
      await loadJob(true);
    } finally {
      setStageBusy(null);
    }
  };

  const recognitionDone = job?.stages?.recognition?.status === 'done';
  const canStartDiarization =
    job !== null && job.status === 'done' && recognitionDone && job.segments.length > 0;
  const canStartSummary =
    job !== null && job.status === 'done' && job.text.trim() !== '';

  const engine = globalModel === 'pyannote-segmentation-3.0' ? 'pyannote' : 'wespeaker';
  const availableProfiles = profiles?.[engine] ?? [];
  const effectiveProfile = profileDraft ?? job?.diarize_profile ?? '';
  const profileOptions: Array<{ value: string; label: string }> = [
    {
      value: '',
      label:
        job?.diarize_profile !== undefined && job.diarize_profile !== ''
          ? 'По настройкам'
          : 'По настройкам (текущая)',
    },
    ...availableProfiles.map((group) => ({ value: group.id, label: group.label })),
  ];
  const globalGroupLabel =
    availableProfiles.find((group) => group.id === globalProfile)?.label ?? globalProfile;
  const profileValue = effectiveProfile === '' ? '' : effectiveProfile;
  if (
    effectiveProfile !== '' &&
    availableProfiles.length > 0 &&
    !availableProfiles.some((group) => group.id === effectiveProfile)
  ) {
    profileOptions.push({ value: effectiveProfile, label: effectiveProfile });
  }
  const profileLabel =
    profileOptions.find((option) => option.value === profileValue)?.label ??
    globalGroupLabel;

  const handleProfileSelect = async (value: string | undefined): Promise<void> => {
    if (job === null || value === undefined) {
      return;
    }
    const previous = profileDraft ?? job.diarize_profile ?? '';
    setProfileDraft(value);
    try {
      const updated = await setJobDiarizeProfile(job.id, value);
      setJob(updated);
      setNotice(
        value === ''
          ? 'Профиль «По настройкам»: при запуске диаризации будет применена глобальная группа.'
          : 'Группа параметров сохранена для этой задачи.',
      );
    } catch (reason: unknown) {
      setProfileDraft(previous);
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      showError(describeError(reason));
    }
  };

  const subtitle =
    job === null
      ? 'Загрузка задачи…'
      : [
          statusLabel(job.status),
          formatDate(job.created_at),
          job.audio_format.toUpperCase(),
          formatBytes(job.size_bytes),
          job.duration_s !== null ? `аудио ${formatDuration(job.duration_s)}` : null,
          job.status !== 'queued'
            ? `время выполнения ${formatDuration(jobElapsedSeconds(job, now))}`
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

      {/*
        Единая верхняя плашка: и сообщения, и ошибки.  Раньше ошибки этапов
        (например, сбой саммаризации) показывались отдельной плашкой внизу
        страницы — теперь всё выводится здесь, рядом с заголовком.
      */}
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
                  setErrorRetry(null);
                }}
              />
            }
          >
            {errorRetry?.kind === 'stage' ? (
              <Button
                appearance="subtle"
                icon={<ArrowClockwiseRegular />}
                disabled={stageBusy !== null}
                onClick={() => {
                  void handleStageAction(errorRetry.stage, 'restart');
                }}
              >
                Перезапустить
              </Button>
            ) : errorRetry?.kind === 'job' ? (
              <Button
                appearance="subtle"
                icon={<HistoryRegular />}
                onClick={() => {
                  void handleRetry();
                }}
              >
                Повторить
              </Button>
            ) : null}
          </MessageBarActions>
        </MessageBar>
      ) : notice !== null ? (
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
          <Button
            size="small"
            icon={<DocumentArrowRightRegular />}
            onClick={() => {
              void handleRevealSource();
            }}
            disabled={job === null}
          >
            Исходный файл
          </Button>
          <Button
            size="small"
            icon={saving ? <Spinner size="tiny" /> : <SaveRegular />}
            disabled={
              job === null ||
              job.status !== 'done' ||
              saving ||
              (job.text.trim() === '' && job.segments.length === 0)
            }
            onClick={() => {
              void handleSave();
            }}
          >
            Сохранить как…
          </Button>
          <Button
            size="small"
            icon={<CopyRegular />}
            disabled={viewText === ''}
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
              now={now}
              onAction={(action) => {
                void handleStageAction('recognition', action);
              }}
            />
            <StageControl
              name="diarization"
              stage={job.stages.diarization}
              canStart={canStartDiarization}
              busy={stageBusy !== null}
              now={now}
              onAction={(action) => {
                void handleStageAction('diarization', action);
              }}
            />
            <StageControl
              name="summary"
              stage={job.stages.summary}
              canStart={canStartSummary}
              busy={stageBusy !== null}
              now={now}
              onAction={(action) => {
                void handleStageAction('summary', action);
              }}
            />
          </div>
        ) : null}

        {job !== null ? (
          <div className="job-profile-row" aria-label="Группа параметров диаризации">
            <span className="job-profile-label">Группа параметров:</span>
            <Dropdown
              aria-label="Группа параметров диаризации для этой задачи"
              value={profileLabel}
              selectedOptions={[profileValue]}
              disabled={stageBusy !== null || profileOptions.length <= 1}
              onOptionSelect={(_event, data) => {
                void handleProfileSelect(data.optionValue);
              }}
            >
              {profileOptions.map((option) => (
                <Option key={option.value} value={option.value}>
                  {option.label}
                </Option>
              ))}
            </Dropdown>
            {effectiveProfile === '' ? (
              <span className="job-stage-profile-hint">
                Глобальная группа: {globalGroupLabel}
              </span>
            ) : null}
          </div>
        ) : null}
      </section>

      {job !== null && isActive(job.status) ? (
        <div className="transcript-progress job-live-progress" role="status" aria-live="polite">
          <ProgressBar value={job.progress} thickness="medium" />
          <span>{job.phase}</span>
        </div>
      ) : null}

      {job !== null &&
      job.status === 'done' &&
      resultViews !== null &&
      view !== null ? (
        <section
          className="transcript-panel job-transcript-panel surface"
          aria-labelledby="job-result-title"
        >
          <div className="transcript-head">
            <div>
              <h2 id="job-result-title" className="files-title">
                {RESULT_LABELS[view]}
              </h2>
              {view !== 'summary' ? (
                <p className="job-meta">
                  {job.segments.length > 0
                    ? `фрагментов: ${job.segments.length}`
                    : 'текст целиком'}
                  {view === 'diarization' && speakerCount > 0
                    ? ` · спикеров: ${speakerCount}`
                    : ''}
                </p>
              ) : null}
            </div>
            <div className="transcript-head-actions">
              {view === 'diarization' && job.segments.length > 0 ? (
                <Button
                  appearance={reviewMode ? 'primary' : 'secondary'}
                  icon={<PlayRegular />}
                  disabled={reviewBusy}
                  onClick={() => void enterReview()}
                >
                  Проверить спикеров
                </Button>
              ) : null}
              {resultViews.views.length > 1 ? (
                <Dropdown
                  aria-label="Результат разбора"
                  className="job-result-switch"
                  value={RESULT_LABELS[view]}
                  selectedOptions={[view]}
                  onOptionSelect={(_event, data) => {
                    if (data.optionValue !== undefined) {
                      setView(data.optionValue as ResultView);
                      setViewPicked(true);
                    }
                  }}
                >
                  {resultViews.views.map((key) => (
                    <Option key={key} value={key}>
                      {RESULT_LABELS[key]}
                    </Option>
                  ))}
                </Dropdown>
              ) : null}
            </div>
          </div>
          {view === 'summary' && job.summary !== null ? (
            <SummaryBody summary={job.summary} />
          ) : reviewMode && view === 'diarization' ? (
            <div className="review-pane">
              {audioUrl !== null ? (
                <audio
                  ref={audioRef}
                  className="review-audio"
                  controls
                  preload="metadata"
                  src={audioUrl}
                />
              ) : (
                <p className="review-hint">
                  Прослушивание недоступно для этого формата — сверяйтесь по
                  времени и тексту.
                </p>
              )}
              <div className="review-toolbar">
                <span className="review-count">
                  Исправлено: {reviewEdits.size} из {reviewFragments.length}
                </span>
                <div className="review-toolbar-buttons">
                  <Button
                    appearance="primary"
                    icon={<SaveRegular />}
                    disabled={reviewBusy}
                    onClick={() => void saveReview()}
                  >
                    Сохранить отчёт
                  </Button>
                  <Button
                    icon={<ArrowClockwiseRegular />}
                    disabled={reviewBusy || reviewEdits.size === 0}
                    onClick={resetReview}
                  >
                    Сбросить
                  </Button>
                  <Button
                    icon={<DismissRegular />}
                    disabled={reviewBusy}
                    onClick={() => setReviewMode(false)}
                  >
                    Готово
                  </Button>
                </div>
              </div>
              <ul className="review-list">
                {reviewFragments.map((fragment, position) => {
                  const original = fragment.speaker ?? '';
                  const effective = reviewEdits.get(position) ?? original;
                  const edited = reviewEdits.has(position);
                  const rowOptions =
                    original === '' || reviewSpeakerOptions.includes(original)
                      ? reviewSpeakerOptions
                      : [...reviewSpeakerOptions, original];
                  return (
                    <li
                      key={position}
                      className={
                        edited ? 'review-item review-item-edited' : 'review-item'
                      }
                    >
                      <div className="review-item-top">
                        <button
                          type="button"
                          className="review-play"
                          aria-label={`Прослушать фрагмент с ${formatClock(fragment.start_s)}`}
                          disabled={audioUrl === null}
                          onClick={() => playFragment(fragment.start_s)}
                        >
                          <PlayRegular />
                        </button>
                        <span className="review-time">
                          {formatClock(fragment.start_s)}–
                          {formatClock(fragment.end_s)}
                        </span>
                        <select
                          className="review-select"
                          value={effective}
                          aria-label="Спикер фрагмента"
                          onChange={(event) => {
                            const value = event.target.value;
                            setReviewEdits((prev) => {
                              const next = new Map(prev);
                              if (value === original) {
                                next.delete(position);
                              } else {
                                next.set(position, value);
                              }
                              return next;
                            });
                          }}
                        >
                          {original === '' ? <option value="">—</option> : null}
                          {rowOptions.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                        {edited ? (
                          <span className="review-was">
                            было: {original === '' ? '—' : original}
                          </span>
                        ) : null}
                      </div>
                      <div className="review-text">{fragment.text}</div>
                      {edited ? (
                        <input
                          className="review-comment"
                          type="text"
                          placeholder="Комментарий (необязательно)"
                          value={reviewComments.get(position) ?? ''}
                          onChange={(event) => {
                            const value = event.target.value;
                            setReviewComments((prev) => {
                              const next = new Map(prev);
                              if (value === '') {
                                next.delete(position);
                              } else {
                                next.set(position, value);
                              }
                              return next;
                            });
                          }}
                        />
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : (
            <Textarea
              className="job-transcript"
              readOnly
              resize="none"
              value={viewText === '' ? 'Речь не распознана.' : viewText}
              aria-label={RESULT_LABELS[view]}
            />
          )}
        </section>
      ) : null}
    </div>
  );
}
