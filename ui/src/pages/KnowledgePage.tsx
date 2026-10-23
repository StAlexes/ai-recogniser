import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import {
  Badge,
  Button,
  Input,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
  Spinner,
} from '@fluentui/react-components';
import {
  ArrowClockwiseRegular,
  BookOpenRegular,
  DatabaseRegular,
  DismissRegular,
  SearchRegular,
  SparkleRegular,
} from '@fluentui/react-icons';
import {
  ApiError,
  AuthError,
  askRag,
  getRagStatus,
  searchRag,
} from '../api';
import type { RagSource, RagStatus } from '../api';
import { PageHeader } from '../components/PageHeader';

interface KnowledgePageProps {
  onAuthError: () => void;
}

function describeError(reason: unknown): string {
  if (reason instanceof ApiError) {
    return reason.message;
  }
  return 'Не удалось получить ответ от RAG.';
}

function formatNumber(value: number | null): string {
  if (value === null) {
    return '—';
  }
  return value.toLocaleString('ru-RU');
}

function formatScore(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) {
    return '';
  }
  return value.toFixed(3).replace('.', ',');
}

function statusColor(status: RagStatus | null): 'success' | 'danger' | 'informative' {
  if (status === null || status.enabled === false) {
    return 'informative';
  }
  return status.reachable ? 'success' : 'danger';
}

function statusLabel(status: RagStatus | null): string {
  if (status === null) {
    return 'Проверка…';
  }
  if (!status.enabled) {
    return 'Выключено';
  }
  return status.reachable ? 'Подключено' : 'Недоступно';
}

function SourceCard({ source }: { source: RagSource }) {
  const score = formatScore(source.score);
  return (
    <article className="rag-source surface">
      <div className="rag-source-head">
        <span className="rag-source-name" title={source.source}>
          {source.source || 'Без имени'}
        </span>
        {score !== '' ? (
          <Badge appearance="tint" color="informative" size="small">
            score {score}
          </Badge>
        ) : null}
      </div>
      {source.breadcrumb !== '' ? (
        <p className="rag-source-breadcrumb">{source.breadcrumb}</p>
      ) : null}
      <p className="rag-source-text">{source.text}</p>
    </article>
  );
}

export function KnowledgePage({ onAuthError }: KnowledgePageProps) {
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [asking, setAsking] = useState(false);
  const [results, setResults] = useState<RagSource[] | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = async (): Promise<void> => {
    setStatusLoading(true);
    try {
      setStatus(await getRagStatus());
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    } finally {
      setStatusLoading(false);
    }
  };

  useEffect(() => {
    void loadStatus();
    // onAuthError is stable in App; loadStatus intentionally runs once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runSearch = async (): Promise<void> => {
    const clean = query.trim();
    if (clean === '' || searching || asking) {
      return;
    }
    setSearching(true);
    setError(null);
    setAnswer(null);
    try {
      const response = await searchRag(clean);
      setResults(response.results);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    } finally {
      setSearching(false);
    }
  };

  const runAsk = async (): Promise<void> => {
    const clean = query.trim();
    if (clean === '' || searching || asking) {
      return;
    }
    setAsking(true);
    setError(null);
    try {
      const response = await askRag(clean);
      setAnswer(response.answer);
      setResults(response.sources);
    } catch (reason: unknown) {
      if (reason instanceof AuthError) {
        onAuthError();
        return;
      }
      setError(describeError(reason));
    } finally {
      setAsking(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    void runSearch();
  };

  const busy = searching || asking;
  const disabled = status === null || !status.enabled;

  return (
    <>
      <PageHeader
        title="База знаний"
        subtitle="Поиск по локальному ragkb и ответы со ссылками на источники"
        badge={
          <Badge appearance="tint" color={statusColor(status)}>
            {statusLoading ? 'Проверка…' : statusLabel(status)}
          </Badge>
        }
        actions={
          <Button
            icon={statusLoading ? <Spinner size="tiny" /> : <ArrowClockwiseRegular />}
            disabled={statusLoading}
            onClick={() => {
              void loadStatus();
            }}
          >
            Проверить связь
          </Button>
        }
      />

      <section className="kb-hero surface" aria-labelledby="kb-hero-title">
        <span className="kb-hero-icon" aria-hidden="true">
          <BookOpenRegular fontSize={26} />
        </span>
        <div className="kb-hero-body">
          <h2 id="kb-hero-title" className="kb-hero-title">
            Подключение к ragkb
          </h2>
          <p className="kb-hero-text">
            Приложение использует уже работающий локальный RAG-сервис: индекс, эмбеддинги и LLM
            остаются в ragkb. Ничего не копируется и не отправляется в облако.
          </p>
          <div className="kb-status-grid">
            <span>
              Адрес: <code>{status?.configured_url ?? '—'}</code>
            </span>
            <span>
              Чанков: <strong>{formatNumber(status?.chunks ?? null)}</strong>
            </span>
            <span>
              Модель ответов: <strong>{status?.gen_model ?? '—'}</strong>
            </span>
            <span>
              Задержка: <strong>{status?.latency_ms !== null && status?.latency_ms !== undefined ? `${status.latency_ms} мс` : '—'}</strong>
            </span>
          </div>
          {status?.error !== null && status?.error !== undefined ? (
            <p className="kb-hero-note kb-status-error">{status.error}</p>
          ) : null}
          {status?.indexing === true ? (
            <p className="kb-hero-note">ragkb сейчас индексирует документы; поиск может быть медленнее.</p>
          ) : null}
        </div>
      </section>

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

      <form className="rag-search surface" onSubmit={handleSubmit}>
        <Input
          className="rag-search-input"
          value={query}
          placeholder="Например: как настроить SSL в T-FLEX DOCs?"
          aria-label="Запрос к базе знаний"
          contentBefore={<SearchRegular />}
          disabled={disabled || busy}
          onChange={(_event, data) => {
            setQuery(data.value);
          }}
        />
        <div className="rag-search-actions">
          <Button
            type="submit"
            appearance="primary"
            icon={searching ? <Spinner size="tiny" /> : <SearchRegular />}
            disabled={disabled || busy || query.trim() === ''}
          >
            {searching ? 'Ищу…' : 'Найти'}
          </Button>
          <Button
            type="button"
            icon={asking ? <Spinner size="tiny" /> : <SparkleRegular />}
            disabled={disabled || busy || query.trim() === ''}
            onClick={() => {
              void runAsk();
            }}
          >
            {asking ? 'Думаю…' : 'Спросить RAG'}
          </Button>
        </div>
      </form>

      {answer !== null ? (
        <section className="rag-answer surface" aria-labelledby="rag-answer-title">
          <h2 id="rag-answer-title" className="files-title">
            Ответ
          </h2>
          <p className="rag-answer-text">{answer}</p>
        </section>
      ) : null}

      {results !== null ? (
        <section className="rag-results" aria-labelledby="rag-results-title" aria-live="polite">
          <div className="rag-results-head">
            <h2 id="rag-results-title" className="files-title">
              Источники
            </h2>
            <Badge appearance="tint" color="informative" size="small">
              {results.length}
            </Badge>
          </div>
          {results.length === 0 ? (
            <p className="empty-note">По этому запросу ничего не найдено.</p>
          ) : (
            results.map((source, index) => (
              <SourceCard key={`${source.id ?? source.source}-${index}`} source={source} />
            ))
          )}
        </section>
      ) : null}

      <div className="kb-grid">
        <article className="feature-card surface">
          <span className="feature-icon" aria-hidden="true">
            <DatabaseRegular fontSize={20} />
          </span>
          <h3 className="feature-title">Индекс снаружи</h3>
          <p className="feature-text">
            ragkb хранит чанки и эмбеддинги в своём каталоге. Приложение подключается к готовому
            сервису по HTTP и не дублирует индекс.
          </p>
        </article>

        <article className="feature-card surface">
          <span className="feature-icon" aria-hidden="true">
            <SearchRegular fontSize={20} />
          </span>
          <h3 className="feature-title">Поиск и источники</h3>
          <p className="feature-text">
            Гибридный поиск ragkb возвращает фрагменты, breadcrumb и score. Ответ LLM строится
            только по найденным источникам.
          </p>
        </article>

        <article className="feature-card surface">
          <span className="feature-icon" aria-hidden="true">
            <SparkleRegular fontSize={20} />
          </span>
          <h3 className="feature-title">Без облака</h3>
          <p className="feature-text">
            По умолчанию адрес — <code>127.0.0.1</code>. Изменить его и выключить интеграцию можно в
            настройках приложения.
          </p>
        </article>
      </div>
    </>
  );
}
