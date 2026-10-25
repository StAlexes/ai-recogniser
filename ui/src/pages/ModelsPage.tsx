import { useEffect, useState } from 'react';
import { Badge, Button } from '@fluentui/react-components';
import {
  ArrowClockwiseRegular,
  CubeRegular,
  DesktopRegular,
} from '@fluentui/react-icons';
import { ApiError, AuthError, getModels, getSystemProfile } from '../api';
import type { ModelItem, ModelsResponse, SystemProfile } from '../api';
import { DisabledAction } from '../components/DisabledAction';
import { ErrorState, LoadingState } from '../components/PageState';
import { PageHeader } from '../components/PageHeader';
import { formatGb, formatModelSize, pluralCores } from '../format';

function kindLabel(kind: string): string {
  if (kind === 'asr') {
    return 'ASR · распознавание';
  }
  if (kind === 'llm') {
    return 'LLM · саммари';
  }
  return kind.toUpperCase();
}

function licenseLabel(license: string): string {
  return license === 'check before release' ? 'Уточняется' : license;
}

function profileSummary(profile: SystemProfile): string {
  const hardware = profile.hardware;
  if (hardware === undefined) {
    return 'Профиль оборудования недоступен';
  }

  const parts: string[] = [];
  if (hardware.cpu_cores !== null) {
    parts.push(pluralCores(hardware.cpu_cores));
  }
  parts.push(`${formatGb(hardware.ram_total_gb)} RAM`);
  const gpu = hardware.gpu?.find((item) => item.virtual !== true) ?? hardware.gpu?.[0];
  if (gpu !== undefined) {
    parts.push(gpu.name);
  }
  return parts.join(' · ');
}

interface ModelCardProps {
  model: ModelItem;
  installed: boolean;
}

function ModelCard({ model, installed }: ModelCardProps) {
  return (
    <article className="model-card surface">
      <div className="model-head">
        <span className="model-icon" aria-hidden="true">
          <CubeRegular fontSize={20} />
        </span>
        <div className="model-heading">
          <h2 className="model-title">{model.title}</h2>
          <span className="model-kind">{kindLabel(model.kind)}</span>
        </div>
        <div className="model-badges">
          {installed ? (
            <Badge appearance="tint" color="success" size="small">
              Установлена
            </Badge>
          ) : null}
        </div>
      </div>

      <p className="model-description">{model.note ?? 'Описание недоступно'}</p>

      <dl className="model-facts">
        <div>
          <dt>Размер</dt>
          <dd>{formatModelSize(model.size_mb ?? 0)}</dd>
        </div>
        <div>
          <dt>Лицензия</dt>
          <dd>{licenseLabel(model.license ?? '—')}</dd>
        </div>
      </dl>

      <div className="model-actions">
        <DisabledAction action="Установить" tooltip="Появится в M1">
          <Button appearance="primary" disabled>
            Установить
          </Button>
        </DisabledAction>
      </div>
    </article>
  );
}

interface ModelsPageProps {
  onAuthError: () => void;
}

export function ModelsPage({ onAuthError }: ModelsPageProps) {
  const [catalog, setCatalog] = useState<ModelsResponse | null>(null);
  const [profile, setProfile] = useState<SystemProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);

    getModels()
      .then((response) => {
        if (!cancelled) {
          setCatalog(response);
        }
      })
      .catch((reason: unknown) => {
        if (cancelled) {
          return;
        }
        if (reason instanceof AuthError) {
          onAuthError();
          return;
        }
        setError(
          reason instanceof ApiError
            ? reason.message
            : 'Не удалось загрузить каталог моделей.',
        );
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    getSystemProfile()
      .then((response) => {
        if (!cancelled) {
          setProfile(response);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled && reason instanceof AuthError) {
          onAuthError();
        }
        // Hardware details are optional; the page works without them.
      });

    return () => {
      cancelled = true;
    };
  }, [reloadKey, onAuthError]);

  const models = catalog?.items ?? [];
  const installed = catalog?.installed ?? [];
  const recommended = models.find((model) => model.kind === 'asr');

  return (
    <>
      <PageHeader
        title="Модели"
        subtitle="Реестр локальных моделей для распознавания и саммари"
        actions={
          <Button
            appearance="subtle"
            icon={<ArrowClockwiseRegular />}
            disabled={loading}
            onClick={() => {
              setReloadKey((value) => value + 1);
            }}
          >
            Обновить
          </Button>
        }
      />

      {profile !== null ? (
        <div className="profile-strip surface">
          <span className="profile-icon" aria-hidden="true">
            <DesktopRegular fontSize={20} />
          </span>
          <div className="profile-text">
            <span className="profile-title">Этот ПК</span>
            <span className="profile-detail">{profileSummary(profile)}</span>
          </div>
          {recommended !== undefined ? (
            <Badge appearance="tint" color="brand" className="profile-badge">
              Рекомендуем: {recommended.title}
            </Badge>
          ) : null}
        </div>
      ) : null}

      {loading ? (
        <LoadingState label="Загружаем каталог моделей…" />
      ) : error !== null ? (
        <ErrorState
          title="Каталог недоступен"
          description={error}
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
      ) : models.length === 0 ? (
        <ErrorState
          title="Каталог пуст"
          description="Ядро не вернуло ни одной модели. Проверьте манифест моделей."
        />
      ) : (
        <div className="models-grid">
          {models.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              installed={installed.includes(model.id)}
            />
          ))}
        </div>
      )}
    </>
  );
}
