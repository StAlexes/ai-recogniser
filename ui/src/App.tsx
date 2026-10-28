import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { FluentProvider } from '@fluentui/react-components';
import { AuthError, getHealth } from './api';
import type { HealthResponse } from './api';
import { AppErrorBoundary } from './components/AppErrorBoundary';
import { AuthErrorPanel } from './components/AuthErrorPanel';
import { Sidebar } from './components/Sidebar';
import type { PageKey } from './components/Sidebar';
import { useAppTheme } from './hooks/useAppTheme';
import { useCoreStatus } from './hooks/useCoreStatus';
import { KnowledgePage } from './pages/KnowledgePage';
import { ModelsPage } from './pages/ModelsPage';
import { QueuePage } from './pages/QueuePage';
import { SettingsPage } from './pages/SettingsPage';
import { darkTheme, lightTheme } from './theme';

const FALLBACK_VERSION = '0.1.0-dev';

const PAGE_TITLES: Record<PageKey, string> = {
  queue: 'Очередь',
  knowledge: 'База знаний',
  settings: 'Настройки',
  models: 'Менеджер ИИ-моделей',
};

function initialJobFromUrl(): string | null {
  const value = new URLSearchParams(window.location.search).get('job');
  return value !== null && value.trim() !== '' ? value : null;
}

export function App() {
  const { resolved, setMode } = useAppTheme();
  const core = useCoreStatus();
  const [page, setPage] = useState<PageKey>('queue');
  const [initialJobId, setInitialJobId] = useState<string | null>(initialJobFromUrl);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [authLost, setAuthLost] = useState(false);
  const mainRef = useRef<HTMLElement>(null);

  const navigate = useCallback((next: PageKey): void => {
    setPage(next);
    if (next !== 'queue') {
      setInitialJobId(null);
    }
  }, []);

  const handleAuthError = useCallback((): void => {
    setAuthLost(true);
  }, []);

  const handleRetry = useCallback((): void => {
    window.location.reload();
  }, []);

  useEffect(() => {
    if (!core.online || authLost) {
      return;
    }
    let cancelled = false;
    getHealth()
      .then((response) => {
        if (!cancelled) {
          setHealth(response);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled && reason instanceof AuthError) {
          setAuthLost(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [core.online, authLost]);

  useEffect(() => {
    const label = authLost ? 'Нет доступа' : PAGE_TITLES[page];
    document.title = `${label} — AI Recogniser`;
    mainRef.current?.scrollTo({ top: 0 });
  }, [page, authLost]);

  const renderPage = (): ReactNode => {
    switch (page) {
      case 'queue':
        return (
          <QueuePage
            onAuthError={handleAuthError}
            initialJobId={initialJobId}
            onInitialJobOpened={() => {
              setInitialJobId(null);
            }}
          />
        );
      case 'knowledge':
        return <KnowledgePage onAuthError={handleAuthError} />;
      case 'settings':
        return (
          <SettingsPage
            onAuthError={handleAuthError}
            onThemeModeChange={setMode}
            onManageModels={() => {
              navigate('models');
            }}
          />
        );
      case 'models':
        return (
          <ModelsPage
            onAuthError={handleAuthError}
            onBack={() => {
              navigate('settings');
            }}
          />
        );
      default:
        return null;
    }
  };

  return (
    <FluentProvider
      theme={resolved === 'dark' ? darkTheme : lightTheme}
      className="app-provider"
      // Portals only need the theme CSS variables. Copying the provider root
      // classes would also copy `.app-provider { height: 100% }` and the opaque
      // surface background, turning every tooltip mount node into a full-screen
      // overlay above the application.
      applyStylesToPortals={false}
    >
      <div className="app-shell">
        <a className="skip-link" href="#main-content">
          Перейти к содержимому
        </a>

        <Sidebar
          active={page === 'models' ? 'settings' : page}
          onNavigate={navigate}
          coreOnline={core.online}
          coreChecked={core.checked}
          version={health?.version ?? FALLBACK_VERSION}
        />

        <main id="main-content" ref={mainRef} className="main-area" tabIndex={-1}>
          <div className="main-inner">
            <AppErrorBoundary key={authLost ? 'auth' : page}>
              {authLost ? <AuthErrorPanel onRetry={handleRetry} /> : renderPage()}
            </AppErrorBoundary>
          </div>
        </main>
      </div>
    </FluentProvider>
  );
}
