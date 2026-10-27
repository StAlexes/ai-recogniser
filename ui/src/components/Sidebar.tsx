import type { ReactNode } from 'react';
import {
  Popover,
  PopoverSurface,
  PopoverTrigger,
} from '@fluentui/react-components';
import {
  BookOpenRegular,
  DismissRegular,
  DoorRegular,
  InfoRegular,
  ListRegular,
  SettingsRegular,
} from '@fluentui/react-icons';
import { minimizeToTray, quitApp } from '../desktop';
import { StatusDot } from './StatusDot';

export type PageKey = 'queue' | 'knowledge' | 'settings';

interface NavItem {
  key: PageKey;
  label: string;
  icon: ReactNode;
}

const NAV_ITEMS: readonly NavItem[] = [
  { key: 'queue', label: 'Очередь', icon: <ListRegular /> },
  { key: 'knowledge', label: 'База знаний', icon: <BookOpenRegular /> },
  { key: 'settings', label: 'Настройки', icon: <SettingsRegular /> },
];

interface SidebarProps {
  active: PageKey;
  onNavigate: (page: PageKey) => void;
  coreOnline: boolean;
  coreChecked: boolean;
  version: string;
}

export function Sidebar({
  active,
  onNavigate,
  coreOnline,
  coreChecked,
  version,
}: SidebarProps) {
  const statusText = !coreChecked
    ? 'Проверка ядра…'
    : coreOnline
      ? 'Ядро доступно'
      : 'Ядро недоступно';

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          AI
        </span>
        <span className="brand-text">
          <span className="brand-name">AI Recogniser</span>
          <span className="brand-sub">Локальное распознавание</span>
        </span>
      </div>

      <nav className="nav" aria-label="Основная навигация">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.key}
            type="button"
            className="nav-item"
            aria-current={active === item.key ? 'page' : undefined}
            onClick={() => {
              onNavigate(item.key);
            }}
          >
            <span className="nav-icon" aria-hidden="true">
              {item.icon}
            </span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-spacer" />

      <Popover withArrow positioning="above-start">
        <PopoverTrigger disableButtonEnhancement>
          <button
            type="button"
            className="sidebar-footer"
            aria-label={`О программе. Версия ${version}. ${statusText}`}
          >
            <span className="footer-version">Версия {version}</span>
            <span className="footer-status">
              <StatusDot online={coreOnline} checked={coreChecked} />
              <span>{statusText}</span>
            </span>
          </button>
        </PopoverTrigger>

        <PopoverSurface className="about-surface" aria-label="О программе">
          <div className="about-head">
            <span className="brand-mark brand-mark-small" aria-hidden="true">
              AI
            </span>
            <div>
              <div className="about-name">AI Recogniser</div>
              <div className="about-note">Локальное распознавание</div>
            </div>
            <button
              type="button"
              className="about-close"
              aria-label="Свернуть в трей"
              title="Свернуть в трей"
              onClick={() => {
                minimizeToTray();
              }}
            >
              <DismissRegular fontSize={16} aria-hidden="true" />
            </button>
          </div>

          <button
            type="button"
            className="about-exit"
            onClick={() => {
              quitApp();
            }}
          >
            <DoorRegular fontSize={16} aria-hidden="true" />
            Выход
          </button>

          <dl className="about-list">
            <div className="about-row">
              <dt>Версия</dt>
              <dd>{version}</dd>
            </div>
            <div className="about-row">
              <dt>Ядро</dt>
              <dd className="about-status">
                <StatusDot online={coreOnline} checked={coreChecked} />
                <span>{statusText}</span>
              </dd>
            </div>
            <div className="about-row">
              <dt>План</dt>
              <dd>
                <span className="link-like">docs/PLAN.md</span>
              </dd>
            </div>
          </dl>

          <div className="about-footer">
            <InfoRegular aria-hidden="true" />
            <span>Данные остаются на этом компьютере.</span>
          </div>
        </PopoverSurface>
      </Popover>
    </aside>
  );
}
