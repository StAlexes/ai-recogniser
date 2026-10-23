import type { ReactNode } from 'react';
import { Spinner } from '@fluentui/react-components';
import { WarningRegular } from '@fluentui/react-icons';

interface LoadingStateProps {
  label: string;
}

export function LoadingState({ label }: LoadingStateProps) {
  return (
    <div className="state-panel surface" role="status" aria-live="polite">
      <Spinner size="medium" label={label} />
    </div>
  );
}

interface ErrorStateProps {
  title: string;
  description: string;
  action?: ReactNode;
}

export function ErrorState({ title, description, action }: ErrorStateProps) {
  return (
    <div className="state-panel surface" role="alert">
      <span className="state-icon" aria-hidden="true">
        <WarningRegular fontSize={24} />
      </span>
      <h2 className="state-title">{title}</h2>
      <p className="state-text">{description}</p>
      {action}
    </div>
  );
}
