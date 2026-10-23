import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { Button } from '@fluentui/react-components';
import { ArrowClockwiseRegular } from '@fluentui/react-icons';
import { ErrorState } from './PageState';

interface AppErrorBoundaryProps {
  children: ReactNode;
}

interface AppErrorBoundaryState {
  failed: boolean;
}

/**
 * Keeps a rendering error inside one section: the sidebar and navigation stay
 * usable instead of the whole window going blank.
 */
export class AppErrorBoundary extends Component<
  AppErrorBoundaryProps,
  AppErrorBoundaryState
> {
  constructor(props: AppErrorBoundaryProps) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError(): AppErrorBoundaryState {
    return { failed: true };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('UI section failed to render', error, info.componentStack);
  }

  override render(): ReactNode {
    if (this.state.failed) {
      return (
        <ErrorState
          title="Раздел не открылся"
          description="Произошла ошибка интерфейса. Перейдите в другой раздел или перезапустите окно."
          action={
            <Button
              icon={<ArrowClockwiseRegular />}
              onClick={() => {
                window.location.reload();
              }}
            >
              Перезагрузить
            </Button>
          }
        />
      );
    }

    return this.props.children;
  }
}
