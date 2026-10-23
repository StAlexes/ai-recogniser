import { useEffect, useState } from 'react';
import { getHealthz } from '../api';

export interface CoreStatus {
  online: boolean;
  checked: boolean;
}

const POLL_INTERVAL_MS = 5_000;

/**
 * Polls the public liveness endpoint. Checks are sequential, so a slow core
 * never stacks requests. The loop stops when the tab is hidden and refreshes
 * immediately when it becomes visible again.
 */
export function useCoreStatus(): CoreStatus {
  const [online, setOnline] = useState(false);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;

    const schedule = (): void => {
      if (disposed || document.hidden) {
        return;
      }
      timer = window.setTimeout(() => {
        void check();
      }, POLL_INTERVAL_MS);
    };

    const check = async (): Promise<void> => {
      try {
        await getHealthz();
        if (!disposed) {
          setOnline(true);
        }
      } catch {
        if (!disposed) {
          setOnline(false);
        }
      } finally {
        if (!disposed) {
          setChecked(true);
          schedule();
        }
      }
    };

    const onVisibilityChange = (): void => {
      if (document.hidden) {
        if (timer !== undefined) {
          window.clearTimeout(timer);
          timer = undefined;
        }
        return;
      }
      if (timer !== undefined) {
        window.clearTimeout(timer);
        timer = undefined;
      }
      void check();
    };

    void check();
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      disposed = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, []);

  return { online, checked };
}
