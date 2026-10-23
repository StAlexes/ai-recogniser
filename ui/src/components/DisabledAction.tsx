import type { ReactElement } from 'react';
import { Tooltip } from '@fluentui/react-components';

interface DisabledActionProps {
  action: string;
  tooltip: string;
  children: ReactElement;
}

/**
 * A disabled Fluent button does not emit pointer or focus events, so the
 * tooltip is anchored to a focusable wrapper. The wrapper keeps the reason
 * available to keyboard and screen-reader users.
 */
export function DisabledAction({ action, tooltip, children }: DisabledActionProps) {
  return (
    <Tooltip content={tooltip} relationship="description" withArrow>
      <span
        className="disabled-tooltip"
        tabIndex={0}
        role="group"
        aria-label={`${action}. ${tooltip}`}
      >
        {children}
      </span>
    </Tooltip>
  );
}
