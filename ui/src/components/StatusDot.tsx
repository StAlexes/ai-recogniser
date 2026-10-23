interface StatusDotProps {
  online: boolean;
  checked: boolean;
}

/** Small state indicator used in the sidebar and the About popover. */
export function StatusDot({ online, checked }: StatusDotProps) {
  const state = checked ? (online ? 'online' : 'offline') : 'unknown';
  return <span className="status-dot" data-state={state} aria-hidden="true" />;
}
