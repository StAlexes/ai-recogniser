/**
 * Bridge to the native desktop shell.
 *
 * The app can be opened either inside the pywebview window (where the shell
 * exposes `window.pywebview.api`) or in a normal browser during development.
 * In the browser the calls are harmless no-ops.
 */

interface PywebviewApi {
  quit?: () => Promise<void> | void;
  minimize?: () => Promise<void> | void;
}

interface PywebviewWindow extends Window {
  pywebview?: { api?: PywebviewApi };
}

function api(): PywebviewApi | undefined {
  return (window as PywebviewWindow).pywebview?.api;
}

/** Close the application completely (the tray icon stops as well). */
export function quitApp(): void {
  const bridge = api();
  if (bridge?.quit) {
    void bridge.quit();
    return;
  }
  window.close();
}

/** Hide the window to the tray without stopping the app. */
export function minimizeToTray(): void {
  const bridge = api();
  if (bridge?.minimize) {
    void bridge.minimize();
  }
}
