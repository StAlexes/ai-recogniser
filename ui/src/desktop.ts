/**
 * Bridge to the native desktop shell.
 *
 * The app can be opened either inside the pywebview window (where the shell
 * exposes `window.pywebview.api`) or in a normal browser during development.
 * In the browser the calls are harmless no-ops.
 */

export interface BridgeResult {
  ok: boolean;
  path?: string;
  error?: string;
  cancelled?: boolean;
}

interface PywebviewApi {
  quit?: () => Promise<void> | void;
  minimize?: () => Promise<void> | void;
  open_result_folder?: (jobId: string) => Promise<BridgeResult> | BridgeResult;
  save_result?: (
    jobId: string,
    defaultName: string,
  ) => Promise<BridgeResult> | BridgeResult;
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

/** True when the UI runs inside the native shell rather than a browser. */
export function hasDesktopBridge(): boolean {
  return api() !== undefined;
}

/** Open the result folder for a job via the native shell. */
export async function openResultFolder(jobId: string): Promise<BridgeResult | null> {
  const bridge = api();
  if (!bridge?.open_result_folder) {
    return null;
  }
  return bridge.open_result_folder(jobId);
}

/** Show the native Save-As dialog and copy the transcript to the chosen path. */
export async function saveResultAs(
  jobId: string,
  defaultName: string,
): Promise<BridgeResult | null> {
  const bridge = api();
  if (!bridge?.save_result) {
    return null;
  }
  return bridge.save_result(jobId, defaultName);
}
