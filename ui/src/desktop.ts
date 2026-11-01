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

export interface BridgeFilesResult extends BridgeResult {
  paths?: string[];
}

interface PywebviewApi {
  quit?: () => Promise<void> | void;
  minimize?: () => Promise<void> | void;
  open_result_folder?: (jobId: string) => Promise<BridgeResult> | BridgeResult;
  reveal_source?: (jobId: string) => Promise<BridgeResult> | BridgeResult;
  add_files?: () => Promise<BridgeFilesResult> | BridgeFilesResult;
  watch_dropped_files?: () => Promise<BridgeResult> | BridgeResult;
  save_result?: (
    jobId: string,
    defaultName: string,
    view?: string,
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

/** Show the original file of a job in Explorer (desktop shell only). */
export async function revealSourceFile(jobId: string): Promise<BridgeResult | null> {
  const bridge = api();
  if (!bridge?.reveal_source) {
    return null;
  }
  return bridge.reveal_source(jobId);
}

/** Open the native file picker; returns full paths of the picked files. */
export async function addFilesViaBridge(): Promise<BridgeFilesResult | null> {
  const bridge = api();
  if (!bridge?.add_files) {
    return null;
  }
  return bridge.add_files();
}

/**
 * Ask the shell to map dropped file names to their full paths.
 *
 * WebView2 knows the real paths of dropped files, but only the Python side
 * receives them (``FilesDropped``).  The shell listens for document drops and
 * stores a name→path list in ``window.__airecDropPaths``; this function just
 * makes sure the listener is installed once.
 */
export async function watchDroppedFiles(): Promise<BridgeResult | null> {
  const bridge = api();
  if (!bridge?.watch_dropped_files) {
    return null;
  }
  return bridge.watch_dropped_files();
}

/** Read the name→path pairs the shell stored for the latest drop. */
export function takeDroppedPaths(): Array<{ name: string; path: string }> {
  const stored = (window as unknown as { __airecDropPaths?: unknown }).__airecDropPaths;
  if (!Array.isArray(stored)) {
    return [];
  }
  return stored
    .filter(
      (item): item is { name: string; path: string } =>
        typeof (item as { name?: unknown })?.name === 'string' &&
        typeof (item as { path?: unknown })?.path === 'string',
    )
    .map((item) => ({ name: item.name, path: item.path }));
}

/**
 * Show the native Save-As dialog and copy the shown result to the chosen path.
 *
 * *view* selects which result to save: recognition (plain transcript),
 * diarization (speaker-attributed) or summary.  The format is derived from
 * the extension the user picked in the dialog, so no format argument is sent.
 */
export async function saveResultAs(
  jobId: string,
  defaultName: string,
  view: string = 'recognition',
): Promise<BridgeResult | null> {
  const bridge = api();
  if (!bridge?.save_result) {
    return null;
  }
  return bridge.save_result(jobId, defaultName, view);
}
