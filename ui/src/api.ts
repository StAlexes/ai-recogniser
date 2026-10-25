/**
 * Typed client for the local AI Recogniser core.
 *
 * The UI is served by the core itself, so the default base URL is same-origin.
 * VITE_API_BASE exists for development against a separately started core.
 * The session token arrives in the URL fragment and is kept in sessionStorage
 * for the lifetime of the window.
 */

export type ThemeMode = 'system' | 'light' | 'dark';
export type Language = 'ru' | 'en';

export interface Settings {
  schema_version: number;
  theme: ThemeMode;
  language: Language;
  autostart: boolean;
  notifications: boolean;
  close_to_tray: boolean;
  pause_on_battery: boolean;
  results_dir: string;
  keep_extracted_audio: boolean;
  asr_model: string;
  llm_model: string;
  cpu_threads: number;
  max_parallel_jobs: number;
  external_llm_enabled: boolean;
  vad_enabled: boolean;
  vad_max_segment_s: number;
  vad_min_silence_ms: number;
  vad_speech_pad_ms: number;
  diarize_enabled: boolean;
  diarize_threshold: number;
  diarize_model_path: string;
  rag_enabled: boolean;
  rag_url: string;
  rag_k: number;
}

export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled';

export interface JobSummary {
  id: string;
  filename: string;
  status: JobStatus;
  progress: number;
  phase: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  model: string;
  duration_s: number | null;
  elapsed_s: number | null;
  rtf: number | null;
  segment_count: number;
  size_bytes: number;
  audio_format: string;
  error: string | null;
}

export interface JobSegment {
  start_s: number;
  end_s: number;
  text: string;
  speaker: string | null;
}

export interface JobDetail extends JobSummary {
  text: string;
  result_file: string | null;
  result_url: string | null;
  segments: JobSegment[];
}

export interface JobsResponse {
  items: JobSummary[];
  total: number;
}

export interface RagStatus {
  enabled: boolean;
  configured_url: string;
  reachable: boolean;
  latency_ms: number | null;
  chunks: number | null;
  indexing: boolean | null;
  gen_model: string | null;
  error: string | null;
}

export interface RagSource {
  id?: number;
  source: string;
  breadcrumb: string;
  text: string;
  position?: number;
  score?: number;
  dense?: number;
  lexical?: number;
  exact?: boolean | string;
  match?: string;
}

export interface RagSearchResponse {
  query: string;
  results: RagSource[];
}

export interface RagAskResponse {
  question: string;
  answer: string;
  sources: RagSource[];
}

export interface HealthResponse {
  status: string;
  version: string;
  pid: number;
  port: number;
  uptime_s: number;
  platform: string;
  python: string;
  portable: boolean;
  data_dir: string;
  ui_built: boolean;
}

export interface HealthzResponse {
  status: string;
}

export interface ModelInfo {
  id: string;
  kind: string;
  title: string;
  size_mb: number;
  license: string;
  languages: string[];
  default: boolean;
  description: string;
}

export interface ModelsResponse {
  items: ModelInfo[];
  installed: string[];
}

export interface DiskInfo {
  mount: string;
  fs: string;
  total_gb: number;
  free_gb: number;
}

export interface GpuInfo {
  name: string;
  vram_gb: number | null;
  driver: string;
  /** True for software/virtual display adapters (e.g. Parsec). */
  virtual?: boolean;
}

export interface HardwareInfo {
  cpu: string;
  cpu_cores: number | null;
  cpu_threads: number | null;
  ram_total_gb: number;
  ram_available_gb: number;
  disks: DiskInfo[];
  gpu: GpuInfo[];
  os: string;
  python: string;
  architecture: string;
}

export interface SystemProfile {
  schema_version: number;
  generated_at: string;
  app_version: string;
  hardware?: HardwareInfo;
  tier?: string | null;
  recommendation?: unknown;
}

/** The core rejected the session token. */
export class AuthError extends Error {
  constructor() {
    super('Нет доступа к ядру');
    this.name = 'AuthError';
  }
}

/** A transport or protocol error returned by the core. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const TOKEN_STORAGE_KEY = 'airec.token';

function resolveBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE;
  if (typeof configured !== 'string') {
    return '';
  }
  return configured.trim().replace(/\/+$/, '');
}

function takeTokenFromHash(): string | null {
  const hash = window.location.hash;
  if (hash.length < 2) {
    return null;
  }

  const params = new URLSearchParams(hash.slice(1));
  const candidate = params.get('token');
  const token = candidate?.trim() ?? '';
  if (token === '') {
    return null;
  }

  // Never leave the token in the address bar or in the WebView history.
  window.history.replaceState(
    null,
    '',
    `${window.location.pathname}${window.location.search}`,
  );
  return token;
}

function readStoredToken(): string | null {
  try {
    return window.sessionStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeToken(token: string): void {
  try {
    window.sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  } catch {
    // Private mode or a locked-down WebView: requests still work without storage.
  }
}

const bootToken = takeTokenFromHash();
if (bootToken !== null) {
  storeToken(bootToken);
}
const token = bootToken ?? readStoredToken();

export { token };

const API_BASE = resolveBaseUrl();
const DEFAULT_TIMEOUT_MS = 15_000;

interface RequestOptions {
  method?: 'GET' | 'PUT' | 'POST';
  body?: unknown;
  timeoutMs?: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

async function readErrorDetail(response: Response): Promise<string | null> {
  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('application/json')) {
    return null;
  }
  try {
    const payload: unknown = await response.json();
    if (isRecord(payload) && typeof payload.detail === 'string') {
      const detail = payload.detail.trim();
      return detail === '' ? null : detail;
    }
  } catch {
    // A malformed error body is not more useful than the HTTP status.
  }
  return null;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  const headers = new Headers();
  headers.set('Accept', 'application/json');
  if (token !== null) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  let body: string | undefined;
  if (options.body !== undefined) {
    headers.set('Content-Type', 'application/json');
    body = JSON.stringify(options.body);
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: options.method ?? 'GET',
      headers,
      body,
      credentials: 'same-origin',
      cache: 'no-store',
      signal: controller.signal,
    });
  } catch {
    if (controller.signal.aborted) {
      throw new ApiError(
        0,
        `Ядро не ответило за ${Math.round(timeoutMs / 1000)} с.`,
      );
    }
    throw new ApiError(0, 'Не удалось связаться с ядром. Проверьте, что приложение запущено.');
  } finally {
    window.clearTimeout(timer);
  }

  if (response.status === 401) {
    throw new AuthError();
  }

  if (!response.ok) {
    const detail = await readErrorDetail(response);
    throw new ApiError(
      response.status,
      detail ?? `Ядро вернуло ошибку (HTTP ${response.status}).`,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  if (text === '') {
    return undefined as T;
  }

  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(response.status, 'Ядро вернуло некорректный ответ.');
  }
}

/** Public liveness probe, used for the status indicator. */
export function getHealthz(): Promise<HealthzResponse> {
  return request<HealthzResponse>('/healthz', { timeoutMs: 4_000 });
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/v1/health');
}

export function getSettings(): Promise<Settings> {
  return request<Settings>('/api/v1/settings');
}

export function putSettings(settings: Settings): Promise<Settings> {
  return request<Settings>('/api/v1/settings', {
    method: 'PUT',
    body: settings,
  });
}

export function getModels(): Promise<ModelsResponse> {
  return request<ModelsResponse>('/api/v1/models');
}

export function getSystemProfile(): Promise<SystemProfile> {
  return request<SystemProfile>('/api/v1/system/profile', {
    timeoutMs: 20_000,
  });
}

export function getRagStatus(): Promise<RagStatus> {
  return request<RagStatus>('/api/v1/rag/status', { timeoutMs: 8_000 });
}

export function searchRag(query: string, k?: number): Promise<RagSearchResponse> {
  const params = new URLSearchParams({ q: query });
  if (k !== undefined) {
    params.set('k', String(k));
  }
  return request<RagSearchResponse>(`/api/v1/rag/search?${params.toString()}`, {
    timeoutMs: 50_000,
  });
}

export function askRag(question: string, k?: number): Promise<RagAskResponse> {
  return request<RagAskResponse>('/api/v1/rag/ask', {
    method: 'POST',
    body: { question, ...(k === undefined ? {} : { k }) },
    timeoutMs: 250_000,
  });
}

export function getJobs(): Promise<JobsResponse> {
  return request<JobsResponse>('/api/v1/jobs', { timeoutMs: 10_000 });
}

export function getJob(jobId: string): Promise<JobDetail> {
  return request<JobDetail>(`/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    timeoutMs: 15_000,
  });
}

export function cancelJob(jobId: string): Promise<JobDetail> {
  return request<JobDetail>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST', timeoutMs: 15_000 },
  );
}

export function retryJob(jobId: string): Promise<JobDetail> {
  return request<JobDetail>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/retry`,
    { method: 'POST', timeoutMs: 15_000 },
  );
}

/** Upload one file to the local core and return the queued job. */
export async function uploadJob(file: File): Promise<JobDetail> {
  const timeoutMs = 600_000;
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  const headers = new Headers();
  headers.set('Accept', 'application/json');
  if (token !== null) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const form = new FormData();
  form.append('file', file, file.name);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/jobs`, {
      method: 'POST',
      headers,
      body: form,
      credentials: 'same-origin',
      cache: 'no-store',
      signal: controller.signal,
    });
  } catch {
    if (controller.signal.aborted) {
      throw new ApiError(0, 'Загрузка файла заняла слишком много времени.');
    }
    throw new ApiError(0, 'Не удалось передать файл ядру приложения.');
  } finally {
    window.clearTimeout(timer);
  }

  if (response.status === 401) {
    throw new AuthError();
  }
  if (!response.ok) {
    const detail = await readErrorDetail(response);
    throw new ApiError(
      response.status,
      detail ?? `Ядро вернуло ошибку загрузки (HTTP ${response.status}).`,
    );
  }
  try {
    return (await response.json()) as JobDetail;
  } catch {
    throw new ApiError(response.status, 'Ядро вернуло некорректный ответ на загрузку.');
  }
}
