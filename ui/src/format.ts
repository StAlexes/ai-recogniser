const BYTES_IN_KB = 1024;
const BYTES_IN_MB = BYTES_IN_KB * 1024;
const BYTES_IN_GB = BYTES_IN_MB * 1024;

function decimal(value: number, digits: number): string {
  return value.toFixed(digits).replace('.', ',');
}

/** Human-readable file size for files picked in the browser. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 КБ';
  }
  if (bytes < BYTES_IN_MB) {
    return `${Math.max(1, Math.round(bytes / BYTES_IN_KB))} КБ`;
  }
  if (bytes < BYTES_IN_GB) {
    return `${decimal(bytes / BYTES_IN_MB, 1)} МБ`;
  }
  return `${decimal(bytes / BYTES_IN_GB, 2)} ГБ`;
}

/** Model sizes come from the catalog in megabytes. */
export function formatModelSize(sizeMb: number): string {
  if (!Number.isFinite(sizeMb) || sizeMb <= 0) {
    return '—';
  }
  if (sizeMb >= 1024) {
    return `${decimal(sizeMb / 1024, 1)} ГБ`;
  }
  return `${Math.round(sizeMb)} МБ`;
}

function plural(count: number, one: string, few: string, many: string): string {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) {
    return one;
  }
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    return few;
  }
  return many;
}

export function pluralFiles(count: number): string {
  return `${count} ${plural(count, 'файл', 'файла', 'файлов')}`;
}

export function pluralCores(count: number): string {
  return `${count} ${plural(count, 'ядро', 'ядра', 'ядер')}`;
}

export function formatGb(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—';
  }
  return `${decimal(value, 1)} ГБ`;
}
