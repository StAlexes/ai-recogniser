import { useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import {
  Badge,
  Button,
  MessageBar,
  MessageBarActions,
  MessageBarBody,
} from '@fluentui/react-components';
import {
  AddRegular,
  CloudRegular,
  DismissRegular,
  InfoRegular,
  MusicNote2Regular,
  ShieldCheckmarkRegular,
  SparkleRegular,
  VideoRegular,
} from '@fluentui/react-icons';
import { Dropzone } from '../components/Dropzone';
import { InfoCard } from '../components/InfoCard';
import { PageHeader } from '../components/PageHeader';
import { formatBytes, pluralFiles } from '../format';

const ACCEPTED_EXTENSIONS = new Set([
  'mp3',
  'mp4',
  'm4a',
  'wav',
  'flac',
  'ogg',
  'opus',
  'mkv',
  'mov',
  'webm',
  'avi',
]);

const VIDEO_EXTENSIONS = new Set(['mp4', 'mkv', 'mov', 'webm', 'avi']);

const ACCEPT_ATTRIBUTE = Array.from(ACCEPTED_EXTENSIONS)
  .map((extension) => `.${extension}`)
  .join(',');

interface SelectedFile {
  id: string;
  name: string;
  size: number;
  kind: 'audio' | 'video';
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf('.');
  if (dot < 1 || dot === name.length - 1) {
    return '';
  }
  return name.slice(dot + 1).toLowerCase();
}

export function QueuePage() {
  const [files, setFiles] = useState<SelectedFile[]>([]);
  const [noticeVisible, setNoticeVisible] = useState(false);
  const [rejectedCount, setRejectedCount] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const nextId = useRef(0);

  const openPicker = (): void => {
    inputRef.current?.click();
  };

  const addFiles = (incoming: FileList): void => {
    const accepted: SelectedFile[] = [];
    let rejected = 0;

    for (const file of Array.from(incoming)) {
      const extension = extensionOf(file.name);
      if (!ACCEPTED_EXTENSIONS.has(extension)) {
        rejected += 1;
        continue;
      }
      nextId.current += 1;
      accepted.push({
        id: `file-${nextId.current}`,
        name: file.name,
        size: file.size,
        kind: VIDEO_EXTENSIONS.has(extension) ? 'video' : 'audio',
      });
    }

    if (accepted.length > 0) {
      setFiles((current) => [...current, ...accepted]);
      setNoticeVisible(true);
    }
    setRejectedCount(rejected);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const selected = event.currentTarget.files;
    if (selected !== null) {
      addFiles(selected);
    }
    event.currentTarget.value = '';
  };

  const removeFile = (id: string): void => {
    setFiles((current) => current.filter((file) => file.id !== id));
  };

  const clearFiles = (): void => {
    setFiles([]);
    setNoticeVisible(false);
    setRejectedCount(0);
  };

  return (
    <>
      <PageHeader
        title="Очередь"
        subtitle="Локальное распознавание аудио и видео"
        actions={
          <Button appearance="primary" icon={<AddRegular />} onClick={openPicker}>
            Добавить файлы
          </Button>
        }
      />

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        multiple
        tabIndex={-1}
        aria-hidden="true"
        onChange={handleInputChange}
      />

      {noticeVisible && files.length > 0 ? (
        <MessageBar intent="info" className="queue-notice">
          <MessageBarBody>Приём файлов и очередь появятся в M1</MessageBarBody>
          <MessageBarActions
            containerAction={
              <Button
                appearance="transparent"
                icon={<DismissRegular />}
                aria-label="Скрыть сообщение"
                onClick={() => {
                  setNoticeVisible(false);
                }}
              />
            }
          />
        </MessageBar>
      ) : null}

      <Dropzone onFiles={addFiles} onBrowse={openPicker} />

      {rejectedCount > 0 ? (
        <p className="queue-hint" role="status">
          Пропущено файлов: {rejectedCount}. Поддерживаются mp3, mp4, m4a, wav, flac, ogg, opus,
          mkv, mov, webm и avi.
        </p>
      ) : null}

      {files.length === 0 ? (
        <div className="empty-note">
          <InfoRegular aria-hidden="true" />
          <span>
            Очередь пуста. Добавьте mp3 или mp4 — приём файлов и обработка появятся в M1.
          </span>
        </div>
      ) : (
        <section className="files-panel surface" aria-labelledby="selected-files-title">
          <div className="files-head">
            <h2 id="selected-files-title" className="files-title">
              Выбрано: {pluralFiles(files.length)}
            </h2>
            <Button appearance="subtle" size="small" onClick={clearFiles}>
              Очистить
            </Button>
          </div>
          <ul className="files-list">
            {files.map((file) => (
              <li key={file.id} className="file-row">
                <span className="file-icon" data-kind={file.kind} aria-hidden="true">
                  {file.kind === 'video' ? (
                    <VideoRegular fontSize={20} />
                  ) : (
                    <MusicNote2Regular fontSize={20} />
                  )}
                </span>
                <span className="file-name" title={file.name}>
                  {file.name}
                </span>
                <span className="file-size">{formatBytes(file.size)}</span>
                <Badge appearance="tint" color="brand" size="small">
                  M1
                </Badge>
                <Button
                  appearance="transparent"
                  size="small"
                  icon={<DismissRegular />}
                  aria-label={`Убрать ${file.name}`}
                  onClick={() => {
                    removeFile(file.id);
                  }}
                />
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="info-grid">
        <InfoCard tone="brand" icon={<SparkleRegular fontSize={18} />} title="GigaAM v3 · 214 МБ">
          Русский язык, пунктуация и нормализация из коробки.
        </InfoCard>
        <InfoCard tone="teal" icon={<CloudRegular fontSize={18} />} title="Локальные саммари">
          Без облака: краткое содержание считает LLM на этом ПК.
        </InfoCard>
        <InfoCard
          tone="purple"
          icon={<ShieldCheckmarkRegular fontSize={18} />}
          title="Приватность"
        >
          Данные не покидают компьютер. Сеть нужна только для загрузки моделей.
        </InfoCard>
      </div>
    </>
  );
}
