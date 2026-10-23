import { useRef, useState } from 'react';
import type { DragEvent } from 'react';
import { ArrowUploadRegular } from '@fluentui/react-icons';

const FORMATS = 'mp3 · mp4 · m4a · wav · flac · ogg · opus · mkv · mov · webm · avi';

interface DropzoneProps {
  onFiles: (files: FileList) => void;
  onBrowse: () => void;
}

/**
 * Clickable drop target. Uses a native button so keyboard users get the same
 * action without extra key handlers.
 */
export function Dropzone({ onFiles, onBrowse }: DropzoneProps) {
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);

  const handleDragEnter = (event: DragEvent<HTMLButtonElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    depth.current += 1;
    setDragging(true);
  };

  const handleDragOver = (event: DragEvent<HTMLButtonElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
  };

  const handleDragLeave = (event: DragEvent<HTMLButtonElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    depth.current = Math.max(0, depth.current - 1);
    if (depth.current === 0) {
      setDragging(false);
    }
  };

  const handleDrop = (event: DragEvent<HTMLButtonElement>): void => {
    event.preventDefault();
    event.stopPropagation();
    depth.current = 0;
    setDragging(false);
    if (event.dataTransfer.files.length > 0) {
      onFiles(event.dataTransfer.files);
    }
  };

  return (
    <button
      type="button"
      className="dropzone"
      data-dragging={dragging}
      onClick={onBrowse}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      aria-label={
        dragging
          ? 'Перетащите mp3 или mp4'
          : 'Добавить файлы: нажмите или перетащите mp3 или mp4'
      }
    >
      <span className="dropzone-icon" aria-hidden="true">
        <ArrowUploadRegular fontSize={24} />
      </span>
      <span className="dropzone-title">
        {dragging ? 'Перетащите mp3 или mp4' : 'Перетащите файлы сюда'}
      </span>
      <span className="dropzone-sub">
        {dragging ? 'Отпустите, чтобы добавить файлы' : 'или нажмите, чтобы выбрать'}
      </span>
      <span className="dropzone-formats">{FORMATS}</span>
    </button>
  );
}
