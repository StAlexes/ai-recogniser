"""Оповещения пользователя (Windows toast / трей / лог).

Реализация без лишних зависимостей:
- Если запущен трей (pystray Icon) — используется ``Icon.notify()``,
  который показывает нативный toast Windows 10/11.
- Иначе — fallback на запись в лог и (опционально) MessageBox.

Используется ядром при завершении задач, ошибках и важных событиях.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_ICON: object | None = None  # pystray.Icon, set by the shell
_lock = threading.Lock()


def set_icon(icon: object) -> None:
    """Задать иконку трея (вызывается из shell при старте)."""
    global _ICON
    with _lock:
        _ICON = icon


def notify(title: str, message: str) -> None:
    """Показать уведомление пользователю.

    Приоритет: трей (toast) → лог.  Никогда не бросает исключений.
    """
    with _lock:
        icon = _ICON
    if icon is not None:
        try:
            icon.notify(message, title)
            return
        except Exception:  # pystray может быть недоступен на старых системах
            log.exception("не удалось показать toast через трей")
    log.info("уведомление: %s — %s", title, message)


def notify_job_done(job_id: str, filename: str) -> None:
    """Уведомление о завершённой задаче."""
    notify("Готово", f"Распознавание завершено: {filename}")


def notify_job_error(job_id: str, filename: str, error: str) -> None:
    """Уведомление об ошибке задачи."""
    notify("Ошибка", f"Не удалось распознать {filename}: {error}")


class Notifier:
    """Прозрачная обёртка для тестирования."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def done(self, job_id: str, filename: str) -> None:
        if self.enabled:
            notify_job_done(job_id, filename)

    def error(self, job_id: str, filename: str, error: str) -> None:
        if self.enabled:
            notify_job_error(job_id, filename, error)
