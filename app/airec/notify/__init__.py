"""AI Recogniser notification module.

Windows toast notifications via tray icon (pystray) with log fallback.
"""

from __future__ import annotations

from airec.notify.notifier import Notifier, notify, notify_job_done, notify_job_error, set_icon

__all__ = ["Notifier", "notify", "notify_job_done", "notify_job_error", "set_icon"]
