"""Native Windows toast notifications with result actions.

The toast is created through PowerShell's built-in WinRT projection, so the
app does not need an extra dependency (winrt/winsdk) and still works inside a
PyInstaller build.  Buttons activate the ``airec://`` protocol, which is
registered by :mod:`airec.integrations.windows`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from xml.sax.saxutils import escape

from airec.integrations.windows import APP_USER_MODEL_ID

log = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000

_SCRIPT = rf"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
$xmlText = [Console]::In.ReadToEnd()
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml($xmlText)
$toast = New-Object Windows.UI.Notifications.ToastNotification $xml
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{APP_USER_MODEL_ID}').Show($toast)
"""


def _toast_xml(job_id: str, filename: str, *, error: bool = False) -> str:
    title = "Ошибка распознавания" if error else "Распознавание завершено"
    message = escape(filename, {'"': "&quot;"})
    result_uri = escape(f"airec://result/{job_id}", {'"': "&quot;"})
    folder_uri = escape(f"airec://folder/{job_id}", {'"': "&quot;"})
    return f"""<toast launch="{result_uri}" activationType="protocol">
  <visual>
    <binding template="ToastGeneric">
      <text>{title}</text>
      <text>{message}</text>
      <text placement="attribution">AI Recogniser</text>
    </binding>
  </visual>
  <actions>
    <action content="Открыть результат" activationType="protocol" arguments="{result_uri}" />
    <action content="Открыть папку" activationType="protocol" arguments="{folder_uri}" />
  </actions>
</toast>"""


def show_job_toast(job_id: str, filename: str, *, error: bool = False) -> bool:
    """Show a Windows toast for a finished job. Never raises."""
    if os.name != "nt":
        return False
    xml = _toast_xml(job_id, filename, error=error)
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                _SCRIPT,
            ],
            input=xml.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        log.exception("cannot show a Windows toast for job %s", job_id)
        return False
    if completed.returncode != 0:
        log.warning(
            "toast failed for job %s: %s",
            job_id,
            completed.stderr.decode("utf-8", errors="replace")[:400],
        )
        return False
    log.info("toast shown for job %s", job_id)
    return True
