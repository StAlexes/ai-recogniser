"""Local API session token and authentication middleware.

The core binds to 127.0.0.1 on a random port and writes {port, token, pid}
to session.json. The desktop shell reads it, health-checks the core and passes
the token to the UI through the URL fragment (never through query strings or logs).
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

PUBLIC_PREFIXES = (
    "/healthz",
    "/assets/",
    "/favicon.ico",
    "/favicon.svg",
    "/icons/",
)
PUBLIC_EXACT = {"/", "/index.html"}


@dataclass(frozen=True)
class Session:
    port: int
    token: str
    pid: int
    version: str
    started_at: str

    @classmethod
    def create(cls, port: int, token: str, version: str) -> Session:
        return cls(
            port=port,
            token=token,
            pid=os.getpid(),
            version=version,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )


def new_token() -> str:
    return secrets.token_urlsafe(32)


def write_session(path: Path, session: Session) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(session), indent=2), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_session(path: Path) -> Session | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Session(
            port=int(raw["port"]),
            token=str(raw["token"]),
            pid=int(raw["pid"]),
            version=str(raw.get("version", "")),
            started_at=str(raw.get("started_at", "")),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def remove_session(path: Path, *, pid: int | None = None) -> None:
    """Remove the session file, but only if it belongs to this process."""
    current = read_session(path)
    if current is None:
        return
    if pid is not None and current.pid != pid:
        return
    try:
        path.unlink()
    except OSError:
        pass


def _extract_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-token")


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Require a bearer token for API calls; static UI assets stay public."""

    def __init__(self, app, token: str, enabled: bool = True) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.token = token
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled or request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        supplied = _extract_token(request)
        if not supplied or not hmac.compare_digest(supplied, self.token):
            return JSONResponse(
                {"detail": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
