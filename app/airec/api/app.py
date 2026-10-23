"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from airec.api import jobs, models, system
from airec.api import settings as settings_api
from airec.core.config import Settings
from airec.core.paths import AppPaths
from airec.core.security import TokenAuthMiddleware
from airec.version import __version__

log = logging.getLogger(__name__)

PLACEHOLDER_HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>AI Recogniser</title>
<style>body{font-family:Segoe UI,sans-serif;background:#f3f3f3;color:#1b1b1b;
display:grid;place-items:center;min-height:100vh;margin:0}main{max-width:640px;padding:32px}
code{background:#e8e8e8;padding:2px 6px;border-radius:4px}</style></head>
<body><main><h1>AI Recogniser</h1>
<p>Ядро запущено, но UI ещё не собран.</p>
<p>Выполните <code>npm.cmd install &amp;&amp; npm.cmd run build</code> в папке <code>ui</code>,
либо запустите dev-сервер Vite.</p></main></body></html>"""


class SpaStaticFiles(StaticFiles):
    """Serve index.html for client-side routes, but never for /api/*."""

    async def get_response(self, path: str, scope):  # type: ignore[no-untyped-def]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            normalized = path.replace("\\", "/")
            if exc.status_code == 404 and not normalized.startswith("api/"):
                return await super().get_response("index.html", scope)
            raise


def create_app(
    *,
    settings: Settings,
    paths: AppPaths,
    token: str,
    auth_enabled: bool = True,
    port: int = 0,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        log.info("core started: version=%s port=%s data=%s", __version__, port, paths.data)
        yield
        log.info("core stopped")

    app = FastAPI(
        title="AI Recogniser Core",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.paths = paths
    app.state.token = token
    app.state.pid = os.getpid()
    app.state.port = port
    app.state.started_at = time.time()

    app.add_middleware(TokenAuthMiddleware, token=token, enabled=auth_enabled)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(settings_api.router)
    app.include_router(jobs.router)
    app.include_router(models.router)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        """Unauthenticated liveness probe for the desktop shell."""
        return {"status": "ok"}

    if paths.ui.exists():
        app.mount("/", SpaStaticFiles(directory=paths.ui, html=True), name="ui")
    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def placeholder() -> str:
            return PLACEHOLDER_HTML

    return app
