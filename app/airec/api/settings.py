"""Settings endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from airec.core.config import Settings, save_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@router.get("")
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.put("")
def put_settings(payload: Settings, request: Request) -> Settings:
    paths = request.app.state.paths
    previous = request.app.state.settings
    save_settings(paths.config_file, payload)
    request.app.state.settings = payload
    if previous.context_menu_enabled != payload.context_menu_enabled:
        from airec.integrations.windows import (
            register_context_menu,
            unregister_context_menu,
        )

        try:
            if payload.context_menu_enabled:
                register_context_menu()
            else:
                unregister_context_menu()
        except Exception:  # registry access must never break settings
            log.exception("cannot apply the context-menu setting")
    log.info("settings updated (theme=%s, asr=%s)", payload.theme, payload.asr_model)
    return payload
