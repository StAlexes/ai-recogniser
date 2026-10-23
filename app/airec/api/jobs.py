"""Job queue endpoints (skeleton for M1)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get("")
def list_jobs() -> dict[str, object]:
    # M1 replaces this with the SQLite-backed queue.
    return {"items": [], "total": 0}


@router.post("")
def create_job() -> None:
    raise HTTPException(
        status_code=501,
        detail="Очередь распознавания появится в M1 (следующий этап).",
    )
