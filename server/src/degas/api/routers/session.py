"""The GPU session."""

from typing import Any

from fastapi import APIRouter

from degas.api.deps import Svc
from degas.api.schemas import StartSession

router = APIRouter(tags=["session"])


@router.get("/session")
async def get_session(svc: Svc) -> dict[str, Any]:
    return svc.sessions.snapshot()


@router.post("/session", status_code=202)
async def start_session(svc: Svc, body: StartSession) -> dict[str, Any]:
    return await svc.sessions.start(body.gpu, body.high_mem)


@router.delete("/session")
async def stop_session(svc: Svc) -> dict[str, Any]:
    await svc.sessions.stop()
    return svc.sessions.snapshot()


@router.post("/session/touch")
async def touch_session(svc: Svc) -> dict[str, Any]:
    svc.sessions.touch()
    return svc.sessions.snapshot()


@router.post("/session/reset-worker", status_code=202)
async def reset_worker(svc: Svc) -> dict[str, Any]:
    await svc.sessions.reset_worker()
    return svc.sessions.snapshot()
