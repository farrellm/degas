"""Health, and the event stream the app follows."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse

from degas import __version__
from degas.api.deps import Svc

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/events", response_class=EventSourceResponse)
async def events(svc: Svc) -> AsyncIterator[dict[str, Any]]:
    yield {"type": "hello", "version": __version__}
    async for event in svc.bus.subscribe():
        yield event
