"""Drive: the access token rclone copies with, and copying assets ahead of a job."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel

from degas_worker.state import Worker

router = APIRouter()


class DriveToken(BaseModel):
    access_token: str
    expiry: str  # RFC 3339
    root: str = "degas"


class FetchAssets(BaseModel):
    assets: list[dict[str, Any]]  # [{path, size?}]


@router.post("/drive-token", status_code=204)
def drive_token(worker: Worker, body: DriveToken) -> None:
    worker.cache.set_token(body.access_token, body.expiry, body.root)


@router.post("/assets/fetch", response_class=EventSourceResponse)
async def fetch_assets(worker: Worker, body: FetchAssets) -> AsyncIterator[dict[str, Any]]:
    """Copy assets into the local cache, streaming `copy` progress."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def work() -> None:
        try:
            for asset in body.assets:
                path, size = asset["path"], asset.get("size")

                def progress(done: int, total: int, path: str = path) -> None:
                    event = {"t": "progress", "phase": "copy", "asset": path, "step": done}
                    event["steps"] = total
                    loop.call_soon_threadsafe(queue.put_nowait, event)

                worker.cache.ensure(path, size, progress)
            loop.call_soon_threadsafe(queue.put_nowait, {"t": "done"})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"t": "error", "message": str(e)})

    task = asyncio.ensure_future(run_in_threadpool(work))
    while True:
        event = await queue.get()
        yield event
        if event["t"] in ("done", "error"):
            break
    await task
