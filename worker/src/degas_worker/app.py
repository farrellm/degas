"""Worker HTTP app, served on 127.0.0.1 inside the VM and reached via SSH tunnel."""

import asyncio
import hashlib
import os
import re
import shutil
import signal
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from degas_worker import __version__
from degas_worker.cache import AssetCache
from degas_worker.families import RUNNERS
from degas_worker.families.base import FamilyRunner
from degas_worker.gpu import gpu_info
from degas_worker.jobs import JobManager, WorkerBusy
from degas_worker.paths import Paths

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StartJob(BaseModel):
    job_id: str
    spec: dict[str, Any]
    seeds: list[int]


class DriveToken(BaseModel):
    access_token: str
    expiry: str  # RFC 3339
    root: str = "degas"


class FetchAssets(BaseModel):
    assets: list[dict[str, Any]]  # [{path, size?}]


def create_app(  # noqa: PLR0915 - route definitions
    paths: Paths | None = None,
    runners: dict[str, Callable[[], FamilyRunner]] | None = None,
    rclone: str | None = None,
    exit_process: Callable[[], None] | None = None,
) -> FastAPI:
    paths = paths or Paths.from_env()
    cache = AssetCache(paths, rclone)
    jobs = JobManager(paths, cache, RUNNERS if runners is None else runners)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        paths.ensure()
        yield
        jobs.unload()

    app = FastAPI(title="Degas worker", version=__version__, lifespan=lifespan)
    app.state.jobs = jobs

    @app.get("/health")
    def health() -> dict[str, Any]:  # sync: may import torch, runs in the threadpool
        disk = shutil.disk_usage(paths.home if paths.home.exists() else "/")
        return {
            "status": "ok",
            "version": __version__,
            **gpu_info(),
            "disk_free": disk.free,
            "loaded": jobs.loaded_family,
            "drive_token": cache.has_token,
        }

    @app.get("/state")
    def state() -> dict[str, Any]:
        return {
            "job": jobs.current.summary() if jobs.current else None,
            "outputs": jobs.outputs(),
        }

    # -- blobs -----------------------------------------------------------------------------

    def blob_path(sha: str) -> Any:
        if not _SHA256.match(sha):
            raise HTTPException(400, "Invalid sha256")
        return paths.blobs / sha

    @app.head("/blobs/{sha}")
    def has_blob(sha: str) -> Response:
        return Response(status_code=200 if blob_path(sha).exists() else 404)

    @app.put("/blobs/{sha}", status_code=204)
    async def put_blob(sha: str, request: Request) -> None:
        path = blob_path(sha)
        data = await request.body()
        if hashlib.sha256(data).hexdigest() != sha:
            raise HTTPException(400, "Content does not match sha256")
        paths.blobs.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    # -- jobs ------------------------------------------------------------------------------

    @app.post("/jobs", status_code=202)
    async def start_job(body: StartJob) -> dict[str, Any]:
        try:
            record = jobs.start(body.job_id, body.spec, body.seeds)
        except WorkerBusy:
            raise HTTPException(409, "A job is already running") from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return record.summary()

    @app.get("/jobs/{job_id}/events", response_class=EventSourceResponse)
    async def job_events(job_id: str) -> AsyncIterator[dict[str, Any]]:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(404, "Unknown job")
        async for event in record.subscribe():
            yield event

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, bool]:
        return {"cancelled": jobs.cancel(job_id)}

    @app.get("/outputs/{job_id}/{item}")
    def get_output(job_id: str, item: int) -> FileResponse:
        try:
            found = jobs.output_file(job_id, item)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        if found is None:
            raise HTTPException(404, "No such output")
        return FileResponse(found[0], media_type=found[1])

    @app.delete("/outputs/{job_id}/{item}", status_code=204)
    def ack_output(job_id: str, item: int) -> None:
        try:
            jobs.ack(job_id, item)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    # -- Drive -----------------------------------------------------------------------------

    @app.post("/drive-token", status_code=204)
    def drive_token(body: DriveToken) -> None:
        cache.set_token(body.access_token, body.expiry, body.root)

    @app.post("/assets/fetch", response_class=EventSourceResponse)
    async def fetch_assets(body: FetchAssets) -> AsyncIterator[dict[str, Any]]:
        """Copy assets into the local cache, streaming `copy` progress."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def work() -> None:
            try:
                for asset in body.assets:
                    path, size = asset["path"], asset.get("size")

                    def progress(done: int, total: int, path: str = path) -> None:
                        event = {"t": "progress", "phase": "copy", "path": path, "step": done}
                        event["steps"] = total
                        loop.call_soon_threadsafe(queue.put_nowait, event)

                    cache.ensure(path, size, progress)
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

    # -- lifecycle -------------------------------------------------------------------------

    @app.post("/shutdown", status_code=202)
    def shutdown(background: BackgroundTasks) -> None:
        jobs.unload()
        background.add_task(exit_process or _terminate)

    return app


def _terminate() -> None:
    os.kill(os.getpid(), signal.SIGTERM)  # uvicorn shuts down gracefully


app = create_app()
