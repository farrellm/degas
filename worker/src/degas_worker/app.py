"""Worker HTTP app, served on 127.0.0.1 inside the VM and reached via SSH tunnel."""

import asyncio
import hashlib
import os
import re
import shutil
import signal
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from degas_worker import __version__
from degas_worker.cache import AssetCache, CacheError
from degas_worker.families import RUNNERS
from degas_worker.families.base import FamilyRunner
from degas_worker.gpu import gpu_info, versions
from degas_worker.jobs import JobManager, WorkerBusy
from degas_worker.paths import Paths
from degas_worker.preprocess import PREPROCESSORS
from degas_worker.preprocess.base import Preprocessor

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


class Preprocess(BaseModel):
    id: str
    image: str  # sha256:… of a staged blob
    asset: dict[str, Any]  # {path, size?}: the preprocessor's model in Drive
    params: dict[str, Any] = {}


def create_app(  # noqa: PLR0915 - route definitions
    paths: Paths | None = None,
    runners: dict[str, Callable[[], FamilyRunner]] | None = None,
    rclone: str | None = None,
    exit_process: Callable[[], None] | None = None,
    cache_budget: int | None = None,
    preprocessors: dict[str, Callable[[], Preprocessor]] | None = None,
) -> FastAPI:
    paths = paths or Paths.from_env()
    cache = AssetCache(paths, rclone, cache_budget)
    jobs = JobManager(paths, cache, RUNNERS if runners is None else runners)
    pre_factories = PREPROCESSORS if preprocessors is None else preprocessors
    pre_loaded: dict[str, Preprocessor] = {}
    pre_lock = threading.Lock()  # one preprocessor call at a time

    def unload_preprocessors() -> None:
        for pre in pre_loaded.values():
            pre.unload()
        pre_loaded.clear()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        paths.ensure()
        yield
        jobs.unload()
        unload_preprocessors()

    app = FastAPI(title="Degas worker", version=__version__, lifespan=lifespan)
    app.state.jobs = jobs
    app.state.cache = cache

    @app.get("/health")
    def health() -> dict[str, Any]:  # sync: may import torch, runs in the threadpool
        disk = shutil.disk_usage(paths.home if paths.home.exists() else "/")
        return {
            "status": "ok",
            "version": __version__,
            **gpu_info(),
            "versions": versions(),
            "disk_free": disk.free,
            "loaded": jobs.loaded_family,
            "drive_token": cache.has_token,
            "cache": cache.status(),
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

    # -- preprocessors ----------------------------------------------------------------------

    @app.post("/preprocess")
    def preprocess(body: Preprocess) -> dict[str, Any]:  # sync: GPU work in the threadpool
        """Runs alongside a generation job; the two share the GPU."""
        factory = pre_factories.get(body.id)
        if factory is None:
            raise HTTPException(400, f"Unknown preprocessor {body.id!r}")
        image = blob_path(body.image.removeprefix("sha256:"))
        if not image.exists():
            raise HTTPException(400, f"Input {body.image} was not staged on the worker")
        path = body.asset.get("path")
        if not isinstance(path, str) or not path:
            raise HTTPException(400, "asset: a path is required")
        try:
            with pre_lock, cache.pinned(path):
                model = cache.ensure(path, body.asset.get("size"))
                pre = pre_loaded.get(body.id)
                if pre is None:
                    pre = pre_loaded[body.id] = factory()
                return pre.run(model, image, body.params)
        except CacheError as e:
            raise HTTPException(503, str(e)) from None
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
                        event = {"t": "progress", "phase": "copy", "asset": path, "step": done}
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
        unload_preprocessors()
        background.add_task(exit_process or _terminate)

    return app


def _terminate() -> None:
    os.kill(os.getpid(), signal.SIGTERM)  # uvicorn shuts down gracefully


app = create_app()
