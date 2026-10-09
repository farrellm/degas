"""Worker HTTP app, served on 127.0.0.1 inside the VM and reached via SSH tunnel."""

import os
import signal
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from degas_worker import __version__
from degas_worker.cache import AssetCache
from degas_worker.families import RUNNERS
from degas_worker.families.base import FamilyRunner
from degas_worker.jobs import JobManager
from degas_worker.paths import Paths
from degas_worker.preprocess import PREPROCESSORS
from degas_worker.preprocess.base import Preprocessor
from degas_worker.preprocess.host import PreprocessorHost
from degas_worker.routers import assets, blobs, jobs, preprocess, system
from degas_worker.state import WorkerState


def create_app(
    paths: Paths | None = None,
    runners: dict[str, Callable[[], FamilyRunner]] | None = None,
    rclone: str | None = None,
    exit_process: Callable[[], None] | None = None,
    cache_budget: int | None = None,
    preprocessors: dict[str, Callable[[], Preprocessor]] | None = None,
) -> FastAPI:
    paths = paths or Paths.from_env()
    cache = AssetCache(paths, rclone, cache_budget)
    worker = WorkerState(
        paths=paths,
        cache=cache,
        jobs=JobManager(paths, cache, RUNNERS if runners is None else runners),
        preprocessors=PreprocessorHost(
            PREPROCESSORS if preprocessors is None else preprocessors, cache
        ),
        exit_process=exit_process or _terminate,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        paths.ensure()
        yield
        worker.unload()

    app = FastAPI(title="Degas worker", version=__version__, lifespan=lifespan)
    app.state.worker = worker
    app.state.jobs = worker.jobs
    app.state.cache = cache
    for module in (system, blobs, jobs, preprocess, assets):
        app.include_router(module.router)
    return app


def _terminate() -> None:
    os.kill(os.getpid(), signal.SIGTERM)  # uvicorn shuts down gracefully
