"""Health, what the worker is doing, and shutdown."""

import shutil
from typing import Any

from fastapi import APIRouter, BackgroundTasks

from degas_worker import __version__
from degas_worker.gpu import gpu_info, versions
from degas_worker.state import Worker

router = APIRouter()


@router.get("/health")
def health(worker: Worker) -> dict[str, Any]:  # sync: may import torch, runs in the threadpool
    home = worker.paths.home
    disk = shutil.disk_usage(home if home.exists() else "/")
    return {
        "status": "ok",
        "version": __version__,
        **gpu_info(),
        "versions": versions(),
        "disk_free": disk.free,
        "loaded": worker.jobs.loaded_family,
        "drive_token": worker.cache.has_token,
        "cache": worker.cache.status(),
    }


@router.get("/state")
def state(worker: Worker) -> dict[str, Any]:
    current = worker.jobs.current
    return {
        "job": current.summary() if current else None,
        "outputs": worker.jobs.outputs.pending(),
    }


@router.post("/shutdown", status_code=202)
def shutdown(worker: Worker, background: BackgroundTasks) -> None:
    worker.unload()
    background.add_task(worker.exit_process)
