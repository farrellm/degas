"""What the worker's routes share."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from degas_worker.cache import AssetCache
from degas_worker.jobs import JobManager
from degas_worker.paths import Paths, is_sha256
from degas_worker.preprocess.host import PreprocessorHost


@dataclass
class WorkerState:
    paths: Paths
    cache: AssetCache
    jobs: JobManager
    preprocessors: PreprocessorHost
    exit_process: Callable[[], None]

    def unload(self) -> None:
        """Free the GPU: drop the resident pipeline and preprocessor."""
        self.jobs.unload()
        self.preprocessors.unload()

    def blob_path(self, sha: str) -> Path:
        """Where an input blob is (or would be) staged."""
        if not is_sha256(sha):
            raise HTTPException(400, "Invalid sha256")
        return self.paths.blobs / sha


def state(request: Request) -> WorkerState:
    worker: WorkerState = request.app.state.worker
    return worker


Worker = Annotated[WorkerState, Depends(state)]
