"""Preprocessors: selection masks, control traces and faces (design §4.4)."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from degas_worker.cache import CacheError
from degas_worker.state import Worker

router = APIRouter()


class Preprocess(BaseModel):
    id: str
    image: str  # sha256:… of a staged blob
    asset: dict[str, Any] | None = None  # {path, size?}: its model in Drive, if it has one
    params: dict[str, Any] = {}


@router.post("/preprocess")
def preprocess(worker: Worker, body: Preprocess) -> dict[str, Any]:  # sync: GPU work, threadpool
    """Runs alongside a generation job; the two share the GPU."""
    if body.id not in worker.preprocessors:
        raise HTTPException(400, f"Unknown preprocessor {body.id!r}")
    image = worker.blob_path(body.image.removeprefix("sha256:"))
    if not image.exists():
        raise HTTPException(400, f"Input {body.image} was not staged on the worker")
    asset = body.asset or {}
    path = asset.get("path")
    if body.asset is not None and (not isinstance(path, str) or not path):
        raise HTTPException(400, "asset: a path is required")
    try:
        return worker.preprocessors.run(body.id, image, body.params, path, asset.get("size"))
    except CacheError as e:
        raise HTTPException(503, str(e)) from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
