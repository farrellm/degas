"""Input blobs the server stages for a job (content-addressed, so sent at most once)."""

import hashlib

from fastapi import APIRouter, HTTPException, Request, Response

from degas_worker.state import Worker

router = APIRouter()


@router.head("/blobs/{sha}")
def has_blob(worker: Worker, sha: str) -> Response:
    return Response(status_code=200 if worker.blob_path(sha).exists() else 404)


@router.put("/blobs/{sha}", status_code=204)
async def put_blob(worker: Worker, sha: str, request: Request) -> None:
    path = worker.blob_path(sha)
    data = await request.body()
    if hashlib.sha256(data).hexdigest() != sha:
        raise HTTPException(400, "Content does not match sha256")
    worker.paths.blobs.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
