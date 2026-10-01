"""Blobs: uploads and imports, the image editors' transforms and masks, preprocessors, and
the files themselves."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from degas.api.deps import Svc, blob_or_404
from degas.api.schemas import Frame, FromUrl, Preprocess, RemapMask, Transform
from degas.inputs import MAX_UPLOAD_BYTES
from degas.media import MediaError

router = APIRouter(tags=["blobs"])

# A blob's address is its content, so it never changes.
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


@router.post("/blobs", status_code=201)
async def upload_blob(svc: Svc, request: Request) -> dict[str, Any]:
    """Upload an image or video as the raw request body (camera roll)."""
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The file is larger than 200 MB")
    if not data:
        raise HTTPException(400, "The upload is empty")
    return await svc.inputs.store(bytes(data), request.headers.get("content-type"))


@router.post("/blobs/from-url", status_code=201)
async def blob_from_url(svc: Svc, body: FromUrl) -> dict[str, Any]:
    return await svc.inputs.fetch(body.url)


@router.post("/blobs/{sha}/transform", status_code=201)
async def transform_blob(svc: Svc, sha: str, body: Transform) -> dict[str, Any]:
    blob_or_404(svc, sha)
    return await svc.inputs.transform(sha, body.ops)


@router.get("/blobs/{sha}/transform")
async def get_transform(svc: Svc, sha: str) -> dict[str, Any]:
    """The original and operations a derived image was made with, to reopen the editor.

    Any other image is its own original, with no operations.
    """
    blob_or_404(svc, sha)
    record = svc.db.get_transform(sha)
    if record is None or svc.blobs.path(record["original"]) is None:
        return {"original": sha, "ops": []}
    return record


@router.post("/blobs/{sha}/mask", status_code=201)
async def upload_mask(svc: Svc, sha: str, request: Request) -> dict[str, Any]:
    """A mask painted over image `sha` (raw PNG body; white or opaque is redrawn)."""
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The mask is larger than 200 MB")
    return await svc.inputs.store_mask(sha, data)


@router.post("/blobs/{sha}/remap", status_code=201)
async def remap_mask(svc: Svc, sha: str, body: RemapMask) -> dict[str, Any]:
    """Carry mask `sha` from the image it was painted on to a new crop of that image."""
    return await svc.inputs.remap_mask(sha, body.source, body.to)


@router.post("/preprocess", status_code=201)
async def preprocess(svc: Svc, body: Preprocess) -> dict[str, Any]:
    return await svc.preprocessing.run(body.id, body.image, body.params)


@router.post("/blobs/{sha}/frame", status_code=201)
async def blob_frame(svc: Svc, sha: str, body: Frame) -> dict[str, Any]:
    return await svc.inputs.frame(sha, body.at)


@router.get("/blobs/{sha}")
async def get_blob(svc: Svc, sha: str) -> FileResponse:
    return FileResponse(blob_or_404(svc, sha), headers=IMMUTABLE)


@router.get("/thumbs/{sha}")
async def get_thumb(svc: Svc, sha: str) -> FileResponse:
    path = svc.blobs.thumb(sha)
    if path is None and svc.blobs.is_video(sha):
        try:
            path = await svc.inputs.poster(sha)
        except MediaError:
            path = None
    if path is None:
        raise HTTPException(404, "No such blob")
    return FileResponse(path, media_type="image/webp", headers=IMMUTABLE)
