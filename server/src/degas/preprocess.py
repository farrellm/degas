"""Preprocessors run on the worker (design §4.4). Phase 6 has one: SAM 3 selection for masks."""

import base64
import binascii
from typing import Any

from degas.colab.worker_client import WorkerError
from degas.inputs import unref
from degas.media import MediaError
from degas.services import Services

SAM_ASSET = "preprocessors/sam3"
MAX_POINTS = 16
MAX_TEXT = 200


class PreprocessError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def sam_params(params: dict[str, Any]) -> dict[str, Any]:
    """Check SAM prompts: points `[{x, y, include}]` in image pixels and/or a text phrase."""
    points = params.get("points") or []
    text = params.get("text") or ""
    if not isinstance(points, list) or len(points) > MAX_POINTS:
        raise PreprocessError(400, f"Use at most {MAX_POINTS} points")
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise PreprocessError(400, "That description is too long")
    clean: list[dict[str, Any]] = []
    for p in points:
        if not isinstance(p, dict) or not all(
            isinstance(p.get(k), int | float) and not isinstance(p.get(k), bool) for k in "xy"
        ):
            raise PreprocessError(400, "Each point needs x and y")
        clean.append({"x": float(p["x"]), "y": float(p["y"]), "include": p.get("include", True)})
    text = text.strip()
    if not clean and not text:
        raise PreprocessError(400, "Tap the image or describe what to select")
    # With a description, points (either kind) pick among the matches; without one, SAM
    # needs something to include.
    if not text and not any(p["include"] for p in clean):
        raise PreprocessError(400, "Tap something to include first")
    return {"points": clean, "text": text}


async def run(svc: Services, kind: str, image: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run a preprocessor on an image blob; returns its outputs as stored blobs."""
    if kind != "sam":
        raise PreprocessError(400, f"Unknown preprocessor {kind!r}")
    sha = unref(image)
    size = svc.blobs.image_size(sha) if not svc.blobs.is_video(sha) else None
    path = svc.blobs.path(sha)
    if size is None or path is None:
        raise PreprocessError(404, "The image is no longer stored")
    clean = sam_params(params)
    worker = svc.sessions.worker
    if worker is None or svc.sessions.state not in ("ready", "busy"):
        raise PreprocessError(409, "Start a session to use Select")
    asset = svc.db.get_asset(SAM_ASSET)
    if asset is None or asset["kind"] != "preprocessor":
        raise PreprocessError(
            400, f"SAM 3 isn't in Drive. Put it under degas/{SAM_ASSET}/, then rescan."
        )
    svc.sessions.touch()
    try:
        if not await worker.has_blob(sha):
            await worker.put_blob(sha, path.read_bytes())
        out = await worker.preprocess(
            {
                "id": kind,
                "image": f"sha256:{sha}",
                "asset": {"path": asset["path"], "size": asset["size"]},
                "params": clean,
            }
        )
    except WorkerError as e:
        raise PreprocessError(502, f"Select failed on the GPU: {e}") from None
    candidates = []
    for c in out.get("candidates") or []:
        try:
            data = base64.b64decode(c["mask"], validate=True)
            stored = await svc.inputs.store_mask(sha, data)
        except (binascii.Error, KeyError, MediaError) as e:
            raise PreprocessError(502, f"The GPU returned an unreadable mask: {e}") from None
        candidates.append({**stored, "score": c.get("score")})
    chosen = out.get("chosen", 0) if candidates else None
    return {"candidates": candidates, "chosen": chosen}
