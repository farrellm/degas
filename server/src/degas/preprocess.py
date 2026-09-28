"""Preprocessors run on the worker (design §4.4): SAM 3 selection for masks, and the depth,
pose and edge traces that ControlNet units read."""

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from degas.colab.worker_client import WorkerError
from degas.inputs import unref
from degas.media import MediaError
from degas.services import Services

SAM_ASSET = "preprocessors/sam3"
DEPTH_ASSET = "preprocessors/depth-anything-v2"
POSE_ASSET = "preprocessors/dwpose"
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


def canny_params(params: dict[str, Any]) -> dict[str, Any]:
    """Canny's hysteresis thresholds, 0 to 255, with `low` below `high`."""
    out: dict[str, int] = {}
    for key, default in (("low", 100), ("high", 200)):
        value = params.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PreprocessError(400, f"{key}: expected a number")
        out[key] = max(0, min(255, round(value)))
    if out["low"] >= out["high"]:
        raise PreprocessError(400, "The low threshold must be below the high one")
    return out


def no_params(_params: dict[str, Any]) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Kind:
    name: str  # what the UI calls it, in errors
    use: str  # "Start a session to …"
    check: Callable[[dict[str, Any]], dict[str, Any]]
    output: Literal["masks", "image"]
    asset: str | None = None  # the model folder in Drive, if it needs one
    model: str = ""  # the model's name, for "… isn't in Drive"


KINDS: dict[str, Kind] = {
    "sam": Kind("Select", "use Select", sam_params, "masks", SAM_ASSET, "SAM 3"),
    "depth": Kind("Depth", "trace depth", no_params, "image", DEPTH_ASSET, "Depth Anything V2"),
    "pose": Kind("Pose", "trace poses", no_params, "image", POSE_ASSET, "DWPose"),
    "canny": Kind("Edges", "trace edges", canny_params, "image"),
}


async def run(svc: Services, kind: str, image: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run a preprocessor on an image blob; returns its outputs as stored blobs.

    SAM answers `{candidates: [mask], chosen}`; the traces answer `{image}`.
    """
    pre = KINDS.get(kind)
    if pre is None:
        raise PreprocessError(400, f"Unknown preprocessor {kind!r}")
    sha = unref(image)
    size = svc.blobs.image_size(sha) if not svc.blobs.is_video(sha) else None
    path = svc.blobs.path(sha)
    if size is None or path is None:
        raise PreprocessError(404, "The image is no longer stored")
    clean = pre.check(params)
    worker = svc.sessions.worker
    if worker is None or svc.sessions.state not in ("ready", "busy"):
        raise PreprocessError(409, f"Start a session to {pre.use}")
    request: dict[str, Any] = {"id": kind, "image": f"sha256:{sha}", "params": clean}
    if pre.asset:
        asset = svc.db.get_asset(pre.asset)
        if asset is None or asset["kind"] != "preprocessor":
            raise PreprocessError(
                400, f"{pre.model} isn't in Drive. Put it under degas/{pre.asset}/, then rescan."
            )
        request["asset"] = {"path": asset["path"], "size": asset["size"]}
    svc.sessions.touch()
    try:
        if not await worker.has_blob(sha):
            await worker.put_blob(sha, path.read_bytes())
        out = await worker.preprocess(request)
    except WorkerError as e:
        raise PreprocessError(502, f"{pre.name} failed on the GPU: {e}") from None
    if pre.output == "image":
        try:
            data = base64.b64decode(out["image"], validate=True)
            return {"image": await svc.inputs.store_trace(data, size)}
        except (binascii.Error, KeyError, TypeError, MediaError) as e:
            raise PreprocessError(502, f"The GPU returned an unreadable image: {e}") from None
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
