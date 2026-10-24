"""Qwen-Image 2.1 jobs for the MCP tools: build the spec from a prompt and images, queue it,
start a GPU session if none is up, and wait for the images."""

import asyncio
import io
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

from PIL import Image
from starlette.concurrency import run_in_threadpool

from degas.blobs import is_ref, ref, unref
from degas.db import ResultRow
from degas.errors import DegasError
from degas.families import FAMILIES
from degas.families.base import SpecError
from degas.families.qwen21 import MAX_REFS, PRESETS, SIZE
from degas.media import MediaError
from degas.services import Services
from degas.submission import submit

FAMILY = "qwen21"
GPU = "L4"  # Qwen's smallest GPU, for a session the tools start themselves
Resolution = Literal["1K", "2K"]
# Qwen's presets come as seven 2K shapes, then the same shapes at about 1 megapixel.
Aspect = Literal["1:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"]
ASPECTS: tuple[str, ...] = get_args(Aspect)
SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "2K": dict(zip(ASPECTS, PRESETS[:7], strict=True)),
    "1K": dict(zip(ASPECTS, PRESETS[7:], strict=True)),
}
MAX_SIDE = 2752  # the schema's maximum width and height
MIN_SIDE = 512
WAIT_S = 45 * 60  # a cold session copies the model to the VM before the first image
POLL_S = 1.0
BARE_SHA = re.compile(r"[0-9a-f]{64}")

Report = Callable[[str], Awaitable[None]]


class GenerateError(DegasError):
    pass


@dataclass
class Made:
    """A finished image: its result row and a small JPEG to show the agent."""

    result: ResultRow
    preview: bytes


async def image_ref(svc: Services, value: str, n: int) -> str:
    """Image `n` as a `sha256:` ref: a stored blob's ref (or bare hash), or an http(s) or
    `data:` link, which is imported."""
    value = value.strip()
    if BARE_SHA.fullmatch(value):
        value = ref(value)
    if is_ref(value):
        sha = unref(value)
        if svc.blobs.is_video(sha) or svc.blobs.image_size(sha) is None:
            raise MediaError(f"Image {n}: {value} isn't a stored image")
        return value
    try:
        info = await svc.inputs.fetch(value)
    except MediaError as e:
        raise MediaError(f"Image {n}: {e}") from None
    if info["media_type"].startswith("video/"):
        raise MediaError(f"Image {n} is a video")
    return ref(info["sha256"])


def default_resolution(images: list[str]) -> Resolution:
    """2K, except with images: a 2K edit with references runs out of memory on an L4, the GPU
    the tools start."""
    return "1K" if images else "2K"


def preset_size(aspect: str, resolution: Resolution) -> tuple[int, int]:
    try:
        return SIZES[resolution][aspect]
    except KeyError:
        raise SpecError(
            f"aspect_ratio must be one of {', '.join(ASPECTS)}, resolution 1K or 2K"
        ) from None


def matching_size(width: int, height: int, resolution: Resolution) -> tuple[int, int]:
    """The output size with an image's own shape, at the resolution's pixel count."""
    side = SIZES[resolution]["1:1"][0]
    aspect = width / height
    m = SIZE.multiple_of
    w = min(MAX_SIDE, max(MIN_SIDE, int(side * math.sqrt(aspect)) // m * m))
    h = min(MAX_SIDE, max(MIN_SIDE, int(side / math.sqrt(aspect)) // m * m))
    while w * h > SIZE.max_pixels:
        w, h = (w - m, h) if w > h else (w, h - m)
    return w, h


def build_spec(
    svc: Services,
    prompt: str,
    images: list[str],
    size: tuple[int, int],
    fit: Literal["crop", "pad"],
    seed: int | None,
    upsample: bool,
    negative_prompt: str = "",
) -> dict[str, Any]:
    """A Qwen spec: text-to-image without images, else an edit of image 1 that refers to the
    rest by number."""
    if len(images) > MAX_REFS + 1:
        raise SpecError(f"At most {MAX_REFS + 1} images can go with the prompt")
    models = svc.db.list_assets(FAMILY, "model")
    if not models:
        raise SpecError("There's no Qwen-Image 2.1 model in Drive")
    params: dict[str, Any] = {
        "prompt": prompt,
        "width": size[0],
        "height": size[1],
        "upsample": upsample,
    }
    if seed is not None:
        params["seed"] = seed
    if negative_prompt:
        params["negative_prompt"] = negative_prompt
    spec: dict[str, Any] = {
        "family": FAMILY,
        "variant": "base",
        "mode": "edit" if images else "t2i",
        "model": {"path": models[0]["path"]},
        "params": params,
    }
    if images:
        spec["inputs"] = {"source": images[0], "fit": fit, "refs": images[1:]}
    return spec


async def run(svc: Services, spec: dict[str, Any], report: Report) -> list[Made]:
    """Queue the spec, start a session if none is active, and wait for its images."""
    job = await submit(svc, FAMILIES[FAMILY], spec)
    if not svc.sessions.active:
        await report(f"Starting a GPU session ({GPU})")
        await svc.sessions.start(GPU, False)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT_S
    last = ""
    while True:
        current = svc.db.get_job(job["id"])
        if current is None:
            raise GenerateError("The job was deleted")
        if current["status"] == "done":
            break
        if current["status"] in ("error", "cancelled"):
            raise GenerateError(current["error"] or f"The job was {current['status']}")
        if current["status"] == "queued" and not svc.sessions.active:
            await svc.dispatcher.cancel(job["id"])
            step = svc.sessions.step
            raise GenerateError(
                "The GPU session ended before the job ran" + (f": {step}" if step else "")
            )
        if loop.time() > deadline:
            raise GenerateError(
                f"Still waiting after {WAIT_S // 60} minutes; job {job['id']} stays queued "
                "and its image will appear in Results"
            )
        message = _status(svc, job["id"], current["status"])
        if message != last:
            await report(message)
            last = message
        await asyncio.sleep(POLL_S)
    rows = sorted(svc.db.list_results(job_id=job["id"]), key=lambda r: r["item_index"])
    made = []
    for row in rows:
        path = svc.blobs.path(row["blob_sha"])
        if path is None:
            raise GenerateError("The image is no longer stored")
        made.append(Made(row, await run_in_threadpool(_preview, path.read_bytes())))
    return made


def _status(svc: Services, job_id: str, status: str) -> str:
    if status == "queued":
        step = svc.sessions.step
        return f"Queued ({step})" if step and svc.sessions.state == "starting" else "Queued"
    progress = svc.dispatcher.progress.get(job_id) or {}
    phase = progress.get("phase")
    if phase == "denoise":
        return f"Denoising, step {progress.get('step')} of {progress.get('steps')}"
    return f"Running ({phase})" if phase else "Running"


PREVIEW_SIDE = 1024


def _preview(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        im.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()
