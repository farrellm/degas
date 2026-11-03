"""Image jobs for the MCP tools, on the model `[mcp]` in degas.toml names: build the spec from
a prompt and images, queue it, start a GPU session if none is up, and wait for the images."""

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
from degas.config import McpConfig
from degas.db import ResultRow
from degas.errors import DegasError
from degas.families import FAMILIES
from degas.families.base import (
    GPUS,
    FamilyDescriptor,
    SizeConstraints,
    SpecError,
    Variant,
)
from degas.families.validation import validate_model, validate_params
from degas.media import MediaError
from degas.services import Services
from degas.submission import submit

Resolution = Literal["1K", "2K"]
PIXELS: dict[str, int] = {"1K": 1024 * 1024, "2K": 2048 * 2048}
Aspect = Literal["1:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"]
ASPECTS: tuple[str, ...] = get_args(Aspect)
# A family's preset stands for an aspect within this much of it, and a resolution within
# PRESET_PIXELS of its pixel count; otherwise the size is worked out.
PRESET_ASPECT = 0.05
PRESET_PIXELS = 1.25
WAIT_S = 45 * 60  # a cold session copies the model to the VM before the first image
POLL_S = 1.0
BARE_SHA = re.compile(r"[0-9a-f]{64}")

Report = Callable[[str], Awaitable[None]]


class GenerateError(DegasError):
    pass


class McpConfigError(ValueError):
    """`[mcp]` in degas.toml names something the family doesn't have."""


@dataclass(frozen=True)
class Target:
    """The configured family and variant the tools generate with."""

    family: FamilyDescriptor
    variant: Variant
    config: McpConfig

    @property
    def label(self) -> str:
        return self.variant.label

    @property
    def gpu(self) -> str:
        return self.config.gpu or self.variant.min_gpu

    @property
    def can_edit(self) -> bool:
        return "edit" in self.variant.modes

    @property
    def max_images(self) -> int:
        """Images a call can take: the edit's source and its references."""
        return self.variant.max_refs + 1 if self.can_edit else 0

    @property
    def size(self) -> SizeConstraints:
        return self.family.size_constraints(self.variant.id)

    @property
    def sides(self) -> tuple[int, int]:
        """The smallest and largest width or height the family takes."""
        width = self.family.param_schema(self.variant.id, "t2i")["properties"]["width"]
        return width["minimum"], width["maximum"]

    def modes(self) -> tuple[str, ...]:
        return ("t2i", "edit") if self.can_edit else ("t2i",)

    def params(self, mode: str) -> dict[str, Any]:
        """The configured parameters this mode takes."""
        props = self.family.param_schema(self.variant.id, mode)["properties"]
        extra = self.config.edit_params if mode == "edit" else self.config.t2i_params
        return {k: v for k, v in {**self.config.params, **extra}.items() if k in props}

    def takes(self, mode: str, param: str) -> bool:
        return param in self.family.param_schema(self.variant.id, mode)["properties"]


def resolve_target(config: McpConfig) -> Target:
    """Check `[mcp]` against its family, so a bad config fails when the app is built."""
    family = FAMILIES.get(config.family)
    if family is None:
        raise McpConfigError(
            f"[mcp] family {config.family!r} isn't one of {', '.join(sorted(FAMILIES))}"
        )
    if family.media != "image":
        raise McpConfigError(f"[mcp] {family.label} makes videos; the tools make images")
    if config.gpu is not None and config.gpu not in GPUS:
        raise McpConfigError(f"[mcp] gpu must be one of {', '.join(GPUS)}")
    target = Target(family, _variant(family, config.variant), config)
    try:
        if config.model is not None:
            validate_model({"path": config.model}, target.variant)
        _check_params(target)
    except SpecError as e:
        raise McpConfigError(f"[mcp] {e}") from None
    return target


def _variant(family: FamilyDescriptor, wanted: str | None) -> Variant:
    """The variant named, else the first that does t2i and edit, else the first t2i one."""
    t2i = [v for v in family.variants if "t2i" in v.modes]
    if wanted is None:
        variant = next((v for v in t2i if "edit" in v.modes), t2i[0] if t2i else None)
        if variant is None:
            raise McpConfigError(f"[mcp] {family.label} has no text-to-image variant")
        return variant
    variant = next((v for v in family.variants if v.id == wanted), None)
    if variant is None:
        ids = ", ".join(v.id for v in family.variants)
        raise McpConfigError(f"[mcp] {family.label} has no variant {wanted!r} ({ids})")
    if variant not in t2i:
        raise McpConfigError(f"[mcp] {variant.label} doesn't do text-to-image")
    return variant


def _check_params(target: Target) -> None:
    """Every configured parameter belongs to a mode that uses it, with a value it takes."""
    config, modes = target.config, target.modes()
    schemas = {mode: target.family.param_schema(target.variant.id, mode) for mode in modes}
    props = {mode: set(schema["properties"]) for mode, schema in schemas.items()}
    for mode, values in (("t2i", config.t2i_params), ("edit", config.edit_params)):
        if values and mode not in modes:
            raise SpecError(f"{target.label} can't do {mode!r}, so {mode}_params is unused")
        unknown = set(values) - props.get(mode, set())
        if unknown:
            raise SpecError(f"{mode}_params: unknown parameters {', '.join(sorted(unknown))}")
    unknown = set(config.params) - set().union(*props.values())
    if unknown:
        raise SpecError(f"params: unknown parameters {', '.join(sorted(unknown))}")
    for mode, schema in schemas.items():
        validate_params(schema, {"prompt": "x", **target.params(mode)})


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


def default_resolution(target: Target, images: list[str]) -> Resolution:
    """The configured resolution, a separate one with images (a 2K Qwen edit with references
    runs out of memory on an L4)."""
    return target.config.resolution_with_images if images else target.config.resolution


def _ratio(aspect: str) -> float:
    w, h = aspect.split(":")
    return int(w) / int(h)


def _pixels(target: Target, resolution: Resolution) -> int:
    return min(PIXELS[resolution], target.size.max_pixels)


def preset_size(target: Target, aspect: str, resolution: Resolution) -> tuple[int, int]:
    """The family's preset for the aspect and resolution, else a size worked out for them."""
    if aspect not in ASPECTS:
        raise SpecError(f"aspect_ratio must be one of {', '.join(ASPECTS)}")
    ratio = _ratio(aspect)
    pixels = _pixels(target, resolution)
    near = [
        (w, h)
        for w, h in target.size.presets
        if abs(w / h / ratio - 1) <= PRESET_ASPECT
        and 1 / PRESET_PIXELS <= w * h / pixels <= PRESET_PIXELS
    ]
    if near:
        return min(near, key=lambda p: abs(p[0] * p[1] - pixels))
    return _size(target, ratio, pixels)


def matching_size(
    target: Target, width: int, height: int, resolution: Resolution
) -> tuple[int, int]:
    """The output size with an image's own shape, at the resolution's pixel count."""
    return _size(target, width / height, _pixels(target, resolution))


def _size(target: Target, ratio: float, pixels: int) -> tuple[int, int]:
    m = target.size.multiple_of
    lo, hi = target.sides
    w, h = math.sqrt(pixels * ratio), math.sqrt(pixels / ratio)
    # A side past the family's maximum shrinks both, keeping the shape.
    scale = min(1, hi / w, hi / h)
    w = min(hi, max(lo, int(w * scale) // m * m))
    h = min(hi, max(lo, int(h * scale) // m * m))
    while w * h > target.size.max_pixels:
        w, h = (w - m, h) if w > h else (w, h - m)
    return w, h


def build_spec(
    svc: Services,
    target: Target,
    prompt: str,
    images: list[str],
    size: tuple[int, int],
    fit: Literal["crop", "pad"],
    seed: int | None,
    upsample: bool | None,
    negative_prompt: str = "",
) -> dict[str, Any]:
    """A spec on the configured model: text-to-image without images, else an edit of image 1
    that refers to the rest by number. The family checks the rest when it's submitted."""
    mode = "edit" if images else "t2i"
    if images and not target.can_edit:
        raise SpecError(f"{target.label} doesn't take images")
    if len(images) > target.max_images:
        raise SpecError(f"At most {target.max_images} images can go with the prompt")
    model = target.config.model
    if model is None:
        prefix = target.variant.model_dir or ""
        models = [
            a["path"]
            for a in svc.db.list_assets(target.family.id, "model")
            if not prefix or a["path"] == prefix or a["path"].startswith(prefix + "/")
        ]
        if not models:
            raise SpecError(f"There's no {target.label} model in Drive")
        model = models[0]
    params: dict[str, Any] = {
        **target.params(mode),
        "prompt": prompt,
        "width": size[0],
        "height": size[1],
    }
    if seed is not None:
        params["seed"] = seed
    if negative_prompt:
        if not target.takes(mode, "negative_prompt"):
            raise SpecError(f"{target.label} doesn't take a negative prompt here")
        params["negative_prompt"] = negative_prompt
    if upsample is not None:
        if target.takes(mode, "upsample"):
            params["upsample"] = upsample
        elif upsample:
            raise SpecError(f"{target.label} can't upsample the prompt here")
    spec: dict[str, Any] = {
        "family": target.family.id,
        "variant": target.variant.id,
        "mode": mode,
        "model": {"path": model},
        "params": params,
    }
    if target.config.loras:
        spec["loras"] = [lora.model_dump() for lora in target.config.loras]
    if images:
        spec["inputs"] = {"source": images[0], "fit": fit, "refs": images[1:]}
    return spec


async def run(svc: Services, target: Target, spec: dict[str, Any], report: Report) -> list[Made]:
    """Queue the spec, start a session if none is active, and wait for its images."""
    job = await submit(svc, target.family, spec)
    if not svc.sessions.active:
        await report(f"Starting a GPU session ({target.gpu})")
        await svc.sessions.start(target.gpu, target.config.high_mem)
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
