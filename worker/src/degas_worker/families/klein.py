"""FLUX.2 [klein] runner: text-to-image, and editing from up to 4 images (design §4.3).

For an edit, `Flux2KleinPipeline` takes the source and its references as condition images,
each scaled down to at most 1 megapixel; with none it makes the image from the prompt alone.
The 9B model is step-distilled: 4 steps and no CFG, so guidance is left at 1. The model is
the official diffusers folder, in bf16.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from degas_worker.families.base import Output, RunContext
from degas_worker.families.offload import place
from degas_worker.families.runtime import (
    fetch_loras,
    free_gpu_memory,
    png_output,
    seeded,
    step_callback,
    sync_loras,
)
from degas_worker.spec import Spec


class KleinRunner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode")
        if mode not in ("t2i", "edit"):
            raise ValueError(f"FLUX.2 [klein] can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras = fetch_loras(spec, ctx)
        inputs = spec.get("inputs") or {}
        images: list[Image.Image] = []
        if mode == "edit":
            images = [_open(ctx.blob(ref)) for ref in [inputs["source"], *inputs.get("refs", [])]]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        sync_loras(self.pipe, self.adapters, loras)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        steps = int(params["steps"])
        kwargs: dict[str, Any] = {
            "prompt": params["prompt"],
            "width": int(params["width"]),
            "height": int(params["height"]),
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
        }
        if images:
            kwargs["image"] = images
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)

            result = self.pipe(
                **kwargs,
                generator=seeded(seed),
                callback_on_step_end=step_callback(ctx, item, steps),
            )
            ctx.progress(item, "encode", steps, steps)
            yield png_output(item, seed, result.images[0])

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        pipe = Flux2KleinPipeline.from_pretrained(
            str(path), torch_dtype=torch.bfloat16, local_files_only=True
        )
        # About 35 GB: an H100 keeps it resident and an L4 or A100 offloads.
        place(pipe)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.adapters = {}

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.adapters = {}
        free_gpu_memory()


def _open(path: Path) -> Image.Image:
    """A condition image, flattened to RGB (the pipeline doesn't take alpha)."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel("A"))
            return flat
        return im.convert("RGB")
