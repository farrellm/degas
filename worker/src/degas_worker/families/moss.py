"""MOSS-Image runner: text-to-image from the Base or SFT checkpoint (design §4.3).

The model is a diffusers custom pipeline: its folder ships `pipeline_moss_image.py`, which
defines the pipeline, transformer and scheduler, so it loads with `trust_remote_code`. The
Qwen3.5 text encoder and the transformer run in bf16 and the FLUX.2 VAE in fp32. The pipeline
takes no step callback, and its shift lives in the scheduler's config, so each job gets a
scheduler built for its shift.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline

from degas_worker.families.base import Output, RunContext
from degas_worker.families.offload import place, reoffload
from degas_worker.families.runtime import (
    free_gpu_memory,
    png_output,
    scheduler_progress,
    seeded,
)
from degas_worker.spec import Spec

PIPELINE_FILE = "pipeline_moss_image.py"


class MossRunner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode")
        if mode != "t2i":
            raise ValueError(f"MOSS-Image can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        steps = int(params["steps"])
        skip = params["skip_layer"]
        kwargs: dict[str, Any] = {
            "prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt") or "",
            "width": int(params["width"]),
            "height": int(params["height"]),
            "num_inference_steps": steps,
            "guidance_scale": float(params["cfg"]),
            # The negative branch runs with this block skipped (none: plain CFG).
            "uncond_skip_layers": [] if skip == "off" else [int(skip)],
        }
        scheduler = self.pipe.scheduler
        self.pipe.scheduler = type(scheduler).from_config(
            scheduler.config, shift=float(params["shift"])
        )
        try:
            for item, seed in enumerate(seeds):
                ctx.check_cancelled()
                ctx.progress(item, "denoise", 0, steps)
                with scheduler_progress(self.pipe.scheduler, ctx, item, steps):
                    result = self.pipe(**kwargs, generator=seeded(seed))
                ctx.progress(item, "encode", steps, steps)
                yield png_output(item, seed, result.images[0])
        except Exception:
            # An out of memory would leave the model that was running on the GPU.
            reoffload(self.pipe)
            raise

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        pipe = DiffusionPipeline.from_pretrained(
            str(path),
            custom_pipeline=str(path / PIPELINE_FILE),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
        )
        # About 27 GB: an 80 GB A100 or an H100 keeps it resident, an L4 or 40 GB A100 offloads.
        place(pipe)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        free_gpu_memory()
