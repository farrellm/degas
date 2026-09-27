"""Stable Diffusion XL runner (t2i)."""

import gc
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)

from degas_worker.families.base import Output, RunContext

# Keep in sync with the server descriptor (degas/families/sdxl.py).
SCHEDULERS: dict[str, tuple[Any, dict[str, Any]]] = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpmpp_2m": (DPMSolverMultistepScheduler, {}),
    "dpmpp_2m_karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "ddim": (DDIMScheduler, {}),
    "unipc": (UniPCMultistepScheduler, {}),
}

# Below this much total VRAM, SDXL fp16 is run with model CPU offload.
_OFFLOAD_BELOW_BYTES = 12 * 1024**3


class SdxlRunner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self._scheduler_config: Any = None

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode", "t2i")
        if mode != "t2i":
            raise ValueError(f"SDXL mode {mode!r} is not supported yet")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        self._set_scheduler(params.get("scheduler", "dpmpp_2m_karras"))
        steps = int(params["steps"])
        clip_skip = int(params.get("clip_skip") or 0)
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)

            def on_step(
                _pipe: Any, i: int, _t: Any, kwargs: dict[str, Any], item: int = item
            ) -> dict[str, Any]:
                ctx.check_cancelled()
                done = i + 1
                ctx.progress(item, "denoise" if done < steps else "decode", done, steps)
                return kwargs

            result = self.pipe(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt") or None,
                width=int(params["width"]),
                height=int(params["height"]),
                num_inference_steps=steps,
                guidance_scale=float(params["cfg"]),
                clip_skip=clip_skip or None,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
            ctx.progress(item, "encode", steps, steps)
            buf = io.BytesIO()
            result.images[0].save(buf, format="PNG")
            yield Output(
                item=item, seed=seed, data=buf.getvalue(), media_type="image/png", ext="png"
            )

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(path), torch_dtype=torch.float16, use_safetensors=path.suffix == ".safetensors"
        )
        _free, total = torch.cuda.mem_get_info()
        if total < _OFFLOAD_BELOW_BYTES:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self._scheduler_config = pipe.scheduler.config

    def _set_scheduler(self, name: str) -> None:
        try:
            cls, kwargs = SCHEDULERS[name]
        except KeyError:
            raise ValueError(f"Unknown scheduler {name!r}") from None
        self.pipe.scheduler = cls.from_config(self._scheduler_config, **kwargs)

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        gc.collect()
        torch.cuda.empty_cache()
