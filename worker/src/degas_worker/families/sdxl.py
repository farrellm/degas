"""Stable Diffusion XL runner: text-to-image, image-to-image, inpaint and outpaint, with LoRAs.

A regular checkpoint is loaded as the text-to-image pipeline, and the image-to-image and
inpaint pipelines are made from it with `from_pipe` (they share its weights and LoRAs). An
inpainting checkpoint (the `inpaint` variant, a 9-channel UNet) only has the inpaint pipeline.
"""

import contextlib
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
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

from degas_worker import masks
from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import plan_loras

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
        self.inpaint_model = False
        self.offload = False
        self.derived: dict[str, Any] = {}  # mode → pipeline made from `pipe` with from_pipe
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name
        self._scheduler_config: Any = None

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode", "t2i")
        if mode not in ("t2i", "i2i", "inpaint", "outpaint"):
            raise ValueError(f"SDXL can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras = [
            (lora["path"], ctx.fetch_asset(lora["path"], lora.get("size")), float(lora["weight"]))
            for lora in spec.get("loras") or []
        ]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path, inpaint=spec.get("variant") == "inpaint")
        self._apply_loras(loras)
        pipe = self._pipe_for(mode)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        self._set_scheduler(pipe, params.get("scheduler", "dpmpp_2m_karras"))
        steps = int(params["steps"])
        size = (int(params["width"]), int(params["height"]))
        kwargs, original, keep = self._inputs(mode, spec, size, ctx)
        # Starting from an image skips the first (1 - strength) of the schedule.
        strength = float(kwargs.get("strength", 1.0))
        runs = max(1, min(steps, int(steps * strength)))
        clip_skip = int(params.get("clip_skip") or 0)
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, runs)

            def on_step(
                _pipe: Any, i: int, _t: Any, kw: dict[str, Any], item: int = item
            ) -> dict[str, Any]:
                ctx.check_cancelled()
                done = min(i + 1, runs)
                ctx.progress(item, "denoise" if done < runs else "decode", done, runs)
                return kw

            result = pipe(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt") or None,
                num_inference_steps=steps,
                guidance_scale=float(params["cfg"]),
                clip_skip=clip_skip or None,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
                **kwargs,
            )
            ctx.progress(item, "encode", runs, runs)
            image = result.images[0]
            if original is not None and keep is not None:
                image = masks.composite(original, image, keep)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            yield Output(
                item=item, seed=seed, data=buf.getvalue(), media_type="image/png", ext="png"
            )

    def _inputs(
        self, mode: str, spec: dict[str, Any], size: tuple[int, int], ctx: RunContext
    ) -> tuple[dict[str, Any], Image.Image | None, Image.Image | None]:
        """Pipeline arguments for the mode, plus the image and mask to composite back onto."""
        params = spec["params"]
        width, height = size
        if mode == "t2i":
            return {"width": width, "height": height}, None, None
        inputs = spec["inputs"]
        with Image.open(ctx.blob(inputs["source"])) as im:
            source = im.convert("RGB")
        if mode == "i2i":
            return {"image": source, "strength": float(params["strength"])}, None, None
        blur = int(params.get("mask_blur") or 0)
        if mode == "inpaint":
            with Image.open(ctx.blob(inputs["mask"])) as im:
                mask = im.convert("L")
            image, strength = source, float(params["strength"])
        else:
            image, mask = masks.outpaint_canvas(
                source, inputs["place"], size, int(params.get("blend") or 0)
            )
            strength = 1.0
        soft = masks.blur(mask, blur)
        kwargs: dict[str, Any] = {
            "image": image,
            "mask_image": soft,
            "strength": strength,
            "width": width,
            "height": height,
        }
        if mode == "inpaint" and params.get("inpaint_area") == "masked":
            kwargs["padding_mask_crop"] = int(params.get("mask_padding") or 0)
        return kwargs, image, soft

    def _load(self, path: Path, inpaint: bool) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        cls = StableDiffusionXLInpaintPipeline if inpaint else StableDiffusionXLPipeline
        pipe = cls.from_single_file(
            str(path), torch_dtype=torch.float16, use_safetensors=path.suffix == ".safetensors"
        )
        _free, total = torch.cuda.mem_get_info()
        self.offload = total < _OFFLOAD_BELOW_BYTES
        if self.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.inpaint_model = inpaint
        self.derived = {}
        self.adapters = {}
        self._scheduler_config = pipe.scheduler.config

    def _pipe_for(self, mode: str) -> Any:
        if self.inpaint_model:
            if mode not in ("inpaint", "outpaint"):
                raise ValueError("An inpainting model can only inpaint or outpaint")
            return self.pipe
        if mode == "t2i":
            return self.pipe
        key = "i2i" if mode == "i2i" else "inpaint"
        if key not in self.derived:
            cls = (
                StableDiffusionXLImg2ImgPipeline
                if key == "i2i"
                else StableDiffusionXLInpaintPipeline
            )
            derived = cls.from_pipe(self.pipe)
            if self.offload:
                derived.enable_model_cpu_offload()
            derived.set_progress_bar_config(disable=True)
            self.derived[key] = derived
        return self.derived[key]

    def _apply_loras(self, loras: list[tuple[str, Path, float]]) -> None:
        """Load only new adapters, delete ones no longer requested, then set the weights."""
        plan = plan_loras(self.adapters, [(path, weight) for path, _, weight in loras])
        if plan.remove:
            self.pipe.delete_adapters(plan.remove)
            self.adapters = {p: n for p, n in self.adapters.items() if n not in plan.remove}
        local = {path: file for path, file, _ in loras}
        for path, name in plan.add:
            try:
                self.pipe.load_lora_weights(str(local[path]), adapter_name=name)
            except Exception as e:
                with contextlib.suppress(Exception):
                    self.pipe.delete_adapters([name])
                raise ValueError(f"Could not load LoRA {path}: {e}") from e
            self.adapters[path] = name
        if plan.names:
            self.pipe.set_adapters(plan.names, adapter_weights=plan.weights)

    def _set_scheduler(self, pipe: Any, name: str) -> None:
        try:
            cls, kwargs = SCHEDULERS[name]
        except KeyError:
            raise ValueError(f"Unknown scheduler {name!r}") from None
        pipe.scheduler = cls.from_config(self._scheduler_config, **kwargs)

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.derived = {}
        self.adapters = {}
        gc.collect()
        torch.cuda.empty_cache()
