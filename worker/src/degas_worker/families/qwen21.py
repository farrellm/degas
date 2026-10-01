"""Qwen-Image 2.1 runner: text-to-image, editing from up to 10 images, and inpaint (design §4.3).

One `QwenImage21Pipeline` does all three: an edit passes the source and its references as the
pipeline's condition images, and an inpaint is an edit whose result is pasted back onto the
source through the blurred mask. The model is the official diffusers folder, in bf16.
"""

import contextlib
import gc
import io
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from PIL import Image

from degas_worker import masks
from degas_worker.degrid import degrid
from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import plan_loras, single_loras
from degas_worker.families.offload import loaded_bytes, module_bytes, place
from degas_worker.spec import Spec

# Keep in sync with the server descriptor (degas/families/qwen21.py).
SCHEDULES: dict[str, dict[str, Any]] = {"default": {}, "beta": {"use_beta_sigmas": True}}

# The transformer's sequence is the prompt, each condition image's latents (16 px each) and its
# vision tokens in the prompt (32 px each), then the output's latents. PROMPT_TOKENS covers the
# text and template.
LATENT_PX = 16 * 16
VISION_PX = 32 * 32
PROMPT_TOKENS = 1024
# Activation memory per token of the whole sequence during a transformer step (MLP gate and
# projection, attention inputs, the block-causal mask). Estimated from 2K edits on an A100.
ACTIVATION_BYTES_PER_TOKEN = 256 * 1024
# Share of the GPU a step may plan to use; the rest is left for the allocator.
GPU_BUDGET = 0.9


class Qwen21Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.offloaded = False
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name
        self._scheduler_config: Any = None

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode", "t2i")
        if mode not in ("t2i", "edit", "inpaint"):
            raise ValueError(f"Qwen-Image 2.1 can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras = [
            (lora["path"], ctx.fetch_asset(lora["path"], lora.get("size")), float(lora["weight"]))
            for lora in single_loras(spec)
        ]
        inputs = spec.get("inputs") or {}
        images: list[Image.Image] = []
        keep: Image.Image | None = None
        if mode != "t2i":
            images = [_open(ctx.blob(ref)) for ref in [inputs["source"], *inputs.get("refs", [])]]
        params = spec["params"]
        if mode == "inpaint":
            with Image.open(ctx.blob(inputs["mask"])) as im:
                keep = masks.blur(im.convert("L"), int(params.get("mask_blur") or 0))
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        self._apply_loras(loras)
        ctx.progress(0, "load", 1, 1)

        extra = SCHEDULES.get(params.get("schedule", "default"))
        if extra is None:
            raise ValueError(f"Unknown schedule {params['schedule']!r}")
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self._scheduler_config, **extra
        )
        steps = int(params["steps"])
        width, height = int(params["width"]), int(params["height"])
        cfg = float(params["cfg"])
        kwargs: dict[str, Any] = {
            "prompt": params["prompt"],
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            # References are sized to the output's pixel count, each at its own aspect.
            "output_resolution": round(math.sqrt(width * height)),
        }
        if cfg > 1:
            kwargs["negative_prompt"] = params.get("negative_prompt") or ""
            kwargs["true_cfg_scale"] = cfg
        if images:
            kwargs["image"] = images
            kwargs["use_kv_cache"] = self._kv_cache_fits(len(images), width, height)
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)

            def on_step(
                _pipe: Any, i: int, _t: Any, kw: dict[str, Any], item: int = item
            ) -> dict[str, Any]:
                ctx.check_cancelled()
                done = i + 1
                ctx.progress(item, "denoise" if done < steps else "decode", done, steps)
                return kw

            result = self.pipe(
                **kwargs,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
            )
            ctx.progress(item, "encode", steps, steps)
            image = result.images[0]
            if params.get("degrid", True):
                image = degrid(image)
            if keep is not None:
                image = masks.composite(images[0], image, keep)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            yield Output(
                item=item, seed=seed, data=buf.getvalue(), media_type="image/png", ext="png"
            )

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        pipe = QwenImage21Pipeline.from_pretrained(
            str(path), torch_dtype=torch.bfloat16, local_files_only=True
        )
        # All of it is about 32 GB, so an H100 keeps it resident and an L4 or A100 offloads.
        self.offloaded = place(pipe)
        # Untiled, a 2K decode needs several GB on top of the edit's KV cache, which the pipeline
        # holds until it returns.
        pipe.vae.enable_tiling(
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=448,
            tile_sample_stride_width=448,
        )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.adapters = {}
        self._scheduler_config = pipe.scheduler.config

    def _kv_cache_fits(self, n_images: int, width: int, height: int) -> bool:
        """Whether an edit can keep the pipeline's KV cache on the GPU.

        The cache holds every block's keys and values for the prompt and condition images, so
        later steps only run the output's tokens. That is 512 KB a token, or about 10 GB for one
        2K image, so a 2K edit with a reference doesn't fit an A100 (40 GB) next to the
        transformer. Without it each step reruns the whole sequence: slower, but it fits.
        """
        # Each condition image is resized to the output's pixel count.
        pixels = width * height
        prefix = PROMPT_TOKENS + n_images * (pixels // LATENT_PX + pixels // VISION_PX)
        tokens = prefix + pixels // LATENT_PX
        config = self.pipe.transformer.config
        width_bytes = config.num_attention_heads * config.attention_head_dim * 2  # bf16
        cache = prefix * 2 * config.num_layers * width_bytes  # keys and values
        # Offloaded, the transformer and VAE are on the GPU while it denoises.
        resident = (
            module_bytes(self.pipe.transformer) + module_bytes(self.pipe.vae)
            if self.offloaded
            else loaded_bytes(self.pipe)
        )
        _free, total = torch.cuda.mem_get_info()
        need = resident + cache + tokens * ACTIVATION_BYTES_PER_TOKEN
        return bool(need <= total * GPU_BUDGET)

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

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.offloaded = False
        self.adapters = {}
        gc.collect()
        torch.cuda.empty_cache()


def _open(path: Path) -> Image.Image:
    """A condition image, keeping its alpha: the pipeline lays it over white for the encoder."""
    with Image.open(path) as im:
        alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        return im.convert("RGBA" if alpha else "RGB")
