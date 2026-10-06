"""Wan 2.2 runner: text-, image- and first-and-last-frame-to-video with LoRAs (design §4.3).

Models are diffusers-format directories. The 5B (TI2V) loads as `WanPipeline` and
does image-to-video through `WanImageToVideoPipeline.from_pipe`, which shares its
components. The A14B variants have two experts (`transformer` for high noise,
`transformer_2` for low noise), and their LoRAs come in high/low pairs. Wan 2.1's 14B
image-to-video and first-and-last-frame models load as `WanImageToVideoPipeline` with one
transformer and a CLIP image encoder. A last frame goes in as the pipeline's `last_image`.
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan, DiffusionPipeline, WanImageToVideoPipeline
from PIL import Image

from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import expert_loras, plan_loras
from degas_worker.families.offload import place
from degas_worker.families.runtime import (
    free_gpu_memory,
    seeded,
    step_callback,
)
from degas_worker.spec import Spec
from degas_worker.video import encode_mp4

# (LoRA asset path, local file, weight, component it goes into)
LoraLoad = tuple[str, Path, float, str]
# The modes that start from a source image.
IMAGE_MODES = frozenset({"i2v", "flf2v"})


class Wan22Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.i2v: Any = None
        self.model_path: Path | None = None
        # "<component>|<asset path>" → adapter name, per expert
        self.adapters: dict[str, str] = {}

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec["mode"]
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras: list[LoraLoad] = [
            (
                part["path"],
                ctx.fetch_asset(part["path"], part.get("size")),
                float(part["weight"]),
                component,
            )
            for part, component in expert_loras(spec)
        ]
        inputs = spec.get("inputs") or {}
        image = _open(ctx.blob(inputs["source"])) if mode in IMAGE_MODES else None
        last = _open(ctx.blob(inputs["end"])) if mode == "flf2v" else None
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        self._apply_loras(loras)
        pipe = self._pipe_for(mode)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        steps = int(params["steps"])
        width, height = int(params["width"]), int(params["height"])
        kwargs: dict[str, Any] = {
            "prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt") or None,
            "width": width,
            "height": height,
            "num_frames": int(params["num_frames"]),
            "num_inference_steps": steps,
            "guidance_scale": float(params["cfg"]),
            "output_type": "np",
        }
        if image is not None:
            kwargs["image"] = image
        if last is not None:
            kwargs["last_image"] = last
        if getattr(pipe, "transformer_2", None) is not None:
            kwargs["guidance_scale_2"] = float(params.get("cfg_low") or params["cfg"])
            if params.get("boundary_ratio"):
                pipe.register_to_config(boundary_ratio=float(params["boundary_ratio"]))

        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)

            frames = pipe(
                **kwargs,
                generator=seeded(seed),
                callback_on_step_end=step_callback(ctx, item, steps),
            ).frames[0]
            ctx.progress(item, "encode", steps, steps)
            pixels = (np.clip(np.asarray(frames), 0, 1) * 255).round().astype(np.uint8)
            data = encode_mp4(
                (frame.tobytes() for frame in pixels),
                pixels.shape[2],
                pixels.shape[1],
                float(params["fps"]),
            )
            yield Output(item=item, seed=seed, data=data, media_type="video/mp4", ext="mp4")

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        vae = AutoencoderKLWan.from_pretrained(
            str(path), subfolder="vae", torch_dtype=torch.float32
        )
        pipe = DiffusionPipeline.from_pretrained(str(path), vae=vae, torch_dtype=torch.bfloat16)
        place(pipe)
        with contextlib.suppress(AttributeError):
            pipe.vae.enable_tiling()  # 720p decodes otherwise peak well above the denoiser
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.adapters = {}

    def _pipe_for(self, mode: str) -> Any:
        if isinstance(self.pipe, WanImageToVideoPipeline):
            if mode not in IMAGE_MODES:
                raise ValueError("This model only makes video from an image")
            return self.pipe
        if mode == "flf2v":
            # The 5B's pipeline would ignore the last frame.
            raise ValueError("This model can't make video between a first and a last frame")
        if mode != "i2v":
            return self.pipe
        if self.i2v is None:
            # Not from_pipe: it casts the shared modules in place to one dtype (float32 by
            # default), doubling the bf16 experts and losing the float32 VAE.
            config = self.pipe.config
            self.i2v = WanImageToVideoPipeline(
                **self.pipe.components,
                boundary_ratio=config.get("boundary_ratio"),
                expand_timesteps=config.get("expand_timesteps", False),
            )
            self.i2v.set_progress_bar_config(disable=True)
        return self.i2v

    def _apply_loras(self, loras: list[LoraLoad]) -> None:
        """Per expert: load new adapters, delete ones no longer requested, set the weights."""
        keyed = {f"{component}|{path}": (file, weight) for path, file, weight, component in loras}
        plan = plan_loras(self.adapters, [(key, w) for key, (_, w) in keyed.items()])
        for key, name in list(self.adapters.items()):
            if name in plan.remove:
                self._component(key).delete_adapters([name])
                del self.adapters[key]
        for key, name in plan.add:
            component, path = key.split("|", 1)
            try:
                self.pipe.load_lora_weights(
                    str(keyed[key][0]),
                    adapter_name=name,
                    load_into_transformer_2=component == "transformer_2",
                )
            except Exception as e:
                with contextlib.suppress(Exception):
                    self._component(key).delete_adapters([name])
                raise ValueError(f"Could not load LoRA {path}: {e}") from e
            self.adapters[key] = name
        for component in ("transformer", "transformer_2"):
            names = [n for key, n in self.adapters.items() if key.startswith(component + "|")]
            weights = [keyed[k][1] for k in self.adapters if k.startswith(component + "|")]
            if names:
                getattr(self.pipe, component).set_adapters(names, weights)

    def _component(self, key: str) -> Any:
        return getattr(self.pipe, key.split("|", 1)[0])

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.i2v = None
        self.model_path = None
        self.adapters = {}
        free_gpu_memory()


def _open(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
