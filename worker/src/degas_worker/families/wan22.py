"""Wan 2.2 runner: text- and image-to-video with LoRAs (design §4.3).

Models are diffusers-format directories. The 5B (TI2V) loads as `WanPipeline` and
does image-to-video through `WanImageToVideoPipeline.from_pipe`, which shares its
components. The A14B variants have two experts (`transformer` for high noise,
`transformer_2` for low noise), and their LoRAs come in high/low pairs.
"""

import contextlib
import gc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan, DiffusionPipeline, WanImageToVideoPipeline
from PIL import Image

from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import plan_loras
from degas_worker.video import encode_mp4

# Offload to the CPU when the loaded weights take more than this share of the GPU's memory.
_OFFLOAD_ABOVE = 0.7

# (LoRA asset path, local file, weight, component it goes into)
LoraLoad = tuple[str, Path, float, str]


class Wan22Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.i2v: Any = None
        self.model_path: Path | None = None
        # "<component>|<asset path>" → adapter name, per expert
        self.adapters: dict[str, str] = {}

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec["mode"]
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras: list[LoraLoad] = []
        for lora in spec.get("loras") or []:
            halves = (
                [(lora.get("high"), "transformer"), (lora.get("low"), "transformer_2")]
                if "high" in lora or "low" in lora
                else [(lora, "transformer")]
            )
            for part, component in halves:
                if part:
                    local = ctx.fetch_asset(part["path"], part.get("size"))
                    loras.append((part["path"], local, float(part["weight"]), component))
        image = None
        if mode == "i2v":
            with Image.open(ctx.blob(spec["inputs"]["source"])) as im:
                image = im.convert("RGB")
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
        if getattr(pipe, "transformer_2", None) is not None:
            kwargs["guidance_scale_2"] = float(params.get("cfg_low") or params["cfg"])
            if params.get("boundary_ratio"):
                pipe.register_to_config(boundary_ratio=float(params["boundary_ratio"]))

        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)

            def on_step(
                _pipe: Any, i: int, _t: Any, cb_kwargs: dict[str, Any], item: int = item
            ) -> dict[str, Any]:
                ctx.check_cancelled()
                done = i + 1
                ctx.progress(item, "denoise" if done < steps else "decode", done, steps)
                return cb_kwargs

            frames = pipe(
                **kwargs,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=on_step,
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
        _free, total = torch.cuda.mem_get_info()
        weights = _loaded_bytes(pipe)
        if weights > total * _OFFLOAD_ABOVE:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        with contextlib.suppress(AttributeError):
            pipe.vae.enable_tiling()  # 720p decodes otherwise peak well above the denoiser
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.adapters = {}

    def _pipe_for(self, mode: str) -> Any:
        if isinstance(self.pipe, WanImageToVideoPipeline):
            if mode != "i2v":
                raise ValueError("This model only makes video from an image")
            return self.pipe
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
        gc.collect()
        torch.cuda.empty_cache()


def _loaded_bytes(pipe: Any) -> int:
    """Size of the pipeline's weights as loaded (Wan's repos store fp32; they load as bf16)."""
    total = 0
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            total += sum(p.numel() * p.element_size() for p in component.parameters())
    return total
