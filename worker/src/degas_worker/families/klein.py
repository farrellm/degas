"""FLUX.2 [klein] runner: editing from up to 4 images (design §4.3).

`Flux2KleinPipeline` takes the source and its references as condition images, each scaled
down to at most 1 megapixel. The 9B model is step-distilled: 4 steps and no CFG, so guidance
is left at 1. The model is the official diffusers folder, in bf16.
"""

import contextlib
import gc
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import plan_loras
from degas_worker.families.offload import place


class KleinRunner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        if spec.get("mode") != "edit":
            raise ValueError(f"FLUX.2 [klein] can't do {spec.get('mode')!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras = [
            (lora["path"], ctx.fetch_asset(lora["path"], lora.get("size")), float(lora["weight"]))
            for lora in spec.get("loras") or []
        ]
        inputs = spec["inputs"]
        images = [_open(ctx.blob(ref)) for ref in [inputs["source"], *inputs.get("refs", [])]]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        self._apply_loras(loras)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        steps = int(params["steps"])
        kwargs: dict[str, Any] = {
            "image": images,
            "prompt": params["prompt"],
            "width": int(params["width"]),
            "height": int(params["height"]),
            "num_inference_steps": steps,
            "guidance_scale": 1.0,
        }
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
            buf = io.BytesIO()
            result.images[0].save(buf, format="PNG")
            yield Output(
                item=item, seed=seed, data=buf.getvalue(), media_type="image/png", ext="png"
            )

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
        self.adapters = {}
        gc.collect()
        torch.cuda.empty_cache()


def _open(path: Path) -> Image.Image:
    """A condition image, flattened to RGB (the pipeline doesn't take alpha)."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel("A"))
            return flat
        return im.convert("RGB")
