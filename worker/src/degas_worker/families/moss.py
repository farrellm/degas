"""MOSS-Image runner: text-to-image from the Base or SFT checkpoint (design §4.3).

The model is a diffusers custom pipeline: its folder ships `pipeline_moss_image.py`, which
defines the pipeline, transformer and scheduler, so it loads with `trust_remote_code`. The
Qwen3.5 text encoder and the transformer run in bf16 and the FLUX.2 VAE in fp32. The pipeline
takes no step callback, and its shift lives in the scheduler's config, so each job gets a
scheduler built for its shift.

There is no LoRA loader either, so LoRAs (kohya files, such as Kromoss's 8-step turbo) are
applied unmerged as forward hooks on the transformer's linears, in fp32: their deltas are close
to bf16's rounding step, so merging them would lose most of them.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file

from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import kohya_modules
from degas_worker.families.offload import place, reoffload
from degas_worker.families.runtime import (
    LoraFile,
    fetch_loras,
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
        self.hooks: dict[str, list[Any]] = {}  # LoRA asset path → its forward hooks
        self.weights: dict[str, float] = {}  # LoRA asset path → weight, read by its hooks

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode")
        if mode != "t2i":
            raise ValueError(f"MOSS-Image can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras = fetch_loras(spec, ctx)
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        self._sync_loras(loras)
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

    def _sync_loras(self, loras: list[LoraFile]) -> None:
        """Hook in newly requested LoRAs, unhook ones no longer requested, and set the weights
        (which the hooks read at each call, so a new weight needs no reload)."""
        wanted = {path for path, _, _ in loras}
        for path in [p for p in self.hooks if p not in wanted]:
            for hook in self.hooks.pop(path):
                hook.remove()
            del self.weights[path]
        for path, file, weight in loras:
            self.weights[path] = weight
            if path not in self.hooks:
                self.hooks[path] = self._hook_lora(path, file)

    def _hook_lora(self, path: str, file: Path) -> list[Any]:
        modules = dict(self.pipe.transformer.named_modules())
        hooks: list[Any] = []
        try:
            state = load_file(str(file), device="cuda")
            for name, (down_key, up_key, alpha_key) in kohya_modules(state).items():
                module = modules.get(name)
                down, up = state[down_key].float(), state[up_key].float()
                if not isinstance(module, torch.nn.Linear):
                    raise ValueError(f"the transformer has no linear {name}")
                shape = (up.shape[0], down.shape[1])
                if shape != tuple(module.weight.shape):
                    raise ValueError(f"{name} is {tuple(module.weight.shape)}, the LoRA {shape}")
                rank = down.shape[0]
                scale = (float(state[alpha_key]) if alpha_key else rank) / rank
                hooks.append(module.register_forward_hook(self._lora_hook(path, down, up, scale)))
        except Exception as e:
            for hook in hooks:
                hook.remove()
            raise ValueError(f"Could not load LoRA {path}: {e}") from e
        return hooks

    def _lora_hook(self, path: str, down: Any, up: Any, scale: float) -> Any:
        def hook(module: Any, args: Any, out: Any) -> Any:
            weight = self.weights[path] * scale
            return out + (weight * (args[0].float() @ down.T) @ up.T).to(out.dtype)

        return hook

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.hooks.clear()  # they go with the transformer
        self.weights.clear()
        free_gpu_memory()
