"""What the runners share around a diffusers pipeline: seeds, progress, outputs and LoRAs."""

import contextlib
import gc
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import plan_loras, single_loras
from degas_worker.spec import Spec

# (LoRA asset path, local file, weight)
LoraFile = tuple[str, Path, float]
# diffusers' `callback_on_step_end`: (pipeline, step index, timestep, tensors) → tensors
StepCallback = Callable[[Any, int, Any, dict[str, Any]], dict[str, Any]]


def free_gpu_memory() -> None:
    """Call after dropping the last reference to a model, so its memory is really freed."""
    gc.collect()
    torch.cuda.empty_cache()


def seeded(seed: int) -> Any:
    """A CPU generator, so a seed gives the same image on every GPU."""
    return torch.Generator("cpu").manual_seed(seed)


def step_callback(
    ctx: RunContext,
    item: int,
    steps: int,
    after: Callable[[Any, int], None] | None = None,
) -> StepCallback:
    """Report each finished step of `item` (the last as `decode`, which comes next), and stop
    a cancelled job. `after(pipeline, steps done)` runs once the step is reported."""

    def on_step(pipe: Any, i: int, _t: Any, tensors: dict[str, Any]) -> dict[str, Any]:
        ctx.check_cancelled()
        done = min(i + 1, steps)
        ctx.progress(item, "denoise" if done < steps else "decode", done, steps)
        if after is not None:
            after(pipe, done)
        return tensors

    return on_step


def png_output(item: int, seed: int, image: Image.Image) -> Output:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Output(item=item, seed=seed, data=buf.getvalue(), media_type="image/png", ext="png")


def fetch_loras(spec: Spec, ctx: RunContext) -> list[LoraFile]:
    """Copy a spec's (single-file) LoRAs into the cache."""
    return [
        (lora["path"], ctx.fetch_asset(lora["path"], lora.get("size")), float(lora["weight"]))
        for lora in single_loras(spec)
    ]


def sync_loras(
    pipe: Any,
    applied: dict[str, str],
    loras: list[LoraFile],
    load: Callable[[Path, str], None] | None = None,
    *,
    activate: bool = True,
) -> None:
    """Make the pipeline's adapters match `loras`: load only new ones, delete ones no longer
    requested, then (unless `activate` is off) set the weights.

    `applied` (asset path → adapter name) is what is loaded now, and is updated as adapters
    come and go. `load(file, adapter name)` defaults to the pipeline's `load_lora_weights`.
    """
    plan = plan_loras(applied, [(path, weight) for path, _, weight in loras])
    if plan.remove:
        pipe.delete_adapters(plan.remove)
        for path in [p for p, name in applied.items() if name in plan.remove]:
            del applied[path]
    local = {path: file for path, file, _ in loras}
    for path, name in plan.add:
        try:
            if load is None:
                pipe.load_lora_weights(str(local[path]), adapter_name=name)
            else:
                load(local[path], name)
        except Exception as e:
            with contextlib.suppress(Exception):
                pipe.delete_adapters([name])
            raise ValueError(f"Could not load LoRA {path}: {e}") from e
        applied[path] = name
    if activate and plan.names:
        pipe.set_adapters(plan.names, adapter_weights=plan.weights)
