"""FLUX 3 Image runner: text-to-image, and editing from up to 10 images (design §4.3).

The model takes full scene captions, so the prompt first goes through BFL's prompt upsampler
(a 9B Qwen3.5 VLM), which sees the condition images of an edit too. Its caption is the same
for every seed (greedy decoding), so it's written once per job and returned with each image.
diffusers runs FLUX 3 as a modular pipeline: its upsampler block is left out so the runner
keeps the caption, and its components load from the Drive folder's subfolders, since the
folder's `modular_model_index.json` names them by Hub repo.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from accelerate import cpu_offload_with_hook
from accelerate.hooks import remove_hook_from_module
from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import Flux3AutoBlocks
from diffusers.modular_pipelines.flux3.prompt_upsampler import (
    prepare_references,
    upsample_prompt,
)
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from degas_worker.families.base import Output, RunContext
from degas_worker.families.klein import _open
from degas_worker.families.offload import OFFLOAD_ABOVE, module_bytes
from degas_worker.families.runtime import (
    fetch_loras,
    free_gpu_memory,
    png_output,
    scheduler_progress,
    seeded,
    sync_loras,
)
from degas_worker.spec import Spec

# Keep in sync with the server descriptor (degas/families/flux3.py).
REF_MAX_PIXELS = 1024 * 1024
# The order the models run in, for offloading one at a time.
MODELS = ("text_encoder", "transformer", "vae")


class Flux3Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.upsampler: Any = None
        self.processor: Any = None
        self.upsampler_path: Path | None = None
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name
        # Whether the models don't all fit on the GPU, and so are chained through offload hooks.
        self.offloaded = False
        self.hooks: list[Any] = []

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode")
        if mode not in ("t2i", "edit"):
            raise ValueError(f"FLUX 3 Image can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        upsampler = spec.get("upsampler")
        pup_path = ctx.fetch_asset(upsampler["path"], upsampler.get("size")) if upsampler else None
        loras = fetch_loras(spec, ctx)
        inputs = spec.get("inputs") or {}
        images: list[Image.Image] = []
        if mode == "edit":
            refs = [inputs["source"], *inputs.get("refs", [])]
            images = [_shrink(_open(ctx.blob(ref))) for ref in refs]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path, pup_path)
        # LoRA loading takes offload hooks off and puts them back only on a DiffusionPipeline.
        self._unhook()
        try:
            sync_loras(self.pipe, self.adapters, loras)
        finally:
            self._hook()
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        prompt = params["prompt"]
        caption: str | None = None
        if pup_path is not None:
            ctx.progress(0, "upsample", 0, 1)
            caption = self._upsample(prompt, images)
            ctx.check_cancelled()
        self.pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=params["cfg"]))
        steps = int(params["steps"])
        kwargs: dict[str, Any] = {
            "prompt": caption or prompt,
            "width": int(params["width"]),
            "height": int(params["height"]),
            "num_inference_steps": steps,
        }
        if params.get("negative_prompt"):
            kwargs["negative_prompt"] = params["negative_prompt"]
        if images:
            kwargs["image"] = images
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)
            with scheduler_progress(self.pipe.scheduler, ctx, item, steps):
                image = self.pipe(**kwargs, generator=seeded(seed), output="images")[0]
            ctx.progress(item, "encode", steps, steps)
            yield png_output(item, seed, image, caption)

    def _upsample(self, prompt: str, images: list[Image.Image]) -> str:
        """The upsampler's scene caption for this request (and the edit's images)."""
        if self.offloaded:
            # Make room: the pipeline's models go back to the CPU. The upsampler goes back when
            # the text encoder, next in the hook chain, runs.
            for hook in self.hooks:
                hook.offload()
            self.upsampler.to("cuda")
        with torch.no_grad():
            caption: str = upsample_prompt(
                self.upsampler,
                self.processor,
                prompt,
                images=prepare_references(images) if images else None,
            )
        return caption

    def _load(self, path: Path, pup_path: Path | None) -> None:
        if self.pipe is not None and self.model_path != path:
            self.unload()
        changed = False
        if self.pipe is None:
            blocks = Flux3AutoBlocks()
            blocks.sub_blocks.pop("prompt_upsampler")
            pipe = blocks.init_pipeline()
            kw: dict[str, Any] = {"torch_dtype": torch.bfloat16, "local_files_only": True}
            pipe.update_components(
                transformer=Flux3Transformer2DModel.from_pretrained(path / "transformer", **kw),
                vae=AutoencoderKLFlux2.from_pretrained(path / "vae", **kw),
                text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                    path / "text_encoder", dtype=torch.bfloat16, local_files_only=True
                ),
                tokenizer=AutoProcessor.from_pretrained(
                    path / "text_encoder", local_files_only=True
                ),
            )
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
            self.model_path = path
            self.adapters = {}
            changed = True
        if pup_path is not None and self.upsampler_path != pup_path:
            self._unhook()
            self.upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(
                pup_path, dtype=torch.bfloat16, local_files_only=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(pup_path, local_files_only=True)
            self.upsampler_path = pup_path
            changed = True
        if changed:
            self._place()

    def _models(self) -> list[torch.nn.Module]:
        """The loaded models in the order they run: the upsampler first, if loaded."""
        models = [getattr(self.pipe, name) for name in MODELS]
        return [self.upsampler, *models] if self.upsampler is not None else models

    def _place(self) -> None:
        """Put every model on the GPU if they fit (about 39 GB with the upsampler: an 80 GB GPU
        does), or else chain offload hooks through them, so each comes in when it runs and
        sends the one before it back to the CPU."""
        self._unhook()
        _free, total = torch.cuda.mem_get_info()
        models = self._models()
        self.offloaded = sum(module_bytes(m) for m in models) > total * OFFLOAD_ABOVE
        for m in models:
            m.to("cpu" if self.offloaded else "cuda")
        free_gpu_memory()
        self._hook()

    def _hook(self) -> None:
        if not self.offloaded or self.hooks:
            return
        hook = None
        for m in self._models():
            _, hook = cpu_offload_with_hook(m, "cuda", prev_module_hook=hook)
            self.hooks.append(hook)

    def _unhook(self) -> None:
        """Take the offload hooks off, with every model on the CPU, until `_hook`."""
        if not self.hooks:
            return
        for hook in self.hooks:
            hook.offload()
        for m in self._models():
            remove_hook_from_module(m, recurse=True)
        self.hooks = []

    def unload(self) -> None:
        if self.pipe is None:
            return
        self._unhook()
        self.pipe = None
        self.model_path = None
        self.adapters = {}
        self.upsampler = None
        self.processor = None
        self.upsampler_path = None
        self.offloaded = False
        free_gpu_memory()


def _shrink(image: Image.Image) -> Image.Image:
    """A condition image scaled down to at most REF_MAX_PIXELS (the pipeline keeps each at its
    own size, up to 4096 x 4096)."""
    w, h = image.size
    if w * h <= REF_MAX_PIXELS:
        return image
    scale = (REF_MAX_PIXELS / (w * h)) ** 0.5
    return image.resize(
        (max(16, round(w * scale)), max(16, round(h * scale))), Image.Resampling.LANCZOS
    )
