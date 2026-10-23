"""Qwen-Image 2.1 runner: text-to-image, editing from up to 10 images, and inpaint (design §4.3).

One `QwenImage21Pipeline` does all three: an edit passes the source and its references as the
pipeline's condition images, and an inpaint is an edit whose result is pasted back onto the
source through the blurred mask. The model is the official diffusers folder, in bf16.

*Upsample prompt* first has the text encoder, Qwen3-VL 8B, rewrite the prompt (seeing an edit's
images) as Kokon does (`qwen21_prompt.py`). It's sampled once per job, with the first seed, and
the rewritten prompt is returned with each image as its caption. A job may name another writer
(Qwen3-VL 8B Heretic, the encoder's base model with its refusals ablated), which is kept on the
CPU beside the pipeline and comes to the GPU only to write.
"""

import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from PIL import Image
from safetensors.torch import load_file
from transformers import LogitsProcessor, Qwen3VLForConditionalGeneration

from degas_worker import masks
from degas_worker.degrid import degrid
from degas_worker.families import qwen21_prompt
from degas_worker.families.base import Output, RunContext
from degas_worker.families.lora import qwen21_lora_state
from degas_worker.families.offload import loaded_bytes, module_bytes, place
from degas_worker.families.runtime import (
    fetch_loras,
    free_gpu_memory,
    png_output,
    seeded,
    step_callback,
    sync_loras,
)
from degas_worker.spec import Spec

log = logging.getLogger(__name__)

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
        # The upsampler a job named in place of the text encoder, on the CPU.
        self.writer: Any = None
        self.writer_path: Path | None = None

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode", "t2i")
        if mode not in ("t2i", "edit", "inpaint"):
            raise ValueError(f"Qwen-Image 2.1 can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras = fetch_loras(spec, ctx)
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
        sync_loras(self.pipe, self.adapters, loras, self._load_lora)
        ctx.progress(0, "load", 1, 1)

        prompt = params["prompt"]
        caption: str | None = None
        if params.get("upsample") and mode != "inpaint":
            ctx.progress(0, "upsample", 0, 1)
            caption = self._upsample(
                prompt,
                images,
                params.get("upsample_style", "kokon"),
                seeds[0],
                self._fetch_writer(spec, ctx),
            )
            ctx.check_cancelled()

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
            "prompt": caption or prompt,
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

            result = self.pipe(
                **kwargs,
                generator=seeded(seed),
                callback_on_step_end=step_callback(ctx, item, steps),
            )
            ctx.progress(item, "encode", steps, steps)
            image = result.images[0]
            if params.get("degrid", True):
                image = degrid(image)
            if keep is not None:
                image = masks.composite(images[0], image, keep)
            yield png_output(item, seed, image, caption)

    def _upsample(
        self,
        prompt: str,
        images: list[Image.Image],
        style: str,
        seed: int,
        writer_path: Path | None = None,
    ) -> str | None:
        """The text encoder's rewrite of the prompt (an edit instruction when there are images),
        or None (with a warning) if it wrote nothing usable: the prompt then goes as written.
        With `writer_path`, that model writes it instead, with the pipeline's processor (the
        same Qwen3-VL 8B tokenizer, template and image processor)."""
        shown = []
        for im in images:
            im = im.convert("RGB")  # noqa: PLW2901
            im.thumbnail((qwen21_prompt.IMAGE_MAX_SIDE, qwen21_prompt.IMAGE_MAX_SIDE))
            shown.append(im)
        processor = self.pipe.processor
        text = processor.apply_chat_template(
            qwen21_prompt.messages(style, prompt, shown),
            tokenize=False,
            add_generation_prompt=True,
        )
        if writer_path is not None:
            encoder = self._writer(writer_path)
            encoder.to("cuda")
        else:
            encoder = self.pipe.text_encoder
            if self.offloaded:
                # Its offload hook would move it only for each forward pass, after `generate`
                # has put its tensors on the CPU. The transformer's hook sends it back.
                encoder.to("cuda")
        batch = processor(text=[text], images=shown or None, padding=True, return_tensors="pt").to(
            encoder.device
        )
        start = batch["input_ids"].shape[-1]
        tokenizer = processor.tokenizer
        torch.manual_seed(seed)
        try:
            with torch.inference_mode():
                ids = encoder.generate(
                    **batch,
                    max_new_tokens=qwen21_prompt.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=qwen21_prompt.TEMPERATURE,
                    top_p=qwen21_prompt.TOP_P,
                    top_k=qwen21_prompt.TOP_K,
                    logits_processor=[_PresencePenalty(start, qwen21_prompt.PRESENCE_PENALTY)],
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        finally:
            if writer_path is not None:
                # Make room for the pipeline.
                encoder.to("cpu")
                free_gpu_memory()
        answer: str = tokenizer.decode(ids[0, start:], skip_special_tokens=True)
        caption = qwen21_prompt.clean(answer)
        if caption is None:
            log.warning("No upsampled prompt: %r", answer[:200])
        return caption

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

    @staticmethod
    def _fetch_writer(spec: Spec, ctx: RunContext) -> Path | None:
        writer = spec.get("upsampler")
        return ctx.fetch_asset(writer["path"], writer.get("size")) if writer else None

    def _writer(self, path: Path) -> Any:
        if self.writer_path != path:
            self.writer = None
            self.writer = Qwen3VLForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, local_files_only=True
            ).eval()
            self.writer_path = path
        return self.writer

    def _load_lora(self, file: Path, name: str) -> None:
        self.pipe.load_lora_weights(qwen21_lora_state(load_file(file)), adapter_name=name)

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

    def unload(self) -> None:
        self.writer = None
        self.writer_path = None
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.offloaded = False
        self.adapters = {}
        free_gpu_memory()


class _PresencePenalty(LogitsProcessor):  # type: ignore[misc]
    """Lowers the logit of every token already generated by `penalty` (vLLM's presence
    penalty). transformers' `repetition_penalty` scales instead, and counts the prompt too."""

    def __init__(self, start: int, penalty: float) -> None:
        self.start = start
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids[:, self.start :]
        if generated.shape[-1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def _open(path: Path) -> Image.Image:
    """A condition image, keeping its alpha: the pipeline lays it over white for the encoder."""
    with Image.open(path) as im:
        alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        return im.convert("RGBA" if alpha else "RGB")
