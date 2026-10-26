"""Krea 2 runner: text-to-image, and editing through an edit LoRA (design §4.3).

`Krea2Pipeline` does text-to-image. For an edit, the spec's edit LoRA is loaded beside the
person's, and the two paths it was trained with are added around the stock pipeline
(`krea2_edit` has the geometry and the sources):

- The prompt and the empty negative are encoded by Qwen3-VL together with the images, and
  passed as `prompt_embeds`.
- The images' clean VAE tokens are put before the noisy target's on every transformer call
  (`prepend_refs`), on their own RoPE frames, and the target-to-image attention logits are
  raised by each image's boost.

Raw runs with CFG, which the form gives in the usual convention: Krea's `guidance_scale` is
`cond + s * (cond - uncond)`, so it gets CFG - 1. Turbo runs without.
"""

import contextlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from diffusers import Krea2Pipeline
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor

from degas_worker.families.base import Output, RunContext
from degas_worker.families.krea2_edit import (
    EditKind,
    fit,
    grounded_prompt,
    grounding_size,
    place,
    placements,
    ref_boosts,
)
from degas_worker.families.lora import diffs_as_loras, fold_alphas
from degas_worker.families.offload import place as place_pipe
from degas_worker.families.offload import reoffload
from degas_worker.families.runtime import (
    LoraFile,
    fetch_loras,
    free_gpu_memory,
    png_output,
    seeded,
    step_callback,
    sync_loras,
)
from degas_worker.spec import Spec


class Krea2Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.processor: Any = None
        self.model_path: Path | None = None
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode")
        if mode not in ("t2i", "edit"):
            raise ValueError(f"Krea 2 can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras: list[LoraFile] = fetch_loras(spec, ctx)
        edit_lora = spec.get("edit_lora")
        if edit_lora:
            file = ctx.fetch_asset(edit_lora["path"], edit_lora.get("size"))
            loras.append((edit_lora["path"], file, float(edit_lora["weight"])))
        inputs = spec.get("inputs") or {}
        images: list[Image.Image] = []
        if mode == "edit":
            images = [_open(ctx.blob(ref)) for ref in [inputs["source"], *inputs.get("refs", [])]]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        sync_loras(self.pipe, self.adapters, loras, self._load_lora)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        steps = int(params["steps"])
        width, height = int(params["width"]), int(params["height"])
        # Turbo has no CFG field; it runs guidance-free.
        guidance = float(params.get("cfg", 1.0)) - 1.0
        kwargs: dict[str, Any] = {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        }
        refs: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
        try:
            if images:
                kind: EditKind = params.get("edit_lora", "general")
                kwargs.update(self._grounded(params["prompt"], images, guidance > 0))
                refs = self._refs(images, kind, width, height)
            else:
                kwargs["prompt"] = params["prompt"]
                kwargs["negative_prompt"] = params.get("negative_prompt") or ""
            with refs:
                for item, seed in enumerate(seeds):
                    ctx.check_cancelled()
                    ctx.progress(item, "denoise", 0, steps)
                    result = self.pipe(
                        **kwargs,
                        generator=seeded(seed),
                        callback_on_step_end=step_callback(ctx, item, steps),
                    )
                    ctx.progress(item, "encode", steps, steps)
                    yield png_output(item, seed, result.images[0])
        except Exception:
            # An out of memory would leave the model that was running on the GPU.
            reoffload(self.pipe)
            raise

    def _grounded(self, prompt: str, images: list[Image.Image], negative: bool) -> dict[str, Any]:
        """The prompt (and with CFG the empty negative) encoded with the images, padded to one
        length: the pipeline builds one set of position ids for both."""
        seen = [im.resize(grounding_size(*im.size), Image.Resampling.BOX) for im in images]
        embeds = [self._encode(prompt, seen)]
        if negative:
            embeds.append(self._encode("", seen))
        length = max(e.shape[1] for e in embeds)
        names = ("prompt_embeds", "negative_prompt_embeds")
        out: dict[str, Any] = {}
        for name, e in zip(names, embeds, strict=False):
            mask = torch.zeros(1, length, dtype=torch.bool, device=e.device)
            mask[:, : e.shape[1]] = True
            pad = e.new_zeros(1, length - e.shape[1], *e.shape[2:])
            out[name] = torch.cat([e, pad], dim=1)
            out[f"{name}_mask"] = mask
        return out

    def _encode(self, prompt: str, images: list[Image.Image]) -> torch.Tensor:
        """`get_text_hidden_states`, with the images in the user turn and no padding."""
        pipe = self.pipe
        device = pipe._execution_device
        batch = self.processor(
            text=[grounded_prompt(prompt, len(images))], images=images, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = pipe.text_encoder(**batch, output_hidden_states=True)
        states = torch.stack([out.hidden_states[i] for i in pipe.text_encoder_select_layers], 2)
        return states[:, pipe.prompt_template_encode_start_idx :].to(pipe.transformer.dtype)

    def _refs(
        self, images: list[Image.Image], kind: EditKind, width: int, height: int
    ) -> contextlib.AbstractContextManager[None]:
        """Encode the images as clean tokens, placed on the target's grid."""
        pipe = self.pipe
        fits = [fit(*im.size, width, height) for im in images]
        bicubic = Image.Resampling.BICUBIC
        tokens = [
            self._vae_tokens(im.crop(f.crop).resize(f.size, bicubic))
            for im, f in zip(images, fits, strict=True)
        ]
        target = (height // 16, width // 16)
        offsets = place([f.grid for f in fits], target, placements(kind, len(images)))
        ids = [
            _grid_ids(frame, f.grid, offset, tokens[0].device)
            for frame, (f, offset) in enumerate(zip(fits, offsets, strict=True), 1)
        ]
        return prepend_refs(pipe.transformer, tokens, ids, ref_boosts(kind, len(images)))

    def _vae_tokens(self, image: Image.Image) -> torch.Tensor:
        """An image as packed, normalized latents: the pipeline's decode, undone."""
        pipe = self.pipe
        vae = pipe.vae
        device = pipe._execution_device
        pixels = pipe.image_processor.preprocess(image, height=image.height, width=image.width)
        pixels = pixels.to(device, vae.dtype)[:, :, None]  # one frame
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.mode()
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents)
        latents = ((latents - mean) / std)[:, :, 0]
        _, channels, h, w = latents.shape
        packed = pipe._pack_latents(latents, 1, channels, h, w)
        return packed.to(pipe.transformer.dtype)

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        pipe = Krea2Pipeline.from_pretrained(
            str(path), torch_dtype=torch.bfloat16, local_files_only=True
        )
        # About 36 GB: an 80 GB A100 or an H100 keeps it resident, a 40 GB A100 offloads.
        place_pipe(pipe)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        # The repos ship only a tokenizer; `processor/` adds Qwen3-VL's image processor.
        self.processor = AutoProcessor.from_pretrained(path / "processor", local_files_only=True)
        self.model_path = path
        self.adapters = {}

    def _load_lora(self, file: Path, name: str) -> None:
        # diffusers' Krea 2 converter takes only lora_A/lora_B. Files in Krea's own names
        # (conradlocke's edit LoRAs) add an alpha per module, and some (Krea2FilterBypass)
        # are a weight difference for the text fusion's layer mix.
        state = diffs_as_loras(load_file(file), lambda n, like: torch.eye(n, dtype=like.dtype))
        self.pipe.load_lora_weights(fold_alphas(state), adapter_name=name)

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.processor = None
        self.model_path = None
        self.adapters = {}
        free_gpu_memory()


@contextlib.contextmanager
def prepend_refs(
    transformer: Any, tokens: list[torch.Tensor], ids: list[torch.Tensor], boosts: list[float]
) -> Iterator[None]:
    """While the pipeline runs, put the images' tokens before the target's on every call.

    `forward` is replaced on the instance, calling through to the one there (accelerate's
    offload hook, when there is one), as `limit_to_areas` does for SDXL's ControlNets. The
    sequence becomes `[text | images | target]`, with the images' rows in the position ids
    between the text's and the target's, and only the target's tokens come back. Each block's
    attention mask gets the images' boosts, as a log added to the target rows' logits for that
    image's keys.
    """
    forward = transformer.__dict__.get("forward")
    inner = transformer.forward
    refs = torch.cat(tokens, dim=1)
    ref_ids = torch.cat(ids, dim=0)
    spans = [t.shape[1] for t in tokens]
    # Per call: the text's length, and the mask built from it for every block.
    call: dict[str, Any] = {}

    def with_refs(*args: Any, **kwargs: Any) -> Any:
        hidden = kwargs.pop("hidden_states")
        positions = kwargs.pop("position_ids")
        text = kwargs["encoder_hidden_states"].shape[1]
        n = hidden.shape[1]
        kwargs["hidden_states"] = torch.cat(
            [refs.to(hidden).expand(hidden.shape[0], -1, -1), hidden], dim=1
        )
        kwargs["position_ids"] = torch.cat(
            [positions[:text], ref_ids.to(positions), positions[text:]], dim=0
        )
        call.clear()
        call["text"] = text
        out = inner(*args, **kwargs)
        if isinstance(out, tuple):
            return (out[0][:, -n:], *out[1:])
        out.sample = out.sample[:, -n:]
        return out

    def boosted(_block: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        # Blocks are called as (hidden_states, temb, rotary, attention_mask).
        if len(args) < 4 or args[3] is None:
            return None
        if "mask" not in call:
            keep = args[3]  # (B, 1, 1, L) bool: False for the text's padding
            total = keep.shape[-1]
            bias = torch.zeros(1, 1, total, total, device=keep.device, dtype=args[0].dtype)
            start = call["text"]
            target = start + sum(spans)
            for span, boost in zip(spans, boosts, strict=True):
                if boost != 1.0:
                    bias[:, :, target:, start : start + span] = math.log(boost)
                start += span
            blocked = torch.finfo(bias.dtype).min
            call["mask"] = bias + torch.where(keep, 0.0, blocked).to(bias.dtype)
        return (*args[:3], call["mask"], *args[4:])

    hooks = []
    if any(b != 1.0 for b in boosts):
        hooks = [b.register_forward_pre_hook(boosted) for b in transformer.transformer_blocks]
    transformer.forward = with_refs
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()
        if forward is None:
            del transformer.forward
        else:
            transformer.forward = forward


def _grid_ids(
    frame: int, grid: tuple[int, int], offset: tuple[float, float], device: torch.device
) -> torch.Tensor:
    """An image's (frame, row, column) RoPE coordinates, one row per token."""
    rows, cols = grid
    top, left = offset
    ids = torch.zeros(rows, cols, 3, device=device)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(rows, device=device)[:, None] + top
    ids[..., 2] = torch.arange(cols, device=device)[None, :] + left
    return ids.reshape(rows * cols, 3)


def _open(path: Path) -> Image.Image:
    """An edit image, flattened to RGB (the VAE and the text encoder don't take alpha)."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel("A"))
            return flat
        return im.convert("RGB")
