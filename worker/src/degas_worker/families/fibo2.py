"""Fibo 2.0 runner: text-to-image, from an image, and editing from up to 5 images
(design §4.3).

One pipeline does all three: with images it edits them. The model reads structured JSON
captions, so the prompt first goes through one of Bria's prompt-to-JSON VLMs (Qwen3-VL 4B
fine-tunes): FIBO-vlm for `t2i`, and for `i2i` (Bria's Inspire), where it sees the source
and the pipeline doesn't; FIBO-edit-vlm for an edit of one image, which sees the image, and
for an `i2i` with a prompt, whose caption of the edit's result is drawn as a new image. Its
JSON is written once per job, sampled with the job's first seed, and returned with each
image. The `ausboss` style writes it with a plain Qwen3-VL 8B and AusBoss's prompts
(`fibo2_prompt_ausboss.py`) instead: Qwen-Image 2.1's text encoder, or Qwen3-VL 8B Heretic.

A job may name another text encoder (Qwen3-VL 4B Heretic), which takes the place of the
pipeline's own without reloading the rest.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from accelerate import cpu_offload_with_hook
from accelerate.hooks import remove_hook_from_module
from diffusers import BriaFibo2Pipeline
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from degas_worker.families import fibo2_prompt_ausboss as ausboss
from degas_worker.families.base import Output, RunContext
from degas_worker.families.fibo2_prompt import (
    NEGATIVE_T2I,
    edit_messages,
    inspire_messages,
    inspired_caption,
    parse_caption,
    t2i_messages,
)
from degas_worker.families.flux3 import _shrink
from degas_worker.families.klein import _open
from degas_worker.families.offload import OFFLOAD_ABOVE, module_bytes
from degas_worker.families.runtime import free_gpu_memory, png_output, seeded, step_callback
from degas_worker.spec import Spec

log = logging.getLogger(__name__)

# The order the models run in, for offloading one at a time.
MODELS = ("text_encoder", "transformer", "vae")
# The fonts the pipeline draws the caption's texts with, in the order training used them
# (the model folder's `fonts/`).
FONTS = ("Aileron-Regular.otf", "DejaVuSans.ttf")
# Bria's settings for each VLM: (image pixels it sees, at least and at most; temperature).
T2I_VLM = (256 * 28 * 28, 1024 * 28 * 28, 0.2)
EDIT_VLM = (4 * 28 * 28, 1280 * 28 * 28, 0.4)
VLM_TOP_P = 0.9
# Bria allows 4096, but a caption takes about 500-1000 tokens. FIBO-edit-vlm can fall into a
# loop (listing the books on a shelf, over and over), and at the worker's rate of about
# 8 tokens a second under offload, 4096 tokens took 9 minutes.
VLM_MAX_TOKENS = 2048


class Fibo2Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.fonts: list[str] = []
        # The text encoder the job named in place of the pipeline's own, if any.
        self.te_path: Path | None = None
        self.vlm: Any = None
        self.processor: Any = None
        self.vlm_path: Path | None = None
        # Whether the models don't all fit on the GPU, and so are chained through offload hooks.
        self.offloaded = False
        self.hooks: list[Any] = []

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode")
        if mode not in ("t2i", "i2i", "edit"):
            raise ValueError(f"Fibo 2.0 can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        upsampler = spec.get("upsampler")
        vlm_path = ctx.fetch_asset(upsampler["path"], upsampler.get("size")) if upsampler else None
        encoder = spec.get("text_encoder")
        te_path = ctx.fetch_asset(encoder["path"], encoder.get("size")) if encoder else None
        inputs = spec.get("inputs") or {}
        images: list[Image.Image] = []
        source: Image.Image | None = None
        if mode == "edit":
            refs = [inputs["source"], *inputs.get("refs", [])]
            images = [_shrink(_open(ctx.blob(ref))) for ref in refs]
            source = images[0]
        elif mode == "i2i":
            # Only the VLM sees it.
            source = _open(ctx.blob(inputs["source"]))
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path, vlm_path, te_path)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        prompt = params["prompt"]
        caption: str | None = None
        if vlm_path is not None:
            ctx.progress(0, "upsample", 0, 1)
            caption = self._caption(prompt, mode, source, seeds[0])
            if caption is None and mode == "i2i":
                # The prompt alone (maybe empty) would leave out the picture.
                raise ValueError(f"{vlm_path.name} wrote no caption of the source image")
            ctx.check_cancelled()
        steps = int(params["steps"])
        cfg = float(params["cfg"])
        kwargs: dict[str, Any] = {
            "prompt": caption or prompt,
            "width": int(params["width"]),
            "height": int(params["height"]),
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "fonts": self.fonts or None,
        }
        if images:
            kwargs["image"] = images
        elif cfg > 1:
            kwargs["negative_prompt"] = NEGATIVE_T2I
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)
            image = self.pipe(
                **kwargs,
                generator=seeded(seed),
                callback_on_step_end=step_callback(ctx, item, steps),
            ).images[0]
            ctx.progress(item, "encode", steps, steps)
            yield png_output(item, seed, image, caption)

    def _caption(self, prompt: str, mode: str, image: Image.Image | None, seed: int) -> str | None:
        """The VLM's JSON caption for this request: of a new image (inspired by `image` for
        `i2i`), or of an edit of `image`. None if its answer isn't one (it ran out of tokens in
        a loop): the prompt is then sent as written, which Fibo 2.0 was also trained on."""
        if self.offloaded:
            # Make room: the pipeline's models go back to the CPU. The VLM goes back when the
            # text encoder, next in the hook chain, runs.
            for hook in self.hooks:
                hook.offload()
            self.vlm.to("cuda")
        if image is not None:
            image = image.copy()
            image.thumbnail((1024, 1024))
        if self._is_writer():
            # AusBoss's style: one prompt for a new image and one for an edit.
            instruction = prompt if mode == "edit" else None
            return self._ask(
                ausboss.messages(prompt, image), image, ausboss.TEMPERATURE, seed, instruction
            )
        if mode == "edit":
            assert image is not None
            return self._ask(edit_messages(image, prompt), image, EDIT_VLM[2], seed, prompt)
        if image is None:
            return self._ask(t2i_messages(prompt), None, T2I_VLM[2], seed)
        if not prompt.strip():
            return self._ask(inspire_messages(image), image, T2I_VLM[2], seed)
        # A prompted Inspire: FIBO-edit-vlm's caption of the edit's result, which the pipeline
        # draws without the source.
        return self._ask(edit_messages(image, prompt), image, EDIT_VLM[2], seed, inspired=True)

    def _ask(
        self,
        messages: list[dict[str, Any]],
        image: Image.Image | None,
        temperature: float,
        seed: int,
        instruction: str | None = None,
        *,
        inspired: bool = False,
    ) -> str | None:
        """The caption in the VLM's answer to `messages`, or None (with a warning) if there's
        none. `instruction` is an edit's, as `parse_caption` takes it; `inspired` takes an
        edit's result caption without its instruction."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(
            text=[text],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)
        tokenizer = self.processor.tokenizer
        writer = self._is_writer()
        # AusBoss samples with temperature alone, over the generation config's top-k, top-p
        # and repetition penalty; Bria adds top-p.
        sampling = (
            {"top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}
            if writer
            else {"top_p": VLM_TOP_P}
        )
        torch.manual_seed(seed)
        with torch.inference_mode():
            ids = self.vlm.generate(
                **batch,
                max_new_tokens=VLM_MAX_TOKENS,
                do_sample=True,
                temperature=temperature,
                **sampling,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        answer: str = tokenizer.decode(
            ids[0, batch["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        if writer:
            answer = ausboss.repair(answer.strip())
        try:
            return inspired_caption(answer) if inspired else parse_caption(answer, instruction)
        except ValueError:
            log.warning("No caption from %s: %r…%r", self.vlm_path, answer[:200], answer[-200:])
            return None

    def _load(self, path: Path, vlm_path: Path | None, te_path: Path | None) -> None:
        if self.pipe is not None and self.model_path != path:
            self.unload()
        changed = False
        if self.pipe is None:
            # The job's text encoder from the start, so the pipeline's own isn't loaded first.
            extra = {"text_encoder": _encoder(te_path)} if te_path is not None else {}
            pipe = BriaFibo2Pipeline.from_pretrained(
                path, dtype=torch.bfloat16, local_files_only=True, **extra
            )
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe
            self.model_path = path
            self.te_path = te_path
            self.fonts = [str(path / "fonts" / f) for f in FONTS if (path / "fonts" / f).exists()]
            changed = True
        elif self.te_path != te_path:
            # Swap only the text encoder: the transformer stays loaded.
            self._unhook()
            self.pipe.text_encoder = _encoder(te_path or path / "text_encoder")
            free_gpu_memory()
            self.te_path = te_path
            changed = True
        if vlm_path is not None and self.vlm_path != vlm_path:
            # One VLM at a time: FIBO-vlm and FIBO-edit-vlm are the same size and load quickly.
            self._unhook()
            self.vlm = None
            free_gpu_memory()
            if (vlm_path / "model_index.json").exists():
                # A pipeline folder (Qwen-Image 2.1), whose text encoder writes the caption.
                self.vlm = _encoder(vlm_path / "text_encoder").eval()
                self.processor = AutoProcessor.from_pretrained(
                    vlm_path / "processor", local_files_only=True
                )
            elif vlm_path.name.startswith("FIBO-"):
                min_pixels, max_pixels, _ = T2I_VLM if vlm_path.name == "FIBO-vlm" else EDIT_VLM
                self.vlm = _encoder(vlm_path).eval()
                self.processor = AutoProcessor.from_pretrained(
                    vlm_path, min_pixels=min_pixels, max_pixels=max_pixels, local_files_only=True
                )
            else:
                self.vlm = _encoder(vlm_path).eval()
                self.processor = AutoProcessor.from_pretrained(vlm_path, local_files_only=True)
            self.vlm_path = vlm_path
            changed = True
        if changed:
            self._place()

    def _is_writer(self) -> bool:
        """Whether the loaded VLM is a plain Qwen3-VL that writes AusBoss's style, rather than
        one of Bria's."""
        return self.vlm_path is not None and not self.vlm_path.name.startswith("FIBO-")

    def _models(self) -> list[torch.nn.Module]:
        """The loaded models in the order they run: the VLM first, if loaded."""
        models = [getattr(self.pipe, name) for name in MODELS]
        return [self.vlm, *models] if self.vlm is not None else models

    def _place(self) -> None:
        """Put every model on the GPU if they fit (about 35 GB with a VLM), or else chain
        offload hooks through them, so each comes in when it runs and sends the one before it
        back to the CPU."""
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
        self.te_path = None
        self.fonts = []
        self.vlm = None
        self.processor = None
        self.vlm_path = None
        self.offloaded = False
        free_gpu_memory()


def _encoder(path: Path) -> Any:
    """A Qwen3-VL from its transformers folder, in bf16, on the CPU."""
    return Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True
    )
