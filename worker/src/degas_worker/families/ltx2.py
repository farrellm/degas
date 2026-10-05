"""LTX-2 runner: text-, image- and first-and-last-frame-to-video, with sound (design §4.3).

Models are diffusers folders (LTX-2.3, LTX-2.3 Distilled, LTX-2.5) that load as `LTX2Pipeline`.
Image-to-video and first-and-last-frame use `LTX2ImageToVideoPipeline` and
`LTX2ConditionPipeline`, built from the same components. The distilled transformers run a
fixed 8-sigma schedule without guidance, and can upscale 2x: the video is made at half size,
`LTX2LatentUpsamplePipeline` doubles its latents (the model folder's `latent_upsampler/`), and
3 more sigmas refine them at full size.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import (
    LTX2ConditionPipeline,
    LTX2ImageToVideoPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2Pipeline,
)
from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from PIL import Image

from degas_worker.families.base import Output, RunContext
from degas_worker.families.offload import place
from degas_worker.families.runtime import (
    StepCallback,
    fetch_loras,
    free_gpu_memory,
    seeded,
    step_callback,
    sync_loras,
)
from degas_worker.spec import Spec
from degas_worker.video import Audio, encode_mp4

DISTILLED = frozenset({"ltx23-distilled", "ltx25"})
UPSAMPLER = "latent_upsampler"
# Distilled passes: no CFG, STG or modality guidance, so each step is one forward pass.
UNGUIDED: dict[str, float] = {
    "guidance_scale": 1.0,
    "audio_guidance_scale": 1.0,
    "stg_scale": 0.0,
    "audio_stg_scale": 0.0,
    "modality_scale": 1.0,
    "audio_modality_scale": 1.0,
}


class Ltx2Runner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name
        self.derived: dict[str, Any] = {}  # mode → its pipeline, sharing `pipe`'s components
        self.upsample: Any = None

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec["mode"]
        if mode not in ("t2v", "i2v", "flf2v"):
            raise ValueError(f"LTX-2 can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        loras = fetch_loras(spec, ctx)
        inputs = spec.get("inputs") or {}
        frames = [_open(ctx.blob(inputs[key])) for key in ("source", "end") if key in inputs]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        sync_loras(self.pipe, self.adapters, loras)
        params = spec["params"]
        distilled = spec["variant"] in DISTILLED
        upscale = distilled and bool(params.get("upscale"))
        if upscale:
            self._load_upsampler(path)
        pipe = self._pipe_for(mode)
        ctx.progress(0, "load", 1, 1)

        width, height = int(params["width"]), int(params["height"])
        num_frames = int(params["num_frames"])
        fps = float(params["fps"])
        kwargs: dict[str, Any] = {
            "prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt") or None,
            "num_frames": num_frames,
            "frame_rate": fps,
        }
        if distilled:
            kwargs.update(UNGUIDED, sigmas=DISTILLED_SIGMA_VALUES)
            steps = len(DISTILLED_SIGMA_VALUES)
        else:
            steps = int(params["steps"])
            stg = float(params["stg"])
            kwargs.update(
                num_inference_steps=steps,
                guidance_scale=float(params["cfg"]),
                audio_guidance_scale=float(params["audio_cfg"]),
                stg_scale=stg,
                audio_stg_scale=stg,
            )
        refine = len(STAGE_2_DISTILLED_SIGMA_VALUES) if upscale else 0
        total = steps + refine

        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, total)
            generator = seeded(seed)
            report = step_callback(ctx, item, total)
            if upscale:
                latents, audio_latents = pipe(
                    **kwargs,
                    **_conditions(mode, frames, width // 2, height // 2),
                    width=width // 2,
                    height=height // 2,
                    generator=generator,
                    callback_on_step_end=report,
                    output_type="latent",
                    return_dict=False,
                )
                ctx.check_cancelled()
                latents = self.upsample(
                    latents=latents,
                    width=width // 2,
                    height=height // 2,
                    num_frames=num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]
                refined = {**kwargs, "sigmas": STAGE_2_DISTILLED_SIGMA_VALUES}
                video, audio = pipe(
                    **refined,
                    **_conditions(mode, frames, width, height),
                    width=width,
                    height=height,
                    latents=latents,
                    audio_latents=audio_latents,
                    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                    generator=generator,
                    callback_on_step_end=_offset(report, steps),
                    output_type="np",
                    return_dict=False,
                )
            else:
                video, audio = pipe(
                    **kwargs,
                    **_conditions(mode, frames, width, height),
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=report,
                    output_type="np",
                    return_dict=False,
                )
            ctx.progress(item, "encode", total, total)
            pixels = (np.clip(np.asarray(video[0]), 0, 1) * 255).round().astype(np.uint8)
            data = encode_mp4(
                (frame.tobytes() for frame in pixels),
                pixels.shape[2],
                pixels.shape[1],
                fps,
                audio=self._audio(audio[0]),
            )
            yield Output(item=item, seed=seed, data=data, media_type="video/mp4", ext="mp4")

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        pipe = LTX2Pipeline.from_pretrained(
            str(path), torch_dtype=torch.bfloat16, local_files_only=True
        )
        # A 22B transformer and a 12B Gemma: offloaded on an A100, resident on an 80 GB card.
        place(pipe)
        # Video decodes otherwise peak well above the denoiser. The audio VAE is small and
        # can't tile (diffusers raises NotImplementedError).
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.adapters = {}

    def _pipe_for(self, mode: str) -> Any:
        if mode == "t2v":
            return self.pipe
        if mode not in self.derived:
            # Not from_pipe: it casts the shared modules in place to one dtype (float32 by
            # default), doubling the bf16 transformer.
            cls = LTX2ImageToVideoPipeline if mode == "i2v" else LTX2ConditionPipeline
            pipe = cls(**self.pipe.components)
            pipe.set_progress_bar_config(disable=True)
            self.derived[mode] = pipe
        return self.derived[mode]

    def _load_upsampler(self, path: Path) -> None:
        if self.upsample is not None:
            return
        folder = path / UPSAMPLER
        if not folder.is_dir():
            raise ValueError(
                f"Upscale needs the model's {UPSAMPLER}/ folder; copy it into {path.name} in Drive"
            )
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            str(folder), torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        # The VAE is shared and placed with the pipeline; only its latent statistics are read.
        self.upsample = LTX2LatentUpsamplePipeline(vae=self.pipe.vae, latent_upsampler=upsampler)
        self.upsample.set_progress_bar_config(disable=True)

    def _audio(self, wave: Any) -> Audio:
        """The vocoder's waveform, `(channels, samples)`, as interleaved float32."""
        samples = wave.float().cpu().numpy()
        if samples.ndim == 1:
            samples = samples[None]
        rate = int(self.pipe.vocoder.config.output_sampling_rate)
        pcm = np.ascontiguousarray(samples.T, dtype="<f4").tobytes()
        return Audio(pcm=pcm, rate=rate, channels=samples.shape[0])

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.adapters = {}
        self.derived = {}
        self.upsample = None
        free_gpu_memory()


def _conditions(mode: str, frames: list[Image.Image], width: int, height: int) -> dict[str, Any]:
    """The pipeline arguments for the mode's images, sized for a pass at `width` x `height`."""
    sized = [im.resize((width, height), Image.Resampling.LANCZOS) for im in frames]
    if mode == "i2v":
        return {"image": sized[0]}
    if mode == "flf2v":
        first, last = sized
        return {
            "conditions": [
                LTX2VideoCondition(frames=first, index=0, strength=1.0),
                LTX2VideoCondition(frames=last, index=-1, strength=1.0),
            ]
        }
    return {}


def _offset(report: StepCallback, done: int) -> StepCallback:
    """`report`, counting on from `done` steps (the refine after an upscale)."""
    return lambda pipe, i, t, tensors: report(pipe, done + i, t, tensors)


def _open(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
