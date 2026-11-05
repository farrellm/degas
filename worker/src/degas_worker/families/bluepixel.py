"""BluePixel-Turbo runner: instruction editing of one image at up to 4K (design §4.3).

vivo's model is FLUX.2 [klein] 9B fine-tuned onto a DC-AE f64 autoencoder (1x1 patches, 128
latent channels) and step-distilled to 4 steps with CFG 3. `BluePixelPipeline` ports the
`FluxKleinPipeline` from vivo's `vins_infer.py` (github.com/vivo/BluePixel, Apache-2.0) onto
diffusers' `Flux2KleinPipeline`, which already encodes the prompt as klein does (Qwen3-8B,
layers 9, 18 and 27): the source is the condition image, at the output size, and the
DC-AE's latents are scaled by its `scaling_factor` instead of klein's patchify and batch norm.
The VAE runs in float32, as vivo's does, and the rest in bf16.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderDC, DiffusionPipeline, Flux2KleinPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from degas_worker.families.base import Output, RunContext
from degas_worker.families.offload import place, reoffload
from degas_worker.families.runtime import (
    StepCallback,
    free_gpu_memory,
    png_output,
    seeded,
    step_callback,
)
from degas_worker.spec import Spec

SPATIAL = 64  # the DC-AE's downscale; the transformer's patches are 1x1
# A VAE pass that runs out of memory is redone in tiles. diffusers' DC-AE defaults (512 px
# tiles, 448 px apart) blend over one latent, which leaves a visible grid at f64; these blend
# over 8 (512 px).
TILE = 2048
TILE_STRIDE = 1536

log = logging.getLogger(__name__)


class BluePixelPipeline(Flux2KleinPipeline):  # type: ignore[misc]
    def __init__(
        self,
        scheduler: Any,
        vae: Any,
        text_encoder: Any,
        tokenizer: Any,
        transformer: Any,
        is_distilled: bool = False,
    ) -> None:
        # Flux2KleinPipeline's own __init__ reads the KL VAE's config.
        DiffusionPipeline.__init__(self)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        # False: CFG still applies. It must be a parameter with a default, or diffusers'
        # `components` takes the config key for a missing module.
        self.register_to_config(is_distilled=is_distilled)
        self.vae_scale_factor = SPATIAL
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=SPATIAL)
        self.tokenizer_max_length = 512
        self.default_sample_size = 64

    def _encode_vae_image(self, image: torch.Tensor, generator: Any) -> torch.Tensor:
        latents = self._vae(lambda: self.vae.encode(image).latent, image.shape[-2:])
        return latents * self.vae.config.scaling_factor

    def prepare_latents(
        self,
        batch_size: int,
        num_latents_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Any,
        generator: Any,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, num_latents_channels, height // SPATIAL, width // SPATIAL)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latent_ids = self._prepare_latent_ids(latents).to(device)
        return self._pack_latents(latents), latent_ids

    def _vae(self, call: Callable[[], torch.Tensor], size: Any) -> torch.Tensor:
        """Run a VAE pass whole, or in tiles if that runs out of memory."""
        self.vae.disable_tiling()
        try:
            return call()
        except torch.OutOfMemoryError:
            free_gpu_memory()
        log.info("VAE out of memory at %dx%d: tiling", int(size[1]), int(size[0]))
        self.vae.enable_tiling(
            tile_sample_min_height=TILE,
            tile_sample_min_width=TILE,
            tile_sample_stride_height=TILE_STRIDE,
            tile_sample_stride_width=TILE_STRIDE,
        )
        try:
            return call()
        finally:
            self.vae.disable_tiling()

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.0,
        generator: Any = None,
        callback_on_step_end: StepCallback | None = None,
        max_sequence_length: int = 512,
    ) -> Image.Image:
        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt, device=device, max_sequence_length=max_sequence_length
        )
        if self.do_classifier_free_guidance:
            negative_embeds, negative_text_ids = self.encode_prompt(
                prompt="", device=device, max_sequence_length=max_sequence_length
            )

        condition = self.image_processor.preprocess(image.convert("RGB"), height, width)
        image_latents, image_latent_ids = self.prepare_image_latents(
            images=[condition],
            batch_size=1,
            generator=generator,
            device=device,
            dtype=self.vae.dtype,
        )
        latents, latent_ids = self.prepare_latents(
            1,
            self.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(0)
        ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        for i, t in enumerate(timesteps):
            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
            model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
            noise_pred = self.transformer(
                hidden_states=model_input,
                timestep=timestep,
                guidance=None,  # BluePixel has no guidance embedding
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=ids,
                return_dict=False,
            )[0][:, : latents.size(1)]
            if self.do_classifier_free_guidance:
                with self.transformer.cache_context("uncond"):
                    negative_pred = self.transformer(
                        hidden_states=model_input,
                        timestep=timestep,
                        guidance=None,
                        encoder_hidden_states=negative_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=ids,
                        return_dict=False,
                    )[0][:, : latents.size(1)]
                noise_pred = negative_pred + guidance_scale * (noise_pred - negative_pred)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if callback_on_step_end is not None:
                callback_on_step_end(self, i, t, {"latents": latents})
        self._current_timestep = None

        latents = self._unpack_latents_with_ids(
            latents, latent_ids, height // SPATIAL, width // SPATIAL
        )
        latents = (latents / self.vae.config.scaling_factor).to(self.vae.dtype)
        decoded = self._vae(lambda: self.vae.decode(latents, return_dict=False)[0], (height, width))
        out: Image.Image = self.image_processor.postprocess(decoded, output_type="pil")[0]
        self.maybe_free_model_hooks()
        return out


class BluePixelRunner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        if spec.get("mode") != "edit":
            raise ValueError(f"BluePixel can't do {spec.get('mode')!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        source = _open(ctx.blob(spec["inputs"]["source"]))
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path)
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        steps = int(params["steps"])
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, steps)
            try:
                image = self.pipe(
                    image=source,
                    prompt=params["prompt"],
                    height=int(params["height"]),
                    width=int(params["width"]),
                    num_inference_steps=steps,
                    guidance_scale=float(params["cfg"]),
                    generator=seeded(seed),
                    callback_on_step_end=step_callback(ctx, item, steps),
                )
            except Exception:
                reoffload(self.pipe)
                raise
            ctx.progress(item, "encode", steps, steps)
            yield png_output(item, seed, image)

    def _load(self, path: Path) -> None:
        if self.pipe is not None and self.model_path == path:
            return
        self.unload()
        vae = AutoencoderDC.from_pretrained(
            str(path), subfolder="vae", torch_dtype=torch.float32, local_files_only=True
        )
        # The folder's model_index names vivo's class; load the parts into ours.
        pipe = BluePixelPipeline.from_pretrained(
            str(path), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True
        )
        # About 37 GB (the transformer is stored in float32 and cast): an 80 GB card keeps it
        # resident, a 40 GB A100 offloads.
        place(pipe)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        free_gpu_memory()


def _open(path: Path) -> Image.Image:
    """The source, flattened onto white (the pipeline doesn't take alpha)."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.getchannel("A"))
            return flat
        return im.convert("RGB")
