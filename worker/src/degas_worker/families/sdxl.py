"""Stable Diffusion XL runner: text-to-image, image-to-image, inpaint and outpaint, with LoRAs
and ControlNets.

A regular checkpoint is loaded as the text-to-image pipeline, and the image-to-image and
inpaint pipelines are made from it with `from_pipe` (they share its weights and LoRAs). An
inpainting checkpoint (the `inpaint` variant, a 9-channel UNet) only has the inpaint pipeline.
With ControlNet units, the ControlNet version of the mode's pipeline is made the same way.

Regional ControlNet: a unit with an area mask has its ControlNet's residuals multiplied by
the mask, downsampled to each residual's size, so it only guides that area.

Image prompts (IP-Adapter): one adapter per unit, loaded into the shared UNet with the CLIP
image encoder they read pictures with, and unloaded when a job has none. Each unit's scale
names the blocks its purpose uses and is set again between steps for its step range.
"""

import contextlib
import gc
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models.embeddings import IPAdapterFaceIDPlusImageProjection
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from degas_worker import masks
from degas_worker.families import ip_adapter
from degas_worker.families.base import Output, RunContext
from degas_worker.families.control import control_kwargs, crop_areas
from degas_worker.families.lora import plan_loras, single_loras, strip_text_model
from degas_worker.spec import ControlUnit, ImagePromptUnit, Spec

# Keep in sync with the server descriptor (degas/families/sdxl.py).
SCHEDULERS: dict[str, tuple[Any, dict[str, Any]]] = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpmpp_2m": (DPMSolverMultistepScheduler, {}),
    "dpmpp_2m_sde": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "dpmpp_3m_sde": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
    ),
    "ddim": (DDIMScheduler, {}),
    "unipc": (UniPCMultistepScheduler, {}),
}

# Noise schedules, for the scheduler classes that take them.
SCHEDULES: dict[str, dict[str, Any]] = {
    "default": {},
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
}
_SCHEDULED = (EulerDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler)

# Highest denoise strength given to the inpainting checkpoint (see `_inputs`).
INPAINT_MAX_STRENGTH = 0.99

# Below this much total VRAM, SDXL fp16 is run with model CPU offload.
_OFFLOAD_BELOW_BYTES = 12 * 1024**3


class SdxlRunner:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.model_path: Path | None = None
        self.vae_path: Path | None = None  # None: the checkpoint's own VAE
        self.inpaint_model = False
        self.offload = False
        self.derived: dict[str, Any] = {}  # mode → pipeline made from `pipe` with from_pipe
        self.adapters: dict[str, str] = {}  # LoRA asset path → loaded adapter name
        self.controlnets: dict[str, Any] = {}  # ControlNet asset path → loaded model
        self.ip_adapters: tuple[str, ...] = ()  # IP-Adapter asset paths, one per unit
        self.encoder_path: Path | None = None  # their image encoder
        self.faces: Any = None  # InsightFace, for FaceID units
        self._scheduler_config: Any = None

    def run(self, spec: Spec, seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode", "t2i")
        if mode not in ("t2i", "i2i", "inpaint", "outpaint"):
            raise ValueError(f"SDXL can't do {mode!r}")
        pipe = self._load_for(mode, spec, ctx)
        params = spec["params"]
        self._set_scheduler(
            pipe, params.get("scheduler", "dpmpp_2m"), params.get("schedule", "karras")
        )
        steps = int(params["steps"])
        size = (int(params["width"]), int(params["height"]))
        kwargs, original, keep = self._inputs(mode, spec, size, ctx)
        regional = self._guidance(pipe, mode, spec, kwargs, size, ctx)
        prompts = spec.get("image_prompts") or []
        # Starting from an image skips the first (1 - strength) of the schedule.
        strength = float(kwargs.get("strength", 1.0))
        runs = max(1, min(steps, int(steps * strength)))
        clip_skip = int(params.get("clip_skip") or 0)
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, runs)

            if prompts:
                pipe.set_ip_adapter_scale(ip_adapter.scales(prompts, 0, runs))

            def on_step(
                _pipe: Any, i: int, _t: Any, kw: dict[str, Any], item: int = item
            ) -> dict[str, Any]:
                ctx.check_cancelled()
                done = min(i + 1, runs)
                ctx.progress(item, "denoise" if done < runs else "decode", done, runs)
                # Image prompts that start or end at the next step.
                if prompts and (scales := ip_adapter.changed(prompts, done, runs)) is not None:
                    _pipe.set_ip_adapter_scale(scales)
                return kw

            with limit_to_areas(regional):
                result = pipe(
                    prompt=params["prompt"],
                    negative_prompt=params.get("negative_prompt") or None,
                    num_inference_steps=steps,
                    guidance_scale=float(params["cfg"]),
                    clip_skip=clip_skip or None,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    callback_on_step_end=on_step,
                    **kwargs,
                )
            ctx.progress(item, "encode", runs, runs)
            image = result.images[0]
            if original is not None and keep is not None:
                image = masks.composite(original, image, keep)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            yield Output(
                item=item, seed=seed, data=buf.getvalue(), media_type="image/png", ext="png"
            )

    def _load_for(self, mode: str, spec: Spec, ctx: RunContext) -> Any:
        """Copy what the spec needs into the cache, load it, and return the mode's pipeline."""
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        config = ctx.fetch_asset(spec["config"]["path"], spec["config"].get("size"))
        vae = spec.get("vae")
        vae_path = ctx.fetch_asset(vae["path"], vae.get("size")) if vae else None
        loras = [
            (lora["path"], ctx.fetch_asset(lora["path"], lora.get("size")), float(lora["weight"]))
            for lora in single_loras(spec)
        ]
        nets = [
            (net["path"], ctx.fetch_asset(net["path"], net.get("size")))
            for net in (u["controlnet"] for u in spec.get("control") or [])
        ]
        prompts = spec.get("image_prompts") or []
        adapters = [
            (a["path"], ctx.fetch_asset(a["path"], a.get("size")))
            for a in (u["adapter"] for u in prompts)
        ]
        encoder = spec.get("image_encoder") if prompts else None
        encoder_path = ctx.fetch_asset(encoder["path"], encoder.get("size")) if encoder else None
        detector = spec.get("face_detector") if prompts else None
        detector_path = (
            ctx.fetch_asset(detector["path"], detector.get("size")) if detector else None
        )
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path, config, vae_path, inpaint=spec.get("variant") == "inpaint")
        self._apply_loras(loras)
        self._load_image_prompts(adapters, encoder_path)
        # Loading a FaceID model activates its own LoRA alone, so the weights go on last.
        self._activate_loras(loras, ip_adapter.face_loras(prompts))
        self._load_faces(detector_path)
        self._load_controlnets(nets)
        pipe = self._pipe_for(mode, [p for p, _ in nets])
        ctx.progress(0, "load", 1, 1)
        return pipe

    def _guidance(
        self,
        pipe: Any,
        mode: str,
        spec: Spec,
        kwargs: dict[str, Any],
        size: tuple[int, int],
        ctx: RunContext,
    ) -> list[tuple[Any, Image.Image]]:
        """Add the ControlNet units' and image prompts' arguments to `kwargs`; returns the
        ControlNets limited to areas, for `limit_to_areas`."""
        # *Around the mask* redraws a crop, so areas get the same crop.
        box = (
            pipe.mask_processor.get_crop_region(
                kwargs["mask_image"], *size, pad=kwargs["padding_mask_crop"]
            )
            if "padding_mask_crop" in kwargs
            else None
        )
        units = spec.get("control") or []
        areas: list[Image.Image | None] = []
        if units:
            images, areas = self._control_inputs(units, ctx)
            if box:
                areas = crop_areas(areas, box, size)
            kwargs.update(control_kwargs(units, images, mode))
        prompts = spec.get("image_prompts") or []
        if prompts:
            cfg = float(spec["params"]["cfg"]) > 1  # diffusers only guides above 1
            kwargs["ip_adapter_image_embeds"] = self._image_prompt_embeds(pipe, prompts, ctx, cfg)
            ip_masks = self._image_prompt_areas(prompts, ctx, size, box)
            if ip_masks:
                kwargs["cross_attention_kwargs"] = {"ip_adapter_masks": ip_masks}
        return [
            (self.controlnets[u["controlnet"]["path"]], area)
            for u, area in zip(units, areas, strict=False)
            if area
        ]

    def _inputs(
        self, mode: str, spec: Spec, size: tuple[int, int], ctx: RunContext
    ) -> tuple[dict[str, Any], Image.Image | None, Image.Image | None]:
        """Pipeline arguments for the mode, plus the image and mask to composite back onto."""
        params = spec["params"]
        width, height = size
        if mode == "t2i":
            return {"width": width, "height": height}, None, None
        inputs = spec["inputs"]
        with Image.open(ctx.blob(inputs["source"])) as im:
            source = im.convert("RGB")
        if mode == "i2i":
            return {"image": source, "strength": float(params["strength"])}, None, None
        blur = int(params.get("mask_blur") or 0)
        if mode == "inpaint":
            with Image.open(ctx.blob(inputs["mask"])) as im:
                mask = im.convert("L")
            image, strength = source, float(params["strength"])
        else:
            image, mask = masks.outpaint_canvas(
                source, inputs["place"], size, int(params.get("blend") or 0)
            )
            strength = 1.0
        if spec.get("variant") == "inpaint":
            # The SDXL inpainting UNet misbehaves at a full-strength schedule; 0.99 is its
            # usual workaround.
            strength = min(strength, INPAINT_MAX_STRENGTH)
        soft = masks.blur(mask, blur)
        kwargs: dict[str, Any] = {
            "image": image,
            "mask_image": soft,
            "strength": strength,
            "width": width,
            "height": height,
        }
        if mode == "inpaint" and params.get("inpaint_area") == "masked":
            kwargs["padding_mask_crop"] = int(params.get("mask_padding") or 0)
        return kwargs, image, soft

    @staticmethod
    def _control_inputs(
        units: list[ControlUnit], ctx: RunContext
    ) -> tuple[list[Image.Image], list[Image.Image | None]]:
        """Each unit's control image, and its area mask if it has one (both at output size)."""
        images: list[Image.Image] = []
        areas: list[Image.Image | None] = []
        for unit in units:
            with Image.open(ctx.blob(unit["image"])) as im:
                images.append(im.convert("RGB"))
            if unit.get("mask"):
                with Image.open(ctx.blob(unit["mask"])) as im:
                    areas.append(im.convert("L"))
            else:
                areas.append(None)
        return images, areas

    def _image_prompt_embeds(
        self, pipe: Any, prompts: list[ImagePromptUnit], ctx: RunContext, cfg: bool
    ) -> list[Any]:
        """Each unit's pictures encoded once for the whole batch (diffusers would encode them
        again for every image).

        A FaceID unit reads each picture's main face: its InsightFace identity goes in as the
        unit's embeddings, and CLIP reads the aligned face crop, which FaceID Plus takes as
        its projection's `clip_embeds`.
        """
        pictures: list[list[Image.Image]] = []
        identities: dict[int, Any] = {}
        for n, unit in enumerate(prompts):
            unit_pictures = []
            for ref in unit["images"]:
                with Image.open(ctx.blob(ref)) as im:
                    unit_pictures.append(im.convert("RGB"))
            if ip_adapter.is_faceid(unit["adapter"]["path"]):
                identities[n], unit_pictures = self._faces_of(unit_pictures, n + 1)
            pictures.append(unit_pictures)
        device = pipe._execution_device
        with torch.no_grad():
            embeds = list(pipe.prepare_ip_adapter_image_embeds(pictures, None, device, 1, cfg))
        layers = pipe.unet.encoder_hid_proj.image_projection_layers
        for n, ids in identities.items():
            unit = prompts[n]
            layer = layers[n]
            if isinstance(layer, IPAdapterFaceIDPlusImageProjection):
                layer.clip_embeds = embeds[n].to(device, torch.float16)
                layer.shortcut = ip_adapter.has_shortcut(unit["adapter"]["path"])
                layer.shortcut_scale = float(unit.get("structure", 1.0))
            found = torch.from_numpy(ids)[None].to(device, torch.float16)  # 1, pictures, 512
            embeds[n] = torch.cat([torch.zeros_like(found), found]) if cfg else found
        return embeds

    def _faces_of(self, pictures: list[Image.Image], unit: int) -> tuple[Any, list[Image.Image]]:
        """Each picture's main face: its identities (pictures by 512) and aligned crops."""
        from degas_worker.preprocess.face import align, main_face  # noqa: PLC0415 - cv2

        if self.faces is None:
            raise ValueError("FaceID needs InsightFace in preprocessors/insightface/")
        ids, crops = [], []
        for k, picture in enumerate(pictures, 1):
            rgb = np.asarray(picture)
            face = main_face(self.faces, rgb)
            if face is None:
                raise ValueError(f"Image prompt {unit}: no face found in picture {k}")
            ids.append(self.faces.identity(rgb, face))
            crops.append(Image.fromarray(align(rgb, face, 224)))
        return np.stack(ids).astype(np.float32), crops

    def _load_faces(self, detector: Path | None) -> None:
        """Keep InsightFace loaded while FaceID units want it."""
        if detector is None:
            self.faces = None
            return
        if self.faces is None or self.faces.model_dir != detector:
            from degas_worker.preprocess.face import FaceAnalyzer  # noqa: PLC0415 - cv2

            self.faces = FaceAnalyzer(detector)

    @staticmethod
    def _image_prompt_areas(
        prompts: list[ImagePromptUnit],
        ctx: RunContext,
        size: tuple[int, int],
        box: tuple[int, int, int, int] | None,
    ) -> list[Any] | None:
        """Each unit's area as diffusers takes it (one mask per picture), None for a unit
        without one; None when no unit has an area."""
        areas: list[Image.Image | None] = []
        for unit in prompts:
            if unit.get("mask"):
                with Image.open(ctx.blob(unit["mask"])) as im:
                    areas.append(im.convert("L"))
            else:
                areas.append(None)
        if not any(areas):
            return None
        if box:
            areas = crop_areas(areas, box, size)
        processor = IPAdapterMaskProcessor()
        out: list[Any] = []
        for unit, area in zip(prompts, areas, strict=True):
            if area is None:
                out.append(None)
                continue
            mask = processor.preprocess([area], height=size[1], width=size[0])
            out.append(mask.reshape(1, 1, *mask.shape[-2:]).repeat(1, len(unit["images"]), 1, 1))
        return out

    def _load_image_prompts(self, adapters: list[tuple[str, Path]], encoder: Path | None) -> None:
        """Load these IP-Adapters (one per unit) and their image encoder, or unload them all.

        The UNet refuses to run with IP-Adapter layers and no pictures, so a job without image
        prompts unloads them. Pipelines made with `from_pipe` copied the old encoder (or none),
        so they are made again.
        """
        wanted = tuple(path for path, _ in adapters)
        if wanted == self.ip_adapters and encoder == self.encoder_path:
            return
        if self.ip_adapters:
            # FaceID models brought their own LoRAs, which outlive `unload_ip_adapter`.
            faceid = [
                n for n in getattr(self.pipe.unet, "peft_config", {}) if n.startswith("faceid_")
            ]
            if faceid:
                self.pipe.delete_adapters(faceid)
            self.pipe.unload_ip_adapter()
            self.ip_adapters = ()
            self.encoder_path = None
            gc.collect()
            torch.cuda.empty_cache()
        self.derived = {}
        if not wanted:
            if self.offload:
                # Drop the old encoder's offload hook along with it.
                self.pipe.enable_model_cpu_offload()
            return
        if encoder is None:
            raise ValueError("Image prompts need their image encoder")
        try:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                str(encoder), dtype=torch.float16, local_files_only=True
            )
        except Exception as e:
            raise ValueError(f"Could not load the image encoder {encoder.name}: {e}") from e
        if not self.offload:
            image_encoder.to("cuda")
        self.pipe.register_modules(
            image_encoder=image_encoder, feature_extractor=CLIPImageProcessor()
        )
        files = [local for _, local in adapters]
        try:
            self.pipe.load_ip_adapter(
                [str(f.parent) for f in files],
                subfolder=[""] * len(files),
                weight_name=[f.name for f in files],
                image_encoder_folder=None,
                local_files_only=True,
            )
        except Exception as e:
            with contextlib.suppress(Exception):
                self.pipe.unload_ip_adapter()
            raise ValueError(f"Could not load image prompt model {', '.join(wanted)}: {e}") from e
        if self.offload:
            # Offload hooks are set per component; the encoder needs its own.
            self.pipe.enable_model_cpu_offload()
        self.ip_adapters = wanted
        self.encoder_path = encoder

    def _load_controlnets(self, nets: list[tuple[str, Path]]) -> None:
        """Keep the requested ControlNets resident and drop the rest."""
        wanted = {path for path, _ in nets}
        gone = [path for path in self.controlnets if path not in wanted]
        if gone:
            for path in gone:
                del self.controlnets[path]
            self.derived = {k: v for k, v in self.derived.items() if "+" not in k}
            gc.collect()
            torch.cuda.empty_cache()
        for path, local in nets:
            if path not in self.controlnets:
                self.controlnets[path] = self._load_controlnet(path, local)

    def _load_controlnet(self, path: str, local: Path) -> Any:
        if _is_union(local):
            raise ValueError(f"{path} is a union ControlNet, which Degas can't use yet")
        try:
            if local.is_dir():
                net = ControlNetModel.from_pretrained(str(local), torch_dtype=torch.float16)
            else:
                net = ControlNetModel.from_single_file(str(local), torch_dtype=torch.float16)
        except Exception as e:
            raise ValueError(f"Could not load ControlNet {path}: {e}") from e
        return net if self.offload else net.to("cuda")

    def _load(self, path: Path, config: Path, vae_path: Path | None, inpaint: bool) -> None:
        if self.pipe is not None and self.model_path == path and self.vae_path == vae_path:
            return
        self.unload()
        cls = StableDiffusionXLInpaintPipeline if inpaint else StableDiffusionXLPipeline
        extra: dict[str, Any] = {}
        if vae_path is not None:
            # The checkpoint's own VAE overflows in float16, so the pipelines would move it to
            # float32 for every encode and decode; the fp16 fix (`force_upcast` off) doesn't.
            extra["vae"] = AutoencoderKL.from_pretrained(
                str(vae_path), torch_dtype=torch.float16, local_files_only=True
            )
        pipe = cls.from_single_file(
            str(path),
            config=str(config),
            local_files_only=True,
            torch_dtype=torch.float16,
            use_safetensors=path.suffix == ".safetensors",
            **extra,
        )
        _free, total = torch.cuda.mem_get_info()
        self.offload = total < _OFFLOAD_BELOW_BYTES
        if self.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.model_path = path
        self.vae_path = vae_path
        self.inpaint_model = inpaint
        self.derived = {}
        self.adapters = {}
        self._scheduler_config = pipe.scheduler.config

    def _pipe_for(self, mode: str, controlnets: list[str]) -> Any:
        """The pipeline for a mode, with these ControlNets (asset paths) if any."""
        if self.inpaint_model and mode not in ("inpaint", "outpaint"):
            raise ValueError("An inpainting model can only inpaint or outpaint")
        base = mode if mode in ("t2i", "i2i") else "inpaint"
        if not controlnets and (self.inpaint_model or base == "t2i"):
            return self.pipe
        key = "+".join([base, *controlnets])
        if key not in self.derived:
            extra: dict[str, Any] = {}
            if controlnets:
                cls = CONTROL_PIPELINES[base]
                nets = [self.controlnets[p] for p in controlnets]
                extra["controlnet"] = nets[0] if len(nets) == 1 else nets
            else:
                cls = PIPELINES[base]
            # from_pipe defaults to float32 and casts the *shared* modules in place, which
            # would double the loaded pipeline's VRAM (OOM on a T4).
            derived = cls.from_pipe(self.pipe, torch_dtype=torch.float16, **extra)
            if self.offload:
                derived.enable_model_cpu_offload()
            derived.set_progress_bar_config(disable=True)
            self.derived[key] = derived
        return self.derived[key]

    def _apply_loras(self, loras: list[tuple[str, Path, float]]) -> None:
        """Load only new adapters and delete ones no longer requested (`_activate_loras`
        weights them)."""
        plan = plan_loras(self.adapters, [(path, weight) for path, _, weight in loras])
        if plan.remove:
            self.pipe.delete_adapters(plan.remove)
            self.adapters = {p: n for p, n in self.adapters.items() if n not in plan.remove}
        local = {path: file for path, file, _ in loras}
        for path, name in plan.add:
            try:
                self._load_lora(local[path], name)
            except Exception as e:
                with contextlib.suppress(Exception):
                    self.pipe.delete_adapters([name])
                raise ValueError(f"Could not load LoRA {path}: {e}") from e
            self.adapters[path] = name

    def _activate_loras(
        self, loras: list[tuple[str, Path, float]], faceid: list[tuple[str, float]]
    ) -> None:
        """Weight the requested LoRAs, and FaceID models' own LoRAs, together."""
        names = [self.adapters[path] for path, _, _ in loras] + [n for n, _ in faceid]
        weights = [w for _, _, w in loras] + [w for _, w in faceid]
        if names:
            self.pipe.set_adapters(names, adapter_weights=weights)

    def _load_lora(self, file: Path, name: str) -> None:
        """`load_lora_weights`, but with the text encoder keys matched to transformers 5's
        flattened `CLIPTextModel` (see `strip_text_model`)."""
        pipe = self.pipe
        state, alphas, metadata = pipe.lora_state_dict(
            str(file), unet_config=pipe.unet.config, return_lora_metadata=True
        )
        encoders = [("text_encoder", pipe.text_encoder), ("text_encoder_2", pipe.text_encoder_2)]
        for prefix, encoder in encoders:
            if not hasattr(encoder, "text_model"):
                state = strip_text_model(state, prefix)
                alphas = alphas and strip_text_model(alphas, prefix)
        common = {"network_alphas": alphas, "adapter_name": name, "metadata": metadata}
        pipe.load_lora_into_unet(state, unet=pipe.unet, _pipeline=pipe, **common)
        for prefix, encoder in encoders:
            pipe.load_lora_into_text_encoder(
                state,
                text_encoder=encoder,
                prefix=prefix,
                lora_scale=pipe.lora_scale,
                _pipeline=pipe,
                **common,
            )

    def _set_scheduler(self, pipe: Any, name: str, schedule: str) -> None:
        try:
            cls, kwargs = SCHEDULERS[name]
            sigmas = SCHEDULES[schedule]
        except KeyError as e:
            raise ValueError(f"Unknown scheduler or schedule {e.args[0]!r}") from None
        if issubclass(cls, _SCHEDULED):
            kwargs = {**kwargs, **sigmas}
        pipe.scheduler = cls.from_config(self._scheduler_config, **kwargs)

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.model_path = None
        self.vae_path = None
        self.derived = {}
        self.adapters = {}
        self.controlnets = {}
        self.ip_adapters = ()
        self.encoder_path = None
        self.faces = None
        gc.collect()
        torch.cuda.empty_cache()


PIPELINES: dict[str, Any] = {
    "i2i": StableDiffusionXLImg2ImgPipeline,
    "inpaint": StableDiffusionXLInpaintPipeline,
}
CONTROL_PIPELINES: dict[str, Any] = {
    "t2i": StableDiffusionXLControlNetPipeline,
    "i2i": StableDiffusionXLControlNetImg2ImgPipeline,
    "inpaint": StableDiffusionXLControlNetInpaintPipeline,
}


def _is_union(local: Path) -> bool:
    """Whether a ControlNet is a union model (one net for many control types)."""
    if local.is_dir():
        try:
            config = json.loads((local / "config.json").read_text())
        except (OSError, ValueError):
            return False
        return bool(config.get("_class_name") == "ControlNetUnionModel")
    if local.suffix != ".safetensors":
        return False
    from safetensors import safe_open  # noqa: PLC0415 - only needed here

    with safe_open(str(local), framework="pt") as f:
        return any(k.startswith(("task_embedding", "control_type_proj")) for k in f.keys())  # noqa: SIM118 - not a dict


@contextlib.contextmanager
def limit_to_areas(units: list[tuple[Any, Image.Image]]) -> Iterator[None]:
    """While the pipeline runs, multiply each ControlNet's residuals by its area mask.

    `forward` is replaced on the instance rather than the model being wrapped in another
    module, so the pipelines' `isinstance` checks keep passing. accelerate's offload hook also
    lives on the instance's `forward`; the masked one calls through it.
    """
    restore: list[tuple[Any, Any]] = []
    try:
        for net, area in units:
            restore.append((net, net.__dict__.get("forward")))
            net.forward = _masked(net.forward, area)
        yield
    finally:
        for net, forward in reversed(restore):
            if forward is None:
                del net.forward
            else:
                net.forward = forward


def _masked(forward: Any, area: Image.Image) -> Any:
    mask = torch.from_numpy(np.asarray(area, dtype=np.float32) / 255.0)[None, None]
    sized: dict[tuple[Any, ...], Any] = {}

    def fit(residual: Any) -> Any:
        key = (tuple(residual.shape[-2:]), residual.device, residual.dtype)
        if key not in sized:
            m = F.interpolate(mask.to(residual.device), size=residual.shape[-2:], mode="area")
            sized[key] = m.to(residual.dtype)
        return residual * sized[key]

    def masked_forward(*args: Any, **kwargs: Any) -> Any:
        out = forward(*args, **kwargs)
        if isinstance(out, tuple):  # return_dict=False, as the pipelines call it
            return ([fit(d) for d in out[0]], fit(out[1]), *out[2:])
        out.down_block_res_samples = [fit(d) for d in out.down_block_res_samples]
        out.mid_block_res_sample = fit(out.mid_block_res_sample)
        return out

    return masked_forward
