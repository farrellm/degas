"""Stable Diffusion XL runner: text-to-image, image-to-image, inpaint and outpaint, with LoRAs
and ControlNets.

A regular checkpoint is loaded as the text-to-image pipeline, and the image-to-image and
inpaint pipelines are made from it with `from_pipe` (they share its weights and LoRAs). An
inpainting checkpoint (the `inpaint` variant, a 9-channel UNet) only has the inpaint pipeline.
With ControlNet units, the ControlNet version of the mode's pipeline is made the same way.

Regional ControlNet: a unit with an area mask has its ControlNet's residuals multiplied by
the mask, downsampled to each residual's size, so it only guides that area.
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
from PIL import Image

from degas_worker import masks
from degas_worker.families.base import Output, RunContext
from degas_worker.families.control import control_kwargs, crop_areas
from degas_worker.families.lora import plan_loras

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
        self._scheduler_config: Any = None

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        mode = spec.get("mode", "t2i")
        if mode not in ("t2i", "i2i", "inpaint", "outpaint"):
            raise ValueError(f"SDXL can't do {mode!r}")
        model = spec["model"]
        path = ctx.fetch_asset(model["path"], model.get("size"))
        config = ctx.fetch_asset(spec["config"]["path"], spec["config"].get("size"))
        vae = spec.get("vae")
        vae_path = ctx.fetch_asset(vae["path"], vae.get("size")) if vae else None
        loras = [
            (lora["path"], ctx.fetch_asset(lora["path"], lora.get("size")), float(lora["weight"]))
            for lora in spec.get("loras") or []
        ]
        units = spec.get("control") or []
        nets = [
            (net["path"], ctx.fetch_asset(net["path"], net.get("size")))
            for net in (u["controlnet"] for u in units)
        ]
        ctx.check_cancelled()
        ctx.progress(0, "load", 0, 1)
        self._load(path, config, vae_path, inpaint=spec.get("variant") == "inpaint")
        self._apply_loras(loras)
        self._load_controlnets(nets)
        pipe = self._pipe_for(mode, [p for p, _ in nets])
        ctx.progress(0, "load", 1, 1)

        params = spec["params"]
        self._set_scheduler(
            pipe, params.get("scheduler", "dpmpp_2m"), params.get("schedule", "karras")
        )
        steps = int(params["steps"])
        size = (int(params["width"]), int(params["height"]))
        kwargs, original, keep = self._inputs(mode, spec, size, ctx)
        areas: list[Image.Image | None] = []
        if units:
            images, areas = self._control_inputs(units, ctx)
            if "padding_mask_crop" in kwargs:
                box = pipe.mask_processor.get_crop_region(
                    kwargs["mask_image"], *size, pad=kwargs["padding_mask_crop"]
                )
                areas = crop_areas(areas, box, size)
            kwargs.update(control_kwargs(units, images, mode))
        regional = [
            (self.controlnets[p], area) for (p, _), area in zip(nets, areas, strict=False) if area
        ]
        # Starting from an image skips the first (1 - strength) of the schedule.
        strength = float(kwargs.get("strength", 1.0))
        runs = max(1, min(steps, int(steps * strength)))
        clip_skip = int(params.get("clip_skip") or 0)
        for item, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(item, "denoise", 0, runs)

            def on_step(
                _pipe: Any, i: int, _t: Any, kw: dict[str, Any], item: int = item
            ) -> dict[str, Any]:
                ctx.check_cancelled()
                done = min(i + 1, runs)
                ctx.progress(item, "denoise" if done < runs else "decode", done, runs)
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

    def _inputs(
        self, mode: str, spec: dict[str, Any], size: tuple[int, int], ctx: RunContext
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
        units: list[dict[str, Any]], ctx: RunContext
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
