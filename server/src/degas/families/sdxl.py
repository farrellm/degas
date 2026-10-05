"""Stable Diffusion XL descriptor."""

from typing import Any

from degas.families.base import (
    ImagePromptOptions,
    JsonSchema,
    LoraFormat,
    MediaKind,
    SizeConstraints,
    SpecError,
    Variant,
    find_variant,
)
from degas.families.schema import (
    choice_prop,
    negative_prompt_prop,
    params_schema,
    prompt_prop,
    seed_prop,
    size_props,
    steps_prop,
)
from degas.families.validation import (
    IP_PURPOSES,
    snap_size,
    validate_control,
    validate_image_prompts,
    validate_inputs,
    validate_model,
    validate_params,
    validate_place,
    validate_single_loras,
)
from degas_worker.spec import ImagePromptUnit, Spec

# Keep in sync with the worker runner (degas_worker/families/sdxl.py).
SCHEDULERS = {
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "euler_a": "Euler a",
    "euler": "Euler",
    "ddim": "DDIM",
    "unipc": "UniPC",
}

# Noise schedules (sigma spacing), for the samplers in SCHEDULED; the rest take "default".
SCHEDULES = {"default": "Default", "karras": "Karras", "exponential": "Exponential"}
SCHEDULED = frozenset({"dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "euler", "unipc"})

MAX_LORAS = 8
MAX_CONTROL = 3
MAX_IMAGE_PROMPTS = 2

# The CLIP image encoders IP-Adapters read their pictures with (transformers folders). Every
# useful SDXL adapter uses ViT-H; only h94's first one, `ip-adapter_sdxl`, uses ViT-bigG.
# InsightFace's detector and recognizer, which FaceID models read faces with.
FACE_DETECTOR = "preprocessors/insightface"
FACE_DEFAULTS = {"structure": 1.0, "lora_weight": 0.6}

ENCODERS = {
    "vit-h": "image_encoders/sdxl/clip-vit-h-14",
    "vit-bigg": "image_encoders/sdxl/clip-vit-bigg-14",
}

# Inpainting checkpoints (9-channel UNets) live apart from the rest, so they don't show up
# where a text-to-image model is expected.
INPAINT_DIR = "models/sdxl/inpaint"

# Ollin's fp16-fix VAE (madebyollin/sdxl-vae-fp16-fix, a diffusers folder), which decodes in
# float16. The stock SDXL VAE overflows in float16, so diffusers runs a checkpoint's own VAE in
# float32.
FP16_VAE = "vae/sdxl/sdxl-vae-fp16-fix"

# The diffusers configs and tokenizers a single-file checkpoint is loaded with (its repo's
# files without the weights), so loading needs nothing from Hugging Face.
CONFIGS = {
    "base": "configs/sdxl/stable-diffusion-xl-base-1.0",
    "inpaint": "configs/sdxl/stable-diffusion-xl-1.0-inpainting-0.1",
}

# How much of the image an inpaint redraws.
INPAINT_AREAS = {"whole": "The whole image", "masked": "Around the mask"}

# Inpainting checkpoints default to DPM++ 2M SDE.
SAMPLER = {"base": "dpmpp_2m", "inpaint": "dpmpp_2m_sde"}

# Denoise strength defaults: an inpainting checkpoint is made to redraw the mask from scratch.
STRENGTH = {("base", "i2i"): 0.6, ("base", "inpaint"): 0.85, ("inpaint", "inpaint"): 1.0}

PRESETS = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
    (640, 1536),
    (1536, 640),
)


class Sdxl:
    id = "sdxl"
    label = "Stable Diffusion XL"
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = True
    supports_image_prompts = True
    image_prompt_options = ImagePromptOptions(
        purposes=IP_PURPOSES, areas=True, steps=True, faces=True
    )
    variants: tuple[Variant, ...] = (
        Variant(id="base", label="SDXL", min_gpu="T4", modes=("t2i", "i2i", "inpaint", "outpaint")),
        Variant(
            id="inpaint",
            label="SDXL inpainting",
            min_gpu="T4",
            modes=("inpaint", "outpaint"),
            model_dir=INPAINT_DIR,
        ),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SizeConstraints(
            multiple_of=8, min_pixels=512 * 512, max_pixels=1536 * 1536, presets=PRESETS
        )

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        self._check(variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": prompt_prop(),
            "negative_prompt": negative_prompt_prop(),
            **size_props((1024, 1024), minimum=512, maximum=2048, multiple_of=8),
            "steps": steps_prop(30, 100),
            "cfg": {
                "type": "number",
                "title": "CFG",
                "default": 5.5,
                "minimum": 1,
                "maximum": 20,
                "multipleOf": 0.5,
                "x-widget": "slider",
            },
            "seed": seed_prop(),
            "scheduler": choice_prop("Sampler", SCHEDULERS, SAMPLER[variant], advanced=True),
            "schedule": choice_prop(
                "Schedule",
                SCHEDULES,
                "karras",
                description="Not used by Euler a or DDIM.",
                advanced=True,
            ),
            "clip_skip": {
                "type": "integer",
                "title": "CLIP skip",
                "default": 0,
                "minimum": 0,
                "maximum": 4,
                "x-widget": "number",
                "x-advanced": True,
            },
            "vae_fp32": {
                "type": "boolean",
                "title": "Built-in VAE in float32",
                "description": "The checkpoint's own VAE instead of the fp16 fix. Slower.",
                "default": False,
                "x-advanced": True,
            },
        }
        props.update(_mode_params(variant, mode))
        return params_schema(props)

    def extend_variant(self, variant: str) -> str | None:
        return None  # images

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "base")
        mode = spec.get("mode", "t2i")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        if variant != "inpaint" and model["path"].startswith(INPAINT_DIR + "/"):
            raise SpecError("An inpainting model can only inpaint or outpaint")
        params = upgrade_params(spec.get("params") or {})
        params = validate_params(self.param_schema(variant, mode), params)
        if params["scheduler"] not in SCHEDULED:
            params["schedule"] = "default"
        snap_size(params, self.size_constraints(variant), "SDXL")
        loras = validate_single_loras(spec.get("loras"), MAX_LORAS)
        inputs = validate_inputs(spec.get("inputs"), mode)
        if mode == "outpaint":
            inputs["place"] = validate_place(inputs.get("place"), params)
        out: Spec = {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "config": {"path": CONFIGS[variant], "size": None},
            "loras": loras,
            "params": params,
            "inputs": inputs,
            "control": validate_control(spec.get("control"), self.id, MAX_CONTROL),
        }
        prompts = validate_image_prompts(spec.get("image_prompts"), self.id, MAX_IMAGE_PROMPTS)
        if prompts:
            out["image_prompts"] = [faceid_unit(n, u) for n, u in enumerate(prompts, 1)]
            out["image_encoder"] = {"path": image_encoder(prompts), "size": None}
            if any(is_faceid(u["adapter"]["path"]) for u in prompts):
                out["face_detector"] = {"path": FACE_DETECTOR, "size": None}
        if not params["vae_fp32"]:
            out["vae"] = {"path": FP16_VAE, "size": None}
        return out

    def _check(self, variant: str, mode: str) -> None:
        find_variant(self, variant, mode)


def encoder_of(adapter: str) -> str:
    """The image encoder an IP-Adapter file was trained with, from its name."""
    stem = adapter.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    return "vit-bigg" if stem == "ip-adapter_sdxl" or "bigg" in stem else "vit-h"


def is_faceid(adapter: str) -> bool:
    """Whether an IP-Adapter reads InsightFace identities (FaceID) rather than CLIP pictures."""
    return "faceid" in adapter.rsplit("/", 1)[-1].lower()


def faceid_unit(n: int, unit: ImagePromptUnit) -> ImagePromptUnit:
    """A FaceID unit gets its structure and LoRA weight; other units don't have them."""
    unit = unit.copy()
    unit.pop("downsample", None)  # Redux only
    if not is_faceid(unit["adapter"]["path"]):
        unit.pop("structure", None)
        unit.pop("lora_weight", None)
        return unit
    if unit["purpose"] != "all":
        raise SpecError(f"Image prompt {n}: a FaceID model reads a face, so it acts everywhere")
    unit.setdefault("structure", FACE_DEFAULTS["structure"])
    unit.setdefault("lora_weight", FACE_DEFAULTS["lora_weight"])
    return unit


def image_encoder(prompts: list[ImagePromptUnit]) -> str:
    """The one image encoder a job's image prompts share (the pipeline holds one)."""
    kinds = {encoder_of(unit["adapter"]["path"]) for unit in prompts}
    if len(kinds) > 1:
        raise SpecError(
            "These image prompt models read pictures with different encoders (ViT-H and "
            "ViT-bigG), so they can't be used together"
        )
    return ENCODERS[kinds.pop()]


def upgrade_params(params: dict[str, Any]) -> dict[str, Any]:
    """Params from before the schedule was its own control (the sampler had it in its name)."""
    if "scheduler" not in params or "schedule" in params:
        return params
    if params["scheduler"] == "dpmpp_2m_karras":
        return {**params, "scheduler": "dpmpp_2m", "schedule": "karras"}
    return {**params, "schedule": "default"}


def _mode_params(variant: str, mode: str) -> dict[str, JsonSchema]:
    """Parameters that only apply when starting from an image."""
    props: dict[str, JsonSchema] = {}
    if mode in ("i2i", "inpaint"):
        props["strength"] = {
            "type": "number",
            "title": "Denoise strength",
            "default": STRENGTH[variant, mode],
            "minimum": 0.05,
            "maximum": 1,
            "multipleOf": 0.05,
            "x-widget": "slider",
        }
    if mode == "inpaint":
        props["inpaint_area"] = choice_prop("Redraw", INPAINT_AREAS, "whole")
        props["mask_padding"] = {
            "type": "integer",
            "title": "Space around the mask",
            "default": 32,
            "minimum": 0,
            "maximum": 256,
            "multipleOf": 8,
            "x-widget": "slider",
            "x-advanced": True,
        }
    if mode in ("inpaint", "outpaint"):
        props["mask_blur"] = {
            "type": "integer",
            "title": "Mask blur",
            "default": 8,
            "minimum": 0,
            "maximum": 64,
            "x-widget": "slider",
            "x-advanced": True,
        }
    if mode == "outpaint":
        props["blend"] = {
            "type": "integer",
            "title": "Blend into the image",
            "default": 32,
            "minimum": 0,
            "maximum": 128,
            "multipleOf": 8,
            "x-widget": "slider",
            "x-advanced": True,
        }
    return props
