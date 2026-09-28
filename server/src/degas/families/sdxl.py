"""Stable Diffusion XL descriptor."""

from typing import Any, Literal

from degas.families.base import (
    JsonSchema,
    SizeConstraints,
    SpecError,
    Variant,
    find_variant,
    snap_size,
    validate_inputs,
    validate_model,
    validate_params,
    validate_place,
    validate_single_loras,
)

# Keep in sync with the worker runner (degas_worker/families/sdxl.py).
SCHEDULERS = {
    "dpmpp_2m_karras": "DPM++ 2M Karras",
    "dpmpp_2m": "DPM++ 2M",
    "euler_a": "Euler a",
    "euler": "Euler",
    "ddim": "DDIM",
    "unipc": "UniPC",
}

MAX_LORAS = 8

# Inpainting checkpoints (9-channel UNets) live apart from the rest, so they don't show up
# where a text-to-image model is expected.
INPAINT_DIR = "models/sdxl/inpaint"

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
    media: Literal["image", "video"] = "image"
    lora_format: Literal["single", "paired_hi_lo"] = "single"
    supports_control = False  # Phase 7
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
            "prompt": {
                "type": "string",
                "title": "Prompt",
                "minLength": 1,
                "x-widget": "prompt",
            },
            "negative_prompt": {
                "type": "string",
                "title": "Negative prompt",
                "default": "",
                "x-widget": "prompt",
            },
            "width": {
                "type": "integer",
                "title": "Width",
                "default": 1024,
                "minimum": 512,
                "maximum": 2048,
                "multipleOf": 8,
                "x-widget": "aspect",
            },
            "height": {
                "type": "integer",
                "title": "Height",
                "default": 1024,
                "minimum": 512,
                "maximum": 2048,
                "multipleOf": 8,
                "x-widget": "aspect",
            },
            "steps": {
                "type": "integer",
                "title": "Steps",
                "default": 30,
                "minimum": 1,
                "maximum": 100,
                "x-widget": "slider",
            },
            "cfg": {
                "type": "number",
                "title": "CFG",
                "default": 5.5,
                "minimum": 1,
                "maximum": 20,
                "multipleOf": 0.5,
                "x-widget": "slider",
            },
            "seed": {
                "type": "integer",
                "title": "Seed",
                "default": -1,
                "minimum": -1,
                "maximum": 2**32 - 1,
                "x-widget": "seed",
            },
            "scheduler": {
                "type": "string",
                "title": "Sampler",
                "default": "dpmpp_2m_karras",
                "enum": list(SCHEDULERS),
                "x-enum-labels": list(SCHEDULERS.values()),
                "x-widget": "select",
                "x-advanced": True,
            },
            "clip_skip": {
                "type": "integer",
                "title": "CLIP skip",
                "default": 0,
                "minimum": 0,
                "maximum": 4,
                "x-widget": "number",
                "x-advanced": True,
            },
        }
        props.update(_mode_params(variant, mode))
        return {"type": "object", "required": ["prompt"], "properties": props}

    def validate(self, spec: dict[str, Any]) -> dict[str, Any]:
        variant = spec.get("variant", "base")
        mode = spec.get("mode", "t2i")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        if variant != "inpaint" and model["path"].startswith(INPAINT_DIR + "/"):
            raise SpecError("An inpainting model can only inpaint or outpaint")
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, self.size_constraints(variant), "SDXL")
        loras = validate_single_loras(spec.get("loras"), MAX_LORAS)
        inputs = validate_inputs(spec.get("inputs"), mode)
        if mode == "outpaint":
            inputs["place"] = validate_place(inputs.get("place"), params)
        return {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": loras,
            "params": params,
            "inputs": inputs,
            "control": [],
        }

    def _check(self, variant: str, mode: str) -> None:
        find_variant(self, variant, mode)


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
        props["inpaint_area"] = {
            "type": "string",
            "title": "Redraw",
            "default": "whole",
            "enum": ["whole", "masked"],
            "x-enum-labels": ["The whole image", "Around the mask"],
            "x-widget": "select",
        }
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
