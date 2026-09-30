"""Qwen-Image 2.1 descriptor: text-to-image and editing from up to 10 images (design §4.3)."""

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
    validate_refs,
    validate_single_loras,
)

MAX_LORAS = 8
# The model takes up to 10 condition images: the source and 9 references.
MAX_REFS = 9

# Keep in sync with the worker runner (degas_worker/families/qwen21.py).
SCHEDULES = {"default": "Default", "beta": "Beta"}

# Qwen's 2K table, then the same shapes at about 1 megapixel. The VAE shrinks 16x and the
# transformer doesn't patch, but the pipeline rounds sizes down to 32.
PRESETS = (
    (2048, 2048),
    (1792, 2400),
    (2400, 1792),
    (1696, 2528),
    (2528, 1696),
    (1536, 2752),
    (2752, 1536),
    (1024, 1024),
    (864, 1152),
    (1152, 864),
    (832, 1248),
    (1248, 832),
    (768, 1376),
    (1376, 768),
)
SIZE = SizeConstraints(
    multiple_of=32, min_pixels=512 * 512, max_pixels=2400 * 1792, presets=PRESETS
)


class Qwen21:
    id = "qwen21"
    label = "Qwen-Image 2.1"
    media: Literal["image", "video"] = "image"
    lora_format: Literal["single", "paired_hi_lo"] = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    variants: tuple[Variant, ...] = (
        Variant(
            id="base",
            label="Qwen-Image 2.1",
            min_gpu="L4",
            modes=("t2i", "edit", "inpaint"),
            max_refs=MAX_REFS,
        ),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": {"type": "string", "title": "Prompt", "minLength": 1, "x-widget": "prompt"},
            "negative_prompt": {
                "type": "string",
                "title": "Negative prompt",
                "description": "Used when CFG is above 1.",
                "default": "",
                "x-widget": "prompt",
            },
            "width": {
                "type": "integer",
                "title": "Width",
                "default": 2048,
                "minimum": 512,
                "maximum": 2752,
                "multipleOf": 32,
                "x-widget": "aspect",
            },
            "height": {
                "type": "integer",
                "title": "Height",
                "default": 2048,
                "minimum": 512,
                "maximum": 2752,
                "multipleOf": 32,
                "x-widget": "aspect",
            },
            "steps": {
                "type": "integer",
                "title": "Steps",
                "default": 40,
                "minimum": 1,
                "maximum": 80,
                "x-widget": "slider",
            },
            "cfg": {
                "type": "number",
                "title": "CFG",
                "description": "Above 1 uses the negative prompt and takes twice as long.",
                "default": 1.0,
                "minimum": 1,
                "maximum": 4,
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
            "schedule": {
                "type": "string",
                "title": "Schedule",
                "default": "default",
                "enum": list(SCHEDULES),
                "x-enum-labels": list(SCHEDULES.values()),
                "x-widget": "select",
                "x-advanced": True,
            },
            "degrid": {
                "type": "boolean",
                "title": "Remove VAE grid",
                "description": "Filters out the faint 2 px lattice the VAE leaves.",
                "default": True,
                "x-advanced": True,
            },
        }
        if mode == "inpaint":
            props["mask_blur"] = {
                "type": "integer",
                "title": "Mask blur",
                "default": 8,
                "minimum": 0,
                "maximum": 64,
                "x-widget": "slider",
                "x-advanced": True,
            }
        return {"type": "object", "required": ["prompt"], "properties": props}

    def validate(self, spec: dict[str, Any]) -> dict[str, Any]:
        variant = spec.get("variant", "base")
        mode = spec.get("mode", "t2i")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, SIZE, self.label)
        if spec.get("control"):
            raise SpecError(f"{self.label} doesn't take control units")
        raw = spec.get("inputs")
        inputs = validate_inputs(raw, mode)
        if mode != "t2i":
            refs = validate_refs((raw or {}).get("refs"), MAX_REFS)
            if refs:
                inputs["refs"] = refs
        return {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": validate_single_loras(spec.get("loras"), MAX_LORAS),
            "params": params,
            "inputs": inputs,
            "control": [],
        }
