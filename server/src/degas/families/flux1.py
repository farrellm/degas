"""FLUX.1 [dev] descriptor: text-to-image (design §4.3).

A checkpoint is a single `.safetensors` transformer (the fp8 `flux1-dev-fp8`, or a Civitai
fine-tune); the text encoders, VAE and configs come from the official diffusers folder in
`configs/flux1/`, without its transformer weights. A whole diffusers folder in `models/flux1/`
loads on its own.
"""

from typing import Any, Literal

from degas.families.base import (
    JsonSchema,
    SizeConstraints,
    SpecError,
    Variant,
    find_variant,
    snap_size,
    validate_model,
    validate_params,
    validate_single_loras,
)

MAX_LORAS = 8
# What a single-file checkpoint is loaded with: FLUX.1-dev's diffusers folder, whose
# `transformer/` keeps only its config.json.
BASE = "configs/flux1/FLUX.1-dev"
SINGLE_FILE = (".safetensors", ".sft")

# About 1 megapixel at the usual aspect ratios, then 1.5K squares. The VAE shrinks 8x and the
# transformer packs 2x2 patches, so sizes are multiples of 16.
PRESETS = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
    (1536, 1536),
)
SIZE = SizeConstraints(
    multiple_of=16, min_pixels=512 * 512, max_pixels=1536 * 1536, presets=PRESETS
)


class Flux1:
    id = "flux1"
    label = "FLUX.1"
    media: Literal["image", "video"] = "image"
    lora_format: Literal["single", "paired_hi_lo"] = "single"
    supports_control = False
    supports_image_prompts = False
    variants: tuple[Variant, ...] = (
        Variant(id="dev", label="FLUX.1 [dev]", min_gpu="L4", modes=("t2i",)),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": {"type": "string", "title": "Prompt", "minLength": 1, "x-widget": "prompt"},
            "width": {
                "type": "integer",
                "title": "Width",
                "default": 1024,
                "minimum": 512,
                "maximum": 2048,
                "multipleOf": 16,
                "x-widget": "aspect",
            },
            "height": {
                "type": "integer",
                "title": "Height",
                "default": 1024,
                "minimum": 512,
                "maximum": 2048,
                "multipleOf": 16,
                "x-widget": "aspect",
            },
            "steps": {
                "type": "integer",
                "title": "Steps",
                "default": 28,
                "minimum": 1,
                "maximum": 50,
                "x-widget": "slider",
            },
            "guidance": {
                "type": "number",
                "title": "Guidance",
                "description": "How closely it follows the prompt. There's no negative prompt.",
                "default": 3.5,
                "minimum": 1,
                "maximum": 10,
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
        }
        return {"type": "object", "required": ["prompt"], "properties": props}

    def validate(self, spec: dict[str, Any]) -> dict[str, Any]:
        variant = spec.get("variant", "dev")
        mode = spec.get("mode", "t2i")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, SIZE, self.label)
        if spec.get("control"):
            raise SpecError(f"{self.label} doesn't take control units")
        out: dict[str, Any] = {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": validate_single_loras(spec.get("loras"), MAX_LORAS),
            "params": params,
            "inputs": {},
            "control": [],
        }
        if model["path"].lower().endswith(SINGLE_FILE):
            out["config"] = {"path": BASE, "size": None}
        return out
