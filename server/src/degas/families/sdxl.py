"""Stable Diffusion XL descriptor."""

from typing import Any, Literal

from degas.families.base import (
    JsonSchema,
    SizeConstraints,
    SpecError,
    Variant,
    validate_params,
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
        Variant(id="base", label="SDXL", min_gpu="T4", modes=("t2i",)),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SizeConstraints(
            multiple_of=8, min_pixels=512 * 512, max_pixels=1536 * 1536, presets=PRESETS
        )

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        self._check(variant, mode)
        return {
            "type": "object",
            "required": ["prompt"],
            "properties": {
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
            },
        }

    def validate(self, spec: dict[str, Any]) -> dict[str, Any]:
        variant = spec.get("variant", "base")
        mode = spec.get("mode", "t2i")
        self._check(variant, mode)
        model = spec.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("path"), str):
            raise SpecError("A model is required")
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        c = self.size_constraints(variant)
        for dim in ("width", "height"):
            params[dim] = max(c.multiple_of, params[dim] // c.multiple_of * c.multiple_of)
        pixels = params["width"] * params["height"]
        if not c.min_pixels <= pixels <= c.max_pixels:
            raise SpecError(
                f"{params['width']}x{params['height']} is outside SDXL's supported pixel count"
            )
        loras = validate_single_loras(spec.get("loras"), MAX_LORAS)
        return {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": {"path": model["path"], "size": model.get("size")},
            "loras": loras,
            "params": params,
            "inputs": {},
            "control": [],
        }

    def _check(self, variant: str, mode: str) -> None:
        v = next((v for v in self.variants if v.id == variant), None)
        if v is None:
            raise SpecError(f"Unknown SDXL variant {variant!r}")
        if mode not in v.modes:
            raise SpecError(f"SDXL mode {mode!r} is not supported yet")
