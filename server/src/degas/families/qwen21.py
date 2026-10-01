"""Qwen-Image 2.1 descriptor: text-to-image and editing from up to 10 images (design §4.3)."""

from typing import Any

from degas.families.base import (
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
    snap_size,
    validate_inputs,
    validate_model,
    validate_params,
    validate_refs,
    validate_single_loras,
)
from degas_worker.spec import Spec

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
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
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
            "prompt": prompt_prop(),
            "negative_prompt": negative_prompt_prop("Used when CFG is above 1."),
            **size_props((2048, 2048), minimum=512, maximum=2752, multiple_of=32),
            "steps": steps_prop(40, 80),
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
            "seed": seed_prop(),
            "schedule": choice_prop("Schedule", SCHEDULES, "default", advanced=True),
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
        return params_schema(props)

    def validate(self, spec: dict[str, Any]) -> Spec:
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
