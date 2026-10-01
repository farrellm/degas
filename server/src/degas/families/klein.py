"""FLUX.2 [klein] descriptor: editing from up to 4 images (design §4.3).

The 9B model, step-distilled to 4 steps, as the official diffusers folder. It reads the
source and up to 3 references (BFL's limit for klein), and has no CFG.
"""

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

MAX_LORAS = 8
MAX_REFS = 3
# The pipeline scales a condition image down to 1 megapixel, never up.
REF_MAX_PIXELS = 1024 * 1024

# About 1 megapixel at the usual aspect ratios, then larger squares up to klein's 4 MP. The
# VAE shrinks 8x and the transformer packs 2x2 patches, so sizes are multiples of 16.
PRESETS = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
    (1536, 1536),
    (2048, 2048),
)
SIZE = SizeConstraints(
    multiple_of=16, min_pixels=512 * 512, max_pixels=2048 * 2048, presets=PRESETS
)


class Klein:
    id = "klein"
    label = "FLUX.2 [klein]"
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    variants: tuple[Variant, ...] = (
        Variant(
            id="9b",
            label="FLUX.2 [klein] 9B",
            min_gpu="L4",
            modes=("edit",),
            max_refs=MAX_REFS,
            ref_max_pixels=REF_MAX_PIXELS,
        ),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": prompt_prop(),
            **size_props((1024, 1024), minimum=512, maximum=2048, multiple_of=16),
            "steps": steps_prop(4, 16, "The model is distilled to 4."),
            "seed": seed_prop(),
        }
        return params_schema(props)

    def validate(self, spec: dict[str, Any]) -> dict[str, Any]:
        variant = spec.get("variant", "9b")
        mode = spec.get("mode", "edit")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, SIZE, self.label)
        if spec.get("control"):
            raise SpecError(f"{self.label} doesn't take control units")
        raw = spec.get("inputs")
        inputs = validate_inputs(raw, mode)
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
