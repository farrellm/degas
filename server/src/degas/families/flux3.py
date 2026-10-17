"""FLUX 3 Image descriptor: text-to-image, and editing from up to 10 images (design §4.3).

Klein Base, the undistilled checkpoint (50 steps, CFG 3.5), as BFL's diffusers folder. The
model expects a full scene caption, so by default the prompt goes through BFL's prompt
upsampler (`upsamplers/flux3/flux-3-image-klein-pup/`, a transformers folder) first, and the
caption it writes is kept with each result. An edit reads the source and up to 9 references.
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
# The model reads up to 10 images: the source and 9 references.
MAX_REFS = 9
# The pipeline keeps each condition image at its own size; the runner scales larger ones down
# to this, which is also what the upsampler sees of a single reference.
REF_MAX_PIXELS = 1024 * 1024
UPSAMPLER = "upsamplers/flux3/flux-3-image-klein-pup"

# About 1 megapixel at the usual aspect ratios, then larger squares. The VAE shrinks 8x and
# the transformer packs 2x2 patches, so sizes are multiples of 16.
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


class Flux3:
    id = "flux3"
    label = "FLUX 3 Image"
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    variants: tuple[Variant, ...] = (
        Variant(
            id="klein-base",
            label="FLUX 3 Image Klein Base",
            min_gpu="L4",
            modes=("t2i", "edit"),
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
            "negative_prompt": negative_prompt_prop(),
            **size_props((1024, 1024), minimum=512, maximum=2048, multiple_of=16),
            "steps": steps_prop(50, 100),
            "cfg": {
                "type": "number",
                "title": "CFG",
                "description": "How closely it follows the caption.",
                "default": 3.5,
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
            },
            "upsample": {
                "type": "boolean",
                "title": "Upsample prompt",
                "description": "Expands the prompt into a full scene caption first. Off sends "
                "the prompt as written.",
                "default": True,
            },
            "seed": seed_prop(),
        }
        return params_schema(props)

    def extend_variant(self, variant: str) -> str | None:
        return None  # images

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "klein-base")
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
        out: Spec = {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": validate_single_loras(spec.get("loras"), MAX_LORAS),
            "params": params,
            "inputs": inputs,
            "control": [],
        }
        if params["upsample"]:
            out["upsampler"] = {"path": UPSAMPLER, "size": None}
        return out
