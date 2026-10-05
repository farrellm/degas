"""FLUX.1 [dev] descriptor: text-to-image (design §4.3).

Image prompts use FLUX.1 Redux (`ip_adapters/flux1/FLUX.1-Redux-dev/`, a diffusers folder):
each picture's tokens are appended to the prompt's, shrunk and weighted.

A checkpoint is a single `.safetensors` transformer (the fp8 `flux1-dev-fp8`, or a Civitai
fine-tune); the text encoders, VAE and configs come from the official diffusers folder in
`configs/flux1/`, without its transformer weights. A whole diffusers folder in `models/flux1/`
loads on its own.
"""

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
    params_schema,
    prompt_prop,
    seed_prop,
    size_props,
    steps_prop,
)
from degas.families.validation import (
    snap_size,
    validate_image_prompts,
    validate_model,
    validate_params,
    validate_single_loras,
)
from degas_worker.spec import ImagePromptUnit, Spec

MAX_LORAS = 8
MAX_IMAGE_PROMPTS = 2
# Redux shrinks each picture's 27 x 27 tokens by 3 unless told: at full size the pictures
# drown the prompt.
DOWNSAMPLE = 3
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
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = True
    image_prompt_options = ImagePromptOptions(detail=True)
    variants: tuple[Variant, ...] = (
        Variant(id="dev", label="FLUX.1 [dev]", min_gpu="L4", modes=("t2i",)),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": prompt_prop(),
            **size_props((1024, 1024), minimum=512, maximum=2048, multiple_of=16),
            "steps": steps_prop(28, 50),
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
            "seed": seed_prop(),
        }
        return params_schema(props)

    def extend_variant(self, variant: str) -> str | None:
        return None  # images

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "dev")
        mode = spec.get("mode", "t2i")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, SIZE, self.label)
        if spec.get("control"):
            raise SpecError(f"{self.label} doesn't take control units")
        out: Spec = {
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
        prompts = validate_image_prompts(spec.get("image_prompts"), self.id, MAX_IMAGE_PROMPTS)
        if prompts:
            out["image_prompts"] = [redux_unit(n, u) for n, u in enumerate(prompts, 1)]
        return out


def redux_unit(n: int, unit: ImagePromptUnit) -> ImagePromptUnit:
    """A Redux unit: its pictures, weight and downsampling, and nothing Redux can't do."""
    if unit.get("mask"):
        raise SpecError(f"Image prompt {n}: FLUX.1 can't limit a picture to an area")
    if unit["purpose"] != "all" or (unit["start"], unit["end"]) != (0.0, 1.0):
        raise SpecError(f"Image prompt {n}: FLUX.1 reads a picture everywhere, at every step")
    return {
        "adapter": unit["adapter"],
        "images": unit["images"],
        "fit": unit["fit"],
        "purpose": unit["purpose"],
        "weight": unit["weight"],
        "start": unit["start"],
        "end": unit["end"],
        "downsample": unit.get("downsample", DOWNSAMPLE),
    }
