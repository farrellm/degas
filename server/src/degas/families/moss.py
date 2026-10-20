"""MOSS-Image descriptor: text-to-image from the Base, SFT or Kromoss checkpoint (design §4.3).

OpenMOSS's 5B DiT with a Qwen3.5 text encoder and the FLUX.2 VAE, as the repos' diffusers
folders (a custom pipeline). Base, SFT and Kromoss (lodestones' fine-tune of Base) share the
architecture and the recipe: 50 steps, CFG 3, shift 5, and the negative branch run with block
22 skipped. Each variant reads its own folder under `models/moss/`. LoRAs are kohya files
under `loras/moss/`, such as Kromoss's 8-step turbo (8 steps, CFG 1). No layout prompts.
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
    validate_model,
    validate_params,
    validate_single_loras,
)
from degas_worker.spec import Spec

MODELS = "models/moss"
MAX_LORAS = 4
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
SKIP_LAYERS = {
    "22": "Block 22",
    "24": "Block 24",
    "26": "Block 26",
    "off": "Off",
}

# About 1 megapixel at the usual aspect ratios, the 1456² square OpenMOSS shows, then 4 MP.
# The VAE shrinks 8x and the transformer packs 2x2 patches, so sizes are multiples of 16.
PRESETS = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
    (1456, 1456),
    (2048, 2048),
)
SIZE = SizeConstraints(
    multiple_of=16, min_pixels=512 * 512, max_pixels=2048 * 2048, presets=PRESETS
)


class Moss:
    id = "moss"
    label = "MOSS-Image"
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    variants: tuple[Variant, ...] = (
        Variant(
            id="base",
            label="MOSS-Image Base",
            min_gpu="L4",
            modes=("t2i",),
            model_dir=f"{MODELS}/MOSS-Image-Base",
        ),
        Variant(
            id="sft",
            label="MOSS-Image SFT",
            min_gpu="L4",
            modes=("t2i",),
            model_dir=f"{MODELS}/MOSS-Image-SFT",
        ),
        Variant(
            id="kromoss",
            label="Kromoss",
            min_gpu="L4",
            modes=("t2i",),
            model_dir=f"{MODELS}/Kromoss",
        ),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": prompt_prop(),
            "negative_prompt": negative_prompt_prop(f"OpenMOSS suggests: {NEGATIVE}"),
            **size_props((1024, 1024), minimum=512, maximum=2048, multiple_of=16),
            "steps": steps_prop(50, 100),
            "cfg": {
                "type": "number",
                "title": "CFG",
                "description": "How closely it follows the prompt. Raise it for short prompts. "
                "The 8-step turbo LoRA wants 1, with 8 steps.",
                "default": 3.0,
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
            },
            "shift": {
                "type": "number",
                "title": "Shift",
                "description": "Higher spends more steps on the overall layout.",
                "default": 5.0,
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
                "x-advanced": True,
            },
            "skip_layer": choice_prop(
                "Skip layer",
                SKIP_LAYERS,
                "22",
                description="The block the negative prompt's pass skips. Off can look more "
                "realistic but lose detail.",
                advanced=True,
            ),
            "seed": seed_prop(),
        }
        return params_schema(props)

    def extend_variant(self, variant: str) -> str | None:
        return None  # images

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "base")
        mode = spec.get("mode", "t2i")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, SIZE, self.label)
        if spec.get("control"):
            raise SpecError(f"{self.label} doesn't take control units")
        return {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": validate_single_loras(spec.get("loras"), MAX_LORAS),
            "params": params,
            "inputs": {},
            "control": [],
        }
