"""BluePixel-Turbo descriptor: instruction editing of one image at up to 4K (design §4.3).

vivo's fine-tune of FLUX.2 [klein] 9B with a DC-AE f64 autoencoder in place of klein's VAE,
step-distilled to 4 steps with CFG 3, as the diffusers folder `openvivo/BluePixel-Turbo`. It
edits only: the source is the one condition image, fitted to the output size.
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
)
from degas_worker.spec import Spec

# The autoencoder shrinks 64x and the transformer takes 1x1 patches, so sizes are multiples
# of 64. The model is trained for native 4K: 4096² by default, then about 16.7 megapixels at
# the usual aspect ratios, with 1K and 2K squares for quicker edits.
PRESETS = (
    (4096, 4096),
    (3520, 4672),
    (4672, 3520),
    (3328, 4992),
    (4992, 3328),
    (3072, 5440),
    (5440, 3072),
    (1024, 1024),
    (2048, 2048),
)
MAX_SIDE = 5440
SIZE = SizeConstraints(
    multiple_of=64, min_pixels=1024 * 1024, max_pixels=4096 * 4096, presets=PRESETS
)


class BluePixel:
    id = "bluepixel"
    label = "BluePixel"
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    # About 37 GB in bf16 (the VAE in float32), and 4K activations: an L4 is too small.
    variants: tuple[Variant, ...] = (
        Variant(id="turbo", label="BluePixel-Turbo", min_gpu="A100", modes=("edit",)),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": prompt_prop(),
            **size_props((4096, 4096), minimum=512, maximum=MAX_SIDE, multiple_of=64),
            "steps": steps_prop(4, 16, "The model is distilled to 4."),
            "cfg": {
                "type": "number",
                "title": "CFG",
                "description": "How closely it follows the instruction. 1 turns it off, "
                "which halves the time.",
                "default": 3.0,
                "minimum": 1,
                "maximum": 8,
                "multipleOf": 0.5,
                "x-widget": "slider",
            },
            "seed": seed_prop(),
        }
        return params_schema(props)

    def extend_variant(self, variant: str) -> str | None:
        return None  # images

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "turbo")
        mode = spec.get("mode", "edit")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, SIZE, self.label)
        if spec.get("control"):
            raise SpecError(f"{self.label} doesn't take control units")
        if spec.get("loras"):
            raise SpecError(f"{self.label} doesn't take LoRAs")
        raw = spec.get("inputs")
        if (raw or {}).get("refs"):
            raise SpecError(f"{self.label} edits one image and reads no references")
        return {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": [],
            "params": params,
            "inputs": validate_inputs(raw, mode),
            "control": [],
        }
