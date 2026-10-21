"""Fibo 2.0 descriptor: text-to-image, and editing from up to 5 images (design §4.3).

Bria's Dev checkpoint, the full undistilled model (50 steps, CFG 5), as its diffusers folder
with the repo's `fonts/`. One model both generates and edits. It's trained on structured JSON
captions, so by default the prompt goes through Bria's prompt-to-JSON VLM first: FIBO-vlm
writes a caption for `t2i`, FIBO-edit-vlm an edit instruction and the result's caption for
an edit of one image (`upsamplers/fibo2/`, transformers folders). FIBO-edit-vlm reads one
image, so an edit with references sends the instruction as written, a form Fibo 2.0 was also
trained on. The JSON is kept with each result. The pipeline has no LoRA loader.
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
)
from degas_worker.spec import Spec

# The pipeline reads up to 5 images: the source and 4 references.
MAX_REFS = 4
# The pipeline crops each reference to about 1 megapixel; the runner scales larger ones
# down to this first.
REF_MAX_PIXELS = 1024 * 1024
UPSAMPLER = "upsamplers/fibo2/FIBO-vlm"
EDIT_UPSAMPLER = "upsamplers/fibo2/FIBO-edit-vlm"

# About 1 megapixel, the area Fibo 2.0 was trained at, at the sizes Bria shows (1424x800 is
# its 16:9), then a larger square. The FLUX.2 VAE shrinks 8x and the transformer packs 2x2
# patches, so sizes are multiples of 16.
PRESETS = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (832, 1248),
    (1248, 832),
    (800, 1424),
    (1424, 800),
    (1536, 1536),
)
SIZE = SizeConstraints(
    multiple_of=16, min_pixels=512 * 512, max_pixels=2048 * 2048, presets=PRESETS
)


class Fibo2:
    id = "fibo2"
    label = "Fibo 2.0"
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    variants: tuple[Variant, ...] = (
        Variant(
            id="dev",
            label="Fibo 2.0 Dev",
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
            **size_props((1024, 1024), minimum=512, maximum=2048, multiple_of=16),
            "steps": steps_prop(50, 100),
            "cfg": {
                "type": "number",
                "title": "CFG",
                "description": "How closely it follows the caption.",
                "default": 5.0,
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
            },
            "upsample": {
                "type": "boolean",
                "title": "Upsample prompt",
                "description": "Writes the structured caption Fibo was trained on first. Off "
                "sends the prompt as written. An edit with more than one image always sends "
                "it as written.",
                "default": True,
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
        if spec.get("loras"):
            raise SpecError(f"{self.label} doesn't take LoRAs")
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
            "loras": [],
            "params": params,
            "inputs": inputs,
            "control": [],
        }
        if params["upsample"] and not inputs.get("refs"):
            path = UPSAMPLER if mode == "t2i" else EDIT_UPSAMPLER
            out["upsampler"] = {"path": path, "size": None}
        return out
