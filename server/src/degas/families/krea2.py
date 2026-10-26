"""Krea 2 descriptor: text-to-image, and editing through an edit LoRA (design §4.3).

Krea's 12.9B MMDiT with a Qwen3-VL 4B text encoder and the Qwen-Image VAE, as the official
diffusers folders: Raw, the undistilled base model (28 steps, CFG), and Turbo, distilled to 8
steps with no CFG. Krea ships no edit model. An edit runs conradlocke's General Edit or
Identity Edit LoRA (`loras/krea2/`) with the source fed in as clean in-context tokens and the
prompt encoded together with the images, as comfyui-krea2edit does; the descriptor adds the
LoRA to the spec as `edit_lora`. Identity Edit takes a second image, the person to put into
the source's scene.
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
    validate_inputs,
    validate_model,
    validate_params,
    validate_refs,
    validate_single_loras,
)
from degas_worker.spec import Spec

MODELS = "models/krea2"
MAX_LORAS = 8
# Identity Edit reads a scene (the source) and a person; General Edit reads the source alone.
MAX_REFS = 1
EDIT_LORAS = {
    "general": "loras/krea2/krea2_general_edit_v2.safetensors",
    "identity": "loras/krea2/krea2_identity_edit_v2.safetensors",
}
EDIT_KINDS = {"general": "General edit", "identity": "Identity"}

# About 1 megapixel at the usual aspect ratios: Krea runs Raw up to 1K. Turbo goes to 2K. The
# VAE shrinks 8x and the transformer packs 2x2 patches, so sizes are multiples of 16.
PRESETS_1K = (
    (1024, 1024),
    (896, 1152),
    (1152, 896),
    (832, 1216),
    (1216, 832),
    (768, 1344),
    (1344, 768),
)
SIZES = {
    "raw": SizeConstraints(
        multiple_of=16,
        min_pixels=512 * 512,
        max_pixels=1536 * 1536,
        presets=(*PRESETS_1K, (1536, 1536)),
    ),
    "turbo": SizeConstraints(
        multiple_of=16,
        min_pixels=512 * 512,
        max_pixels=2048 * 2048,
        presets=(*PRESETS_1K, (1536, 1536), (2048, 2048)),
    ),
}


class Krea2:
    id = "krea2"
    label = "Krea 2"
    media: MediaKind = "image"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    # The bf16 transformer alone (26 GB) is more than an L4 holds.
    variants: tuple[Variant, ...] = (
        Variant(
            id="raw",
            label="Krea 2 Raw",
            min_gpu="A100",
            modes=("t2i", "edit"),
            model_dir=f"{MODELS}/Krea-2-Raw",
            max_refs=MAX_REFS,
        ),
        Variant(
            id="turbo",
            label="Krea 2 Turbo",
            min_gpu="A100",
            modes=("t2i", "edit"),
            model_dir=f"{MODELS}/Krea-2-Turbo",
            max_refs=MAX_REFS,
        ),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZES[variant]

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        edit = mode == "edit"
        props: dict[str, JsonSchema] = {"prompt": prompt_prop()}
        if variant == "raw":
            props["negative_prompt"] = negative_prompt_prop()
        top = SIZES[variant].presets[-1][0]
        props.update(size_props((1024, 1024), minimum=512, maximum=top, multiple_of=16))
        if variant == "raw":
            props["steps"] = steps_prop(20 if edit else 28, 100)
            props["cfg"] = {
                "type": "number",
                "title": "CFG",
                "description": "How closely it follows the prompt. 1 turns it off, which "
                "halves the time and drops the negative prompt.",
                "default": 3.0 if edit else 5.5,
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
            }
        else:
            props["steps"] = steps_prop(8, 16, "The model is distilled to 8.")
        if edit:
            props["edit_lora"] = choice_prop(
                "Edit LoRA",
                EDIT_KINDS,
                "general",
                description="General edit follows an instruction. Identity puts the person "
                "from the photo into the scene or pose the prompt describes; a second image "
                "is the person, and the source the scene.",
            )
        props["seed"] = seed_prop()
        return params_schema(props)

    def extend_variant(self, variant: str) -> str | None:
        return None  # images

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "raw")
        mode = spec.get("mode", "t2i")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, SIZES[variant], self.label)
        if spec.get("control"):
            raise SpecError(f"{self.label} doesn't take control units")
        raw = spec.get("inputs")
        inputs = validate_inputs(raw, mode)
        loras = validate_single_loras(spec.get("loras"), MAX_LORAS)
        out: Spec = {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": loras,
            "params": params,
            "inputs": inputs,
            "control": [],
        }
        if mode == "edit":
            kind = params["edit_lora"]
            refs = validate_refs((raw or {}).get("refs"), MAX_REFS)
            if refs and kind != "identity":
                raise SpecError("Only Identity reads a second image, the person")
            if refs:
                inputs["refs"] = refs
            # A LoRA row naming the edit LoRA sets its weight; otherwise it goes in at 1.
            path = EDIT_LORAS[kind]
            if not any(lora["path"] == path for lora in loras):
                out["edit_lora"] = {"path": path, "size": None, "weight": 1.0}
        return out
