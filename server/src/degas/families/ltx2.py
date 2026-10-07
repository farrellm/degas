"""LTX-2 descriptor: text-, image- and first-and-last-frame-to-video with audio (design §4.3).

LTX-2.3 and LTX-2.5 share diffusers' LTX2 pipelines, so they are variants of one family. The
distilled checkpoints run a fixed 8-step schedule without guidance, and can upscale 2x: the
video is made at half size, then its latents are upsampled and refined in 3 more steps.
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
    validate_single_loras,
)
from degas_worker.spec import Spec

MAX_LORAS = 6
MODELS = "models/ltx2"
MODES = ("t2v", "i2v", "flf2v")
# The variants with a distilled transformer: a fixed schedule, no CFG, and the 2x upscale.
DISTILLED = frozenset({"ltx23-distilled", "ltx25"})

# The VAE compresses 32x; an upscaled video is made at half size, so it steps by 64 (validate).
SIZE = SizeConstraints(
    multiple_of=32,
    min_pixels=256 * 256,
    max_pixels=1536 * 1024,
    # The smaller sizes (about 6,000 video tokens at 121 frames) are for i2v on a 40 GB A100,
    # where 768x768 runs out of memory.
    presets=(
        (960, 544),
        (544, 960),
        (832, 480),
        (480, 832),
        (768, 512),
        (512, 768),
        (768, 768),
        (640, 640),
    ),
)
UPSCALE_MULTIPLE = 64
# The VAE packs 8 frames per latent: counts are 8k + 1.
FRAME_STEP = 8


class Ltx2:
    id = "ltx2"
    label = "LTX-2"
    media: MediaKind = "video"
    lora_format: LoraFormat = "single"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    variants: tuple[Variant, ...] = (
        Variant(
            id="ltx25",
            label="LTX-2.5",
            min_gpu="A100",
            modes=MODES,
            model_dir=f"{MODELS}/ltx25",
        ),
        Variant(
            id="ltx23-distilled",
            label="LTX-2.3 Distilled",
            min_gpu="A100",
            modes=MODES,
            model_dir=f"{MODELS}/ltx23-distilled",
        ),
        Variant(
            id="ltx23",
            label="LTX-2.3",
            min_gpu="A100",
            modes=MODES,
            model_dir=f"{MODELS}/ltx23",
        ),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        props: dict[str, JsonSchema] = {
            "prompt": prompt_prop(),
            "negative_prompt": negative_prompt_prop(),
            **size_props(SIZE.presets[0], minimum=256, maximum=1536, multiple_of=32),
            "num_frames": {
                "type": "integer",
                "title": "Frames",
                "default": 121,
                "minimum": 17,
                "maximum": 257,
                "x-step": FRAME_STEP,
                "x-widget": "slider",
            },
            "fps": {
                "type": "integer",
                "title": "Frame rate",
                "default": 24,
                "minimum": 8,
                "maximum": 50,
                "x-widget": "slider",
            },
        }
        if variant in DISTILLED:
            props["upscale"] = {
                "type": "boolean",
                "title": "Upscale 2×",  # noqa: RUF001 - shown in the form
                "description": "Make the video at half size, then upsample and refine it",
                "default": False,
                "x-advanced": True,
            }
        else:
            props["steps"] = steps_prop(30, 60)
            props["cfg"] = {
                "type": "number",
                "title": "CFG",
                "default": 3.0,
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
            }
            props["audio_cfg"] = {
                "type": "number",
                "title": "Audio CFG",
                "default": 7.0,
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
                "x-advanced": True,
            }
            props["stg"] = {
                "type": "number",
                "title": "STG",
                "description": "Spatio-temporal guidance: steadier motion, slower steps",
                "default": 1.0,
                "minimum": 0,
                "maximum": 3,
                "multipleOf": 0.25,
                "x-widget": "slider",
                "x-advanced": True,
            }
        props["seed"] = seed_prop()
        return params_schema(props)

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "ltx25")
        mode = spec.get("mode", "t2v")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        size = SIZE
        if params.get("upscale"):
            size = SizeConstraints(UPSCALE_MULTIPLE, SIZE.min_pixels, SIZE.max_pixels, SIZE.presets)
        snap_size(params, size, v.label)
        params["num_frames"] = (params["num_frames"] - 1) // FRAME_STEP * FRAME_STEP + 1
        if spec.get("control"):
            raise SpecError("LTX-2 doesn't take control units")
        return {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": validate_single_loras(spec.get("loras"), MAX_LORAS),
            "params": params,
            "inputs": validate_inputs(spec.get("inputs"), mode),
            "control": [],
        }

    def extend_variant(self, variant: str) -> str | None:
        return variant
