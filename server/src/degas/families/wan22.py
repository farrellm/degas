"""Wan 2.2 descriptor: text-, image- and first-and-last-frame-to-video (design §4.3).

It also has Wan 2.1's 14B image-to-video and first-and-last-frame models as variants: the same
pipeline and runner, with one transformer in place of the A14B's two experts. The 5B can't take
a last frame (diffusers conditions it on the first frame only).
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
    validate_paired_loras,
    validate_params,
    validate_single_loras,
)
from degas_worker.spec import Lora, PairedLora, Spec

MAX_LORAS = 6
MODELS = "models/wan22"
WAN21_I2V = "wan21-i2v-14b"
WAN21_FLF2V = "wan21-flf2v-14b"

# The 5B's VAE compresses 16x and its transformer patches 2x2, so sizes step by 32.
# It is trained at 720p only.
SIZE_5B = SizeConstraints(
    multiple_of=32,
    min_pixels=480 * 832,
    max_pixels=1280 * 736,
    presets=((1280, 704), (704, 1280), (960, 960)),
)
# The 14B models, Wan 2.1's too.
SIZE_A14B = SizeConstraints(
    multiple_of=16,
    min_pixels=480 * 480,
    max_pixels=1280 * 720,
    presets=((1280, 720), (720, 1280), (832, 480), (480, 832), (960, 960)),
)

# Defaults from the Wan 2.2 and 2.1 model cards.
DEFAULTS: dict[str, dict[str, Any]] = {
    "ti2v-5b": {"num_frames": 121, "fps": 24, "steps": 50, "cfg": 5.0},
    "t2v-a14b": {"num_frames": 81, "fps": 16, "steps": 40, "cfg": 4.0, "cfg_low": 3.0},
    "i2v-a14b": {"num_frames": 81, "fps": 16, "steps": 40, "cfg": 3.5, "cfg_low": 3.5},
    WAN21_I2V: {"num_frames": 81, "fps": 16, "steps": 40, "cfg": 5.0},
    WAN21_FLF2V: {"num_frames": 81, "fps": 16, "steps": 50, "cfg": 5.5},
}
# Where the A14B variants hand over from the high-noise expert to the low-noise one.
BOUNDARY = {"t2v-a14b": 0.875, "i2v-a14b": 0.9}


class Wan22:
    id = "wan22"
    label = "Wan 2.2"
    media: MediaKind = "video"
    lora_format: LoraFormat = "paired_hi_lo"
    supports_control = False
    supports_image_prompts = False
    image_prompt_options = None
    variants: tuple[Variant, ...] = (
        Variant(
            id="ti2v-5b",
            label="Wan 2.2 TI2V 5B",
            min_gpu="L4",
            modes=("t2v", "i2v"),
            model_dir=f"{MODELS}/ti2v-5b",
            lora_format="single",
        ),
        Variant(
            id="t2v-a14b",
            label="Wan 2.2 T2V A14B",
            min_gpu="A100",
            modes=("t2v",),
            model_dir=f"{MODELS}/t2v-a14b",
        ),
        Variant(
            id="i2v-a14b",
            label="Wan 2.2 I2V A14B",
            min_gpu="A100",
            modes=("i2v", "flf2v"),
            model_dir=f"{MODELS}/i2v-a14b",
        ),
        Variant(
            id=WAN21_I2V,
            label="Wan 2.1 I2V 14B",
            min_gpu="A100",
            modes=("i2v",),
            model_dir=f"{MODELS}/{WAN21_I2V}",
            lora_format="single",
        ),
        # Its image embedding expects two pictures, so it can't make video from one.
        Variant(
            id=WAN21_FLF2V,
            label="Wan 2.1 FLF2V 14B",
            min_gpu="A100",
            modes=("flf2v",),
            model_dir=f"{MODELS}/{WAN21_FLF2V}",
            lora_format="single",
        ),
    )

    def size_constraints(self, variant: str) -> SizeConstraints:
        return SIZE_5B if variant == "ti2v-5b" else SIZE_A14B

    def param_schema(self, variant: str, mode: str) -> JsonSchema:
        find_variant(self, variant, mode)
        d = DEFAULTS[variant]
        c = self.size_constraints(variant)
        width, height = c.presets[0]
        props: dict[str, JsonSchema] = {
            "prompt": prompt_prop(),
            "negative_prompt": negative_prompt_prop(),
            **size_props((width, height), minimum=256, maximum=1280, multiple_of=c.multiple_of),
            "num_frames": {
                "type": "integer",
                "title": "Frames",
                "default": d["num_frames"],
                "minimum": 17,
                "maximum": 121,
                "x-step": 4,  # the VAE packs 4 frames per latent: counts are 4k + 1
                "x-widget": "slider",
            },
            "fps": {
                "type": "integer",
                "title": "Frame rate",
                "default": d["fps"],
                "minimum": 8,
                "maximum": 30,
                "x-widget": "slider",
            },
            "steps": steps_prop(d["steps"], 80),
            "cfg": {
                "type": "number",
                "title": "CFG, high noise" if variant in BOUNDARY else "CFG",
                "default": d["cfg"],
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
            },
            "seed": seed_prop(),
        }
        if variant in BOUNDARY:
            props["cfg_low"] = {
                "type": "number",
                "title": "CFG, low noise",
                "default": d["cfg_low"],
                "minimum": 1,
                "maximum": 10,
                "multipleOf": 0.5,
                "x-widget": "slider",
                "x-advanced": True,
            }
            props["boundary_ratio"] = {
                "type": "number",
                "title": "Expert switch",
                "default": BOUNDARY[variant],
                "minimum": 0.5,
                "maximum": 1,
                "multipleOf": 0.025,
                "x-widget": "slider",
                "x-advanced": True,
            }
        return params_schema(props)

    def validate(self, spec: dict[str, Any]) -> Spec:
        variant = spec.get("variant", "ti2v-5b")
        mode = spec.get("mode", "t2v")
        v = find_variant(self, variant, mode)
        model = validate_model(spec.get("model"), v)
        params = validate_params(self.param_schema(variant, mode), spec.get("params") or {})
        snap_size(params, self.size_constraints(variant), v.label)
        params["num_frames"] = (params["num_frames"] - 1) // 4 * 4 + 1
        loras: list[Lora] | list[PairedLora]
        if v.lora_format == "single":
            loras = validate_single_loras(spec.get("loras"), MAX_LORAS)
        else:
            loras = validate_paired_loras(spec.get("loras"), MAX_LORAS)
        if spec.get("control"):
            raise SpecError("Wan 2.2 doesn't take control units")
        return {
            "family": self.id,
            "variant": variant,
            "mode": mode,
            "model": model,
            "loras": loras,
            "params": params,
            "inputs": validate_inputs(spec.get("inputs"), mode),
            "control": [],
        }

    def extend_variant(self, variant: str) -> str | None:
        return {"t2v-a14b": "i2v-a14b", WAN21_FLF2V: WAN21_I2V}.get(variant, variant)
