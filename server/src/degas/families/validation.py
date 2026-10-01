"""Checks shared by the family descriptors: each takes part of a submitted spec and returns
it normalized (defaults filled, values clamped), or raises `SpecError` with a message for
the form."""

from typing import Any, Literal

from degas.blobs import is_ref
from degas.families.base import JsonSchema, SizeConstraints, SpecError, Variant
from degas.media import FIT_MODES
from degas_worker.spec import (
    AssetRef,
    ControlUnit,
    ImagePromptUnit,
    Lora,
    PairedLora,
    Place,
    SpecInputs,
)

LORA_WEIGHT: JsonSchema = {"type": "number", "minimum": -2, "maximum": 2}
# Modes that start from a source image.
SOURCE_MODES = frozenset({"i2i", "i2v", "edit", "inpaint", "outpaint"})


def validate_model(model: Any, variant: Variant) -> AssetRef:
    if not isinstance(model, dict) or not isinstance(model.get("path"), str) or not model["path"]:
        raise SpecError("A model is required")
    path: str = model["path"]
    if variant.model_dir and not (
        path == variant.model_dir or path.startswith(variant.model_dir + "/")
    ):
        raise SpecError(f"{path} isn't a {variant.label} model")
    return {"path": path, "size": model.get("size")}


def validate_inputs(inputs: Any, mode: str) -> SpecInputs:
    """Source-image inputs: `{source, fit, extends?}`. Transforms are recorded by the server."""
    if mode not in SOURCE_MODES:
        return {}
    inputs = inputs if isinstance(inputs, dict) else {}
    source = inputs.get("source")
    if not is_ref(source):
        raise SpecError("Choose a source image")
    fit = inputs.get("fit") or "crop"
    if fit not in FIT_MODES:
        raise SpecError(f"fit: must be one of {', '.join(FIT_MODES)}")
    out: SpecInputs = {"source": source, "fit": fit}
    if mode == "inpaint":
        mask = inputs.get("mask")
        if not is_ref(mask):
            raise SpecError("Paint the area to redraw")
        out["mask"] = mask
    if mode == "outpaint":
        out["place"] = inputs.get("place")
    extends = inputs.get("extends")
    if extends is not None:
        if not is_ref(extends):
            raise SpecError("extends: expected a sha256 reference")
        out["extends"] = extends
    return out


def validate_refs(refs: Any, limit: int) -> list[str]:
    """The reference images after the source: `["sha256:…", …]`, in the order the prompt
    numbers them (the source is image 1)."""
    if refs is None:
        return []
    if not isinstance(refs, list):
        raise SpecError("refs: expected a list")
    if len(refs) > limit:
        raise SpecError(f"At most {limit} images can go with the source")
    for n, ref in enumerate(refs, 2):
        if not is_ref(ref):
            raise SpecError(f"Image {n}: expected a sha256 reference")
    return list(refs)


MIN_PLACE = 64


def validate_place(place: Any, params: dict[str, Any]) -> Place:
    """Where an outpaint puts its source on the canvas: `{x, y, w, h}` in canvas pixels."""
    if not isinstance(place, dict):
        raise SpecError("Place the image on the canvas")

    def pixels(key: str) -> int:
        value = place.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SpecError(f"place: {key} must be a number")
        return round(value)

    out: Place = {"x": pixels("x"), "y": pixels("y"), "w": pixels("w"), "h": pixels("h")}
    width, height = params["width"], params["height"]
    if out["w"] < MIN_PLACE or out["h"] < MIN_PLACE:
        raise SpecError(f"The image must be at least {MIN_PLACE} px on each side")
    if out["x"] < 0 or out["y"] < 0 or out["x"] + out["w"] > width or out["y"] + out["h"] > height:
        raise SpecError(f"The image must sit inside the {width}x{height} canvas")
    if out["w"] == width and out["h"] == height:
        raise SpecError("The image fills the canvas: there's nothing to outpaint")
    return out


def snap_size(params: dict[str, Any], c: SizeConstraints, label: str) -> None:
    """Round width and height down to `multiple_of` and check the pixel count."""
    for dim in ("width", "height"):
        params[dim] = max(c.multiple_of, params[dim] // c.multiple_of * c.multiple_of)
    pixels = params["width"] * params["height"]
    if not c.min_pixels <= pixels <= c.max_pixels:
        raise SpecError(
            f"{params['width']}x{params['height']} is outside {label}'s supported pixel count"
        )


# -- ControlNet units (design §4.3) ----------------------------------------------------------

TRACES = ("depth", "pose", "canny")  # preprocessors whose output is a control image
CONTROL_SCALE: JsonSchema = {"type": "number", "minimum": 0, "maximum": 2}
CONTROL_FRACTION: JsonSchema = {"type": "number", "minimum": 0, "maximum": 1}


def _sha(value: Any, what: str) -> str:
    if not is_ref(value):
        raise SpecError(f"{what}: expected a sha256 reference")
    return value


def validate_control(control: Any, family: str, limit: int) -> list[ControlUnit]:
    """Validate ControlNet units:
    `[{controlnet: {path}, image, fit?, scale, start, end, mask?, preprocessor?}]`."""
    if control is None:
        return []
    if not isinstance(control, list):
        raise SpecError("control: expected a list")
    if len(control) > limit:
        raise SpecError(f"At most {limit} ControlNets can be used at once")
    folder = f"controlnets/{family}/"
    out: list[ControlUnit] = []
    seen: set[str] = set()
    for n, unit in enumerate(control, 1):
        if not isinstance(unit, dict):
            raise SpecError("Each ControlNet unit must be an object")
        net = unit.get("controlnet")
        path = net.get("path") if isinstance(net, dict) else net
        if not isinstance(path, str) or not path:
            raise SpecError(f"ControlNet {n}: choose a model")
        if not path.startswith(folder):
            raise SpecError(f"{path} isn't a ControlNet for this model")
        # Area masks are applied per ControlNet model, so each model guides one unit.
        if path in seen:
            raise SpecError(f"{path} is used twice. Each ControlNet can guide one image.")
        seen.add(path)
        size = net.get("size") if isinstance(net, dict) else None
        out.append(_control_unit(n, unit, {"path": path, "size": size}))
    return out


def _control_unit(n: int, unit: dict[str, Any], net: AssetRef) -> ControlUnit:
    """A unit's model, image, fit, weight, step range, area and preprocessor record."""
    if not isinstance(unit.get("image"), str) or not unit["image"]:
        raise SpecError(f"ControlNet {n}: choose a control image")
    entry: ControlUnit = {
        "controlnet": net,
        "image": _sha(unit["image"], f"ControlNet {n} image"),
        "fit": unit.get("fit") or "crop",
        "scale": _coerce("scale", CONTROL_SCALE, unit.get("scale", 0.7)),
        "start": _coerce("start", CONTROL_FRACTION, unit.get("start", 0.0)),
        "end": _coerce("end", CONTROL_FRACTION, unit.get("end", 1.0)),
    }
    if entry["fit"] not in FIT_MODES:
        raise SpecError(f"fit: must be one of {', '.join(FIT_MODES)}")
    if entry["start"] >= entry["end"]:
        raise SpecError(f"ControlNet {n}: its steps must start before they end")
    if unit.get("mask") is not None:
        entry["mask"] = _sha(unit["mask"], f"ControlNet {n} area")
    pre = unit.get("preprocessor")
    if pre is not None:
        if not isinstance(pre, dict) or pre.get("id") not in TRACES:
            raise SpecError(f"preprocessor: must be one of {', '.join(TRACES)}")
        params = pre.get("params") or {}
        if not isinstance(params, dict):
            raise SpecError("preprocessor params: expected an object")
        entry["preprocessor"] = {
            "id": pre["id"],
            "source": _sha(pre.get("source"), "preprocessor source"),
            "params": params,
        }
    return entry


# -- Image prompts (IP-Adapter, docs/ip-adapter.md) -------------------------------------------

# What an image prompt takes from its pictures: which of the UNet's attention blocks the
# adapter acts in (the runner maps these to blocks).
IP_PURPOSES = ("all", "style", "layout", "style_layout")
IP_WEIGHT: JsonSchema = {"type": "number", "minimum": 0, "maximum": 2}
MAX_PROMPT_IMAGES = 4
# FaceID: how much of CLIP's reading of the face Plus v2 adds (its `shortcut_scale`), and the
# weight of the LoRA the model carries.
FACE_STRUCTURE: JsonSchema = {"type": "number", "minimum": 0, "maximum": 2}
FACE_LORA: JsonSchema = {"type": "number", "minimum": 0, "maximum": 1.5}
# Redux: its 27 x 27 grid of picture tokens is shrunk by this factor before the prompt sees it.
DOWNSAMPLE: JsonSchema = {"type": "integer", "minimum": 1, "maximum": 5}


def validate_image_prompts(prompts: Any, family: str, limit: int) -> list[ImagePromptUnit]:
    """Validate image prompts:
    `[{adapter: {path}, images: [sha…], fit?, purpose, weight, start, end, mask?}]`."""
    if prompts is None:
        return []
    if not isinstance(prompts, list):
        raise SpecError("image_prompts: expected a list")
    if len(prompts) > limit:
        raise SpecError(f"At most {limit} image prompts can be used at once")
    return [_image_prompt(n, unit, f"ip_adapters/{family}/") for n, unit in enumerate(prompts, 1)]


def _image_prompt(n: int, unit: Any, folder: str) -> ImagePromptUnit:
    """One image prompt's model, pictures, fit, purpose, weight, step range and area."""
    if not isinstance(unit, dict):
        raise SpecError("Each image prompt must be an object")
    adapter = unit.get("adapter")
    path = adapter.get("path") if isinstance(adapter, dict) else adapter
    if not isinstance(path, str) or not path:
        raise SpecError(f"Image prompt {n}: choose a model")
    if not path.startswith(folder):
        raise SpecError(f"{path} isn't an image prompt model for this model")
    images = unit.get("images")
    if not isinstance(images, list) or not images:
        raise SpecError(f"Image prompt {n}: add a picture")
    if len(images) > MAX_PROMPT_IMAGES:
        raise SpecError(f"Image prompt {n}: at most {MAX_PROMPT_IMAGES} pictures")
    size = adapter.get("size") if isinstance(adapter, dict) else None
    entry: ImagePromptUnit = {
        "adapter": {"path": path, "size": size},
        "images": [_sha(image, f"Image prompt {n} picture") for image in images],
        "fit": unit.get("fit") or "crop",
        "purpose": unit.get("purpose") or "all",
        "weight": _coerce("weight", IP_WEIGHT, unit.get("weight", 0.6)),
        "start": _coerce("start", CONTROL_FRACTION, unit.get("start", 0.0)),
        "end": _coerce("end", CONTROL_FRACTION, unit.get("end", 1.0)),
    }
    if entry["fit"] not in FIT_MODES:
        raise SpecError(f"fit: must be one of {', '.join(FIT_MODES)}")
    if entry["purpose"] not in IP_PURPOSES:
        raise SpecError(f"purpose: must be one of {', '.join(IP_PURPOSES)}")
    if entry["start"] >= entry["end"]:
        raise SpecError(f"Image prompt {n}: its steps must start before they end")
    if unit.get("mask") is not None:
        entry["mask"] = _sha(unit["mask"], f"Image prompt {n} area")
    if unit.get("structure") is not None:
        entry["structure"] = _coerce("structure", FACE_STRUCTURE, unit["structure"])
    if unit.get("lora_weight") is not None:
        entry["lora_weight"] = _coerce("lora_weight", FACE_LORA, unit["lora_weight"])
    if unit.get("downsample") is not None:
        entry["downsample"] = _coerce("downsample", DOWNSAMPLE, unit["downsample"])
    return entry


def validate_single_loras(loras: Any, limit: int) -> list[Lora]:
    """Validate a list of single-file LoRAs: `[{path, weight}]`."""
    if loras is None:
        return []
    if not isinstance(loras, list):
        raise SpecError("loras: expected a list")
    if len(loras) > limit:
        raise SpecError(f"At most {limit} LoRAs can be applied at once")
    out: list[Lora] = []
    for lora in loras:
        if not isinstance(lora, dict) or not isinstance(lora.get("path"), str):
            raise SpecError("Each LoRA needs a path")
        if any(o["path"] == lora["path"] for o in out):
            raise SpecError(f"LoRA {lora['path']} is listed twice")
        weight = _coerce("LoRA weight", LORA_WEIGHT, lora.get("weight", 1.0))
        out.append({"path": lora["path"], "weight": weight, "size": lora.get("size")})
    return out


HALVES: tuple[Literal["high", "low"], ...] = ("high", "low")


def validate_paired_loras(loras: Any, limit: int) -> list[PairedLora]:
    """Validate Wan A14B LoRAs: `[{high: {path, weight}, low: {path, weight}}]`.

    Either half may be left out, for a LoRA trained for one expert only.
    """
    if loras is None:
        return []
    if not isinstance(loras, list):
        raise SpecError("loras: expected a list")
    if len(loras) > limit:
        raise SpecError(f"At most {limit} LoRAs can be applied at once")
    out: list[PairedLora] = []
    seen: set[str] = set()
    for lora in loras:
        if not isinstance(lora, dict) or not (lora.get("high") or lora.get("low")):
            raise SpecError("Each LoRA needs a high-noise or low-noise file")
        entry: PairedLora = {}
        for half in HALVES:
            part = lora.get(half)
            if not part:
                continue
            if not isinstance(part, dict) or not isinstance(part.get("path"), str):
                raise SpecError(f"LoRA {half}-noise half needs a path")
            if part["path"] in seen:
                raise SpecError(f"LoRA {part['path']} is listed twice")
            seen.add(part["path"])
            weight = _coerce("LoRA weight", LORA_WEIGHT, part.get("weight", 1.0))
            entry[half] = {"path": part["path"], "weight": weight, "size": part.get("size")}
        out.append(entry)
    return out


def validate_params(schema: JsonSchema, params: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and clamp/coerce values according to a (flat) param schema."""
    out: dict[str, Any] = {}
    props: dict[str, JsonSchema] = schema["properties"]
    unknown = set(params) - set(props)
    if unknown:
        raise SpecError(f"Unknown parameters: {', '.join(sorted(unknown))}")
    for name, prop in props.items():
        value = params.get(name, prop.get("default"))
        if value is None:
            if name in schema.get("required", ()):
                raise SpecError(f"Missing parameter: {name}")
            out[name] = None
            continue
        out[name] = _coerce(name, prop, value)
    return out


def _coerce(name: str, prop: JsonSchema, value: Any) -> Any:
    kind = prop.get("type")
    if "enum" in prop:
        if value not in prop["enum"]:
            raise SpecError(f"{name}: must be one of {', '.join(map(str, prop['enum']))}")
        return value
    if kind == "string":
        if not isinstance(value, str):
            raise SpecError(f"{name}: expected a string")
        if prop.get("minLength") and len(value.strip()) < prop["minLength"]:
            raise SpecError(f"{name}: must not be empty")
        return value
    if kind in ("integer", "number"):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SpecError(f"{name}: expected a number")
        num: int | float = round(value) if kind == "integer" else float(value)
        if "minimum" in prop:
            num = max(num, prop["minimum"])
        if "maximum" in prop:
            num = min(num, prop["maximum"])
        return num
    if kind == "boolean":
        if not isinstance(value, bool):
            raise SpecError(f"{name}: expected true or false")
        return value
    return value
