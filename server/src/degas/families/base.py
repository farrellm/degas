"""Family descriptors: what a model family offers and how its job specs are validated."""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

GPUS = ("T4", "L4", "A100", "H100")  # ascending capability
JsonSchema = dict[str, Any]


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    min_gpu: str
    modes: tuple[str, ...]
    default_params: dict[str, Any] = field(default_factory=dict)
    # Drive folder holding this variant's models (families with one variant use the whole
    # `models/<family>/` folder).
    model_dir: str | None = None
    # Overrides the family's LoRA format (Wan 2.2: the 5B takes single files, A14B pairs).
    lora_format: Literal["single", "paired_hi_lo"] | None = None
    # How many reference images an edit reads after its source (0: none).
    max_refs: int = 0
    # References bigger than this are scaled down to it and smaller ones kept as they are.
    # None: the pipeline scales each to the output's pixel count (Qwen).
    ref_max_pixels: int | None = None


@dataclass(frozen=True)
class SizeConstraints:
    multiple_of: int
    min_pixels: int
    max_pixels: int
    presets: tuple[tuple[int, int], ...]


class FamilyDescriptor(Protocol):
    id: str
    label: str
    media: Literal["image", "video"]
    variants: tuple[Variant, ...]
    lora_format: Literal["single", "paired_hi_lo"]
    supports_control: bool
    supports_image_prompts: bool

    def param_schema(self, variant: str, mode: str) -> JsonSchema: ...

    def size_constraints(self, variant: str) -> SizeConstraints: ...

    def validate(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Return a normalized spec (defaults filled, values clamped) or raise SpecError."""
        ...


def gpu_rank(gpu: str) -> int:
    return GPUS.index(gpu) if gpu in GPUS else -1


def describe(family: FamilyDescriptor) -> dict[str, Any]:
    return {
        "id": family.id,
        "label": family.label,
        "media": family.media,
        "lora_format": family.lora_format,
        "supports_control": family.supports_control,
        "supports_image_prompts": family.supports_image_prompts,
        "variants": [
            {
                "id": v.id,
                "label": v.label,
                "min_gpu": v.min_gpu,
                "modes": list(v.modes),
                "model_dir": v.model_dir,
                "lora_format": v.lora_format or family.lora_format,
                "max_refs": v.max_refs,
                "ref_max_pixels": v.ref_max_pixels,
                "size_constraints": family.size_constraints(v.id).__dict__,
            }
            for v in family.variants
        ],
    }


LORA_WEIGHT: JsonSchema = {"type": "number", "minimum": -2, "maximum": 2}
SHA_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
FIT_MODES = ("crop", "pad", "stretch")
# Modes that start from a source image.
SOURCE_MODES = frozenset({"i2i", "i2v", "edit", "inpaint", "outpaint"})


def find_variant(family: FamilyDescriptor, variant: str, mode: str) -> Variant:
    v = next((v for v in family.variants if v.id == variant), None)
    if v is None:
        raise SpecError(f"Unknown {family.label} variant {variant!r}")
    if mode not in v.modes:
        raise SpecError(f"{v.label} can't do {mode!r}")
    return v


def validate_model(model: Any, variant: Variant) -> dict[str, Any]:
    if not isinstance(model, dict) or not isinstance(model.get("path"), str) or not model["path"]:
        raise SpecError("A model is required")
    path: str = model["path"]
    if variant.model_dir and not (
        path == variant.model_dir or path.startswith(variant.model_dir + "/")
    ):
        raise SpecError(f"{path} isn't a {variant.label} model")
    return {"path": path, "size": model.get("size")}


def validate_inputs(inputs: Any, mode: str) -> dict[str, Any]:
    """Source-image inputs: `{source, fit, extends?}`. Transforms are recorded by the server."""
    if mode not in SOURCE_MODES:
        return {}
    inputs = inputs if isinstance(inputs, dict) else {}
    source = inputs.get("source")
    if not isinstance(source, str) or not SHA_REF.match(source):
        raise SpecError("Choose a source image")
    fit = inputs.get("fit") or "crop"
    if fit not in FIT_MODES:
        raise SpecError(f"fit: must be one of {', '.join(FIT_MODES)}")
    out: dict[str, Any] = {"source": source, "fit": fit}
    if mode == "inpaint":
        mask = inputs.get("mask")
        if not isinstance(mask, str) or not SHA_REF.match(mask):
            raise SpecError("Paint the area to redraw")
        out["mask"] = mask
    if mode == "outpaint":
        out["place"] = inputs.get("place")
    extends = inputs.get("extends")
    if extends is not None:
        if not isinstance(extends, str) or not SHA_REF.match(extends):
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
        if not isinstance(ref, str) or not SHA_REF.match(ref):
            raise SpecError(f"Image {n}: expected a sha256 reference")
    return list(refs)


MIN_PLACE = 64


def validate_place(place: Any, params: dict[str, Any]) -> dict[str, int]:
    """Where an outpaint puts its source on the canvas: `{x, y, w, h}` in canvas pixels."""
    if not isinstance(place, dict):
        raise SpecError("Place the image on the canvas")
    out: dict[str, int] = {}
    for key in ("x", "y", "w", "h"):
        value = place.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SpecError(f"place: {key} must be a number")
        out[key] = round(value)
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


def spec_assets(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The Drive assets a validated spec needs on the GPU: `[{path, size, kind}]`."""
    assets = [{**spec["model"], "kind": "model"}]
    for kind in ("config", "vae"):
        if spec.get(kind):
            assets.append({**spec[kind], "kind": kind})
    for lora in spec.get("loras") or []:
        for part in lora_files(lora):
            assets.append({"path": part["path"], "size": part.get("size"), "kind": "lora"})
    for unit in spec.get("control") or []:
        net = unit["controlnet"]
        if all(a["path"] != net["path"] for a in assets):
            assets.append({"path": net["path"], "size": net.get("size"), "kind": "controlnet"})
    for unit in spec.get("image_prompts") or []:
        adapter = unit["adapter"]
        if all(a["path"] != adapter["path"] for a in assets):
            assets.append(
                {"path": adapter["path"], "size": adapter.get("size"), "kind": "ip_adapter"}
            )
    if spec.get("image_encoder"):
        assets.append({**spec["image_encoder"], "kind": "image_encoder"})
    if spec.get("face_detector"):
        assets.append({**spec["face_detector"], "kind": "preprocessor"})
    return assets


# -- ControlNet units (design §4.3) ----------------------------------------------------------

TRACES = ("depth", "pose", "canny")  # preprocessors whose output is a control image
CONTROL_SCALE: JsonSchema = {"type": "number", "minimum": 0, "maximum": 2}
CONTROL_FRACTION: JsonSchema = {"type": "number", "minimum": 0, "maximum": 1}


def _sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or not SHA_REF.match(value):
        raise SpecError(f"{what}: expected a sha256 reference")
    return value


def validate_control(control: Any, family: str, limit: int) -> list[dict[str, Any]]:
    """Validate ControlNet units:
    `[{controlnet: {path}, image, fit?, scale, start, end, mask?, preprocessor?}]`."""
    if control is None:
        return []
    if not isinstance(control, list):
        raise SpecError("control: expected a list")
    if len(control) > limit:
        raise SpecError(f"At most {limit} ControlNets can be used at once")
    folder = f"controlnets/{family}/"
    out: list[dict[str, Any]] = []
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
        out.append({"controlnet": {"path": path, "size": size}, **_control_unit(n, unit)})
    return out


def _control_unit(n: int, unit: dict[str, Any]) -> dict[str, Any]:
    """A unit's image, fit, weight, step range, area and preprocessor record."""
    if not isinstance(unit.get("image"), str) or not unit["image"]:
        raise SpecError(f"ControlNet {n}: choose a control image")
    entry: dict[str, Any] = {
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


def validate_image_prompts(prompts: Any, family: str, limit: int) -> list[dict[str, Any]]:
    """Validate image prompts:
    `[{adapter: {path}, images: [sha…], fit?, purpose, weight, start, end, mask?}]`."""
    if prompts is None:
        return []
    if not isinstance(prompts, list):
        raise SpecError("image_prompts: expected a list")
    if len(prompts) > limit:
        raise SpecError(f"At most {limit} image prompts can be used at once")
    return [_image_prompt(n, unit, f"ip_adapters/{family}/") for n, unit in enumerate(prompts, 1)]


def _image_prompt(n: int, unit: Any, folder: str) -> dict[str, Any]:
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
    entry: dict[str, Any] = {
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
    for key, schema in (("structure", FACE_STRUCTURE), ("lora_weight", FACE_LORA)):
        if unit.get(key) is not None:
            entry[key] = _coerce(key, schema, unit[key])
    return entry


def lora_files(lora: dict[str, Any]) -> list[dict[str, Any]]:
    """The file entries of a LoRA: itself, or the high/low halves of a Wan A14B pair."""
    if "high" in lora or "low" in lora:
        return [part for part in (lora.get("high"), lora.get("low")) if part]
    return [lora]


def validate_single_loras(loras: Any, limit: int) -> list[dict[str, Any]]:
    """Validate a list of single-file LoRAs: `[{path, weight}]`."""
    if loras is None:
        return []
    if not isinstance(loras, list):
        raise SpecError("loras: expected a list")
    if len(loras) > limit:
        raise SpecError(f"At most {limit} LoRAs can be applied at once")
    out: list[dict[str, Any]] = []
    for lora in loras:
        if not isinstance(lora, dict) or not isinstance(lora.get("path"), str):
            raise SpecError("Each LoRA needs a path")
        if any(o["path"] == lora["path"] for o in out):
            raise SpecError(f"LoRA {lora['path']} is listed twice")
        weight = _coerce("LoRA weight", LORA_WEIGHT, lora.get("weight", 1.0))
        out.append({"path": lora["path"], "weight": weight, "size": lora.get("size")})
    return out


def validate_paired_loras(loras: Any, limit: int) -> list[dict[str, Any]]:
    """Validate Wan A14B LoRAs: `[{high: {path, weight}, low: {path, weight}}]`.

    Either half may be left out, for a LoRA trained for one expert only.
    """
    if loras is None:
        return []
    if not isinstance(loras, list):
        raise SpecError("loras: expected a list")
    if len(loras) > limit:
        raise SpecError(f"At most {limit} LoRAs can be applied at once")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lora in loras:
        if not isinstance(lora, dict) or not (lora.get("high") or lora.get("low")):
            raise SpecError("Each LoRA needs a high-noise or low-noise file")
        entry: dict[str, Any] = {}
        for half in ("high", "low"):
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
