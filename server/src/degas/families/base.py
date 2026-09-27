"""Family descriptors: what a model family offers and how its job specs are validated."""

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
        "variants": [
            {
                "id": v.id,
                "label": v.label,
                "min_gpu": v.min_gpu,
                "modes": list(v.modes),
                "size_constraints": family.size_constraints(v.id).__dict__,
            }
            for v in family.variants
        ],
    }


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
    return value
