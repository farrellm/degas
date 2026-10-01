"""Family descriptors: what a model family offers and how its job specs are validated."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, cast

from degas.errors import DegasError
from degas_worker.spec import AssetRef, Lora, PairedLora, Spec

GPUS = ("T4", "L4", "A100", "H100")  # ascending capability
JsonSchema = dict[str, Any]
MediaKind = Literal["image", "video"]
# A LoRA is one file, or (Wan 2.2 A14B) a pair for the high- and low-noise experts.
LoraFormat = Literal["single", "paired_hi_lo"]


class SpecError(DegasError, ValueError):
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
    lora_format: LoraFormat | None = None
    # How many reference images an edit reads after its source (0: none).
    max_refs: int = 0
    # References bigger than this are scaled down to it and smaller ones kept as they are.
    # None: the pipeline scales each to the output's pixel count (Qwen).
    ref_max_pixels: int | None = None


@dataclass(frozen=True)
class ImagePromptOptions:
    """What a family's image prompts can do, for the form (docs/ip-adapter.md)."""

    purposes: tuple[str, ...] = ("all",)  # which UNet blocks a unit can act in
    areas: bool = False  # limited to an area of the output
    steps: bool = False  # limited to a range of steps
    faces: bool = False  # FaceID models
    detail: bool = False  # Redux: how many of the picture's tokens the prompt gets (`downsample`)


@dataclass(frozen=True)
class SizeConstraints:
    multiple_of: int
    min_pixels: int
    max_pixels: int
    presets: tuple[tuple[int, int], ...]


class FamilyDescriptor(Protocol):
    id: str
    label: str
    media: MediaKind
    variants: tuple[Variant, ...]
    lora_format: LoraFormat
    supports_control: bool
    supports_image_prompts: bool

    @property
    def image_prompt_options(self) -> ImagePromptOptions | None: ...

    def param_schema(self, variant: str, mode: str) -> JsonSchema: ...

    def size_constraints(self, variant: str) -> SizeConstraints: ...

    def validate(self, spec: dict[str, Any]) -> Spec:
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
        "image_prompt_options": (
            asdict(family.image_prompt_options) if family.image_prompt_options else None
        ),
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
                "size_constraints": asdict(family.size_constraints(v.id)),
            }
            for v in family.variants
        ],
    }


def find_variant(family: FamilyDescriptor, variant: str, mode: str) -> Variant:
    v = next((v for v in family.variants if v.id == variant), None)
    if v is None:
        raise SpecError(f"Unknown {family.label} variant {variant!r}")
    if mode not in v.modes:
        raise SpecError(f"{v.label} can't do {mode!r}")
    return v


class NeededAsset(AssetRef):
    kind: str  # as the Drive index names it: model, lora, controlnet …


def spec_assets(spec: Spec) -> list[NeededAsset]:
    """The Drive assets a validated spec needs on the GPU, each once."""
    assets: list[NeededAsset] = []

    def need(asset: AssetRef | None, kind: str, *, shared: bool = False) -> None:
        # Units may share a ControlNet or an image prompt model.
        if not asset or (shared and any(a["path"] == asset["path"] for a in assets)):
            return
        assets.append({"path": asset["path"], "size": asset.get("size"), "kind": kind})

    need(spec["model"], "model")
    need(spec.get("config"), "config")
    need(spec.get("vae"), "vae")
    for lora in spec.get("loras") or []:
        for part in lora_files(lora):
            need(part, "lora")
    for unit in spec.get("control") or []:
        need(unit["controlnet"], "controlnet", shared=True)
    for prompt in spec.get("image_prompts") or []:
        need(prompt["adapter"], "ip_adapter", shared=True)
    need(spec.get("image_encoder"), "image_encoder")
    need(spec.get("face_detector"), "preprocessor")
    return assets


def lora_files(lora: Lora | PairedLora) -> list[Lora]:
    """The file entries of a LoRA: itself, or the high/low halves of a Wan A14B pair."""
    if "high" in lora or "low" in lora:
        pair = cast("PairedLora", lora)
        return [part for part in (pair.get("high"), pair.get("low")) if part]
    return [cast("Lora", lora)]
