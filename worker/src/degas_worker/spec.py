"""The shape of a job spec: what the server sends, and a runner reads.

A spec has been validated and resolved by the server (`degas.families`, `degas.inputs`) by the
time a runner sees it: defaults are filled, every image is a `sha256:<hex>` blob reference
staged on the worker, and every asset is a path under the Drive root with its size.
"""

from typing import Any, NotRequired, TypedDict


class AssetRef(TypedDict):
    """A file or folder under the Drive root. The size, once the server has looked it up, lets
    the cache tell a changed file from the one it has."""

    path: str
    size: int | None


class Lora(AssetRef):
    weight: float


class PairedLora(TypedDict, total=False):
    """A Wan 2.2 A14B LoRA: a file for each expert. Either may be missing."""

    high: Lora
    low: Lora


class Transform(TypedDict):
    """How a derived image was made from its original (design §6.5)."""

    original: str
    ops: list[dict[str, Any]]


class Place(TypedDict):
    """Where an outpaint puts its source on the canvas, in canvas pixels."""

    x: int
    y: int
    w: int
    h: int


class SpecInputs(TypedDict, total=False):
    source: str
    mask: str
    refs: list[str]  # reference images after the source, in the order the prompt numbers them
    extends: str  # the clip this one continues
    place: Place
    fit: str  # how the server fits images to the output size; gone once resolved
    transforms: dict[str, Transform]  # by derived blob reference


class Trace(TypedDict):
    """The preprocessor a control image was traced with, and the image it was traced from."""

    id: str
    source: str
    params: dict[str, Any]


class ControlUnit(TypedDict):
    controlnet: AssetRef
    image: str
    scale: float
    start: float  # the share of the steps it guides, from …
    end: float  # … to
    fit: NotRequired[str]
    mask: NotRequired[str]  # the area it is limited to
    preprocessor: NotRequired[Trace]


class ImagePromptUnit(TypedDict):
    adapter: AssetRef
    images: list[str]
    purpose: str
    weight: float
    start: float
    end: float
    fit: NotRequired[str]
    mask: NotRequired[str]
    structure: NotRequired[float]  # FaceID
    lora_weight: NotRequired[float]  # FaceID
    downsample: NotRequired[int]  # Redux


class Spec(TypedDict):
    family: str
    variant: str
    mode: str
    model: AssetRef
    loras: list[Lora] | list[PairedLora]
    params: dict[str, Any]  # per the family's parameter schema
    inputs: SpecInputs
    control: list[ControlUnit]
    image_prompts: NotRequired[list[ImagePromptUnit]]
    # Assets a family adds to what the person chose.
    config: NotRequired[AssetRef]
    vae: NotRequired[AssetRef]
    image_encoder: NotRequired[AssetRef]
    face_detector: NotRequired[AssetRef]
