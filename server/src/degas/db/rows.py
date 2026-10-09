"""The rows the metadata store returns: plain dicts, typed.

They are what the API sends, so a route may add to one (a result's `spec`, a job's
`progress`) but nothing here is renamed on the way out.
"""

from typing import Any, Literal, NotRequired, TypedDict

from degas_worker.spec import (
    AssetRef,
    ControlUnit,
    ImagePromptUnit,
    Lora,
    PairedLora,
    Spec,
    SpecInputs,
)

SessionState = Literal["starting", "ready", "busy", "stopping", "stopped", "error"]
JobStatus = Literal["queued", "running", "done", "error", "cancelled"]


class SessionRow(TypedDict):
    id: str
    gpu: str
    high_mem: bool
    state: SessionState
    started_at: str
    ended_at: str | None
    last_activity_at: str
    error: str | None


class SessionUpdate(TypedDict, total=False):
    """The session columns that change after it starts."""

    state: SessionState
    ended_at: str | None
    last_activity_at: str
    error: str | None


class JobRuntime(TypedDict, total=False):
    """What a job ran on, kept with saved configs (design §6.4)."""

    gpu: str | None
    diffusers: str
    torch: str
    duration_s: float


class JobRow(TypedDict):
    id: str
    session_id: str | None
    status: JobStatus
    queue_position: int
    spec: Spec
    seeds: list[int]
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    log: str | None  # a failed job's traceback from the worker
    runtime: JobRuntime | None


class JobUpdate(TypedDict, total=False):
    """The job columns that change after it is queued."""

    session_id: str | None
    status: JobStatus
    started_at: str | None
    finished_at: str | None
    error: str | None
    log: str | None
    runtime: JobRuntime | None


class SavedConfig(TypedDict):
    """The self-contained, replayable config of one result: its job's spec, pinned to its seed."""

    degas_version: int
    family: str
    variant: str
    mode: str
    model: AssetRef
    loras: list[Lora] | list[PairedLora]
    params: dict[str, Any]
    inputs: SpecInputs
    control: list[ControlUnit]
    image_prompts: NotRequired[list[ImagePromptUnit]]
    runtime: NotRequired[JobRuntime]
    segments: NotRequired[list["SavedConfig"]]  # a stitched video: the clips it chains


class ResultRow(TypedDict):
    id: str
    job_id: str
    item_index: int
    blob_sha: str
    media_type: str
    seed: int | None
    width: int | None
    height: int | None
    duration: float | None
    created_at: str
    expires_at: str | None  # set when its session ends
    segments: list[SavedConfig] | None
    library_id: str | None  # the library item that keeps it, if any


class LibraryItem(TypedDict):
    id: str
    kind: Literal["image", "video"]
    blob_sha: str
    media_type: str
    width: int | None
    height: int | None
    duration: float | None
    config: SavedConfig
    title: str | None
    tags: list[str]
    created_at: str
    source_result_id: str | None


class LibraryItemUpdate(TypedDict, total=False):
    """What can be edited on a kept item."""

    title: str | None
    tags: list[str]


class SavedPrompt(TypedDict):
    id: str
    name: str
    prompt: str
    negative_prompt: str
    family: str | None
    tags: list[str]
    created_at: str


class PromptUpdate(TypedDict, total=False):
    """What can be edited on a saved prompt."""

    name: str
    tags: list[str]


class AssetRow(TypedDict):
    """A file or folder in Drive, by its path under the Drive root."""

    path: str
    family: str | None  # None for what families share (preprocessors)
    kind: str
    drive_file_id: str
    size: int | None
    mtime: str | None
    md5: str | None
    sha256: str | None
    sidecar: dict[str, Any] | None
    sidecar_rev: str | None
    preview_thumb: str | None  # blob sha
    preview_rev: str | None
    indexed_at: str


class PushSubscriptionRow(TypedDict):
    endpoint: str
    keys: dict[str, str]
    created_at: str


class TransformRecord(TypedDict):
    """A derived image's original (a blob sha) and the operations that made it."""

    original: str
    ops: list[dict[str, Any]]


class Cleared(TypedDict):
    """What a deletion removed, and the blobs that lost a reference to it."""

    results: int
    jobs: int
    blobs: list[str]
