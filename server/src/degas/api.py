"""REST API under /api (design §7)."""

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel, Field

from degas import __version__
from degas.colab.session import SessionError
from degas.drive import DriveError
from degas.families import FAMILIES
from degas.families.base import SpecError, describe, lora_files, spec_assets
from degas.inputs import MAX_UPLOAD_BYTES
from degas.library import input_blobs, release, saved_config
from degas.media import MediaError
from degas.preprocess import PreprocessError
from degas.preprocess import run as run_preprocess
from degas.services import Services

router = APIRouter(prefix="/api")

SEED_MAX = 2**32


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


Svc = Annotated[Services, Depends(services)]


class StartSession(BaseModel):
    gpu: str
    high_mem: bool = False


class SubmitJob(BaseModel):
    spec: dict[str, Any]
    batch_count: int = Field(1, ge=1, le=16)
    seed_mode: Literal["increment", "random"] = "increment"


class MoveJob(BaseModel):
    position: Annotated[int, Field(ge=0)]  # index in the queue; 0 runs next


class PushKeys(BaseModel):
    p256dh: Annotated[str, Field(min_length=1, max_length=200)]
    auth: Annotated[str, Field(min_length=1, max_length=100)]


class PushSubscription(BaseModel):
    endpoint: Annotated[str, Field(pattern=r"^https://", max_length=2000)]
    keys: PushKeys


class PushEndpoint(BaseModel):
    endpoint: str


Tags = Annotated[list[Annotated[str, Field(min_length=1, max_length=40)]], Field(max_length=20)]


class FromUrl(BaseModel):
    url: Annotated[str, Field(min_length=1, max_length=20_000_000)]


class Transform(BaseModel):
    ops: list[dict[str, Any]]


class RemapMask(BaseModel):
    source: str  # the image the mask was painted on
    to: str  # another crop of the same original


class Preprocess(BaseModel):
    id: str
    image: str
    params: dict[str, Any] = {}


class Frame(BaseModel):
    at: Literal["first", "last"] | Annotated[float, Field(ge=0)] = "first"


class LibraryEdit(BaseModel):
    title: Annotated[str, Field(max_length=200)] | None = None
    tags: Tags | None = None


class NewPrompt(BaseModel):
    name: Annotated[str, Field(max_length=200)] = ""
    prompt: str
    negative_prompt: str = ""
    family: str | None = None
    tags: Tags = []


class PromptEdit(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    tags: Tags | None = None


PROMPT_NAME_WORDS = 6


def prompt_name(prompt: str) -> str:
    """A saved prompt's default name: its first few words."""
    words = prompt.replace(",", " ").split()
    name = " ".join(words[:PROMPT_NAME_WORDS])
    return name + ("…" if len(words) > PROMPT_NAME_WORDS else "")


def _clean_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in (t.strip() for t in tags):
        if tag and tag.lower() not in (o.lower() for o in out):
            out.append(tag)
    return out


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


# -- families ------------------------------------------------------------------------------


@router.get("/families")
async def families() -> list[dict[str, Any]]:
    return [describe(f) for f in FAMILIES.values()]


@router.get("/families/{family_id}/schema")
async def family_schema(family_id: str, variant: str, mode: str) -> dict[str, Any]:
    family = FAMILIES.get(family_id)
    if family is None:
        raise HTTPException(404, "Unknown family")
    try:
        return family.param_schema(variant, mode)
    except SpecError as e:
        raise HTTPException(400, str(e)) from None


# -- Drive assets --------------------------------------------------------------------------


@router.get("/assets")
async def assets(
    svc: Svc, family: str | None = None, kind: str | None = None
) -> list[dict[str, Any]]:
    return svc.db.list_assets(family, kind)


@router.post("/assets/rescan")
async def rescan(svc: Svc) -> dict[str, Any]:
    try:
        count = await svc.rescan()
    except DriveError as e:
        raise HTTPException(502, str(e)) from None
    return {"count": count, "indexed_at": svc.db.get_setting("drive.indexed_at")}


@router.get("/drive")
async def drive_status(svc: Svc) -> dict[str, Any]:
    return {**svc.drive.status(), "indexed_at": svc.db.get_setting("drive.indexed_at")}


# -- session -------------------------------------------------------------------------------


@router.get("/session")
async def get_session(svc: Svc) -> dict[str, Any]:
    return svc.sessions.snapshot()


@router.post("/session", status_code=202)
async def start_session(svc: Svc, body: StartSession) -> dict[str, Any]:
    try:
        return await svc.sessions.start(body.gpu, body.high_mem)
    except SessionError as e:
        raise HTTPException(409, str(e)) from None


@router.delete("/session")
async def stop_session(svc: Svc) -> dict[str, Any]:
    await svc.sessions.stop()
    return svc.sessions.snapshot()


@router.post("/session/touch")
async def touch_session(svc: Svc) -> dict[str, Any]:
    svc.sessions.touch()
    return svc.sessions.snapshot()


@router.post("/session/reset-worker", status_code=202)
async def reset_worker(svc: Svc) -> dict[str, Any]:
    try:
        await svc.sessions.reset_worker()
    except SessionError as e:
        raise HTTPException(409, str(e)) from None
    return svc.sessions.snapshot()


# -- jobs ----------------------------------------------------------------------------------


@router.get("/jobs")
async def list_jobs(svc: Svc) -> list[dict[str, Any]]:
    return [svc.dispatcher.describe(j) for j in svc.db.list_jobs()]


@router.post("/jobs", status_code=201)
async def submit_job(svc: Svc, body: SubmitJob) -> dict[str, Any]:
    family = FAMILIES.get(str(body.spec.get("family")))
    if family is None:
        raise HTTPException(400, "Unknown family")
    try:
        spec = family.validate(body.spec)
    except SpecError as e:
        raise HTTPException(400, str(e)) from None
    _resolve_assets(svc, family.id, spec)
    try:
        await svc.inputs.resolve(spec)
    except MediaError as e:
        raise HTTPException(400, str(e)) from None

    seed = spec["params"].pop("seed", -1)
    base = seed if seed is not None and seed >= 0 else svc.rng.randrange(SEED_MAX)
    if body.seed_mode == "random":
        seeds = [base, *(svc.rng.randrange(SEED_MAX) for _ in range(body.batch_count - 1))]
    else:
        seeds = [(base + i) % SEED_MAX for i in range(body.batch_count)]
    job = svc.dispatcher.submit(spec, seeds)
    return svc.dispatcher.describe(job)


def _resolve_assets(svc: Services, family: str, spec: dict[str, Any]) -> None:
    """Check that every asset the spec names is in the Drive index, and record its size."""
    sizes: dict[str, int | None] = {}
    for need in spec_assets(spec):
        asset = svc.db.get_asset(need["path"])
        if asset is None or asset["kind"] != need["kind"] or asset["family"] != family:
            what = {"model": "Model", "lora": "LoRA", "controlnet": "ControlNet"}[need["kind"]]
            raise HTTPException(400, f"{what} {need['path']} is not in the Drive index")
        sizes[need["path"]] = asset["size"]
    spec["model"]["size"] = sizes[spec["model"]["path"]]
    for lora in spec["loras"]:
        for part in lora_files(lora):
            part["size"] = sizes[part["path"]]
    for unit in spec.get("control") or []:
        unit["controlnet"]["size"] = sizes[unit["controlnet"]["path"]]


@router.delete("/jobs/{job_id}")
async def cancel_job(svc: Svc, job_id: str) -> dict[str, Any]:
    if svc.db.get_job(job_id) is None:
        raise HTTPException(404, "Unknown job")
    cancelled = await svc.dispatcher.cancel(job_id)
    return {"cancelled": cancelled}


@router.patch("/jobs/{job_id}")
async def move_job(svc: Svc, job_id: str, body: MoveJob) -> dict[str, Any]:
    """Reorder the queue."""
    if svc.db.get_job(job_id) is None:
        raise HTTPException(404, "Unknown job")
    if not svc.dispatcher.move(job_id, body.position):
        raise HTTPException(409, "Only a queued job can be moved")
    job = svc.db.get_job(job_id)
    assert job is not None
    return svc.dispatcher.describe(job)


@router.post("/jobs/{job_id}/restore")
async def restore_job(svc: Svc, job_id: str) -> dict[str, Any]:
    """Undo cancelling a queued job."""
    if svc.db.get_job(job_id) is None:
        raise HTTPException(404, "Unknown job")
    if not svc.dispatcher.restore(job_id):
        raise HTTPException(409, "Only a job cancelled before it started can be restored")
    job = svc.db.get_job(job_id)
    assert job is not None
    return svc.dispatcher.describe(job)


# -- results -----------------------------------------------------------------------------


@router.get("/results")
async def list_results(
    svc: Svc,
    cursor: str | None = None,
    job: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 60,
) -> dict[str, Any]:
    results = svc.db.list_results(cursor, limit, job)
    specs: dict[str, Any] = {}
    for r in results:
        if r["job_id"] not in specs:
            j = svc.db.get_job(r["job_id"])
            specs[r["job_id"]] = j["spec"] if j else None
        r["spec"] = specs[r["job_id"]]
    next_cursor = results[-1]["created_at"] if len(results) == limit else None
    return {"results": results, "cursor": next_cursor}


@router.post("/results/{result_id}/save")
async def save_result(svc: Svc, result_id: str) -> dict[str, Any]:
    """Keep a result in the library, with the config that reproduces it."""
    result = svc.db.get_result(result_id)
    if result is None:
        raise HTTPException(404, "Unknown result")
    existing = svc.db.library_item_for_result(result_id)
    if existing is not None:
        return existing
    job = svc.db.get_job(result["job_id"])
    if job is None:
        raise HTTPException(409, "The job for this result is gone")
    config = saved_config(job, result)
    item = svc.db.insert_library_item(result, config, input_blobs(config))
    svc.bus.publish({"type": "library", "result": result_id, "item": item["id"]})
    return item


@router.post("/results/{result_id}/extend")
async def extend_result(svc: Svc, result_id: str) -> dict[str, Any]:
    """A spec that continues this clip from its last frame, for editing in Create."""
    result = svc.db.get_result(result_id)
    if result is None:
        raise HTTPException(404, "Unknown result")
    job = svc.db.get_job(result["job_id"])
    if job is None:
        raise HTTPException(409, "The job for this result is gone")
    return await _extend(svc, result["blob_sha"], job["spec"])


async def _extend(svc: Services, sha: str, spec: dict[str, Any]) -> dict[str, Any]:
    family = FAMILIES.get(str(spec.get("family")))
    if family is None:
        raise HTTPException(400, "Unknown family")
    try:
        return await svc.inputs.extend(family, sha, spec)
    except MediaError as e:
        raise HTTPException(400, str(e)) from None


# -- library -------------------------------------------------------------------------------


@router.get("/library")
async def list_library(
    svc: Svc,
    q: str | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 60,
) -> dict[str, Any]:
    items = svc.db.list_library(q, cursor, limit)
    next_cursor = items[-1]["created_at"] if len(items) == limit else None
    return {"items": items, "cursor": next_cursor}


def _library_item(svc: Services, item_id: str) -> dict[str, Any]:
    item = svc.db.get_library_item(item_id)
    if item is None:
        raise HTTPException(404, "Unknown library item")
    return item


@router.get("/library/{item_id}")
async def get_library_item(svc: Svc, item_id: str) -> dict[str, Any]:
    return _library_item(svc, item_id)


@router.patch("/library/{item_id}")
async def edit_library_item(svc: Svc, item_id: str, body: LibraryEdit) -> dict[str, Any]:
    _library_item(svc, item_id)
    fields: dict[str, Any] = {}
    if body.title is not None:
        fields["title"] = body.title.strip() or None
    if body.tags is not None:
        fields["tags"] = _clean_tags(body.tags)
    svc.db.update_library_item(item_id, **fields)
    svc.bus.publish({"type": "library", "item": item_id})
    return _library_item(svc, item_id)


@router.post("/library/{item_id}/extend")
async def extend_library_item(svc: Svc, item_id: str) -> dict[str, Any]:
    item = _library_item(svc, item_id)
    return await _extend(svc, item["blob_sha"], item["config"])


@router.delete("/library/{item_id}")
async def delete_library_item(svc: Svc, item_id: str) -> dict[str, Any]:
    item = _library_item(svc, item_id)
    release(svc.db, svc.blobs, svc.db.delete_library_item(item_id))
    svc.bus.publish({"type": "library", "result": item["source_result_id"], "item": item_id})
    return {"deleted": True}


# -- saved prompts -------------------------------------------------------------------------


@router.get("/prompts")
async def list_prompts(svc: Svc, q: str | None = None) -> list[dict[str, Any]]:
    return svc.db.list_prompts(q)


@router.post("/prompts", status_code=201)
async def save_prompt(svc: Svc, body: NewPrompt) -> dict[str, Any]:
    if not body.prompt.strip():
        raise HTTPException(400, "The prompt is empty")
    saved = svc.db.insert_prompt(
        body.name.strip() or prompt_name(body.prompt),
        body.prompt,
        body.negative_prompt,
        body.family,
        _clean_tags(body.tags),
    )
    svc.bus.publish({"type": "prompts"})
    return saved


@router.patch("/prompts/{prompt_id}")
async def edit_prompt(svc: Svc, prompt_id: str, body: PromptEdit) -> dict[str, Any]:
    if svc.db.get_prompt(prompt_id) is None:
        raise HTTPException(404, "Unknown prompt")
    fields: dict[str, Any] = {}
    if body.name is not None and body.name.strip():
        fields["name"] = body.name.strip()
    if body.tags is not None:
        fields["tags"] = _clean_tags(body.tags)
    svc.db.update_prompt(prompt_id, **fields)
    svc.bus.publish({"type": "prompts"})
    saved = svc.db.get_prompt(prompt_id)
    assert saved is not None
    return saved


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(svc: Svc, prompt_id: str) -> dict[str, Any]:
    if not svc.db.delete_prompt(prompt_id):
        raise HTTPException(404, "Unknown prompt")
    svc.bus.publish({"type": "prompts"})
    return {"deleted": True}


# -- blobs ---------------------------------------------------------------------------------


@router.post("/blobs", status_code=201)
async def upload_blob(svc: Svc, request: Request) -> dict[str, Any]:
    """Upload an image or video as the raw request body (camera roll)."""
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "The file is larger than 200 MB")
    if not data:
        raise HTTPException(400, "The upload is empty")
    try:
        return await svc.inputs.store(bytes(data), request.headers.get("content-type"))
    except MediaError as e:
        raise HTTPException(400, str(e)) from None


@router.post("/blobs/from-url", status_code=201)
async def blob_from_url(svc: Svc, body: FromUrl) -> dict[str, Any]:
    try:
        return await svc.inputs.fetch(body.url)
    except MediaError as e:
        raise HTTPException(400, str(e)) from None


@router.post("/blobs/{sha}/transform", status_code=201)
async def transform_blob(svc: Svc, sha: str, body: Transform) -> dict[str, Any]:
    if svc.blobs.path(sha) is None:
        raise HTTPException(404, "No such blob")
    try:
        return await svc.inputs.transform(sha, body.ops)
    except MediaError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/blobs/{sha}/transform")
async def get_transform(svc: Svc, sha: str) -> dict[str, Any]:
    """The original and operations a derived image was made with, to reopen the editor.

    Any other image is its own original, with no operations.
    """
    if svc.blobs.path(sha) is None:
        raise HTTPException(404, "No such blob")
    record = svc.db.get_transform(sha)
    if record is None or svc.blobs.path(record["original"]) is None:
        return {"original": sha, "ops": []}
    return record


@router.post("/blobs/{sha}/mask", status_code=201)
async def upload_mask(svc: Svc, sha: str, request: Request) -> dict[str, Any]:
    """A mask painted over image `sha` (raw PNG body; white or opaque is redrawn)."""
    data = await request.body()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The mask is larger than 200 MB")
    try:
        return await svc.inputs.store_mask(sha, data)
    except MediaError as e:
        raise HTTPException(400, str(e)) from None


@router.post("/blobs/{sha}/remap", status_code=201)
async def remap_mask(svc: Svc, sha: str, body: RemapMask) -> dict[str, Any]:
    """Carry mask `sha` from the image it was painted on to a new crop of that image."""
    try:
        return await svc.inputs.remap_mask(sha, body.source, body.to)
    except MediaError as e:
        raise HTTPException(400, str(e)) from None


@router.post("/preprocess", status_code=201)
async def preprocess(svc: Svc, body: Preprocess) -> dict[str, Any]:
    try:
        return await run_preprocess(svc, body.id, body.image, body.params)
    except PreprocessError as e:
        raise HTTPException(e.status, str(e)) from None


@router.post("/blobs/{sha}/frame", status_code=201)
async def blob_frame(svc: Svc, sha: str, body: Frame) -> dict[str, Any]:
    try:
        return await svc.inputs.frame(sha, body.at)
    except MediaError as e:
        raise HTTPException(400, str(e)) from None


@router.get("/blobs/{sha}")
async def get_blob(svc: Svc, sha: str) -> FileResponse:
    path = svc.blobs.path(sha)
    if path is None:
        raise HTTPException(404, "No such blob")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/thumbs/{sha}")
async def get_thumb(svc: Svc, sha: str) -> FileResponse:
    path = svc.blobs.thumb(sha)
    if path is None and svc.blobs.is_video(sha):
        try:
            path = await svc.inputs.poster(sha)
        except MediaError:
            path = None
    if path is None:
        raise HTTPException(404, "No such blob")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# -- Web Push ------------------------------------------------------------------------------


@router.get("/push")
async def push_key(svc: Svc) -> dict[str, Any]:
    return {"public_key": svc.push.public_key}


@router.post("/push/subscribe", status_code=201)
async def push_subscribe(svc: Svc, body: PushSubscription) -> dict[str, Any]:
    svc.push.subscribe(body.endpoint, body.keys.model_dump())
    return {"subscribed": True}


@router.post("/push/unsubscribe")
async def push_unsubscribe(svc: Svc, body: PushEndpoint) -> dict[str, Any]:
    return {"unsubscribed": svc.push.unsubscribe(body.endpoint)}


# -- events --------------------------------------------------------------------------------


@router.get("/events", response_class=EventSourceResponse)
async def events(svc: Svc) -> AsyncIterator[dict[str, Any]]:
    yield {"type": "hello", "version": __version__}
    async for event in svc.bus.subscribe():
        yield event
