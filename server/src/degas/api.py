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
from degas.families.base import SpecError, describe
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
    model = svc.db.get_asset(spec["model"]["path"])
    if model is None or model["kind"] != "model" or model["family"] != family.id:
        raise HTTPException(400, f"Model {spec['model']['path']} is not in the Drive index")
    spec["model"]["size"] = model["size"]

    seed = spec["params"].pop("seed", -1)
    base = seed if seed is not None and seed >= 0 else svc.rng.randrange(SEED_MAX)
    if body.seed_mode == "random":
        seeds = [base, *(svc.rng.randrange(SEED_MAX) for _ in range(body.batch_count - 1))]
    else:
        seeds = [(base + i) % SEED_MAX for i in range(body.batch_count)]
    job = svc.dispatcher.submit(spec, seeds)
    return svc.dispatcher.describe(job)


@router.delete("/jobs/{job_id}")
async def cancel_job(svc: Svc, job_id: str) -> dict[str, Any]:
    if svc.db.get_job(job_id) is None:
        raise HTTPException(404, "Unknown job")
    cancelled = await svc.dispatcher.cancel(job_id)
    return {"cancelled": cancelled}


# -- results and blobs ---------------------------------------------------------------------


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


@router.get("/blobs/{sha}")
async def get_blob(svc: Svc, sha: str) -> FileResponse:
    path = svc.blobs.path(sha)
    if path is None:
        raise HTTPException(404, "No such blob")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/thumbs/{sha}")
async def get_thumb(svc: Svc, sha: str) -> FileResponse:
    path = svc.blobs.thumb(sha)
    if path is None:
        raise HTTPException(404, "No such blob")
    return FileResponse(
        path,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# -- events --------------------------------------------------------------------------------


@router.get("/events", response_class=EventSourceResponse)
async def events(svc: Svc) -> AsyncIterator[dict[str, Any]]:
    yield {"type": "hello", "version": __version__}
    async for event in svc.bus.subscribe():
        yield event
