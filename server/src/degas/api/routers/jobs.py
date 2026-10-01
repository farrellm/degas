"""The job queue."""

from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import APIRouter, HTTPException

from degas.api.deps import Svc, family_or_400, job_or_404
from degas.api.schemas import MoveJob, SubmitJob
from degas.submission import batch_seeds, resolve_assets

router = APIRouter(tags=["jobs"])


@router.get("/jobs")
async def list_jobs(svc: Svc) -> Sequence[Mapping[str, Any]]:
    return [svc.dispatcher.describe(j) for j in svc.db.list_jobs()]


@router.post("/jobs", status_code=201)
async def submit_job(svc: Svc, body: SubmitJob) -> Mapping[str, Any]:
    family = family_or_400(body.spec)
    spec = family.validate(body.spec)
    resolve_assets(svc.db, family.id, spec)
    await svc.inputs.resolve(spec)
    seed = spec["params"].pop("seed", -1)
    seeds = batch_seeds(seed, body.batch_count, body.seed_mode, svc.rng)
    job = svc.dispatcher.submit(spec, seeds)
    return svc.dispatcher.describe(job)


@router.delete("/jobs/{job_id}")
async def cancel_job(svc: Svc, job_id: str) -> dict[str, Any]:
    job_or_404(svc, job_id)
    cancelled = await svc.dispatcher.cancel(job_id)
    return {"cancelled": cancelled}


@router.patch("/jobs/{job_id}")
async def move_job(svc: Svc, job_id: str, body: MoveJob) -> Mapping[str, Any]:
    """Reorder the queue."""
    job_or_404(svc, job_id)
    if not svc.dispatcher.move(job_id, body.position):
        raise HTTPException(409, "Only a queued job can be moved")
    return svc.dispatcher.describe(job_or_404(svc, job_id))


@router.post("/jobs/{job_id}/restore")
async def restore_job(svc: Svc, job_id: str) -> Mapping[str, Any]:
    """Undo cancelling a queued job."""
    job_or_404(svc, job_id)
    if not svc.dispatcher.restore(job_id):
        raise HTTPException(409, "Only a job cancelled before it started can be restored")
    return svc.dispatcher.describe(job_or_404(svc, job_id))
