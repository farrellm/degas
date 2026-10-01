"""Results: listing, clearing, keeping one in the library, and extending a clip."""

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from degas.api.deps import Svc, family_or_400, job_or_404, result_or_404
from degas.db import PENDING_JOB_STATUSES
from degas.library import input_blobs, release, saved_config
from degas.services import Services

router = APIRouter(tags=["results"])


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


@router.delete("/results")
async def clear_results(svc: Svc) -> dict[str, Any]:
    """Delete finished jobs and their results now; queued and running jobs stay."""
    return _swept(svc, svc.db.clear_results())


@router.delete("/jobs/{job_id}/results")
async def delete_job_results(svc: Svc, job_id: str, chain: bool = False) -> dict[str, Any]:
    """Delete one finished job's results: its stitched chain with `chain`, else the rest."""
    job = job_or_404(svc, job_id)
    if job["status"] in PENDING_JOB_STATUSES:
        raise HTTPException(409, "Cancel the job before deleting it")
    return _swept(svc, svc.db.delete_job_results(job_id, chain))


def _swept(svc: Services, deleted: dict[str, Any]) -> dict[str, Any]:
    """Free the deleted results' blobs and tell the app what went."""
    release(svc.db, svc.blobs, deleted["blobs"])
    counts = {"results": deleted["results"], "jobs": deleted["jobs"]}
    svc.bus.publish({"type": "swept", **counts})
    return counts


@router.post("/results/{result_id}/save")
async def save_result(svc: Svc, result_id: str) -> dict[str, Any]:
    """Keep a result in the library, with the config that reproduces it."""
    result = result_or_404(svc, result_id)
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
    result = result_or_404(svc, result_id)
    job = svc.db.get_job(result["job_id"])
    if job is None:
        raise HTTPException(409, "The job for this result is gone")
    spec = job["spec"]
    return await svc.inputs.extend(family_or_400(spec), result["blob_sha"], spec)
