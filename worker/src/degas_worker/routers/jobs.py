"""The generation job, its events, and the outputs it leaves until the server has them."""

from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse
from pydantic import BaseModel

from degas_worker.jobs import WorkerBusyError
from degas_worker.spec import Spec
from degas_worker.state import Worker

router = APIRouter()


class StartJob(BaseModel):
    job_id: str
    spec: dict[str, Any]
    seeds: list[int]


@router.post("/jobs", status_code=202)
async def start_job(worker: Worker, body: StartJob) -> dict[str, Any]:
    try:
        record = worker.jobs.start(body.job_id, cast("Spec", body.spec), body.seeds)
    except WorkerBusyError:
        raise HTTPException(409, "A job is already running") from None
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return record.summary()


@router.get("/jobs/{job_id}/events", response_class=EventSourceResponse)
async def job_events(worker: Worker, job_id: str) -> AsyncIterator[dict[str, Any]]:
    record = worker.jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "Unknown job")
    async for event in record.subscribe():
        yield event


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(worker: Worker, job_id: str) -> dict[str, bool]:
    return {"cancelled": worker.jobs.cancel(job_id)}


@router.get("/outputs/{job_id}/{item}")
def get_output(worker: Worker, job_id: str, item: int) -> FileResponse:
    try:
        found = worker.jobs.outputs.file(job_id, item)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if found is None:
        raise HTTPException(404, "No such output")
    return FileResponse(found[0], media_type=found[1])


@router.delete("/outputs/{job_id}/{item}", status_code=204)
def ack_output(worker: Worker, job_id: str, item: int) -> None:
    try:
        worker.jobs.outputs.ack(job_id, item)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
