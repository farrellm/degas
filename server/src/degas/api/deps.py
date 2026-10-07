"""What routes share: the services, and lookups that answer 404 for what isn't there.

Routes that return a stored row are annotated `Mapping[str, Any]`, not the row's TypedDict:
FastAPI would take the TypedDict for a response model and drop whatever a route adds to the
row.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request

from degas.db import JobRow, LibraryItem, ResultRow
from degas.db.rows import Cleared
from degas.families import FAMILIES
from degas.families.base import FamilyDescriptor
from degas.library import release
from degas.services import Services


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


Svc = Annotated[Services, Depends(services)]


def job_or_404(svc: Services, job_id: str) -> JobRow:
    job = svc.db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


def result_or_404(svc: Services, result_id: str) -> ResultRow:
    result = svc.db.get_result(result_id)
    if result is None:
        raise HTTPException(404, "Unknown result")
    return result


def library_item_or_404(svc: Services, item_id: str) -> LibraryItem:
    item = svc.db.get_library_item(item_id)
    if item is None:
        raise HTTPException(404, "Unknown library item")
    return item


def blob_or_404(svc: Services, sha: str) -> Path:
    path = svc.blobs.path(sha)
    if path is None:
        raise HTTPException(404, "No such blob")
    return path


def family_or_400(spec: Mapping[str, Any]) -> FamilyDescriptor:
    family = FAMILIES.get(str(spec.get("family")))
    if family is None:
        raise HTTPException(400, "Unknown family")
    return family


def swept(svc: Services, deleted: Cleared) -> dict[str, Any]:
    """Free the deleted results' blobs and tell the app what went."""
    release(svc.db, svc.blobs, deleted["blobs"])
    counts = {"results": deleted["results"], "jobs": deleted["jobs"]}
    svc.bus.publish({"type": "swept", **counts})
    return counts
