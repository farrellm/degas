"""The library of kept results."""

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Query

from degas.api.deps import Svc, family_or_400, library_item_or_404
from degas.api.schemas import LibraryEdit
from degas.db import LibraryItemUpdate
from degas.library import clean_tags, release

router = APIRouter(tags=["library"])


@router.get("/library")
async def list_library(
    svc: Svc,
    q: str | None = None,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 60,
) -> Mapping[str, Any]:
    items = svc.db.list_library(q, cursor, limit)
    next_cursor = items[-1]["created_at"] if len(items) == limit else None
    return {"items": items, "cursor": next_cursor}


@router.get("/library/{item_id}")
async def get_library_item(svc: Svc, item_id: str) -> Mapping[str, Any]:
    return library_item_or_404(svc, item_id)


@router.patch("/library/{item_id}")
async def edit_library_item(svc: Svc, item_id: str, body: LibraryEdit) -> Mapping[str, Any]:
    library_item_or_404(svc, item_id)
    fields: LibraryItemUpdate = {}
    if body.title is not None:
        fields["title"] = body.title.strip() or None
    if body.tags is not None:
        fields["tags"] = clean_tags(body.tags)
    svc.db.update_library_item(item_id, **fields)
    svc.bus.publish({"type": "library", "item": item_id})
    return library_item_or_404(svc, item_id)


@router.post("/library/{item_id}/extend")
async def extend_library_item(svc: Svc, item_id: str) -> dict[str, Any]:
    item = library_item_or_404(svc, item_id)
    config = item["config"]
    return await svc.inputs.extend(family_or_400(config), item["blob_sha"], config)


@router.delete("/library/{item_id}")
async def delete_library_item(svc: Svc, item_id: str) -> dict[str, Any]:
    item = library_item_or_404(svc, item_id)
    release(svc.db, svc.blobs, svc.db.delete_library_item(item_id))
    svc.bus.publish({"type": "library", "result": item["source_result_id"], "item": item_id})
    return {"deleted": True}
