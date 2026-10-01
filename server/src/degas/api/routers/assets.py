"""Drive assets: the index, rescans, deleting LoRAs, and importing them from Civitai."""

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from degas.api.deps import Svc
from degas.api.schemas import CivitaiImport

router = APIRouter(tags=["assets"])


@router.get("/assets")
async def assets(
    svc: Svc, family: str | None = None, kind: str | None = None
) -> list[dict[str, Any]]:
    return svc.db.list_assets(family, kind)


@router.post("/assets/rescan")
async def rescan(svc: Svc) -> dict[str, Any]:
    count = await svc.rescan()
    return {"count": count, "indexed_at": svc.db.get_setting("drive.indexed_at")}


@router.delete("/assets")
async def delete_assets(svc: Svc, path: Annotated[list[str], Query()]) -> dict[str, Any]:
    """Move LoRAs to Drive's trash (both halves of a pair in one call)."""
    for p in path:
        asset = svc.db.get_asset(p)
        if asset is None:
            raise HTTPException(404, f"{p} isn't in the Drive index")
        if asset["kind"] != "lora":
            raise HTTPException(400, "Only LoRAs can be deleted from Degas")
    await svc.delete_loras(path)
    return {"deleted": path}


@router.post("/civitai/plan")
async def civitai_plan(body: CivitaiImport, svc: Svc) -> dict[str, Any]:
    """What importing a link would do, without doing it."""
    plan = await svc.imports.plan(body.url, **body.options())
    return plan.to_dict()


@router.post("/civitai/import", status_code=202)
async def civitai_import(body: CivitaiImport, svc: Svc) -> dict[str, Any]:
    """Start copying a LoRA from Civitai into Drive; `import` events follow it."""
    return await svc.imports.start(body.url, **body.options())


@router.get("/civitai/import")
async def civitai_import_state(svc: Svc) -> dict[str, Any] | None:
    return svc.imports.current


@router.get("/drive")
async def drive_status(svc: Svc) -> dict[str, Any]:
    return {**svc.drive.status(), "indexed_at": svc.db.get_setting("drive.indexed_at")}
