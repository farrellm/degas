"""Model families and the parameter schemas the Create form is rendered from."""

from typing import Any

from fastapi import APIRouter, HTTPException

from degas.families import FAMILIES
from degas.families.base import describe

router = APIRouter(tags=["families"])


@router.get("/families")
async def families() -> list[dict[str, Any]]:
    return [describe(f) for f in FAMILIES.values()]


@router.get("/families/{family_id}/schema")
async def family_schema(family_id: str, variant: str, mode: str) -> dict[str, Any]:
    family = FAMILIES.get(family_id)
    if family is None:
        raise HTTPException(404, "Unknown family")
    return family.param_schema(variant, mode)
