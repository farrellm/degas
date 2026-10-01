"""REST API under /api (design §7)."""

from fastapi import APIRouter

from degas.api.errors import register_error_handlers
from degas.api.routers import (
    assets,
    blobs,
    families,
    jobs,
    library,
    prompts,
    push,
    results,
    session,
    system,
)

router = APIRouter(prefix="/api")
for _module in (system, families, assets, session, jobs, results, library, prompts, blobs, push):
    router.include_router(_module.router)

__all__ = ["register_error_handlers", "router"]
