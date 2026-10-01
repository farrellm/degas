"""FastAPI application: REST API and static PWA."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from degas import __version__
from degas.api import register_error_handlers, router
from degas.config import Config, load_config
from degas.services import Services, build_services


def create_app(
    config: Config | None = None,
    services: Callable[[Config], Services] = build_services,
) -> FastAPI:
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services(config)
        app.state.services = svc
        await svc.start()
        try:
            yield
        finally:
            await svc.stop()

    app = FastAPI(title="Degas", version=__version__, lifespan=lifespan)
    app.include_router(router)
    register_error_handlers(app)
    if config.web_dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=config.web_dist_dir, html=True), name="web")
    return app
