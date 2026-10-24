"""FastAPI application: REST API and static PWA."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from degas import __version__
from degas.api import register_error_handlers, router
from degas.config import Config, load_config
from degas.mcp_tools import build_mcp, security
from degas.services import Services, build_services


def create_app(
    config: Config | None = None,
    services: Callable[[Config], Services] = build_services,
) -> FastAPI:
    config = config or load_config()
    mcp = build_mcp(lambda: app.state.services)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp", stateless_http=True, transport_security=security()
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = services(config)
        app.state.services = svc
        await svc.start()
        try:
            # A mounted app's own lifespan doesn't run.
            async with mcp.session_manager.run():
                yield
        finally:
            await svc.stop()

    app = FastAPI(title="Degas", version=__version__, lifespan=lifespan)
    app.include_router(router)
    register_error_handlers(app)
    # The route itself, not a mount, so `/mcp` needs no redirect to `/mcp/`.
    app.router.routes.extend(mcp_app.routes)
    if config.web_dist_dir.is_dir():
        app.mount("/", StaticFiles(directory=config.web_dist_dir, html=True), name="web")
    return app
