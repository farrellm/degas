"""FastAPI application: REST API and static PWA."""

from fastapi import FastAPI

from degas import __version__

app = FastAPI(title="Degas", version=__version__)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
