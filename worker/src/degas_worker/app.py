"""Worker HTTP app, served on 127.0.0.1 inside the VM and reached via SSH tunnel."""

from fastapi import FastAPI

from degas_worker import __version__

app = FastAPI(title="Degas worker", version=__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
