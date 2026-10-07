"""How Degas's errors answer a request: each kind has one HTTP status, and its message is the
response's `detail`."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from degas.civitai.civarchive import CivArchiveError
from degas.civitai.client import CivitaiError
from degas.civitai.huggingface import HuggingFaceError
from degas.civitai.importer import ImportBusyError
from degas.civitai.plan import PlanError
from degas.colab.session import SessionError
from degas.drive import DriveError
from degas.errors import DegasError
from degas.families.base import SpecError
from degas.media import MediaError
from degas.preprocess import PreprocessError
from degas.rclone import RcloneError

# Errors not listed here are bugs or outages a request can't fix, and stay a 500.
STATUS: dict[type[DegasError], int] = {
    MediaError: 400,
    SpecError: 400,
    SessionError: 409,
    ImportBusyError: 409,
    CivitaiError: 422,
    CivArchiveError: 422,
    HuggingFaceError: 422,
    PlanError: 422,
    DriveError: 502,
    RcloneError: 502,
}


async def _answer(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, PreprocessError):
        status = exc.status
    else:
        status = next(STATUS[kind] for kind in type(exc).__mro__ if kind in STATUS)
    return JSONResponse({"detail": str(exc)}, status_code=status)


def register_error_handlers(app: FastAPI) -> None:
    for kind in (*STATUS, PreprocessError):
        app.add_exception_handler(kind, _answer)
