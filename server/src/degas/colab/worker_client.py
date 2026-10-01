"""HTTP client for the worker (through the SSH tunnel's forwarded port)."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx2

from degas.errors import DegasError


class WorkerError(DegasError):
    pass


class WorkerBusyError(WorkerError):
    pass


class WorkerClient:
    def __init__(self, base_url: str, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        self._http = httpx2.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=httpx2.Timeout(30, read=120),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = await self._http.request(method, path, **kwargs)
        except httpx2.HTTPError as e:
            raise WorkerError(f"{method} {path}: {e!r}") from e
        if resp.status_code == 409:
            raise WorkerBusyError(resp.text)
        if resp.status_code >= 400:
            raise WorkerError(f"{method} {path}: HTTP {resp.status_code} {resp.text[:300]}")
        return resp.json() if resp.content else None

    async def health(self, timeout: float = 10) -> dict[str, Any]:
        result: dict[str, Any] = await self._json("GET", "/health", timeout=timeout)
        return result

    async def state(self) -> dict[str, Any]:
        result: dict[str, Any] = await self._json("GET", "/state")
        return result

    async def has_blob(self, sha: str) -> bool:
        resp = await self._http.head(f"/blobs/{sha}")
        return resp.status_code == 200

    async def put_blob(self, sha: str, data: bytes) -> None:
        await self._json("PUT", f"/blobs/{sha}", content=data)

    async def start_job(self, job_id: str, spec: dict[str, Any], seeds: list[int]) -> None:
        await self._json("POST", "/jobs", json={"job_id": job_id, "spec": spec, "seeds": seeds})

    async def events(self, job_id: str) -> AsyncIterator[dict[str, Any]]:
        """Job events; replays from the start of the job on every call."""
        timeout = httpx2.Timeout(30, read=None)
        try:
            async with self._http.sse(f"/jobs/{job_id}/events", timeout=timeout) as source:
                if source.response.status_code >= 400:
                    await source.response.aread()
                    raise WorkerError(
                        f"events {job_id}: HTTP {source.response.status_code}"
                        f" {source.response.text[:300]}"
                    )
                async for sse in source:
                    if sse.data:
                        yield json.loads(sse.data)
        except httpx2.HTTPError as e:
            raise WorkerError(f"events {job_id}: {e!r}") from e

    async def fetch_assets(self, assets: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Copy Drive assets into the VM's cache, yielding `copy` progress, then done/error."""
        timeout = httpx2.Timeout(30, read=None)
        try:
            async with self._http.sse(
                "/assets/fetch", method="POST", json={"assets": assets}, timeout=timeout
            ) as source:
                if source.response.status_code >= 400:
                    await source.response.aread()
                    raise WorkerError(
                        f"fetch assets: HTTP {source.response.status_code}"
                        f" {source.response.text[:300]}"
                    )
                async for sse in source:
                    if sse.data:
                        yield json.loads(sse.data)
        except httpx2.HTTPError as e:
            raise WorkerError(f"fetch assets: {e!r}") from e

    async def preprocess(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run a preprocessor; the first use copies and loads its model, so allow minutes."""
        result: dict[str, Any] = await self._json(
            "POST", "/preprocess", json=body, timeout=httpx2.Timeout(30, read=600)
        )
        return result

    async def cancel(self, job_id: str) -> bool:
        body = await self._json("POST", f"/jobs/{job_id}/cancel")
        return bool(body["cancelled"])

    async def get_output(self, job_id: str, item: int) -> tuple[bytes, str]:
        try:
            resp = await self._http.get(f"/outputs/{job_id}/{item}")
        except httpx2.HTTPError as e:
            raise WorkerError(f"output {job_id}/{item}: {e!r}") from e
        if resp.status_code >= 400:
            raise WorkerError(f"output {job_id}/{item}: HTTP {resp.status_code}")
        return resp.content, resp.headers.get("content-type", "application/octet-stream")

    async def ack_output(self, job_id: str, item: int) -> None:
        await self._json("DELETE", f"/outputs/{job_id}/{item}")

    async def drive_token(self, access_token: str, expiry: str, root: str) -> None:
        await self._json(
            "POST",
            "/drive-token",
            json={"access_token": access_token, "expiry": expiry, "root": root},
        )

    async def shutdown(self) -> None:
        await self._json("POST", "/shutdown", timeout=30)
