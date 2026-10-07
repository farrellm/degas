"""Civitai's public REST API (`/api/v1`) and its downloads.

The API token (`civitai.token_file`) goes in an `Authorization` header, never in a URL, so it
stays out of logs. httpx drops the header when a download redirects to the storage host.
"""

import re
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2

from degas.errors import DegasError

CHUNK = 1 << 20
MAX_PREVIEW_BYTES = 32 * 1024 * 1024
HOSTS = ("civitai.com", "civitai.red", "civitai.green")


class CivitaiError(DegasError):
    pass


def parse_ref(ref: str) -> tuple[int | None, int | None]:
    """(model id, version id) from a link, an AIR (`urn:air:…:civitai:<model>@<version>`)
    or a bare version id. One of the two may be None."""
    ref = ref.strip()
    if ref.isdigit():
        return None, int(ref)
    if m := re.fullmatch(r"urn:air:[^:]+:[^:]+:civitai:(\d+)(?:@(\d+))?", ref):
        return int(m[1]), int(m[2]) if m[2] else None
    url = urllib.parse.urlsplit(ref if "://" in ref else f"https://{ref}")
    host = (url.hostname or "").removeprefix("www.")
    if host not in HOSTS:
        raise CivitaiError(f"Not a Civitai link: {ref}")
    query = urllib.parse.parse_qs(url.query)
    if m := re.match(r"/api/(?:download/models|v1/model-versions)/(\d+)", url.path):
        return None, int(m[1])
    if m := re.match(r"/models/(\d+)", url.path):
        version = query.get("modelVersionId", [""])[0]
        return int(m[1]), int(version) if version.isdigit() else None
    raise CivitaiError(f"Couldn't find a model in {ref}")


def still_url(url: str, width: int = 768) -> str:
    """A resized still of an example image or video, from Civitai's image CDN."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname != "image.civitai.com":
        return url
    head, _params, name = parts.path.rsplit("/", 2)
    path = f"{head}/anim=false,transcode=true,width={width}/{Path(name).stem}.jpeg"
    return urllib.parse.urlunsplit(parts._replace(path=path, query=""))


class Civitai:
    def __init__(
        self,
        token_file: Path | None = None,
        api_base: str = "https://civitai.com",
        http: httpx2.AsyncClient | None = None,
    ) -> None:
        self.token_file = token_file.expanduser() if token_file else None
        self.api_base = api_base.rstrip("/")
        self._http = http or httpx2.AsyncClient(timeout=httpx2.Timeout(60, read=120))

    def _headers(self) -> dict[str, str]:
        if self.token_file is not None and self.token_file.exists():
            token = self.token_file.read_text().strip()
            if token:
                return {"Authorization": f"Bearer {token}"}
        return {}

    @property
    def has_token(self) -> bool:
        return bool(self._headers())

    async def _get(self, path: str, missing_ok: bool = False) -> Any:
        try:
            resp = await self._http.get(f"{self.api_base}/api/v1/{path}", headers=self._headers())
        except httpx2.HTTPError as e:
            raise CivitaiError(f"Civitai didn't answer: {e}") from None
        if resp.status_code == 404:
            if missing_ok:
                return None
            raise CivitaiError("Civitai has no such model or version")
        if resp.status_code >= 400:
            raise CivitaiError(f"Civitai API error {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    async def version(self, ref: str) -> dict[str, Any]:
        """The model version a link points to; a model link means its newest version."""
        model_id, version_id = parse_ref(ref)
        if version_id is None:
            assert model_id is not None  # parse_ref gives one or the other
            model = await self.model(model_id)
            versions = model.get("modelVersions") or []
            if not versions:
                raise CivitaiError(f"Model {model_id} has no published versions")
            version_id = int(versions[0]["id"])
        data: dict[str, Any] = await self._get(f"model-versions/{version_id}")
        if model_id is not None and data.get("modelId") != model_id:
            raise CivitaiError(f"Version {version_id} isn't a version of model {model_id}")
        return data

    async def model(self, model_id: int) -> dict[str, Any]:
        """A model with all its versions (`modelVersions`, newest first)."""
        data: dict[str, Any] = await self._get(f"models/{model_id}")
        return data

    async def by_hash(self, sha256: str) -> dict[str, Any] | None:
        data: dict[str, Any] | None = await self._get(
            f"model-versions/by-hash/{sha256}", missing_ok=True
        )
        return data

    @asynccontextmanager
    async def download(self, url: str) -> AsyncIterator[tuple[int | None, AsyncIterator[bytes]]]:
        """Stream a file: (its size if known, its chunks)."""
        try:
            async with self._http.stream(
                "GET", url, headers=self._headers(), follow_redirects=True
            ) as resp:
                if resp.status_code in (401, 403):
                    hint = "" if self.has_token else f"; put an API token in {self.token_file}"
                    raise CivitaiError(f"Civitai refused the download ({resp.status_code}){hint}")
                if resp.status_code >= 400:
                    raise CivitaiError(f"Download failed with HTTP {resp.status_code}")
                length = resp.headers.get("content-length")
                yield (int(length) if length else None), resp.aiter_bytes(CHUNK)
        except httpx2.HTTPError as e:
            raise CivitaiError(f"Download failed: {e}") from None

    async def fetch(self, url: str, limit: int = MAX_PREVIEW_BYTES) -> bytes:
        """A small file, such as an example image."""
        async with self.download(url) as (_size, chunks):
            data = bytearray()
            async for chunk in chunks:
                data += chunk
                if len(data) > limit:
                    raise CivitaiError(f"{url} is larger than {limit} bytes")
            return bytes(data)

    async def aclose(self) -> None:
        await self._http.aclose()
