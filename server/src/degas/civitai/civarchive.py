"""CivArchive (civarchive.com), an archive of Civitai's models, for LoRAs Civitai took down.

`GET /api/models/<id>?modelVersionId=<v>` gives a model and one of its versions, shaped much
like Civitai's API, with each file's SHA-256 and its mirrors: Civitai itself (gone, for a
deleted model), copies in Hugging Face repos, and other sites. `civitai_version` turns it into
Civitai's shape, so `plan_import` plans it, and a file's live mirrors are tried in turn.
"""

import re
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2

from degas.civitai import client, huggingface
from degas.errors import DegasError

CHUNK = 1 << 20
MAX_PREVIEW_BYTES = 32 * 1024 * 1024
HOSTS = ("civarchive.com", "civitaiarchive.com")


class CivArchiveError(DegasError):
    pass


def _host(url: str) -> str:
    parts = urllib.parse.urlsplit(url if "://" in url else f"https://{url}")
    return (parts.hostname or "").removeprefix("www.")


def is_civarchive_link(ref: str) -> bool:
    return _host(ref.strip()) in HOSTS


def parse_civarchive_ref(ref: str) -> tuple[int, int | None]:
    """(model id, version id or None) from a CivArchive model link."""
    ref = ref.strip()
    url = urllib.parse.urlsplit(ref if "://" in ref else f"https://{ref}")
    if _host(ref) not in HOSTS:
        raise CivArchiveError(f"Not a CivArchive link: {ref}")
    if m := re.match(r"/models/(\d+)", url.path):
        version = urllib.parse.parse_qs(url.query).get("modelVersionId", [""])[0]
        return int(m[1]), int(version) if version.isdigit() else None
    raise CivArchiveError(f"Couldn't find a model in {ref}")


def live_mirrors(file: dict[str, Any]) -> list[str]:
    """A file's download links that still work and need no payment: Civitai's first, then
    Hugging Face's, in CivArchive's order. Other sites' links aren't direct downloads."""
    found = [
        str(m["url"])
        for m in file.get("mirrors") or []
        if not m.get("deletedAt")
        and not m.get("deleted_at")
        and not m.get("is_paid")
        and not m.get("is_gated")
        and str(m.get("url", "")).startswith("https://")
    ]
    civitai = [u for u in found if _host(u) in client.HOSTS]
    hf = [u for u in found if _host(u) in huggingface.HOSTS]
    return civitai + hf


def civitai_version(data: dict[str, Any]) -> dict[str, Any]:
    """A CivArchive model (with its `version`) as Civitai's `GET /model-versions/<id>` would
    give it. Each file's `downloadUrl` is its first live mirror, and `mirrors` all of them."""
    v = data.get("version") or {}
    files = []
    for f in v.get("files") or []:
        mirrors = live_mirrors(f)
        files.append(
            {
                "name": f.get("name"),
                "type": f.get("type"),
                "sizeKB": f.get("sizeKB"),
                "primary": f.get("is_primary"),
                "hashes": {"SHA256": f.get("sha256")},
                "downloadUrl": mirrors[0] if mirrors else None,
                "mirrors": mirrors,
            }
        )
    images = [
        {"url": i.get("image_url") or i.get("url")}
        for i in v.get("images") or []
        if i.get("image_url") or i.get("url")
    ]
    return {
        "id": int(v["id"]),
        "modelId": int(v.get("modelId") or data["id"]),
        "name": v.get("name"),
        "baseModel": v.get("baseModel"),
        "model": {"name": data.get("name"), "type": data.get("type")},
        "trainedWords": [w for w in v.get("trigger") or [] if isinstance(w, str)],
        "images": images,
        "files": files,
    }


class CivArchive:
    def __init__(
        self, api_base: str = "https://civarchive.com", http: httpx2.AsyncClient | None = None
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self._http = http or httpx2.AsyncClient(timeout=httpx2.Timeout(60, read=120))

    def page(self, model_id: int, version_id: int) -> str:
        return f"https://civarchive.com/models/{model_id}?modelVersionId={version_id}"

    async def model(self, model_id: int, version_id: int | None = None) -> dict[str, Any]:
        """A model with one `version` (its newest, if `version_id` is None) and the ids and
        names of all its `versions`."""
        params = {"modelVersionId": str(version_id)} if version_id is not None else {}
        try:
            resp = await self._http.get(f"{self.api_base}/api/models/{model_id}", params=params)
        except httpx2.HTTPError as e:
            raise CivArchiveError(f"CivArchive didn't answer: {e}") from None
        if resp.status_code == 404:
            raise CivArchiveError("CivArchive has no such model or version")
        if resp.status_code >= 400:
            raise CivArchiveError(f"CivArchive API error {resp.status_code}: {resp.text[:200]}")
        data: dict[str, Any] = resp.json()
        version = data.get("version") or {}
        if version_id is not None and str(version.get("id")) != str(version_id):
            raise CivArchiveError(f"Version {version_id} isn't a version of model {model_id}")
        return data

    @asynccontextmanager
    async def download(self, url: str) -> AsyncIterator[tuple[int | None, AsyncIterator[bytes]]]:
        """Stream a file from a site that needs no token (an example image)."""
        try:
            async with self._http.stream("GET", url, follow_redirects=True) as resp:
                if resp.status_code >= 400:
                    raise CivArchiveError(f"Download failed with HTTP {resp.status_code}")
                length = resp.headers.get("content-length")
                yield (int(length) if length else None), resp.aiter_bytes(CHUNK)
        except httpx2.HTTPError as e:
            raise CivArchiveError(f"Download failed: {e}") from None

    async def fetch(self, url: str, limit: int = MAX_PREVIEW_BYTES) -> bytes:
        async with self.download(url) as (_size, chunks):
            data = bytearray()
            async for chunk in chunks:
                data += chunk
                if len(data) > limit:
                    raise CivArchiveError(f"{url} is larger than {limit} bytes")
            return bytes(data)

    async def aclose(self) -> None:
        await self._http.aclose()
