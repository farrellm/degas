"""Hugging Face's Hub API (`/api/models`) and its downloads, for LoRAs published there.

The token (`huggingface.token_file`) goes in an `Authorization` header, never in a URL. A
download redirects to Hugging Face's CDN, and httpx drops the header on the way.
"""

import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2

from degas.errors import DegasError

CHUNK = 1 << 20
MAX_PREVIEW_BYTES = 32 * 1024 * 1024
HOSTS = ("huggingface.co", "hf.co")


class HuggingFaceError(DegasError):
    pass


@dataclass(frozen=True)
class HfRef:
    repo: str  # owner/name
    revision: str  # a branch, tag or commit
    path: str | None = None  # a file in the repo
    folder: str = ""  # a folder in the repo (a `tree` link)


def is_hf_link(ref: str) -> bool:
    ref = ref.strip()
    url = urllib.parse.urlsplit(ref if "://" in ref else f"https://{ref}")
    return (url.hostname or "").removeprefix("www.") in HOSTS


def parse_hf_ref(ref: str) -> HfRef:
    """A repo, and maybe a file or folder in it, from a Hugging Face link."""
    ref = ref.strip()
    url = urllib.parse.urlsplit(ref if "://" in ref else f"https://{ref}")
    if (url.hostname or "").removeprefix("www.") not in HOSTS:
        raise HuggingFaceError(f"Not a Hugging Face link: {ref}")
    parts = [urllib.parse.unquote(p) for p in url.path.split("/") if p]
    if parts and parts[0] in ("datasets", "spaces"):
        raise HuggingFaceError(f"Not a model repo: {ref}")
    if len(parts) < 2:
        raise HuggingFaceError(f"Couldn't find a repo in {ref}")
    repo = "/".join(parts[:2])
    if len(parts) == 2:
        return HfRef(repo, "main")
    if parts[2] not in ("blob", "resolve", "tree") or len(parts) < 4:
        raise HuggingFaceError(f"Couldn't find a file in {ref}")
    rest = "/".join(parts[4:])
    if parts[2] == "tree":
        return HfRef(repo, parts[3], folder=rest)
    if not rest:
        raise HuggingFaceError(f"Couldn't find a file in {ref}")
    return HfRef(repo, parts[3], path=rest)


def quote_path(path: str) -> str:
    return urllib.parse.quote(path, safe="/")


class HuggingFace:
    def __init__(
        self,
        token_file: Path | None = None,
        api_base: str = "https://huggingface.co",
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

    def file_url(self, repo: str, revision: str, path: str) -> str:
        return f"{self.api_base}/{repo}/resolve/{revision}/{quote_path(path)}"

    async def model(self, ref: HfRef) -> dict[str, Any]:
        """The repo at `ref.revision`: its commit (`sha`), `cardData` and `siblings`, with
        each LFS file's SHA-256."""
        rev = urllib.parse.quote(ref.revision, safe="")
        url = f"{self.api_base}/api/models/{ref.repo}/revision/{rev}"
        try:
            resp = await self._http.get(url, params={"blobs": "true"}, headers=self._headers())
        except httpx2.HTTPError as e:
            raise HuggingFaceError(f"Hugging Face didn't answer: {e}") from None
        if resp.status_code in (401, 404):
            # Hugging Face answers 401 for a repo that doesn't exist, too.
            raise HuggingFaceError(f"Hugging Face has no repo {ref.repo} (or it's private)")
        if resp.status_code >= 400:
            raise HuggingFaceError(f"Hugging Face API error {resp.status_code}: {resp.text[:200]}")
        data: dict[str, Any] = resp.json()
        return data

    @asynccontextmanager
    async def download(self, url: str) -> AsyncIterator[tuple[int | None, AsyncIterator[bytes]]]:
        """Stream a file: (its size if known, its chunks)."""
        try:
            async with self._http.stream(
                "GET", url, headers=self._headers(), follow_redirects=True
            ) as resp:
                if resp.status_code in (401, 403):
                    hint = (
                        "accept its license on its page"
                        if self.has_token
                        else f"accept its license and put a token in {self.token_file}"
                    )
                    raise HuggingFaceError(
                        f"Hugging Face refused the download ({resp.status_code}): {hint}"
                    )
                if resp.status_code >= 400:
                    raise HuggingFaceError(f"Download failed with HTTP {resp.status_code}")
                length = resp.headers.get("content-length")
                yield (int(length) if length else None), resp.aiter_bytes(CHUNK)
        except httpx2.HTTPError as e:
            raise HuggingFaceError(f"Download failed: {e}") from None

    async def fetch(self, url: str, limit: int = MAX_PREVIEW_BYTES) -> bytes:
        """A small file, such as an example image or the README."""
        async with self.download(url) as (_size, chunks):
            data = bytearray()
            async for chunk in chunks:
                data += chunk
                if len(data) > limit:
                    raise HuggingFaceError(f"{url} is larger than {limit} bytes")
            return bytes(data)

    async def aclose(self) -> None:
        await self._http.aclose()
