"""Source images for i2v (and later i2i, inpaint and control): design §6.1 and §6.5.

Uploads, URL imports, video frames and transformed images are ordinary blobs,
held for 24 h by an `input` ref until a job or a kept item takes its own.
"""

import base64
import binascii
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

import httpx2
from starlette.concurrency import run_in_threadpool

from degas import media
from degas.blobs import THUMB_SIZE, BlobStore
from degas.db import Database
from degas.families.base import FamilyDescriptor
from degas.families.wan22 import extend_variant
from degas.media import MediaError

MAX_FETCH_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
FETCH_TIMEOUT_S = 20
MAX_REDIRECTS = 5
USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 19_0 like Mac OS X) AppleWebKit/605.1.15"
    " (KHTML, like Gecko) Version/19.0 Mobile/15E148 Safari/604.1"
)


def ref(sha: str) -> str:
    return f"sha256:{sha}"


def unref(value: str) -> str:
    return value.removeprefix("sha256:")


class Inputs:
    def __init__(self, db: Database, blobs: BlobStore) -> None:
        self.db = db
        self.blobs = blobs

    # -- storing -----------------------------------------------------------------------------

    async def store(self, data: bytes, content_type: str | None = None) -> dict[str, Any]:
        """Store an uploaded or fetched image or video; returns its blob info."""
        try:
            image, media_type, w, h = await run_in_threadpool(media.normalize_image, data)
        except MediaError:
            pass
        else:
            sha = self.blobs.put(image, media_type)
            self.db.hold_input(sha)
            return {"sha256": sha, "media_type": media_type, "width": w, "height": h}

        with tempfile.NamedTemporaryFile() as tmp:
            Path(tmp.name).write_bytes(data)  # noqa: ASYNC240 - local temp file
            info = await media.probe(Path(tmp.name))
        if info is None:
            raise MediaError("That isn't an image or a video Degas can read")
        media_type = (
            content_type if content_type in media.VIDEO_TYPES else media.video_media_type(info)
        )
        sha = self.blobs.put(data, media_type)
        self.db.hold_input(sha)
        await self.poster(sha)
        return {
            "sha256": sha,
            "media_type": media_type,
            "width": info["width"],
            "height": info["height"],
            "duration": info["duration"],
        }

    async def fetch(self, url: str) -> dict[str, Any]:
        """Import an image or a direct video link (or a `data:` URI)."""
        url = url.strip()
        if url.startswith("data:"):
            return await self.store(*_decode_data_uri(url))
        if not url.lower().startswith(("http://", "https://")):
            raise MediaError("Only http, https and data: links can be imported")
        chunks: list[bytes] = []
        total = 0
        try:
            async with (
                httpx2.AsyncClient(
                    follow_redirects=True,
                    max_redirects=MAX_REDIRECTS,
                    timeout=FETCH_TIMEOUT_S,
                    headers={"User-Agent": USER_AGENT, "Accept": "image/*,video/*;q=0.9,*/*;q=0.5"},
                ) as http,
                http.stream("GET", url) as resp,
            ):
                if resp.status_code >= 400:
                    raise MediaError(f"The link returned HTTP {resp.status_code}")
                content_type = resp.headers.get("content-type", "").split(";")[0].strip()
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_FETCH_BYTES:
                        raise MediaError("The file at that link is larger than 50 MB")
                    chunks.append(chunk)
        except httpx2.HTTPError as e:
            raise MediaError(f"Couldn't fetch the link: {e}") from e
        try:
            return await self.store(b"".join(chunks), content_type)
        except MediaError:
            if content_type.startswith("text/html"):
                raise MediaError(
                    "That link is a web page, not an image. Open the image itself and copy"
                    " its address."
                ) from None
            raise

    async def poster(self, sha: str) -> Path | None:
        """A video blob's poster frame, as its thumbnail."""
        path = self.blobs.path(sha)
        if path is None:
            return None
        frame = await media.extract_frame(path, "first")
        webp = await run_in_threadpool(media.thumbnail, frame, THUMB_SIZE)
        return self.blobs.put_thumb(sha, webp)

    async def frame(self, sha: str, at: str | float) -> dict[str, Any]:
        """Extract one frame of a video blob as a new image blob."""
        path = self.blobs.path(sha)
        if path is None or not self.blobs.is_video(sha):
            raise MediaError("That blob isn't a stored video")
        png = await media.extract_frame(path, at)
        return self._put_image(png)

    def _put_image(self, png: bytes) -> dict[str, Any]:
        sha = self.blobs.put(png, "image/png")
        self.db.hold_input(sha)
        w, h = media.image_size(png) or (None, None)
        return {"sha256": sha, "media_type": "image/png", "width": w, "height": h}

    # -- transforms ----------------------------------------------------------------------------

    async def derive(self, original: str, ops: list[media.Op]) -> dict[str, Any]:
        """Apply `ops` to the original blob; the derived blob records where it came from."""
        path = self.blobs.path(original)
        if path is None or self.blobs.is_video(original):
            raise MediaError("The original image is no longer stored")
        self.db.hold_input(original)
        if not ops:
            w, h = self.blobs.image_size(original) or (None, None)
            return {"sha256": original, "media_type": "image/png", "width": w, "height": h}
        data, w, h = await run_in_threadpool(media.apply_ops, path.read_bytes(), ops)
        sha = self.blobs.put(data, "image/png")
        self.db.add_transform(sha, original, ops)
        self.db.hold_input(sha)
        return {"sha256": sha, "media_type": "image/png", "width": w, "height": h}

    async def transform(self, sha: str, ops: Any) -> dict[str, Any]:
        """Transform an image; for a derived image, `ops` replace its operations on the original."""
        ops = media.validate_ops(ops)
        record = self.db.get_transform(sha)
        original = record["original"] if record and self.blobs.path(record["original"]) else sha
        return await self.derive(original, ops)

    async def resolve(self, spec: dict[str, Any]) -> None:
        """Fit the source to the output size and record every transform in the spec (§6.5).

        After this the spec has no `fit` and its source is exactly the output size.
        """
        inputs = spec.get("inputs") or {}
        fit = inputs.pop("fit", "crop")
        extends = inputs.get("extends")
        if extends and not self.blobs.is_video(unref(extends)):
            raise MediaError("The clip to extend is no longer stored")
        if not inputs.get("source"):
            return
        sha = unref(inputs["source"])
        size = self.blobs.image_size(sha) if not self.blobs.is_video(sha) else None
        if size is None:
            raise MediaError("The source image is no longer stored. Choose it again.")
        target = (int(spec["params"]["width"]), int(spec["params"]["height"]))
        record = self.db.get_transform(sha)
        if record and self.blobs.path(record["original"]) is None:
            record = None  # the original is gone: treat the derived image as the original
        transforms: dict[str, Any] = {}
        if size != target:
            original, previous = (record["original"], record["ops"]) if record else (sha, [])
            ops = [*previous, *media.fit_ops(*size, *target, fit)]
            derived = await self.derive(original, ops)
            inputs["source"] = ref(derived["sha256"])
            transforms[inputs["source"]] = {"original": ref(original), "ops": ops}
        elif record:
            transforms[inputs["source"]] = {
                "original": ref(record["original"]),
                "ops": record["ops"],
            }
        if transforms:
            inputs["transforms"] = transforms
        else:
            inputs.pop("transforms", None)

    # -- video extension -------------------------------------------------------------------

    async def extend(
        self, family: FamilyDescriptor, sha: str, spec: dict[str, Any]
    ) -> dict[str, Any]:
        """A spec that continues a clip from its last frame (design §6.4), for editing in Create."""
        if family.id != "wan22" or not self.blobs.is_video(sha):
            raise MediaError("Only Wan 2.2 videos can be extended")
        path = self.blobs.path(sha)
        assert path is not None
        frame = self._put_image(await media.extract_frame(path, "last"))
        variant = extend_variant(spec["variant"])
        model: dict[str, Any] = {"path": spec["model"]["path"]}
        if variant != spec["variant"]:
            v = next(v for v in family.variants if v.id == variant)
            models = [
                a["path"]
                for a in self.db.list_assets(family.id, "model")
                if v.model_dir and a["path"].startswith(v.model_dir)
            ]
            model = {"path": models[0] if models else ""}
        schema = family.param_schema(variant, "i2v")["properties"]
        params = {k: v for k, v in spec["params"].items() if k in schema}
        params.update(width=frame["width"], height=frame["height"], seed=-1)
        loras = [_strip_sizes(lora) for lora in spec.get("loras") or []]
        return {
            "spec": {
                "family": family.id,
                "variant": variant,
                "mode": "i2v",
                "model": model,
                "loras": loras,
                "params": params,
                "inputs": {"source": ref(frame["sha256"]), "extends": ref(sha)},
            },
            "source": frame,
        }


def _strip_sizes(lora: dict[str, Any]) -> dict[str, Any]:
    if "path" in lora:
        return {"path": lora["path"], "weight": lora["weight"]}
    return {k: {"path": v["path"], "weight": v["weight"]} for k, v in lora.items() if v}


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    header, sep, payload = uri.partition(",")
    if not sep:
        raise MediaError("That data: link is malformed")
    content_type = header.removeprefix("data:").split(";")[0]
    try:
        if header.endswith(";base64"):
            return base64.b64decode(payload, validate=False), content_type
        return unquote_to_bytes(payload), content_type
    except (binascii.Error, ValueError) as e:
        raise MediaError("That data: link is malformed") from e
