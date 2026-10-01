"""Source images, references and masks for i2i, edit, inpaint, outpaint and i2v: design §6.1
and §6.5.

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
from degas.blobs import THUMB_SIZE, BlobStore, ref, unref
from degas.db import Database
from degas.families.base import FamilyDescriptor
from degas.families.wan22 import extend_variant
from degas.media import MediaError

MAX_FETCH_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
FETCH_TIMEOUT_S = 20
MAX_REDIRECTS = 5
# Image prompt pictures are fitted to squares no bigger than this (the encoder sees 224 px).
PROMPT_IMAGE_MAX = 1024
USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 19_0 like Mac OS X) AppleWebKit/605.1.15"
    " (KHTML, like Gecko) Version/19.0 Mobile/15E148 Safari/604.1"
)


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
        """Fit the source (and its mask) and every control image (and its area) to the output
        size, image prompts' pictures to squares (and their areas to the output size), and
        record every transform in the spec, references' crops included (§6.5).

        After this the spec has no `fit`, its source is exactly the output size (for an
        outpaint, the size it is placed at), each control image exactly the output size, and
        each mask exactly its image's size.
        """
        inputs = spec.get("inputs") or {}
        fit = inputs.pop("fit", "crop")
        extends = inputs.get("extends")
        if extends and not self.blobs.is_video(unref(extends)):
            raise MediaError("The clip to extend is no longer stored")
        transforms: dict[str, Any] = {}
        output = (int(spec["params"]["width"]), int(spec["params"]["height"]))
        if inputs.get("source"):
            place = inputs.get("place")
            target = (int(place["w"]), int(place["h"])) if place else output
            inputs["source"], inputs["mask"] = await self._fit(
                inputs["source"],
                inputs.get("mask"),
                target,
                fit,
                transforms,
                "The source image is no longer stored. Choose it again.",
            )
            if inputs["mask"] is None:
                del inputs["mask"]
            elif await self._mask_empty(inputs["mask"]):
                raise MediaError("The mask is empty. Paint the area to redraw.")
        # References go as they are: the pipeline sizes each one itself. A cropped one keeps
        # its original, so a remix can crop it again.
        for n, value in enumerate(inputs.get("refs") or [], 2):
            sha = unref(value)
            if self.blobs.is_video(sha) or self.blobs.image_size(sha) is None:
                raise MediaError(f"Image {n} is no longer stored. Choose it again.")
            record = self.db.get_transform(sha)
            if record and self.blobs.path(record["original"]):
                transforms[value] = {"original": ref(record["original"]), "ops": record["ops"]}
        for n, unit in enumerate(spec.get("control") or [], 1):
            await self._fit_control(n, unit, output, transforms)
        for n, unit in enumerate(spec.get("image_prompts") or [], 1):
            await self._fit_image_prompt(n, unit, output, transforms)
        if transforms:
            spec.setdefault("inputs", inputs)["transforms"] = transforms
        else:
            inputs.pop("transforms", None)

    async def _fit_control(
        self,
        n: int,
        unit: dict[str, Any],
        output: tuple[int, int],
        transforms: dict[str, Any],
    ) -> None:
        """Fit a ControlNet unit's image, and the area painted over it, to the output size."""
        unit["image"], mask = await self._fit(
            unit["image"],
            unit.get("mask"),
            output,
            unit.pop("fit", "crop"),
            transforms,
            f"ControlNet {n}'s image is no longer stored. Choose it again.",
        )
        if mask is not None:
            if await self._mask_empty(mask):
                raise MediaError(f"ControlNet {n}'s area is empty. Paint it, or remove it.")
            unit["mask"] = mask

    async def _fit_image_prompt(
        self,
        n: int,
        unit: dict[str, Any],
        output: tuple[int, int],
        transforms: dict[str, Any],
    ) -> None:
        """Fit an image prompt's pictures to squares and its area to the output size.

        The image encoder sees a 224 px square from the middle of each picture, so fitting
        here changes nothing it sees, but the transforms record exactly what it was.
        """
        fit = unit.pop("fit", "crop")
        gone = f"A picture in image prompt {n} is no longer stored. Choose it again."
        fitted: list[str] = []
        for picture in unit["images"]:
            sha = unref(picture)
            size = self.blobs.image_size(sha) if not self.blobs.is_video(sha) else None
            if size is None:
                raise MediaError(gone)
            side = min(max(size) if fit == "pad" else min(size), PROMPT_IMAGE_MAX)
            square, _ = await self._fit(picture, None, (side, side), fit, transforms, gone)
            fitted.append(square)
        unit["images"] = fitted
        if unit.get("mask"):
            sha = unref(unit["mask"])
            size = self.blobs.image_size(sha)
            if size is None:
                raise MediaError(f"Image prompt {n}'s area is no longer stored. Paint it again.")
            fit_ops = media.fit_ops(*size, *output, "crop")
            unit["mask"] = await self._fit_mask(unit["mask"], size, fit_ops, transforms)
            if await self._mask_empty(unit["mask"]):
                raise MediaError(f"Image prompt {n}'s area is empty. Paint it, or remove it.")

    async def _fit(
        self,
        image: str,
        mask: str | None,
        target: tuple[int, int],
        fit: str,
        transforms: dict[str, Any],
        gone: str,
    ) -> tuple[str, str | None]:
        """Fit an image (and the mask painted over it) to `target`; returns the new refs."""
        sha = unref(image)
        size = self.blobs.image_size(sha) if not self.blobs.is_video(sha) else None
        if size is None:
            raise MediaError(gone)
        record = self.db.get_transform(sha)
        if record and self.blobs.path(record["original"]) is None:
            record = None  # the original is gone: treat the derived image as the original
        fit_ops = media.fit_ops(*size, *target, fit)
        if mask:
            mask = await self._fit_mask(mask, size, fit_ops, transforms)
        if size != target:
            original, previous = (record["original"], record["ops"]) if record else (sha, [])
            ops = [*previous, *fit_ops]
            derived = await self.derive(original, ops)
            image = ref(derived["sha256"])
            transforms[image] = {"original": ref(original), "ops": ops}
        elif record:
            transforms[image] = {"original": ref(record["original"]), "ops": record["ops"]}
        return image, mask

    async def _fit_mask(
        self,
        mask: str,
        size: tuple[int, int],
        fit_ops: list[media.Op],
        transforms: dict[str, Any],
    ) -> str:
        """Give the mask the fit its image gets, so the two stay pixel for pixel."""
        sha = unref(mask)
        path = self.blobs.path(sha)
        if path is None:
            raise MediaError("The mask is no longer stored. Paint it again.")
        data = path.read_bytes()
        if media.image_size(data) != size:
            raise MediaError("The mask was painted on a different image. Paint it again.")
        if not fit_ops:
            return mask
        data = await run_in_threadpool(media.apply_mask_ops, data, fit_ops)
        fitted = self.blobs.put(data, "image/png")
        self.db.hold_input(fitted)
        transforms[ref(fitted)] = {"original": ref(sha), "ops": fit_ops}
        return ref(fitted)

    async def _mask_empty(self, mask: str) -> bool:
        path = self.blobs.path(unref(mask))
        assert path is not None
        return await run_in_threadpool(media.mask_is_empty, path.read_bytes())

    # -- masks -------------------------------------------------------------------------------

    async def store_mask(self, source: str, data: bytes) -> dict[str, Any]:
        """Store a mask painted over `source`, at the source's pixel size."""
        size = self.blobs.image_size(source) if not self.blobs.is_video(source) else None
        if size is None:
            raise MediaError("The image the mask was painted on is no longer stored")
        png = await run_in_threadpool(media.normalize_mask, data, size)
        return self._put_image(png)

    async def remap_mask(self, mask: str, source: str, to: str) -> dict[str, Any]:
        """Carry a mask painted over `source` onto `to`, another crop of the same original.

        What the new crop leaves out of the mask is dropped (`empty` says if all of it was).
        """
        mask_path = self.blobs.path(mask)
        if mask_path is None:
            raise MediaError("The mask is no longer stored")
        was = self.db.get_transform(source)
        now = self.db.get_transform(to)
        original = was["original"] if was else source
        if (now["original"] if now else to) != original:
            raise MediaError("The new image isn't a crop of the one the mask was painted on")
        size = self.blobs.image_size(original)
        data = mask_path.read_bytes()
        if size is None or media.image_size(data) != self.blobs.image_size(source):
            raise MediaError("The mask no longer matches its image")
        ops = [*media.invert_ops(was["ops"] if was else [], *size), *(now["ops"] if now else [])]
        png = await run_in_threadpool(media.apply_mask_ops, data, ops)
        return {**self._put_image(png), "empty": await run_in_threadpool(media.mask_is_empty, png)}

    async def store_trace(self, data: bytes, size: tuple[int, int]) -> dict[str, Any]:
        """Store a preprocessor's trace of an image (a depth map, a pose, edges), which must be
        the image's size so an area painted over one fits the other."""
        if media.image_size(data) != size:
            raise MediaError("the trace isn't the size of its image")
        return self._put_image(data)

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
