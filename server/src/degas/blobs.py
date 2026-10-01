"""Content-addressed blob store and thumbnails."""

import hashlib
import io
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TypeGuard

from PIL import Image

SHA256 = re.compile(r"[0-9a-f]{64}")
REF_PREFIX = "sha256:"
SHA_REF = re.compile(r"sha256:[0-9a-f]{64}")
THUMB_SIZE = 512

EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
}
VIDEO_EXTENSIONS = {"mp4", "webm", "mov"}


def ref(sha: str) -> str:
    """How specs and saved configs name a blob: `sha256:<hex>`."""
    return f"{REF_PREFIX}{sha}"


def unref(value: str) -> str:
    return value.removeprefix(REF_PREFIX)


def is_ref(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and SHA_REF.fullmatch(value) is not None


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.blobs = root / "blobs"
        self.thumbs = root / "thumbs"

    def ensure(self) -> None:
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.thumbs.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path | None:
        """Path of a stored blob, or None if absent."""
        if not SHA256.fullmatch(sha):
            return None
        d = self.blobs / sha[:2]
        matches = list(d.glob(f"{sha}.*")) if d.exists() else []
        return matches[0] if matches else None

    def put(self, data: bytes, media_type: str) -> str:
        sha = hashlib.sha256(data).hexdigest()
        if self.path(sha) is None:
            dest = self.blobs / sha[:2] / f"{sha}.{EXTENSIONS.get(media_type, 'bin')}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{sha}.tmp")  # not matched by path()'s glob
            tmp.write_bytes(data)
            tmp.replace(dest)
        return sha

    def image_size(self, sha: str) -> tuple[int, int] | None:
        path = self.path(sha)
        if path is None:
            return None
        try:
            with Image.open(path) as im:
                return im.size
        except OSError:
            return None

    def is_video(self, sha: str) -> bool:
        path = self.path(sha)
        return path is not None and path.suffix.lstrip(".") in VIDEO_EXTENSIONS

    def put_thumb(self, sha: str, webp: bytes) -> Path:
        """Store a thumbnail made elsewhere (a video's poster frame)."""
        dest = self.thumbs / f"{sha}.webp"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(webp)
        tmp.replace(dest)
        return dest

    def thumb(self, sha: str) -> Path | None:
        """WebP thumbnail of an image blob, generated on first use.

        A video's poster is made with ffmpeg (see `put_thumb`); until then this is None.
        """
        dest = self.thumbs / f"{sha}.webp"
        if dest.exists():
            return dest
        src = self.path(sha)
        if src is None or self.is_video(sha):
            return None
        try:
            with Image.open(src) as im:
                im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="WEBP", quality=82)
        except OSError:
            return None
        self.thumbs.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(dest)
        return dest

    def delete(self, sha: str) -> bool:
        """Remove a blob and its thumbnail."""
        path = self.path(sha)
        (self.thumbs / f"{sha}.webp").unlink(missing_ok=True)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        return True

    def stored(self, older_than_s: float = 0) -> Iterator[str]:
        """Shas of stored blobs last written more than `older_than_s` ago."""
        cutoff = time.time() - older_than_s
        for path in self.blobs.glob("*/*"):
            name = path.name.split(".", 1)[0]
            if SHA256.fullmatch(name) and path.stat().st_mtime <= cutoff:
                yield name
