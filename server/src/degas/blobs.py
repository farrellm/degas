"""Content-addressed blob store and thumbnails."""

import hashlib
import io
import re
from pathlib import Path

from PIL import Image

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
THUMB_SIZE = 512

EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "video/mp4": "mp4",
}


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
        if not _SHA256.match(sha):
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

    def thumb(self, sha: str) -> Path | None:
        """WebP thumbnail of an image blob, generated on first use."""
        dest = self.thumbs / f"{sha}.webp"
        if dest.exists():
            return dest
        src = self.path(sha)
        if src is None:
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
