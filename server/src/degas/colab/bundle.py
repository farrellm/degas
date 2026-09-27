"""The worker bundle: a reproducible tarball of the `degas_worker` package."""

import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

import degas_worker


@dataclass(frozen=True)
class Bundle:
    data: bytes
    sha256: str


def _files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
    )


def build_bundle(package_dir: Path | None = None) -> Bundle:
    """Tar `degas_worker/` with fixed metadata so identical sources hash identically."""
    root = package_dir or Path(degas_worker.__file__).parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in _files(root):
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{root.name}/{path.relative_to(root).as_posix()}")
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    data = buf.getvalue()
    return Bundle(data=data, sha256=hashlib.sha256(data).hexdigest())
