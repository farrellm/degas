"""The local rclone, for writing to Drive (Degas's own Drive token is read-only).

`lora.rclone_remote` names a remote with write access to the same Drive (`gdrive:`).
"""

import asyncio
import contextlib
import io
import subprocess
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Protocol

from PIL import Image

from degas.errors import DegasError

PREVIEW_SIDE = 768


class RcloneError(DegasError):
    pass


Rclone = Callable[..., str]


def rclone(*args: str, stdin: bytes | None = None) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv
        ["rclone", *args],  # noqa: S607 - the user's rclone, from PATH
        input=stdin,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RcloneError(
            f"rclone {args[0]} failed: {proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode()


class Remote(Protocol):
    """rclone, asynchronously: commands, and uploads streamed from memory."""

    async def run(self, *args: str, stdin: bytes | None = None) -> str: ...

    async def rcat(self, target: str, chunks: AsyncIterator[bytes], size: int | None) -> None: ...


class AsyncRclone:
    def __init__(self, binary: str = "rclone") -> None:
        self.binary = binary

    async def run(self, *args: str, stdin: bytes | None = None) -> str:
        proc = await asyncio.create_subprocess_exec(
            self.binary,
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(stdin)
        if proc.returncode != 0:
            raise RcloneError(f"rclone {args[0]} failed: {err.decode(errors='replace').strip()}")
        return out.decode()

    async def rcat(self, target: str, chunks: AsyncIterator[bytes], size: int | None) -> None:
        """Upload a stream to `target`. If the stream fails, rclone is killed rather than
        given EOF, so it never finishes a partial file."""
        sized = ["--size", str(size)] if size is not None else []
        proc = await asyncio.create_subprocess_exec(
            self.binary,
            "rcat",
            *sized,
            target,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stderr is not None
        stderr = asyncio.create_task(proc.stderr.read())
        try:
            async for chunk in chunks:
                proc.stdin.write(chunk)
                await proc.stdin.drain()
            proc.stdin.close()
            code = await proc.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            stderr.cancel()
            raise
        err = await stderr
        if code != 0:
            raise RcloneError(f"rclone rcat failed: {err.decode(errors='replace').strip()}")


def preview_jpeg(image: Path | bytes) -> bytes:
    """A preview for a sidecar: the image as a JPEG, at most `PREVIEW_SIDE` on a side."""
    with Image.open(io.BytesIO(image) if isinstance(image, bytes) else image) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
