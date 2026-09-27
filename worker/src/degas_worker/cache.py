"""Local model cache: copies Drive assets to local disk with rclone.

The server pushes short-lived Drive access tokens (no refresh token), which are
written to an rclone config for a `drive:` remote.
"""

import json
import os
import shutil
import subprocess
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from degas_worker.paths import Paths

ProgressFn = Callable[[int, int], None]  # (bytes done, bytes total)


class CacheError(RuntimeError):
    pass


class AssetCache:
    def __init__(self, paths: Paths, rclone: str | None = None) -> None:
        self.paths = paths
        self._rclone = rclone
        self._root: str | None = None
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    @property
    def has_token(self) -> bool:
        return self._root is not None and self.paths.rclone_conf.exists()

    def set_token(self, access_token: str, expiry: str, root: str) -> None:
        """Store a Drive access token. `root` is the Drive folder under My Drive (e.g. `degas`)."""
        token = json.dumps({"access_token": access_token, "token_type": "Bearer", "expiry": expiry})
        conf = f"[drive]\ntype = drive\nscope = drive.readonly\ntoken = {token}\n"
        self.paths.home.mkdir(parents=True, exist_ok=True)
        tmp = self.paths.rclone_conf.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(conf)
        tmp.replace(self.paths.rclone_conf)
        self._root = root.strip("/")

    def local_path(self, rel: str) -> Path:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise CacheError(f"Invalid asset path: {rel}")
        return self.paths.models / rel_path

    def is_cached(self, rel: str, size: int | None) -> bool:
        dest = self.local_path(rel)
        if dest.is_dir():
            return (dest / ".complete").exists()
        return dest.is_file() and (size is None or dest.stat().st_size == size)

    def ensure(
        self, rel: str, size: int | None = None, on_progress: ProgressFn | None = None
    ) -> Path:
        """Return the local path of a Drive asset, copying it first if needed."""
        with self._locks[rel]:
            dest = self.local_path(rel)
            if self.is_cached(rel, size):
                os.utime(dest)  # LRU bookkeeping for eviction
                return dest
            if not self.has_token:
                raise CacheError("No Drive access token; the server has not pushed one yet")
            self._copy(rel, dest, size, on_progress)
            if dest.is_dir():
                (dest / ".complete").touch()
            elif size is not None and dest.stat().st_size != size:
                raise CacheError(f"Copied {rel} but its size does not match the Drive index")
            return dest

    def _rclone_bin(self) -> str:
        if self._rclone:
            return self._rclone
        if self.paths.rclone_bin.exists():
            return str(self.paths.rclone_bin)
        found = shutil.which("rclone")
        if not found:
            raise CacheError("rclone binary not found")
        return found

    def _copy(self, rel: str, dest: Path, size: int | None, on_progress: ProgressFn | None) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._rclone_bin(),
            "copyto",
            f"drive:{self._root}/{rel}",
            str(dest),
            "--config",
            str(self.paths.rclone_conf),
            "--multi-thread-streams",
            "8",
            "--use-json-log",
            "--stats",
            "1s",
            "--stats-log-level",
            "NOTICE",
        ]
        tail: list[str] = []
        with subprocess.Popen(  # noqa: S603 - fixed argv
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        ) as proc:
            assert proc.stderr is not None
            try:
                for line in proc.stderr:
                    stats = _parse_stats(line)
                    if stats is not None:
                        done, total = stats
                        if on_progress:
                            on_progress(done, total or size or 0)  # may raise to cancel
                    else:
                        tail = [*tail[-9:], line.rstrip()]
            except BaseException:
                proc.kill()
                raise
            code = proc.wait()
        if code != 0:
            raise CacheError(f"rclone failed copying {rel} (exit {code}): " + " | ".join(tail))


def _parse_stats(line: str) -> tuple[int, int] | None:
    try:
        entry = json.loads(line)
    except ValueError:
        return None
    stats = entry.get("stats") if isinstance(entry, dict) else None
    if not isinstance(stats, dict):
        return None
    return int(stats.get("bytes", 0)), int(stats.get("totalBytes", 0))
