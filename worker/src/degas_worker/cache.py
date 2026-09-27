"""Local model cache: copies Drive assets to local disk with rclone.

The server pushes short-lived Drive access tokens (no refresh token), which are
written to an rclone config for a `drive:` remote.

The cache is evicted least-recently-used above a byte budget (design §5). An
asset's last use is its modification time, which is set on every use. Assets
pinned by the running job are never evicted.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from degas_worker.paths import Paths

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]  # (bytes done, bytes total)

GB = 1000**3
DEFAULT_BUDGET = 150 * GB
DISK_RESERVE = 5 * GB  # left free for outputs, blobs and pip
COMPLETE = ".complete"  # marks a fully copied directory asset


@dataclass(frozen=True)
class CachedAsset:
    path: str  # relative, as in the Drive index
    size: int
    last_used: float  # epoch seconds

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "last_used": self.last_used}


class CacheError(RuntimeError):
    pass


class AssetCache:
    def __init__(self, paths: Paths, rclone: str | None = None, budget: int | None = None) -> None:
        self.paths = paths
        self._rclone = rclone
        self._root: str | None = None
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self.budget = budget if budget is not None else _budget_from_env()
        self._pins: Counter[str] = Counter()
        self._evict_lock = threading.Lock()

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
            return (dest / COMPLETE).exists()
        return dest.is_file() and (size is None or dest.stat().st_size == size)

    # -- pins, listing and eviction --------------------------------------------------------

    @contextmanager
    def pinned(self, rel: str) -> Iterator[None]:
        """Keep `rel` from being evicted while the block runs."""
        self.pin(rel)
        try:
            yield
        finally:
            self.unpin(rel)

    def pin(self, rel: str) -> None:
        with self._evict_lock:
            self._pins[rel] += 1

    def unpin(self, rel: str) -> None:
        with self._evict_lock:
            self._pins[rel] -= 1
            if self._pins[rel] <= 0:
                del self._pins[rel]

    def _is_pinned(self, rel: str, keep: str | None) -> bool:
        # Also covers the files of a directory asset that is still being copied.
        held = [*self._pins, *([keep] if keep else [])]
        return any(rel == p or rel.startswith(p + "/") for p in held)

    def entries(self) -> list[CachedAsset]:
        """Every fully copied asset, least recently used first."""
        found: list[CachedAsset] = []
        root = self.paths.models
        for dirpath, dirnames, filenames in os.walk(root):
            d = Path(dirpath)
            if COMPLETE in filenames:
                dirnames.clear()
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                found.append(CachedAsset(d.relative_to(root).as_posix(), size, d.stat().st_mtime))
                continue
            for name in filenames:
                if name.startswith(".") or name.endswith(".partial"):
                    continue
                f = d / name
                st = f.stat()
                found.append(CachedAsset(f.relative_to(root).as_posix(), st.st_size, st.st_mtime))
        return sorted(found, key=lambda a: a.last_used)

    def status(self) -> dict[str, Any]:
        files = self.entries()
        return {
            "used": sum(a.size for a in files),
            "budget": self.budget,
            "files": [a.as_dict() for a in reversed(files)],  # most recent first
        }

    def make_room(self, incoming: int, keep: str | None = None) -> list[str]:
        """Evict least-recently-used assets so `incoming` more bytes fit the budget and disk.

        Pinned assets and `keep` are never evicted. Returns the evicted paths.
        """
        with self._evict_lock:
            files = self.entries()
            over_budget = sum(a.size for a in files) + incoming - self.budget
            short_disk = incoming + DISK_RESERVE - _disk_free(self.paths.models)
            evicted: list[str] = []
            for asset in files:
                if over_budget <= 0 and short_disk <= 0:
                    break
                if self._is_pinned(asset.path, keep):
                    continue
                path = self.local_path(asset.path)
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                over_budget -= asset.size
                short_disk -= asset.size
                evicted.append(asset.path)
        for rel in evicted:
            log.info("evicted %s from the model cache", rel)
        return evicted

    def ensure(
        self, rel: str, size: int | None = None, on_progress: ProgressFn | None = None
    ) -> Path:
        """Return the local path of a Drive asset, copying it first if needed."""
        with self._locks[rel], self.pinned(rel):
            dest = self.local_path(rel)
            if self.is_cached(rel, size):
                os.utime(dest)  # LRU bookkeeping for eviction
                return dest
            if not self.has_token:
                raise CacheError("No Drive access token; the server has not pushed one yet")
            self.make_room(size or 0, keep=rel)
            self._copy(rel, dest, size, on_progress)
            if dest.is_dir():
                (dest / COMPLETE).touch()
            elif size is not None and dest.stat().st_size != size:
                raise CacheError(f"Copied {rel} but its size does not match the Drive index")
            os.utime(dest)  # rclone keeps Drive's modification time; this use is the latest
            if size is None:
                self.make_room(0, keep=rel)
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


def _disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _budget_from_env() -> int:
    value = os.environ.get("DEGAS_CACHE_BUDGET_GB")
    return int(float(value) * GB) if value else DEFAULT_BUDGET
