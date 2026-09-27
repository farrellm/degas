import stat
from pathlib import Path

import pytest

from degas_worker.cache import AssetCache, CacheError
from degas_worker.paths import Paths

# Stand-in for rclone: `copyto SRC DEST ...` writes 10 bytes to DEST and logs JSON stats.
FAKE_RCLONE = """#!/bin/sh
dest="$3"
echo '{"level":"notice","msg":"stats","stats":{"bytes":5,"totalBytes":10}}' >&2
echo "$2" > "$dest.src"
printf '0123456789' > "$dest"
echo '{"level":"notice","msg":"stats","stats":{"bytes":10,"totalBytes":10}}' >&2
"""


@pytest.fixture
def cache(tmp_path: Path) -> AssetCache:
    rclone = tmp_path / "rclone"
    rclone.write_text(FAKE_RCLONE)
    rclone.chmod(rclone.stat().st_mode | stat.S_IEXEC)
    paths = Paths(home=tmp_path / "degas", models=tmp_path / "models")
    paths.ensure()
    return AssetCache(paths, str(rclone))


def test_requires_token(cache: AssetCache) -> None:
    with pytest.raises(CacheError, match="token"):
        cache.ensure("models/sdxl/a.safetensors")


def test_copy_and_reuse(cache: AssetCache) -> None:
    cache.set_token("tok", "2026-09-27T12:00:00Z", "/degas/")
    seen: list[tuple[int, int]] = []
    path = cache.ensure("models/sdxl/a.safetensors", 10, lambda d, t: seen.append((d, t)))
    assert path == cache.paths.models / "models/sdxl/a.safetensors"
    assert path.read_bytes() == b"0123456789"
    assert path.with_name("a.safetensors.src").read_text().strip() == (
        "drive:degas/models/sdxl/a.safetensors"
    )
    assert seen == [(5, 10), (10, 10)]
    seen.clear()
    cache.ensure("models/sdxl/a.safetensors", 10, lambda d, t: seen.append((d, t)))
    assert seen == []  # cached


def test_size_mismatch(cache: AssetCache) -> None:
    cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    with pytest.raises(CacheError, match="size"):
        cache.ensure("models/sdxl/a.safetensors", 11)


def test_rejects_escaping_paths(cache: AssetCache) -> None:
    with pytest.raises(CacheError):
        cache.local_path("../etc/passwd")


def test_cancel_kills_the_copy(cache: AssetCache) -> None:
    cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")

    def cancel(_done: int, _total: int) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cache.ensure("models/sdxl/b.safetensors", 10, cancel)
