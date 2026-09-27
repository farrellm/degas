import os
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
    return AssetCache(paths, str(rclone), budget=25)


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


def put(cache: AssetCache, rel: str, last_used: float, size: int = 10) -> None:
    path = cache.local_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (last_used, last_used))


def test_status_lists_complete_assets(cache: AssetCache) -> None:
    put(cache, "models/sdxl/a.safetensors", 100)
    put(cache, "loras/sdxl/b.safetensors", 200, size=4)
    wan = cache.local_path("models/wan22/ti2v-5b")
    (wan / "transformer").mkdir(parents=True)
    (wan / "transformer" / "w.safetensors").write_bytes(b"x" * 6)
    put(cache, "models/sdxl/c.safetensors.partial", 300)  # a copy in progress

    # Most recent first; a directory still being copied shows as loose files.
    assert [f["path"] for f in cache.status()["files"]] == [
        "models/wan22/ti2v-5b/transformer/w.safetensors",
        "loras/sdxl/b.safetensors",
        "models/sdxl/a.safetensors",
    ]
    (wan / ".complete").touch()
    paths = {f["path"]: f["size"] for f in cache.status()["files"]}
    assert paths["models/wan22/ti2v-5b"] == 6
    assert cache.status()["used"] == 20
    assert cache.status()["budget"] == 25


def test_eviction_is_least_recently_used(cache: AssetCache) -> None:
    cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    put(cache, "models/sdxl/old.safetensors", 100)
    put(cache, "models/sdxl/new.safetensors", 200)
    cache.ensure("loras/sdxl/c.safetensors", 10)  # 30 bytes would exceed the budget of 25
    assert not cache.local_path("models/sdxl/old.safetensors").exists()
    assert cache.local_path("models/sdxl/new.safetensors").exists()
    # The copy counts as the most recent use, whatever time rclone gave the file.
    assert cache.entries()[-1].path == "loras/sdxl/c.safetensors"


def test_pinned_assets_are_not_evicted(cache: AssetCache) -> None:
    cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    put(cache, "models/sdxl/old.safetensors", 100)
    put(cache, "models/sdxl/new.safetensors", 200)
    with cache.pinned("models/sdxl/old.safetensors"):
        cache.ensure("loras/sdxl/c.safetensors", 10)
    assert cache.local_path("models/sdxl/old.safetensors").exists()
    assert not cache.local_path("models/sdxl/new.safetensors").exists()


def test_eviction_also_keeps_disk_free(cache: AssetCache, monkeypatch: pytest.MonkeyPatch) -> None:
    cache.budget = 10**12
    put(cache, "models/sdxl/a.safetensors", 100)
    monkeypatch.setattr("degas_worker.cache._disk_free", lambda _p: 5)
    monkeypatch.setattr("degas_worker.cache.DISK_RESERVE", 0)
    assert cache.make_room(10) == ["models/sdxl/a.safetensors"]
