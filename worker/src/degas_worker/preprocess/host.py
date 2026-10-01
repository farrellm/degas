"""Running preprocessors on request, next to the generation pipeline."""

import contextlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from degas_worker.cache import AssetCache
from degas_worker.preprocess.base import Preprocessor


class PreprocessorHost:
    """Loads a preprocessor the first time it is asked for and keeps it until another kind is.

    One is resident at a time, since it shares the GPU with the pipeline, and calls run one at
    a time.
    """

    def __init__(self, factories: dict[str, Callable[[], Preprocessor]], cache: AssetCache) -> None:
        self._factories = factories
        self._cache = cache
        self._loaded: dict[str, Preprocessor] = {}
        self._lock = threading.Lock()

    def __contains__(self, kind: str) -> bool:
        return kind in self._factories

    def run(
        self,
        kind: str,
        image: Path,
        params: dict[str, Any],
        asset: str | None = None,
        size: int | None = None,
    ) -> dict[str, Any]:
        """Run `kind` on `image`, with its model (the Drive asset `asset`, if it has one)
        copied into the cache first."""
        with self._lock, self._cache.pinned(asset) if asset else contextlib.nullcontext():
            model = self._cache.ensure(asset, size) if asset else None
            pre = self._loaded.get(kind)
            if pre is None:
                self.unload()
                pre = self._loaded[kind] = self._factories[kind]()
            return pre.run(model, image, params)

    def unload(self) -> None:
        for pre in self._loaded.values():
            pre.unload()
        self._loaded.clear()
