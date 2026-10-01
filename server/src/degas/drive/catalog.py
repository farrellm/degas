"""The asset index as the app keeps it: rescans of Drive, and LoRAs deleted from it."""

import asyncio

from degas.blobs import BlobStore
from degas.db import Database
from degas.drive.index import DriveIndexer, companions
from degas.events import EventBus
from degas.library import release
from degas.rclone import Remote


class AssetCatalog:
    """Changes to the index run one at a time, and each is published as an `assets` event."""

    def __init__(
        self,
        db: Database,
        blobs: BlobStore,
        bus: EventBus,
        indexer: DriveIndexer,
        remote: Remote,
        base: str,
    ) -> None:
        self.db = db
        self.blobs = blobs
        self.bus = bus
        self.indexer = indexer
        self.remote = remote  # rclone with write access (Degas's own Drive token is read-only)
        self.base = base  # the Drive root as that remote names it: `gdrive:degas`
        self._lock = asyncio.Lock()

    async def rescan(self) -> int:
        async with self._lock:
            assets = await self.indexer.scan()
            previous = {a["path"]: a for a in self.db.list_assets()}
            await self.indexer.enrich(assets, previous, self.blobs.put)
            count = self.db.replace_assets(assets)
        self.bus.publish({"type": "assets", "count": count})
        return count

    async def delete_loras(self, paths: list[str]) -> None:
        """Move LoRAs (with their sidecars and previews) to Drive's trash and drop them from
        the index. Every path must be an indexed LoRA."""
        async with self._lock:
            for path in paths:
                folder, file = path.rsplit("/", 1)
                target = f"{self.base}/{folder}"
                names = (await self.remote.run("lsf", "--files-only", target)).splitlines()
                for name in [file, *companions(file, names)]:
                    if name in names:
                        await self.remote.run("deletefile", f"{target}/{name}")
                release(self.db, self.blobs, self.db.delete_assets([path]))
        self.bus.publish({"type": "assets", "count": len(self.db.list_assets())})
