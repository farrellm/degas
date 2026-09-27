"""Wiring of the server's long-lived components."""

import asyncio
import contextlib
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from degas.blobs import BlobStore
from degas.colab.cli import Colab, ColabCli
from degas.colab.session import SessionManager
from degas.colab.tunnel import SshTunnel, Tunnel
from degas.colab.worker_client import WorkerClient
from degas.config import Config
from degas.db import Database
from degas.dispatcher import Dispatcher
from degas.drive import DriveAuth, DriveIndexer
from degas.events import EventBus

log = logging.getLogger(__name__)

RESCAN_INTERVAL_S = 6 * 3600


@dataclass
class Services:
    config: Config
    db: Database
    blobs: BlobStore
    bus: EventBus
    drive: DriveAuth
    indexer: DriveIndexer
    sessions: SessionManager
    dispatcher: Dispatcher
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.SystemRandom)

    async def start(self) -> None:
        await self.sessions.recover()
        self._tasks.append(asyncio.create_task(self.dispatcher.run()))
        self._tasks.append(asyncio.create_task(self._periodic_rescan()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        await self.sessions.aclose()
        await self.indexer.aclose()
        await self.drive.aclose()
        self.db.close()

    async def rescan(self) -> int:
        assets = await self.indexer.scan()
        count = self.db.replace_assets(assets)
        self.bus.publish({"type": "assets", "count": count})
        return count

    async def _periodic_rescan(self) -> None:
        while True:
            if self.drive.authorized:
                try:
                    await self.rescan()
                except Exception as e:
                    log.warning("Drive rescan failed: %s", e)
            await asyncio.sleep(RESCAN_INTERVAL_S)


def build_services(
    config: Config,
    colab: Colab | None = None,
    tunnel_factory: Callable[[], Tunnel] | None = None,
    worker_factory: Callable[[Tunnel], WorkerClient] | None = None,
    drive: DriveAuth | None = None,
    indexer: DriveIndexer | None = None,
) -> Services:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(config.data_dir / "degas.sqlite")
    blobs = BlobStore(config.data_dir)
    blobs.ensure()
    bus = EventBus()
    drive = drive or DriveAuth(config.drive.client_file, config.drive_token_file)
    indexer = indexer or DriveIndexer(drive, config.drive.root)
    cli = colab or ColabCli(config.colab.binary, config.colab.session_name, config.colab.auth)

    def default_tunnel() -> Tunnel:
        return SshTunnel(
            cli.proxy_command(str(config.ssh_key)),
            config.ssh_key,
            config.colab.worker_port,
            log_file=config.data_dir / "ssh.log",
        )

    def default_worker(tunnel: Tunnel) -> WorkerClient:
        return WorkerClient(f"http://127.0.0.1:{tunnel.local_port}")

    sessions = SessionManager(
        config,
        db,
        bus,
        cli,
        drive,
        tunnel_factory or default_tunnel,
        worker_factory or default_worker,
    )
    dispatcher = Dispatcher(db, blobs, bus, sessions)
    return Services(config, db, blobs, bus, drive, indexer, sessions, dispatcher)
