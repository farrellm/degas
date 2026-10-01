"""Wiring of the server's long-lived components."""

import asyncio
import contextlib
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import httpx2

from degas.blobs import BlobStore
from degas.civitai.client import Civitai
from degas.civitai.importer import Importer, Imports
from degas.colab.cli import Colab, ColabCli
from degas.colab.session import SessionManager
from degas.colab.tunnel import SshTunnel, Tunnel
from degas.colab.worker_client import WorkerClient
from degas.config import Config
from degas.db import Database
from degas.dispatcher import Dispatcher
from degas.drive import DriveAuth, DriveIndexer
from degas.drive.catalog import AssetCatalog
from degas.events import EventBus
from degas.families import FAMILIES
from degas.inputs import Inputs
from degas.library import sweep
from degas.notices import RESULTS_URL, SESSION_URL, idle_notice, job_notice
from degas.preprocess import Preprocessing
from degas.push import Push, Sender
from degas.rclone import AsyncRclone, Remote

log = logging.getLogger(__name__)

RESCAN_INTERVAL_S = 6 * 3600
SWEEP_INTERVAL_S = 3600


@dataclass
class Services:
    config: Config
    db: Database
    blobs: BlobStore
    bus: EventBus
    drive: DriveAuth
    assets: AssetCatalog
    sessions: SessionManager
    dispatcher: Dispatcher
    inputs: Inputs
    preprocessing: Preprocessing
    push: Push
    imports: Imports
    rng: random.Random = field(default_factory=random.SystemRandom)
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)

    async def start(self) -> None:
        await self.sessions.recover()
        self._tasks.append(asyncio.create_task(self.dispatcher.run()))
        self._tasks.append(asyncio.create_task(self._periodic_rescan()))
        self._tasks.append(asyncio.create_task(self._periodic_sweep()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        await self.push.drain()
        await self.imports.aclose()
        await self.sessions.aclose()
        await self.assets.indexer.aclose()
        await self.drive.aclose()
        self.db.close()

    def sweep(self) -> dict[str, int]:
        counts = sweep(self.db, self.blobs)
        if counts["results"] or counts["jobs"]:
            self.bus.publish({"type": "swept", **counts})
        return counts

    async def _periodic_sweep(self) -> None:
        while True:
            try:
                self.sweep()
            except Exception:
                log.exception("retention sweep failed")
            await asyncio.sleep(SWEEP_INTERVAL_S)

    async def _periodic_rescan(self) -> None:
        while True:
            if self.drive.authorized:
                try:
                    await self.assets.rescan()
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
    push_sender: Sender | None = None,
    civitai_http: httpx2.AsyncClient | None = None,
    remote: Remote | None = None,
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
    inputs = Inputs(db, blobs)
    dispatcher = Dispatcher(db, blobs, bus, sessions, inputs)
    push = Push(db, config.vapid_key_file, config.push.subject, push_sender)

    def job_finished(job: dict[str, Any]) -> None:
        notice = job_notice(job)
        if notice is not None:
            push.notify(*notice, tag=job["id"], url=RESULTS_URL)

    def idle_warning(left: timedelta) -> None:
        gpu = sessions.session["gpu"] if sessions.session else "The GPU"
        push.notify(
            idle_notice(gpu, left),
            "Open the GPU session to keep it running.",
            tag="idle",
            url=SESSION_URL,
        )

    dispatcher.on_finish.append(job_finished)
    sessions.on_idle_warning.append(idle_warning)

    remote = remote or AsyncRclone()
    importer = Importer(
        Civitai(config.civitai.token_file, config.civitai.api_base, civitai_http),
        remote,
        config.lora.rclone_remote,
        config.drive.root,
        set(FAMILIES),
    )
    assets = AssetCatalog(db, blobs, bus, indexer, remote, importer.base)
    imports = Imports(importer, lambda: db.list_assets(kind="lora"), bus.publish, assets.rescan)
    return Services(
        config=config,
        db=db,
        blobs=blobs,
        bus=bus,
        drive=drive,
        assets=assets,
        sessions=sessions,
        dispatcher=dispatcher,
        inputs=inputs,
        preprocessing=Preprocessing(db, blobs, sessions, inputs),
        push=push,
        imports=imports,
    )
