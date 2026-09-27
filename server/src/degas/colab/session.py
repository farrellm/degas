"""Colab session lifecycle: bootstrap, liveness, idle shutdown and restart recovery.

States: starting → ready ⇄ busy → stopping → stopped, plus error (design §3.1).
"""

import asyncio
import contextlib
import logging
import tempfile
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from degas.colab.bundle import Bundle, build_bundle
from degas.colab.cli import Colab, ColabError
from degas.colab.tunnel import Tunnel, TunnelError
from degas.colab.worker_client import WorkerClient, WorkerError
from degas.config import Config
from degas.db import ACTIVE_SESSION_STATES, Database, now
from degas.drive import DriveAuth, DriveError
from degas.events import EventBus
from degas.families.base import GPUS

log = logging.getLogger(__name__)

REMOTE = "/content/degas"
WORKER_MATCH = "degas_worker.app:app"
STARTED_MARKER = "@@degas-worker-started"
RCLONE_URL = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"

# Runs in the Colab kernel so the worker inherits its CUDA environment (Phase 0, finding 6).
BOOTSTRAP = f"""
import os, subprocess, sys
subprocess.run(["pkill", "-f", {WORKER_MATCH!r}])
_env = dict(os.environ, PYTHONPATH="{REMOTE}/worker", DEGAS_WORKER_HOME="{REMOTE}",
            PATH="{REMOTE}/bin:" + os.environ.get("PATH", ""))
_log = open("{REMOTE}/worker.log", "ab")
_p = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", {WORKER_MATCH!r}, "--host", "127.0.0.1",
     "--port", "{{port}}"],
    env=_env, cwd="{REMOTE}", stdin=subprocess.DEVNULL, stdout=_log, stderr=subprocess.STDOUT,
    start_new_session=True)
print("{STARTED_MARKER}", _p.pid)
"""


class SessionError(RuntimeError):
    pass


class Intervals:
    """Liveness timings in seconds (overridable in tests)."""

    tick = 5.0
    health = 30.0
    colab_status = 300.0
    exec_heartbeat = 300.0
    drive_token = 45 * 60.0
    health_wait = 180.0  # worker start-up, including the first torch import


class SessionManager:
    def __init__(
        self,
        config: Config,
        db: Database,
        bus: EventBus,
        colab: Colab,
        drive: DriveAuth,
        tunnel_factory: Callable[[], Tunnel],
        worker_factory: Callable[[Tunnel], WorkerClient],
        bundle: Callable[[], Bundle] = build_bundle,
        intervals: Intervals | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.bus = bus
        self.colab = colab
        self.drive = drive
        self._tunnel_factory = tunnel_factory
        self._worker_factory = worker_factory
        self._bundle = bundle
        self.iv = intervals or Intervals()

        self.session: dict[str, Any] | None = None
        self.tunnel: Tunnel | None = None
        self.worker: WorkerClient | None = None
        self.health: dict[str, Any] | None = None
        self.step: str | None = None
        self.drive_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self.on_ready: list[Callable[[], None]] = []
        self.on_end: list[Callable[[], None]] = []
        self.has_pending_jobs: Callable[[], bool] = lambda: False

    # -- state -----------------------------------------------------------------------------

    @property
    def state(self) -> str | None:
        return self.session["state"] if self.session else None

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_SESSION_STATES

    @property
    def idle_timeout(self) -> timedelta:
        return timedelta(minutes=self.config.idle_timeout_min)

    def idle_deadline(self) -> datetime | None:
        if self.state not in ("ready", "busy") or self.session is None:
            return None
        last = datetime.fromisoformat(self.session["last_activity_at"])
        return last + self.idle_timeout

    def snapshot(self) -> dict[str, Any]:
        deadline = self.idle_deadline()
        return {
            "session": self.session,
            "step": self.step,
            "worker": self.health,
            "idle_deadline": deadline.isoformat(timespec="seconds") if deadline else None,
            "idle_timeout_min": self.config.idle_timeout_min,
            "drive": {**self.drive.status(), "push_error": self.drive_error},
            "gpus": list(GPUS),
        }

    def _publish(self) -> None:
        self.bus.publish({"type": "session", **self.snapshot()})

    def _update(self, **fields: Any) -> None:
        assert self.session is not None
        self.db.update_session(self.session["id"], **fields)
        self.session = self.db.get_session(self.session["id"])
        if self.state in ("ready", "busy") and self.worker is not None:
            if not self._ready.is_set():
                self._ready.set()
                for cb in self.on_ready:
                    cb()
        else:
            self._ready.clear()
        self._publish()

    def _step(self, text: str | None) -> None:
        self.step = text
        log.info("session: %s", text)
        self._publish()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    def touch(self) -> None:
        """User interaction or job activity: resets the idle countdown."""
        if self.state in ("ready", "busy", "starting"):
            self._update(last_activity_at=now())

    def set_busy(self, busy: bool) -> None:
        if self.state in ("ready", "busy"):
            self._update(state="busy" if busy else "ready", last_activity_at=now())

    # -- start / stop ----------------------------------------------------------------------

    async def start(self, gpu: str, high_mem: bool) -> dict[str, Any]:
        if gpu not in GPUS:
            raise SessionError(f"Unknown GPU {gpu!r}")
        async with self._lock:
            if self.active:
                raise SessionError("A session is already active")
            self.session = self.db.create_session(gpu, high_mem)
            self.health = None
            self.drive_error = None
            self._step("Allocating VM")
            self._task = asyncio.create_task(self._bootstrap(allocate=True))
        return self.snapshot()

    async def stop(self, reason: str | None = None) -> None:
        async with self._lock:
            if not self.active or self.session is None:
                return
            await self._cancel_tasks()
            self._update(state="stopping")
            self._step("Stopping" + (f" ({reason})" if reason else ""))
            if self.worker is not None:
                with contextlib.suppress(WorkerError):
                    await self.worker.shutdown()
            await self._teardown()
            try:
                await self.colab.stop()
            except ColabError as e:
                log.warning("colab stop failed: %s", e)
            self._end("stopped", None)

    async def reset_worker(self) -> None:
        """Force reset: kill the worker process and start a fresh one (models reload)."""
        async with self._lock:
            if self.state not in ("ready", "busy") or self.tunnel is None:
                raise SessionError("No running session")
            await self._cancel_tasks()
            self._update(state="starting")
            self._task = asyncio.create_task(self._bootstrap(allocate=False, restart_worker=True))

    async def _cancel_tasks(self) -> None:
        current = asyncio.current_task()
        for task in (self._task, self._monitor):
            if task is not None and task is not current and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._task = None
        self._monitor = None

    async def _teardown(self) -> None:
        if self.worker is not None:
            await self.worker.aclose()
            self.worker = None
        if self.tunnel is not None:
            await self.tunnel.close()
            self.tunnel = None
        self._ready.clear()

    def _end(self, state: str, error: str | None) -> None:
        assert self.session is not None
        self.db.end_session(self.session["id"], state, error)
        self.session = self.db.get_session(self.session["id"])
        self._ready.clear()
        self.step = None
        self.health = None
        self._publish()
        for cb in self.on_end:
            cb()

    async def _fail(self, error: str, stop_vm: bool) -> None:
        log.error("session error: %s", error)
        await self._teardown()
        if stop_vm:
            try:
                await self.colab.stop()
            except ColabError as e:
                log.warning("colab stop failed: %s", e)
        self._end("error", error)

    # -- bootstrap -------------------------------------------------------------------------

    async def _bootstrap(
        self, allocate: bool, restart_worker: bool = False, reattach: bool = False
    ) -> None:
        assert self.session is not None
        try:
            if allocate:
                await self.colab.new(self.session["gpu"], self.session["high_mem"])
            if self.tunnel is None:
                self._step("Connecting")
                await self._connect()
            assert self.worker is not None
            healthy = False
            if reattach and not restart_worker:
                try:
                    self.health = await self.worker.health()
                    healthy = True
                except WorkerError:
                    pass
            if not healthy:
                self._step("Installing worker")
                await self._install()
                self._step("Starting worker")
                await self._start_worker()
                self._step("Waiting for worker")
                await self._wait_health()
            self._step("Sending Drive token")
            await self._push_token()
            self._step(None)
            self._update(state="ready", last_activity_at=now())
            self._monitor = asyncio.create_task(self._watch())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("bootstrap failed")
            self._task = None
            await self._fail(f"Session start failed: {e}", stop_vm=True)

    async def _ensure_key(self) -> None:
        """Degas's own SSH key, generated on first use."""
        key = self.config.ssh_key
        if key.exists():
            return
        key.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        proc = await asyncio.create_subprocess_exec(
            "ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "degas", "-f", str(key),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )  # fmt: skip
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise SessionError(f"ssh-keygen failed: {out.decode(errors='replace').strip()}")

    async def _connect(self) -> None:
        await self._ensure_key()
        tunnel = self._tunnel_factory()
        await tunnel.open()
        self.tunnel = tunnel
        self.worker = self._worker_factory(tunnel)

    async def _install(self) -> None:
        assert self.tunnel is not None
        t = self.tunnel
        bundle = self._bundle()
        current = await t.run(f"mkdir -p {REMOTE}/bin && cat {REMOTE}/bundle.sha 2>/dev/null; true")
        if current.strip() != bundle.sha256:
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "bundle.tar"
                local.write_bytes(bundle.data)
                await t.upload(local, f"{REMOTE}/bundle.tar")
            await t.run(
                f"cd {REMOTE} && rm -rf worker && mkdir worker && tar xf bundle.tar -C worker"
                f" && rm bundle.tar && echo {bundle.sha256} > bundle.sha"
            )
        rclone = self.config.colab.rclone_binary
        if rclone is not None:
            has = await t.run(f"test -x {REMOTE}/bin/rclone && echo yes; true")
            if has.strip() != "yes":
                await t.upload(rclone, f"{REMOTE}/bin/rclone")
                await t.run(f"chmod +x {REMOTE}/bin/rclone")
        else:
            await t.run(
                f"test -x {REMOTE}/bin/rclone || (cd /tmp && curl -fsSLo rclone.zip {RCLONE_URL}"
                f" && unzip -oq rclone.zip && cp rclone-*-linux-amd64/rclone {REMOTE}/bin/"
                f" && chmod +x {REMOTE}/bin/rclone)",
                timeout=300,
            )
        # The Colab image already has these (Phase 0); install only if missing.
        await t.run(
            "python3 -c 'import fastapi, uvicorn' 2>/dev/null"
            " || python3 -m pip install -q fastapi uvicorn",
            timeout=300,
        )

    async def _start_worker(self) -> None:
        code = BOOTSTRAP.replace("{port}", str(self.config.colab.worker_port))
        out = await self.colab.exec(code, timeout=120)
        if STARTED_MARKER not in out:
            raise SessionError(f"Worker did not start: {out.strip()[-500:]}")

    async def _wait_health(self) -> None:
        assert self.worker is not None
        deadline = time.monotonic() + self.iv.health_wait
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.health = await self.worker.health(timeout=60)
                self._publish()
                return
            except WorkerError as e:
                last = e
                await asyncio.sleep(1)
        raise SessionError(f"Worker did not become healthy: {last}")

    async def _push_token(self) -> None:
        if not self.drive.authorized:
            self.drive_error = "Drive is not authorized: run `degas auth drive`"
            return
        assert self.worker is not None
        try:
            token = await self.drive.access_token(min_valid_s=20 * 60)
            await self.worker.drive_token(token.token, token.expiry_rfc3339, self.config.drive.root)
            self.drive_error = None
        except (DriveError, WorkerError, OSError) as e:
            log.warning("could not push Drive token: %s", e)
            self.drive_error = str(e)

    # -- liveness --------------------------------------------------------------------------

    async def _watch(self) -> None:
        clock = time.monotonic
        last_health = last_status = last_exec = last_token = clock()
        while True:
            await asyncio.sleep(self.iv.tick)
            t = clock()
            if t - last_health >= self.iv.health:
                last_health = t
                if not await self._check_health():
                    return
            if t - last_status >= self.iv.colab_status:
                last_status = t
                if not await self._still_allocated():
                    return
            if self.config.colab.exec_heartbeat and t - last_exec >= self.iv.exec_heartbeat:
                last_exec = t
                await self._guard(self.colab.exec("pass", timeout=60), "exec heartbeat")
            if t - last_token >= self.iv.drive_token:
                last_token = t
                await self._push_token()
                self._publish()
            if self._idle_expired():
                self._monitor = None
                asyncio.create_task(self.stop("idle"))  # noqa: RUF006 - stop() awaits this task
                return

    async def _guard(self, coro: Awaitable[Any], what: str) -> None:
        try:
            await coro
        except (ColabError, TunnelError) as e:
            log.warning("%s failed: %s", what, e)

    def _idle_expired(self) -> bool:
        deadline = self.idle_deadline()
        return (
            self.state == "ready"
            and deadline is not None
            and datetime.now(UTC) >= deadline
            and not self.has_pending_jobs()
        )

    async def _still_allocated(self) -> bool:
        try:
            alive = await self.colab.is_alive()
        except ColabError as e:
            log.warning("colab status failed: %s", e)
            return True
        if not alive:
            self._monitor = None
            await self._fail("The Colab VM was reclaimed", stop_vm=False)
        return alive

    async def _check_health(self) -> bool:
        """Health check with recovery: reconnect the tunnel, then restart the worker."""
        assert self.worker is not None
        try:
            self.health = await self.worker.health()
            self._publish()
            return True
        except WorkerError as e:
            log.warning("worker health failed: %s", e)
        if not await self._still_allocated():
            return False
        try:
            await self._teardown()
            await self._connect()
            assert self.worker is not None
            try:
                self.health = await self.worker.health()
            except WorkerError:
                self._step("Restarting worker")
                await self._install()
                await self._start_worker()
                await self._wait_health()
                await self._push_token()
                self._step(None)
            self._update(state=self.state)  # re-arm readiness with the new worker
            return True
        except Exception as e:
            log.exception("session recovery failed")
            self._monitor = None
            await self._fail(f"Lost the worker: {e}", stop_vm=True)
            return False

    # -- server restart --------------------------------------------------------------------

    async def recover(self) -> None:
        """Reconcile the last session row with Colab after a server (re)start."""
        row = self.db.latest_session()
        if row is None or row["state"] not in ACTIVE_SESSION_STATES:
            self.session = row
            return
        self.session = row
        try:
            alive = await self.colab.is_alive()
        except ColabError as e:
            log.warning("colab status failed during recovery: %s", e)
            alive = False
        if not alive:
            self._end("stopped", "The VM was gone when the server restarted")
            return
        if row["state"] == "stopping":
            self.session = {**row, "state": "ready"}
            await self.stop("server restart")
            return
        self._update(state="starting", last_activity_at=now())
        self._step("Reconnecting")
        self._task = asyncio.create_task(self._bootstrap(allocate=False, reattach=True))

    async def aclose(self) -> None:
        """Server shutdown: leave the VM running; the next start reattaches to it."""
        await self._cancel_tasks()
        await self._teardown()
