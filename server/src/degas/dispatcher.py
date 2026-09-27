"""Job dispatcher: runs queued jobs on the worker, one at a time.

Worker events are relayed to the phone over `/api/events`. Each output is fetched
as soon as it is reported, stored as a blob, and then acknowledged on the worker.

Once the running job is past copying its own assets, the next queued job's
models and LoRAs are prefetched into the VM's cache (design §5).
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from degas.blobs import BlobStore
from degas.colab.session import SessionManager
from degas.colab.worker_client import WorkerBusyError, WorkerClient, WorkerError
from degas.db import Database, now
from degas.events import EventBus
from degas.families.base import spec_assets
from degas.library import input_blobs

log = logging.getLogger(__name__)

TERMINAL = ("done", "error", "cancelled")
MAX_REATTACH = 5


class Dispatcher:
    def __init__(
        self, db: Database, blobs: BlobStore, bus: EventBus, sessions: SessionManager
    ) -> None:
        self.db = db
        self.blobs = blobs
        self.bus = bus
        self.sessions = sessions
        self.progress: dict[str, dict[str, Any]] = {}  # job id → last progress event
        self._running: str | None = None
        self._cancel_requested: set[str] = set()
        self._wake = asyncio.Event()
        self._prefetch: asyncio.Task[None] | None = None
        self._prefetched_after: str | None = None  # the running job that triggered a prefetch
        sessions.on_ready.append(self.wake)
        sessions.on_end.append(self._orphans)
        sessions.has_pending_jobs = lambda: self.db.count_pending() > 0

    def wake(self) -> None:
        self._wake.set()

    def publish_job(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is not None:
            self.bus.publish({"type": "job", "job": self.describe(job)})

    def describe(self, job: dict[str, Any]) -> dict[str, Any]:
        return {**job, "progress": self.progress.get(job["id"])}

    # -- queue -----------------------------------------------------------------------------

    def submit(self, spec: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
        job = self.db.insert_job(spec, seeds)
        for sha in input_blobs(spec):
            self.db.add_blob_ref(sha, "job", job["id"])
        self.publish_job(job["id"])
        self.sessions.touch()
        self.wake()
        return job

    async def cancel(self, job_id: str) -> bool:
        job = self.db.get_job(job_id)
        if job is None:
            return False
        if job["status"] == "queued":
            self.db.update_job(job_id, status="cancelled", finished_at=now())
            self.publish_job(job_id)
            return True
        if job["status"] == "running":
            self._cancel_requested.add(job_id)
            worker = self.sessions.worker
            if worker is not None and self._running == job_id:
                try:
                    await worker.cancel(job_id)
                except WorkerError as e:
                    log.warning("cancel %s failed: %s", job_id, e)
            return True
        return False

    def _orphans(self) -> None:
        """The session ended: a running job that nothing is following has failed."""
        for job in self.db.jobs_with_status("running"):
            if job["id"] != self._running:
                self._finish(job["id"], "error", "The session ended while the job was running")

    # -- main loop -------------------------------------------------------------------------

    async def run(self) -> None:
        if not self.sessions.active:
            self._orphans()
        while True:
            await self.sessions.wait_ready()
            worker = self.sessions.worker
            if worker is None:
                await asyncio.sleep(0.5)
                continue
            # After a server restart, re-attach to a job the worker may still be running.
            resumed = self.db.jobs_with_status("running")
            job = resumed[0] if resumed else self.db.next_queued()
            if job is None:
                self._wake.clear()
                if self.db.next_queued() is None:
                    await self._wake.wait()
                continue
            try:
                await self._run_job(worker, job, resume=bool(resumed))
            except Exception:
                log.exception("dispatcher error on job %s", job["id"])
                self._finish(job["id"], "error", "Internal dispatcher error")

    async def _run_job(self, worker: WorkerClient, job: dict[str, Any], resume: bool) -> None:
        job_id = job["id"]
        self._running = job_id
        session = self.sessions.session
        if not resume:
            self.db.update_job(
                job_id,
                status="running",
                session_id=session["id"] if session else None,
                started_at=now(),
            )
            self.publish_job(job_id)
        self.sessions.set_busy(True)
        try:
            if not resume:
                await self._stage_inputs(worker, job["spec"])
                await worker.start_job(job_id, job["spec"], job["seeds"])
            if job_id in self._cancel_requested:
                await worker.cancel(job_id)
            status, error = await self._follow(worker, job_id)
        except WorkerBusyError:
            status, error = "error", "The worker is busy with another job"
        except WorkerError as e:
            status, error = "error", str(e)
        finally:
            self._running = None
            self._cancel_requested.discard(job_id)
        self._record_runtime(job_id)
        self._finish(job_id, status, error)
        self.sessions.set_busy(False)
        await self.sessions.refresh_health()  # the model cache may have changed

    def _record_runtime(self, job_id: str) -> None:
        """What the job ran on, for saved configs (design §6.4)."""
        job = self.db.get_job(job_id)
        session = self.sessions.session
        health = self.sessions.health or {}
        runtime: dict[str, Any] = {"gpu": session["gpu"] if session else None}
        for key in ("diffusers", "torch"):
            if (health.get("versions") or {}).get(key):
                runtime[key] = health["versions"][key]
        if job and job.get("started_at"):
            started = datetime.fromisoformat(job["started_at"])
            runtime["duration_s"] = round((datetime.now(UTC) - started).total_seconds(), 1)
        self.db.update_job(job_id, runtime=runtime)

    def _finish(self, job_id: str, status: str, error: str | None) -> None:
        self.db.update_job(job_id, status=status, error=error, finished_at=now())
        self.progress.pop(job_id, None)
        self.publish_job(job_id)

    async def _stage_inputs(self, worker: WorkerClient, spec: dict[str, Any]) -> None:
        """Send input blobs (content-addressed, so at most once per session)."""
        for sha in _input_blobs(spec):
            if await worker.has_blob(sha):
                continue
            path = self.blobs.path(sha)
            if path is None:
                raise WorkerError(f"Input blob {sha} is missing")
            await worker.put_blob(sha, path.read_bytes())

    async def _follow(self, worker: WorkerClient, job_id: str) -> tuple[str, str | None]:
        for _attempt in range(MAX_REATTACH):
            terminal: dict[str, Any] | None = None
            try:
                async for event in worker.events(job_id):
                    kind = event.get("t")
                    if kind == "progress":
                        self.progress[job_id] = event
                        self.bus.publish({"type": "progress", **event})
                        if event.get("phase") != "copy":
                            self._start_prefetch(worker, job_id)
                    elif kind == "output":
                        await self._fetch_output(worker, job_id, event)
                    elif kind in TERMINAL:
                        terminal = event
                        break
            except WorkerError as e:
                log.warning("event stream for %s broke: %s", job_id, e)
            if terminal is not None:
                await self._collect(worker, job_id)
                if terminal["t"] == "error":
                    return "error", str(terminal.get("message") or "Job failed")
                return str(terminal["t"]), None

            # The stream ended without a terminal event: reconcile via /state.
            try:
                state = await worker.state()
            except WorkerError as e:
                return "error", f"Lost contact with the worker: {e}"
            current = state.get("job") or {}
            if current.get("id") != job_id:
                await self._collect(worker, job_id, state.get("outputs"))
                return "error", "The worker lost the job"
            if current.get("status") != "running":
                continue  # finished meanwhile: replaying the events gives the outcome
            await asyncio.sleep(1)
        return "error", "Could not follow the job's progress"

    # -- prefetch --------------------------------------------------------------------------

    def _start_prefetch(self, worker: WorkerClient, running: str) -> None:
        if self._prefetched_after == running:
            return
        if self._prefetch is not None and not self._prefetch.done():
            return
        self._prefetched_after = running
        self._prefetch = asyncio.create_task(self._prefetch_next(worker))

    async def _prefetch_next(self, worker: WorkerClient) -> None:
        """Copy the next queued job's assets that aren't on the VM yet."""
        job = self.db.next_queued()
        if job is None:
            return
        cache = (self.sessions.health or {}).get("cache") or {}
        cached = {f["path"]: f["size"] for f in cache.get("files", [])}
        needed = [
            {"path": a["path"], "size": a.get("size")}
            for a in spec_assets(job["spec"])
            if a["path"] not in cached or a.get("size") not in (None, cached[a["path"]])
        ]
        if not needed:
            return
        job_id = job["id"]
        try:
            async for event in worker.fetch_assets(needed):
                if event.get("t") == "error":
                    log.warning("prefetch for %s failed: %s", job_id, event.get("message"))
                if event.get("t") != "progress" or self._running == job_id:
                    continue  # once the job runs, its own events report progress
                progress = {
                    "job": job_id,
                    "item": 0,
                    "phase": "copy",
                    "step": event.get("step", 0),
                    "steps": event.get("steps", 0),
                    "asset": event.get("asset"),
                }
                self.progress[job_id] = progress
                self.bus.publish({"type": "progress", **progress})
        except WorkerError as e:
            log.warning("prefetch for %s failed: %s", job_id, e)
        finally:
            if self._running != job_id and self.progress.pop(job_id, None) is not None:
                self.publish_job(job_id)
        await self.sessions.refresh_health()

    async def _collect(
        self, worker: WorkerClient, job_id: str, outputs: list[dict[str, Any]] | None = None
    ) -> None:
        """Fetch any outputs of this job still held by the worker."""
        if outputs is None:
            try:
                outputs = (await worker.state()).get("outputs", [])
            except WorkerError:
                return
        for output in outputs or []:
            if output.get("job") == job_id:
                await self._fetch_output(worker, job_id, output)

    async def _fetch_output(self, worker: WorkerClient, job_id: str, event: dict[str, Any]) -> None:
        item = int(event["item"])
        if not self.db.has_result(job_id, item):
            try:
                data, content_type = await worker.get_output(job_id, item)
            except WorkerError as e:
                log.warning("could not fetch output %s/%s: %s", job_id, item, e)
                return
            media_type = str(event.get("media_type") or content_type)
            sha = self.blobs.put(data, media_type)
            size = self.blobs.image_size(sha) if media_type.startswith("image/") else None
            result = self.db.insert_result(
                job_id,
                item,
                sha,
                media_type,
                event.get("seed"),
                size[0] if size else None,
                size[1] if size else None,
            )
            self.bus.publish({"type": "result", "result": result})
        try:
            await worker.ack_output(job_id, item)
        except WorkerError as e:
            log.warning("could not acknowledge output %s/%s: %s", job_id, item, e)


def _input_blobs(spec: dict[str, Any]) -> list[str]:
    shas: list[str] = []
    inputs = spec.get("inputs") or {}
    for key in ("source", "mask"):
        value = inputs.get(key)
        if isinstance(value, str) and value.startswith("sha256:"):
            shas.append(value.removeprefix("sha256:"))
    for unit in spec.get("control") or []:
        for key in ("image", "mask"):
            value = unit.get(key)
            if isinstance(value, str) and value.startswith("sha256:"):
                shas.append(value.removeprefix("sha256:"))
    return shas
