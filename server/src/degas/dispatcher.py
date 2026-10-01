"""Job dispatcher: runs queued jobs on the worker, one at a time.

Worker events are relayed to the phone over `/api/events`. Each output is fetched
as soon as it is reported, stored as a blob, and then acknowledged on the worker.

Once the running job is past copying its own assets, the next queued job's
models and LoRAs are prefetched into the VM's cache (design §5).

A finished video extension also gets a stitched result: the clip it extends
followed by the continuation (design §6.4).
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from degas import media
from degas.blobs import BlobStore, unref
from degas.colab.session import SessionManager
from degas.colab.worker_client import WorkerBusyError, WorkerClient, WorkerError
from degas.db import Database, JobRow, JobRuntime, SavedConfig, now
from degas.events import EventBus
from degas.families.base import spec_assets
from degas.inputs import Inputs
from degas.library import input_blobs, saved_config, staged_blobs
from degas_worker.spec import Spec

log = logging.getLogger(__name__)

TERMINAL = ("done", "error", "cancelled")
MAX_REATTACH = 5


class Dispatcher:
    def __init__(
        self,
        db: Database,
        blobs: BlobStore,
        bus: EventBus,
        sessions: SessionManager,
        inputs: Inputs,
    ) -> None:
        self.db = db
        self.blobs = blobs
        self.inputs = inputs
        self.bus = bus
        self.sessions = sessions
        self.progress: dict[str, dict[str, Any]] = {}  # job id → last progress event
        self._running: str | None = None
        self._cancel_requested: set[str] = set()
        self._wake = asyncio.Event()
        self._prefetch: asyncio.Task[None] | None = None
        self._prefetched_after: str | None = None  # the running job that triggered a prefetch
        # Called with the job when it finishes (done, error or cancelled).
        self.on_finish: list[Callable[[JobRow], None]] = []
        sessions.on_ready.append(self.wake)
        sessions.on_end.append(self._orphans)
        sessions.has_pending_jobs = lambda: self.db.count_pending() > 0

    def wake(self) -> None:
        self._wake.set()

    def publish_job(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is not None:
            self.bus.publish({"type": "job", "job": self.describe(job)})

    def describe(self, job: JobRow) -> dict[str, Any]:
        # The log (a failed job's worker traceback) is for debugging, not for the phone.
        fields = {k: v for k, v in job.items() if k != "log"}
        return {**fields, "progress": self.progress.get(job["id"])}

    # -- queue -----------------------------------------------------------------------------

    def submit(self, spec: Spec, seeds: list[int]) -> JobRow:
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

    def move(self, job_id: str, index: int) -> bool:
        """Reorder the queue: put a queued job at `index` (0 runs next)."""
        job = self.db.get_job(job_id)
        if job is None or job["status"] != "queued":
            return False
        before = {j["id"]: j["queue_position"] for j in self.db.jobs_with_status("queued")}
        self.db.move_job(job_id, index)
        for j in self.db.jobs_with_status("queued"):
            if before.get(j["id"]) != j["queue_position"]:
                self.publish_job(j["id"])
        self._prefetched_after = None  # the next job may have changed: prefetch it instead
        self.sessions.touch()
        return True

    def restore(self, job_id: str) -> bool:
        """Undo cancelling a job that never started: it goes back to its place in the queue."""
        job = self.db.get_job(job_id)
        if job is None or job["status"] != "cancelled" or job["started_at"] is not None:
            return False
        self.db.update_job(job_id, status="queued", finished_at=None)
        self.publish_job(job_id)
        self.sessions.touch()
        self.wake()
        return True

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

    async def _run_job(self, worker: WorkerClient, job: JobRow, resume: bool) -> None:
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
        if status == "done" and (job["spec"].get("inputs") or {}).get("extends"):
            try:
                await self._stitch(job_id)
            except media.MediaError as e:
                log.warning("stitching %s failed: %s", job_id, e)
                error = f"The clip was made, but joining it to the one it extends failed: {e}"
        self._finish(job_id, status, error)
        self.sessions.set_busy(False)
        await self.sessions.refresh_health()  # the model cache may have changed

    def _record_runtime(self, job_id: str) -> None:
        """What the job ran on, for saved configs (design §6.4)."""
        job = self.db.get_job(job_id)
        session = self.sessions.session
        health = self.sessions.health or {}
        versions = health.get("versions") or {}
        runtime: JobRuntime = {"gpu": session["gpu"] if session else None}
        if versions.get("diffusers"):
            runtime["diffusers"] = versions["diffusers"]
        if versions.get("torch"):
            runtime["torch"] = versions["torch"]
        if job and (started_at := job.get("started_at")):
            started = datetime.fromisoformat(started_at)
            runtime["duration_s"] = round((datetime.now(UTC) - started).total_seconds(), 1)
        self.db.update_job(job_id, runtime=runtime)

    def _finish(self, job_id: str, status: str, error: str | None) -> None:
        self.db.update_job(job_id, status=status, error=error, finished_at=now())
        self.progress.pop(job_id, None)
        self.publish_job(job_id)
        job = self.db.get_job(job_id)
        if job is None:
            return
        for cb in self.on_finish:
            try:
                cb(job)
            except Exception:
                log.exception("job finish callback failed")

    async def _stage_inputs(self, worker: WorkerClient, spec: Spec) -> None:
        """Send input blobs (content-addressed, so at most once per session)."""
        for sha in staged_blobs(spec):
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
                    return "error", self._failed(job_id, terminal)
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

    def _failed(self, job_id: str, event: dict[str, Any]) -> str:
        """Keep the worker's traceback in the job's log, and return the message to show."""
        if trace := event.get("trace"):
            log.warning("job %s failed on the worker:\n%s", job_id, trace)
            self.db.update_job(job_id, log=trace)
        return str(event.get("message") or "Job failed")

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
            w, h, duration = await self._measure(sha, media_type)
            result = self.db.insert_result(
                job_id, item, sha, media_type, event.get("seed"), w, h, duration
            )
            self.bus.publish({"type": "result", "result": result})
        try:
            await worker.ack_output(job_id, item)
        except WorkerError as e:
            log.warning("could not acknowledge output %s/%s: %s", job_id, item, e)

    async def _measure(
        self, sha: str, media_type: str
    ) -> tuple[int | None, int | None, float | None]:
        """Width, height and (for a video) duration of a stored output; makes a video's poster."""
        if media_type.startswith("image/"):
            size = self.blobs.image_size(sha)
            return (size[0], size[1], None) if size else (None, None, None)
        path = self.blobs.path(sha)
        info = await media.probe(path) if path else None
        if info is None:
            return None, None, None
        try:
            await self.inputs.poster(sha)
        except media.MediaError as e:
            log.warning("no poster for %s: %s", sha, e)
        return info["width"], info["height"], info["duration"]

    # -- video extension -------------------------------------------------------------------

    async def _stitch(self, job_id: str) -> None:
        """For each new clip, store the chain: the extended clip, then this one."""
        job = self.db.get_job(job_id)
        if job is None:
            return
        spec = job["spec"]
        parent_sha = unref(spec["inputs"]["extends"])
        parent = self.blobs.path(parent_sha)
        if parent is None:
            raise media.MediaError("the extended clip is no longer stored")
        before = self._segments(parent_sha)
        n = len(job["seeds"])
        for clip in self.db.list_results(job_id=job_id, limit=200):
            if clip["item_index"] >= n or self.db.has_result(job_id, n + clip["item_index"]):
                continue
            path = self.blobs.path(clip["blob_sha"])
            if path is None:
                continue
            data = await media.stitch(parent, path, float(spec["params"].get("fps") or 0))
            sha = self.blobs.put(data, "video/mp4")
            w, h, duration = await self._measure(sha, "video/mp4")
            result = self.db.insert_result(
                job_id,
                n + clip["item_index"],
                sha,
                "video/mp4",
                clip["seed"],
                w,
                h,
                duration,
                segments=[*before, saved_config(job, clip)],
            )
            self.bus.publish({"type": "result", "result": result})

    def _segments(self, sha: str) -> list[SavedConfig]:
        """The configs of the clips making up a stored video, oldest first."""
        result = self.db.result_for_blob(sha)
        if result is not None:
            if segments := result.get("segments"):
                return list(segments)
            job = self.db.get_job(result["job_id"])
            if job is not None:
                return [saved_config(job, result)]
        item = self.db.library_item_for_blob(sha)
        if item is not None:
            config = item["config"]
            return list(config.get("segments") or [config])
        return []
