"""Job execution: one generation slot, run in a background thread.

Events for each job are kept in memory so an SSE subscriber can (re)attach at any
time and replay from the start. Outputs are written to disk with a small JSON
sidecar and stay there until the server acknowledges them (DELETE), so they
survive a dropped connection.
"""

import asyncio
import json
import logging
import re
import shutil
import threading
import traceback
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from degas_worker.cache import AssetCache
from degas_worker.families.base import FamilyRunner, JobCancelled, Output
from degas_worker.paths import Paths
from degas_worker.spec import Spec

log = logging.getLogger(__name__)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TERMINAL = frozenset({"done", "error", "cancelled"})
Event = dict[str, Any]


class WorkerBusy(Exception):  # noqa: N818
    pass


@dataclass
class JobRecord:
    id: str
    spec: Spec
    seeds: list[int]
    loop: asyncio.AbstractEventLoop
    status: str = "running"
    events: list[Event] = field(default_factory=list)
    progress: Event | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _waiters: set[asyncio.Event] = field(default_factory=set)

    def emit(self, event: Event) -> None:
        """Record an event (thread-safe) and wake subscribers."""
        event = {**event, "job": self.id}
        with self._lock:
            self.events.append(event)
            if event["t"] == "progress":
                self.progress = event
            elif event["t"] in TERMINAL:
                self.status = event["t"]
        self.loop.call_soon_threadsafe(self._wake)

    def _wake(self) -> None:
        for waiter in self._waiters:
            waiter.set()

    async def subscribe(self) -> AsyncIterator[Event]:
        """Replay all events so far, then follow until a terminal event."""
        waiter = asyncio.Event()
        self._waiters.add(waiter)
        try:
            i = 0
            while True:
                while i < len(self.events):
                    event = self.events[i]
                    i += 1
                    yield event
                    if event["t"] in TERMINAL:
                        return
                await waiter.wait()
                waiter.clear()
        finally:
            self._waiters.discard(waiter)

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "progress": self.progress}


class _Context:
    def __init__(self, record: JobRecord, cache: AssetCache) -> None:
        self._record = record
        self._cache = cache
        self.pins: list[str] = []  # assets this job uses, kept out of eviction until it ends

    def progress(
        self, item: int, phase: str, step: int, steps: int, asset: str | None = None
    ) -> None:
        event: Event = {"t": "progress", "item": item, "phase": phase, "step": step}
        event["steps"] = steps
        if asset is not None:
            event["asset"] = asset
        self._record.emit(event)

    def check_cancelled(self) -> None:
        if self._record.cancel.is_set():
            raise JobCancelled

    def fetch_asset(self, path: str, size: int | None = None, item: int = 0) -> Path:
        self._cache.pin(path)
        self.pins.append(path)
        if self._cache.is_cached(path, size):
            return self._cache.ensure(path, size)
        self.progress(item, "copy", 0, size or 0, path)

        def on_progress(done: int, total: int) -> None:
            self.check_cancelled()
            self.progress(item, "copy", done, total, path)

        return self._cache.ensure(path, size, on_progress)

    def blob(self, ref: str) -> Path:
        sha = ref.removeprefix("sha256:")
        path = self._cache.paths.blobs / sha
        if not _SHA256.match(sha) or not path.exists():
            raise ValueError(f"Input {ref} was not staged on the worker")
        return path

    def release(self) -> None:
        for path in self.pins:
            self._cache.unpin(path)
        self.pins.clear()


class JobManager:
    def __init__(
        self,
        paths: Paths,
        cache: AssetCache,
        runners: dict[str, Callable[[], FamilyRunner]],
    ) -> None:
        self.paths = paths
        self.cache = cache
        self._factories = runners
        self._runner: FamilyRunner | None = None
        self._runner_family: str | None = None
        self.current: JobRecord | None = None  # running, or the most recent job
        self._jobs: dict[str, JobRecord] = {}

    @property
    def loaded_family(self) -> str | None:
        return self._runner_family

    @property
    def busy(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def start(self, job_id: str, spec: Spec, seeds: list[int]) -> JobRecord:
        if self.busy:
            raise WorkerBusy
        if job_id in self._jobs:
            raise ValueError(f"Job {job_id} already exists")
        if spec.get("family") not in self._factories:
            raise ValueError(f"Unknown family {spec.get('family')!r}")
        record = JobRecord(id=job_id, spec=spec, seeds=seeds, loop=asyncio.get_running_loop())
        self._jobs = {job_id: record}  # keep only the latest finished job for /state
        self.current = record
        threading.Thread(
            target=self._run, args=(record,), name=f"job-{job_id}", daemon=True
        ).start()
        return record

    def cancel(self, job_id: str) -> bool:
        record = self._jobs.get(job_id)
        if record is None or record.status != "running":
            return False
        record.cancel.set()
        return True

    def _runner_for(self, family: str) -> FamilyRunner:
        if self._runner is not None and self._runner_family != family:
            self._runner.unload()
            self._runner = None
        if self._runner is None:
            self._runner = self._factories[family]()
            self._runner_family = family
        return self._runner

    def unload(self) -> None:
        if self._runner is not None:
            self._runner.unload()
            self._runner = None
            self._runner_family = None

    def _run(self, record: JobRecord) -> None:
        ctx = _Context(record, self.cache)
        try:
            runner = self._runner_for(record.spec["family"])
            for output in runner.run(record.spec, record.seeds, ctx):
                self._store(record.id, output)
                record.emit(
                    {
                        "t": "output",
                        "item": output.item,
                        "seed": output.seed,
                        "media_type": output.media_type,
                    }
                )
            record.emit({"t": "done"})
        except JobCancelled:
            record.emit({"t": "cancelled"})
        except Exception as e:
            log.exception("job %s failed", record.id)
            message = str(e) or type(e).__name__
            record.emit({"t": "error", "message": message, "trace": traceback.format_exc()})
        finally:
            ctx.release()

    # -- outputs ---------------------------------------------------------------------------

    def _output_dir(self, job_id: str) -> Path:
        if not job_id or "/" in job_id or job_id.startswith("."):
            raise ValueError(f"Invalid job id {job_id!r}")
        return self.paths.outputs / job_id

    def _store(self, job_id: str, output: Output) -> None:
        d = self._output_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{output.item}.{output.ext}").write_bytes(output.data)
        meta = {
            "job": job_id,
            "item": output.item,
            "seed": output.seed,
            "media_type": output.media_type,
            "file": f"{output.item}.{output.ext}",
        }
        (d / f"{output.item}.json").write_text(json.dumps(meta))

    def outputs(self) -> list[dict[str, Any]]:
        """Outputs not yet acknowledged by the server."""
        found = []
        if self.paths.outputs.exists():
            for meta in sorted(self.paths.outputs.glob("*/*.json")):
                try:
                    found.append(json.loads(meta.read_text()))
                except (OSError, ValueError):
                    continue
        return found

    def output_file(self, job_id: str, item: int) -> tuple[Path, str] | None:
        meta_path = self._output_dir(job_id) / f"{item}.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        return self._output_dir(job_id) / meta["file"], meta["media_type"]

    def ack(self, job_id: str, item: int) -> bool:
        found = self.output_file(job_id, item)
        if found is None:
            return False
        found[0].unlink(missing_ok=True)
        (self._output_dir(job_id) / f"{item}.json").unlink(missing_ok=True)
        d = self._output_dir(job_id)
        if not any(d.iterdir()):
            shutil.rmtree(d, ignore_errors=True)
        return True
