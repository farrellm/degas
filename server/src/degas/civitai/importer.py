"""Copy a Civitai LoRA into Drive, checked against Civitai's hash, with a sidecar and preview.

The file streams from Civitai straight into `rclone rcat`, so nothing large touches the disk.
Its SHA-256 is compared with Civitai's and its md5 with Drive's, and a mismatch deletes it.
"""

import asyncio
import contextlib
import hashlib
import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from degas.civitai.client import Civitai, CivitaiError
from degas.civitai.plan import ImportPlan, PlanError, PlannedFile, plan_import
from degas.rclone import RcloneError, Remote, preview_jpeg

log = logging.getLogger(__name__)

PROGRESS_INTERVAL_S = 0.5

# (stage, bytes done, bytes in all): stage is "download", "sidecar" or "preview".
Progress = Callable[[str, int, int], None]


class CivitaiImportError(RuntimeError):
    """The import can't go ahead, or failed partway."""


def find_duplicates(plan: ImportPlan, index: list[dict[str, Any]]) -> list[str]:
    """Why the plan would duplicate what's in the Drive index, if it would."""
    problems = []
    by_sha = {a["sha256"]: a["path"] for a in index if a.get("sha256")}
    paths = {a["path"] for a in index}
    for f in plan.files:
        if f.sha256 in by_sha:
            problems.append(f"{f.civitai_name} is already in Drive as {by_sha[f.sha256]}")
        elif f.path in paths:
            problems.append(f"{f.path} already exists")
    return problems


class Importer:
    def __init__(
        self,
        civitai: Civitai,
        remote: Remote,
        rclone_remote: str,
        drive_root: str,
        families: set[str],
    ) -> None:
        self.civitai = civitai
        self.remote = remote
        self.base = f"{rclone_remote}{drive_root.strip('/')}"
        self.families = families

    async def plan(
        self,
        ref: str,
        index: list[dict[str, Any]],
        *,
        family: str | None = None,
        name: str | None = None,
        weight: float | None = None,
        force: bool = False,
    ) -> ImportPlan:
        """Resolve a link to a plan; duplicates are errors unless `force`."""
        version = await self.civitai.version(ref)
        plan = plan_import(version, families=self.families, family=family, name=name, weight=weight)
        duplicates = find_duplicates(plan, index)
        if duplicates and not force:
            raise PlanError("; ".join(duplicates))
        plan.warnings.extend(f"Going ahead anyway: {d}" for d in duplicates)
        return plan

    async def run(self, plan: ImportPlan, progress: Progress | None = None) -> list[str]:
        """Upload the plan's files, sidecars and preview; return the LoRAs' Drive paths."""

        def report(stage: str, done: int, total: int) -> None:
            if progress is not None:
                progress(stage, done, total)

        folder = f"{self.base}/{plan.folder}"
        await self.remote.run("mkdir", folder)  # first: parallel uploads duplicate folders
        total = sum(f.size for f in plan.files)
        done = 0
        for f in plan.files:

            def downloaded(n: int, before: int = done) -> None:
                report("download", before + n, total)

            await self._upload(f, downloaded)
            done += f.size
        report("sidecar", total, total)
        sidecar = plan.sidecar().encode()
        for f in plan.files:
            await self.remote.run("rcat", f"{folder}/{f.stem}.yaml", stdin=sidecar)
        if plan.preview_url:
            report("preview", total, total)
            # The picker shows a pair's high-noise preview first; name it after that half.
            await self._preview(plan, f"{folder}/{plan.files[0].stem}.jpg")
        return [f.path for f in plan.files]

    async def backfill(self, asset: dict[str, Any]) -> ImportPlan | None:
        """Write a sidecar (and a preview, if it has none) for a LoRA already in Drive that
        Civitai knows by its SHA-256. None if Civitai doesn't know it."""
        if not asset.get("sha256"):
            return None
        version = await self.civitai.by_hash(asset["sha256"])
        if version is None:
            return None
        plan = plan_import(version, families=self.families, family=asset["family"])
        folder, file = asset["path"].rsplit("/", 1)
        stem = file.rsplit(".", 1)[0]
        await self.remote.run(
            "rcat", f"{self.base}/{folder}/{stem}.yaml", stdin=plan.sidecar().encode()
        )
        if plan.preview_url and not asset.get("preview_thumb"):
            await self._preview(plan, f"{self.base}/{folder}/{stem}.jpg")
        return plan

    async def _preview(self, plan: ImportPlan, target: str) -> None:
        assert plan.preview_url
        try:
            jpeg = preview_jpeg(await self.civitai.fetch(plan.preview_url))
        except (CivitaiError, OSError) as e:  # PIL's UnidentifiedImageError is an OSError
            log.warning("no preview for %s: %s", plan.source, e)
            return
        await self.remote.run("rcat", target, stdin=jpeg)

    async def _upload(self, f: PlannedFile, progress: Callable[[int], None]) -> None:
        target = f"{self.base}/{f.path}"
        sha = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        count = 0

        async with self.civitai.download(f.url) as (size, chunks):
            if size is not None and f.size and abs(size - f.size) > 1024:
                raise CivitaiImportError(
                    f"Civitai sent {size} bytes for {f.civitai_name}, expected about {f.size}"
                )

            async def hashed() -> AsyncIterator[bytes]:
                nonlocal count
                last = 0.0
                async for chunk in chunks:
                    sha.update(chunk)
                    md5.update(chunk)
                    count += len(chunk)
                    if time.monotonic() - last >= PROGRESS_INTERVAL_S:
                        last = time.monotonic()
                        progress(count)
                    yield chunk

            await self.remote.rcat(target, hashed(), size)
        progress(count)
        try:
            if sha.hexdigest() != f.sha256:
                raise CivitaiImportError(
                    f"{f.civitai_name}: SHA-256 {sha.hexdigest()} doesn't match"
                    f" Civitai's {f.sha256}"
                )
            remote_md5 = (await self.remote.run("md5sum", target)).split(" ")[0].strip()
            if remote_md5 != md5.hexdigest():
                raise CivitaiImportError(
                    f"Upload check failed for {f.path}: md5 {remote_md5}"
                    f" in Drive, {md5.hexdigest()} sent"
                )
        except (CivitaiImportError, RcloneError):
            try:
                await self.remote.run("deletefile", target)
            except RcloneError as e:
                log.warning("couldn't delete %s: %s", target, e)
            raise


class ImportBusyError(RuntimeError):
    pass


class Imports:
    """The app's Civitai imports: one at a time, each state change published as an
    `import` event. The finished or failed one stays until the next starts."""

    def __init__(
        self,
        importer: Importer,
        index: Callable[[], list[dict[str, Any]]],
        publish: Callable[[dict[str, Any]], None],
        rescan: Callable[[], Awaitable[int]],
    ) -> None:
        self.importer = importer
        self.index = index
        self.publish = publish
        self.rescan = rescan
        self.current: dict[str, Any] | None = None
        self._task: asyncio.Task[None] | None = None

    async def plan(self, ref: str, **options: Any) -> ImportPlan:
        return await self.importer.plan(ref, self.index(), **options)

    async def start(self, ref: str, **options: Any) -> dict[str, Any]:
        if self._task is not None and not self._task.done():
            raise ImportBusyError("Another import is still running")
        plan = await self.plan(ref, **options)
        self.current = {
            "id": secrets.token_hex(6),
            "source": plan.source,
            "label": plan.label,
            "family": plan.family,
            "paths": [f.path for f in plan.files],
            "state": "copying",
            "done": 0,
            "total": sum(f.size for f in plan.files),
            "error": None,
            "warnings": plan.warnings,
        }
        self._task = asyncio.create_task(self._run(plan, self.current))
        return dict(self.current)

    def _update(self, job: dict[str, Any], **changes: Any) -> None:
        job.update(changes)
        self.publish({"type": "import", "import": dict(job)})

    async def _run(self, plan: ImportPlan, job: dict[str, Any]) -> None:
        def progress(stage: str, done: int, total: int) -> None:
            self._update(
                job,
                state="copying" if stage == "download" else "finishing",
                done=done,
                total=total,
            )

        try:
            await self.importer.run(plan, progress)
        except (CivitaiError, CivitaiImportError, RcloneError) as e:
            self._update(job, state="failed", error=str(e))
            return
        except Exception as e:
            log.exception("Civitai import of %s failed", plan.source)
            self._update(job, state="failed", error=f"Import failed: {e}")
            return
        self._update(job, state="finishing")
        try:
            await self.rescan()
        except Exception as e:
            log.warning("rescan after importing %s failed: %s", plan.source, e)
            job["warnings"] = [*job["warnings"], "Imported, but Drive couldn't be rescanned"]
        self._update(job, state="done")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.importer.civitai.aclose()
