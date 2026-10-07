"""Copy a LoRA from Civitai, CivArchive or Hugging Face into Drive, checked against the site's
hash, with a sidecar and preview.

The file streams from the site straight into `rclone rcat`, so nothing large touches the disk.
Its SHA-256 is compared with the site's and its md5 with Drive's, and a mismatch deletes it.
"""

import asyncio
import contextlib
import hashlib
import logging
import secrets
import tempfile
import time
import urllib.parse
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Protocol

from degas.civitai import client
from degas.civitai.civarchive import (
    CivArchive,
    CivArchiveError,
    civitai_version,
    is_civarchive_link,
    parse_civarchive_ref,
)
from degas.civitai.client import Civitai, CivitaiError
from degas.civitai.huggingface import HuggingFace, HuggingFaceError, is_hf_link, parse_hf_ref
from degas.civitai.plan import (
    VIDEO,
    ImportPlan,
    PlanError,
    PlannedFile,
    pair_key,
    plan_hf_import,
    plan_import,
)
from degas.errors import DegasError
from degas.media import MediaError, extract_frame
from degas.rclone import RcloneError, Remote, preview_jpeg

log = logging.getLogger(__name__)

PROGRESS_INTERVAL_S = 0.5

# (stage, bytes done, bytes in all): stage is "download", "sidecar" or "preview".
Progress = Callable[[str, int, int], None]


class CivitaiImportError(DegasError):
    """The import can't go ahead, or failed partway."""


class Source(Protocol):
    """Where a plan's files come from: Civitai, CivArchive or Hugging Face."""

    def download(
        self, url: str
    ) -> AbstractAsyncContextManager[tuple[int | None, AsyncIterator[bytes]]]: ...

    async def fetch(self, url: str) -> bytes: ...

    async def aclose(self) -> None: ...


SOURCE_ERRORS = (CivitaiError, CivArchiveError, HuggingFaceError)


def find_duplicates(plan: ImportPlan, index: Sequence[Mapping[str, Any]]) -> list[str]:
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


def drop_halves_in_drive(
    plan: ImportPlan, index: Sequence[Mapping[str, Any]], *, rename: bool = True
) -> None:
    """Leave out a Wan half taken from another version when it's already in Drive (imported
    by itself earlier), naming the half still to copy so the two pair up, if `rename`."""
    by_sha = {a["sha256"]: a["path"] for a in index if a.get("sha256")}
    for f in [f for f in plan.files if f.from_sibling and f.sha256 in by_sha]:
        existing = by_sha[f.sha256]
        plan.files.remove(f)
        plan.paired_version = None
        plan.warnings.append(f"The {f.half}-noise half is already in Drive as {existing}")
        folder, _, file = existing.rpartition("/")
        stem = file.removesuffix(f"_{f.half}_noise.safetensors")
        if rename and folder == plan.folder and stem != file:
            for kept in plan.files:
                kept.path = f"{folder}/{stem}_{kept.half}_noise.safetensors"


class Importer:
    def __init__(
        self,
        civitai: Civitai,
        remote: Remote,
        rclone_remote: str,
        drive_root: str,
        families: set[str],
        hf: HuggingFace | None = None,
        civarchive: CivArchive | None = None,
    ) -> None:
        self.civitai = civitai
        self.hf = hf or HuggingFace()
        self.civarchive = civarchive or CivArchive()
        self.remote = remote
        self.base = f"{rclone_remote}{drive_root.strip('/')}"
        self.families = families

    async def plan(
        self,
        ref: str,
        index: Sequence[Mapping[str, Any]],
        *,
        family: str | None = None,
        name: str | None = None,
        weight: float | None = None,
        force: bool = False,
        hint: str | None = None,
    ) -> ImportPlan:
        """Resolve a link to a plan; duplicates are errors unless `force`. `hint` is the
        family for a Hugging Face LoRA that nothing says the base model of."""
        options: dict[str, Any] = {"family": family, "name": name, "weight": weight}
        if is_hf_link(ref):
            plan = await self._plan_hf(ref, hint=hint, **options)
        elif is_civarchive_link(ref):
            plan = await self._plan_civarchive(ref, index, name=name, options=options)
        else:
            plan = await self._plan_civitai(ref, index, name=name, options=options)
        duplicates = find_duplicates(plan, index)
        if duplicates and not force:
            raise PlanError("; ".join(duplicates))
        plan.warnings.extend(f"Going ahead anyway: {d}" for d in duplicates)
        return plan

    async def _plan_hf(self, link: str, **options: Any) -> ImportPlan:
        ref = parse_hf_ref(link)
        info = await self.hf.model(ref)
        readme = None
        commit = str(info.get("sha") or ref.revision)
        if any(f.get("rfilename") == "README.md" for f in info.get("siblings") or []):
            try:
                url = self.hf.file_url(ref.repo, commit, "README.md")
                readme = (await self.hf.fetch(url, limit=1 << 20)).decode(errors="replace")
            except HuggingFaceError as e:
                log.warning("no README for %s: %s", ref.repo, e)
        return plan_hf_import(
            info,
            ref,
            families=self.families,
            readme=readme,
            base_url=self.hf.api_base,
            **options,
        )

    async def _plan_civitai(
        self,
        ref: str,
        index: Sequence[Mapping[str, Any]],
        *,
        name: str | None,
        options: dict[str, Any],
    ) -> ImportPlan:
        version = await self.civitai.version(ref)
        plan = plan_import(version, families=self.families, **options)
        if len(plan.files) == 1 and plan.files[0].half is not None:
            # Civitai often has a Wan pair's halves as two versions: find the other one.
            model = await self.civitai.model(int(version["modelId"]))
            siblings = model.get("modelVersions") or []
            plan = plan_import(version, families=self.families, siblings=siblings, **options)
            drop_halves_in_drive(plan, index, rename=name is None)
        return plan

    async def _plan_civarchive(
        self,
        ref: str,
        index: Sequence[Mapping[str, Any]],
        *,
        name: str | None,
        options: dict[str, Any],
    ) -> ImportPlan:
        model_id, version_id = parse_civarchive_ref(ref)
        data = await self.civarchive.model(model_id, version_id)
        version = civitai_version(data)
        plan = plan_import(version, families=self.families, **options)
        if len(plan.files) == 1 and plan.files[0].half is not None:
            # As on Civitai, a Wan pair's other half may be another version: fetch the ones
            # named the same but for high/low (CivArchive lists only their names).
            key = pair_key(plan.version_name)
            siblings = [
                civitai_version(await self.civarchive.model(model_id, int(v["id"])))
                for v in data.get("versions") or []
                if int(v["id"]) != version["id"] and pair_key(str(v.get("name") or "")) == key
            ]
            plan = plan_import(version, families=self.families, siblings=siblings, **options)
            drop_halves_in_drive(plan, index, rename=name is None)
        plan.origin = "civarchive"
        plan.source = self.civarchive.page(model_id, version["id"])
        return plan

    def _source(self, plan: ImportPlan, url: str) -> Source:
        if plan.origin == "huggingface":
            return self.hf
        if plan.origin == "civarchive":
            # A file's mirrors are on Civitai or Hugging Face; its example images elsewhere.
            if is_hf_link(url):
                return self.hf
            if urllib.parse.urlsplit(url).hostname not in client.HOSTS:
                return self.civarchive
        return self.civitai

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

            await self._upload_any(plan, f, downloaded)
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

    async def backfill(self, asset: Mapping[str, Any]) -> ImportPlan | None:
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
            data = await self._source(plan, plan.preview_url).fetch(plan.preview_url)
            if VIDEO.search(plan.preview_url):
                data = await _first_frame(data)
            jpeg = preview_jpeg(data)
        # PIL's UnidentifiedImageError is an OSError
        except (*SOURCE_ERRORS, MediaError, OSError) as e:
            log.warning("no preview for %s: %s", plan.source, e)
            return
        await self.remote.run("rcat", target, stdin=jpeg)

    async def _upload_any(
        self, plan: ImportPlan, f: PlannedFile, progress: Callable[[int], None]
    ) -> None:
        """Upload from the file's first mirror that works and has the right file."""
        *mirrors, last = [f.url, *f.mirrors]
        for url in mirrors:
            try:
                await self._upload(self._source(plan, url), f, url, progress)
                return
            except (*SOURCE_ERRORS, CivitaiImportError) as e:
                log.warning("%s from %s failed, trying the next mirror: %s", f.path, url, e)
        await self._upload(self._source(plan, last), f, last, progress)

    async def _upload(
        self, source: Source, f: PlannedFile, url: str, progress: Callable[[int], None]
    ) -> None:
        target = f"{self.base}/{f.path}"
        sha = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        count = 0

        async with source.download(url) as (size, chunks):
            if size is not None and f.size and abs(size - f.size) > 1024:
                raise CivitaiImportError(
                    f"The download is {size} bytes for {f.civitai_name}, expected about {f.size}"
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
                    f" the expected {f.sha256}"
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


async def _first_frame(video: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "example"
        path.write_bytes(video)
        return await extract_frame(path, "first")


class ImportBusyError(DegasError):
    pass


class Imports:
    """The app's LoRA imports: one at a time, each state change published as an
    `import` event. The finished or failed one stays until the next starts."""

    def __init__(
        self,
        importer: Importer,
        index: Callable[[], Sequence[Mapping[str, Any]]],
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
        except (*SOURCE_ERRORS, CivitaiImportError, RcloneError) as e:
            self._update(job, state="failed", error=str(e))
            return
        except Exception as e:
            log.exception("Import of %s failed", plan.source)
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
        await self.importer.hf.aclose()
        await self.importer.civarchive.aclose()
