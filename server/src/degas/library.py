"""Keeping results: saved generation configs (design §6.4), retention (§6.3), and the
names and tags of what is kept."""

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from degas.blobs import REF_PREFIX, BlobStore, unref
from degas.db import Database, JobRow, ResultRow, SavedConfig
from degas_worker.spec import Spec

log = logging.getLogger(__name__)

CONFIG_VERSION = 1

# A freshly written blob may not be referenced yet (e.g. a preview during a rescan).
BLOB_GRACE_S = 3600


def saved_config(job: JobRow, result: ResultRow) -> SavedConfig:
    """The self-contained, replayable config of one result: its job's spec, pinned to its seed."""
    spec = job["spec"]
    config: SavedConfig = {
        "degas_version": CONFIG_VERSION,
        "family": spec["family"],
        "variant": spec["variant"],
        "mode": spec["mode"],
        "model": spec["model"],
        "loras": spec.get("loras") or [],
        "params": {**spec["params"], "seed": result["seed"]},
        "inputs": spec.get("inputs") or {},
        "control": spec.get("control") or [],
    }
    if spec.get("image_prompts"):
        config["image_prompts"] = spec["image_prompts"]
    if runtime := job.get("runtime"):
        config["runtime"] = runtime
    if segments := result.get("segments"):
        config["segments"] = segments
    return config


def input_blobs(spec: Spec | SavedConfig) -> list[str]:
    """Blob shas a spec uses as inputs: source, mask, references, control images, image prompts
    and their originals."""
    return _shas(_refs(spec, provenance=True))


def staged_blobs(spec: Spec) -> list[str]:
    """The input blobs the worker reads: `input_blobs` without what only records where an
    input came from (the extended clip, originals and traced sources)."""
    return _shas(_refs(spec, provenance=False))


def _refs(spec: Spec | SavedConfig, *, provenance: bool) -> Iterator[Any]:
    inputs = spec.get("inputs") or {}
    yield inputs.get("source")
    yield inputs.get("mask")
    if provenance:
        yield inputs.get("extends")
    yield from inputs.get("refs") or []
    if provenance:
        for derived, transform in (inputs.get("transforms") or {}).items():
            yield derived
            if transform:
                yield transform.get("original")
    for unit in spec.get("control") or []:
        yield unit.get("image")
        yield unit.get("mask")
        trace = unit.get("preprocessor")
        if provenance and trace:
            yield trace.get("source")
    for prompt in spec.get("image_prompts") or []:
        yield from prompt.get("images", [])
        yield prompt.get("mask")


def _shas(values: Iterable[Any]) -> list[str]:
    shas: list[str] = []
    for value in values:
        if isinstance(value, str) and value.startswith(REF_PREFIX):
            sha = unref(value)
            if sha not in shas:
                shas.append(sha)
    return shas


def sweep(db: Database, blobs: BlobStore, grace_s: float = BLOB_GRACE_S) -> dict[str, int]:
    """Delete expired results and jobs, then every blob nothing references any more."""
    counts = db.expire()
    referenced = db.referenced_blobs()
    removed = [
        sha
        for sha in list(blobs.stored(older_than_s=grace_s))
        if sha not in referenced and blobs.delete(sha)
    ]
    db.forget_transforms(removed)
    counts["blobs"] = len(removed)
    if any(counts.values()):
        log.info("retention sweep: %s", counts)
    return counts


def release(db: Database, blobs: BlobStore, shas: list[str]) -> None:
    """Delete blobs that just lost a reference, if nothing else holds them."""
    gone = [sha for sha in shas if not db.is_referenced(sha)]
    for sha in gone:
        blobs.delete(sha)
    db.forget_transforms(gone)


# -- saved prompts and tags ------------------------------------------------------------------

PROMPT_NAME_WORDS = 6


def prompt_name(prompt: str) -> str:
    """A saved prompt's default name: its first few words."""
    words = prompt.replace(",", " ").split()
    name = " ".join(words[:PROMPT_NAME_WORDS])
    return name + ("…" if len(words) > PROMPT_NAME_WORDS else "")


def clean_tags(tags: list[str]) -> list[str]:
    """Tags trimmed, without blanks or repeats (whatever their case)."""
    out: list[str] = []
    for tag in (t.strip() for t in tags):
        if tag and tag.lower() not in (o.lower() for o in out):
            out.append(tag)
    return out
