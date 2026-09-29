"""Keeping results: saved generation configs (design §6.4) and retention (§6.3)."""

import logging
from typing import Any

from degas.blobs import BlobStore
from degas.db import Database

log = logging.getLogger(__name__)

CONFIG_VERSION = 1

# A freshly written blob may not be referenced yet (e.g. a preview during a rescan).
BLOB_GRACE_S = 3600


def saved_config(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """The self-contained, replayable config of one result: its job's spec, pinned to its seed."""
    spec = job["spec"]
    config: dict[str, Any] = {
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
    if job.get("runtime"):
        config["runtime"] = job["runtime"]
    if result.get("segments"):
        config["segments"] = result["segments"]
    return config


def input_blobs(spec: dict[str, Any]) -> list[str]:
    """Blob shas a spec uses as inputs: source, mask, references, control images and their
    originals."""
    shas: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.startswith("sha256:"):
            sha = value.removeprefix("sha256:")
            if sha not in shas:
                shas.append(sha)

    inputs = spec.get("inputs") or {}
    for key in ("source", "mask", "extends"):
        add(inputs.get(key))
    for value in inputs.get("refs") or []:
        add(value)
    for derived, transform in (inputs.get("transforms") or {}).items():
        add(derived)
        add((transform or {}).get("original"))
    for unit in spec.get("control") or []:
        for key in ("image", "mask"):
            add(unit.get(key))
        add((unit.get("preprocessor") or {}).get("source"))
    return shas


def sweep(db: Database, blobs: BlobStore, grace_s: float = BLOB_GRACE_S) -> dict[str, int]:
    """Delete expired results and jobs, then every blob nothing references any more."""
    counts = db.expire()
    referenced = db.referenced_blobs()
    removed: list[str] = []
    for sha in list(blobs.stored(older_than_s=grace_s)):
        if sha not in referenced and blobs.delete(sha):
            removed.append(sha)
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
