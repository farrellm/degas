"""The wording of push notifications (docs/ux.md, Phase 5)."""

import math
from datetime import timedelta

from degas.db import JobRow
from degas.families import FAMILIES

BODY_CHARS = 140

RESULTS_URL = "/?tab=results"
SESSION_URL = "/?sheet=session"


def job_notice(job: JobRow) -> tuple[str, str] | None:
    """(title, body) for a finished job, or None when there's nothing to tell (cancelled)."""
    if job["status"] == "error":
        return "Job failed", _clip(job.get("error") or "The job failed.")
    if job["status"] != "done":
        return None
    spec = job["spec"]
    family = FAMILIES.get(str(spec.get("family")))
    video = family is not None and family.media == "video"
    n = len(job["seeds"])
    noun = ("clip" if n == 1 else "clips") if video else ("image" if n == 1 else "images")
    title = f"{noun.capitalize()} finished" if n == 1 else f"{n} {noun} finished"
    return title, _clip(str(spec.get("params", {}).get("prompt") or "").strip())


def idle_notice(gpu: str, left: timedelta) -> str:
    """The idle warning, worded like the session sheet: "L4 stops in 2:00 if no job runs."."""
    seconds = math.ceil(left.total_seconds() / 10) * 10
    return f"{gpu} stops in {seconds // 60}:{seconds % 60:02d} if no job runs."


def _clip(text: str) -> str:
    return text if len(text) <= BODY_CHARS else text[: BODY_CHARS - 1].rstrip() + "…"
