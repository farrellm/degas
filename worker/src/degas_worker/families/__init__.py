"""Family runners, keyed by family id."""

from collections.abc import Callable

from degas_worker.families.base import FamilyRunner


def _sdxl() -> FamilyRunner:
    from degas_worker.families.sdxl import SdxlRunner  # noqa: PLC0415 - imports diffusers lazily

    return SdxlRunner()


RUNNERS: dict[str, Callable[[], FamilyRunner]] = {"sdxl": _sdxl}
