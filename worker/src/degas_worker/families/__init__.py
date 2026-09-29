"""Family runners, keyed by family id."""

from collections.abc import Callable

from degas_worker.families.base import FamilyRunner


def _sdxl() -> FamilyRunner:
    from degas_worker.families.sdxl import SdxlRunner  # noqa: PLC0415 - imports diffusers lazily

    return SdxlRunner()


def _qwen21() -> FamilyRunner:
    from degas_worker.families.qwen21 import Qwen21Runner  # noqa: PLC0415 - diffusers, lazily

    return Qwen21Runner()


def _wan22() -> FamilyRunner:
    from degas_worker.families.wan22 import Wan22Runner  # noqa: PLC0415 - imports diffusers lazily

    return Wan22Runner()


RUNNERS: dict[str, Callable[[], FamilyRunner]] = {
    "sdxl": _sdxl,
    "qwen21": _qwen21,
    "wan22": _wan22,
}
