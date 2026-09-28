"""Preprocessors (design §4.4), keyed by id. Each factory imports its module lazily."""

from collections.abc import Callable

from degas_worker.preprocess.base import Preprocessor


def _sam() -> Preprocessor:
    from degas_worker.preprocess.sam import Sam3  # noqa: PLC0415 - imports torch lazily

    return Sam3()


PREPROCESSORS: dict[str, Callable[[], Preprocessor]] = {"sam": _sam}
