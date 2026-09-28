"""Preprocessors (design §4.4), keyed by id. Each factory imports its module lazily."""

from collections.abc import Callable

from degas_worker.preprocess.base import Preprocessor


def _sam() -> Preprocessor:
    from degas_worker.preprocess.sam import Sam3  # noqa: PLC0415 - imports torch lazily

    return Sam3()


def _depth() -> Preprocessor:
    from degas_worker.preprocess.depth import Depth  # noqa: PLC0415 - imports torch lazily

    return Depth()


def _pose() -> Preprocessor:
    from degas_worker.preprocess.dwpose import DwPose  # noqa: PLC0415 - imports cv2 lazily

    return DwPose()


def _canny() -> Preprocessor:
    from degas_worker.preprocess.canny import Canny  # noqa: PLC0415 - imports cv2 lazily

    return Canny()


PREPROCESSORS: dict[str, Callable[[], Preprocessor]] = {
    "sam": _sam,
    "depth": _depth,
    "pose": _pose,
    "canny": _canny,
}
