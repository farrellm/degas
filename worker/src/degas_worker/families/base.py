"""Runner interface shared by all model families."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class JobCancelled(Exception):  # noqa: N818 - control flow, not an error
    pass


@dataclass(frozen=True)
class Output:
    item: int
    seed: int
    data: bytes
    media_type: str
    ext: str


class RunContext(Protocol):
    def progress(self, item: int, phase: str, step: int, steps: int) -> None: ...

    def check_cancelled(self) -> None:
        """Raise `JobCancelled` if the job has been cancelled."""

    def fetch_asset(self, path: str, size: int | None = None, item: int = 0) -> Path:
        """Local path of a Drive asset, copying it (with `copy` progress) if needed."""
        ...


class FamilyRunner(Protocol):
    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        """Load what the spec needs (reusing what is resident) and yield one output per seed."""
        ...

    def unload(self) -> None: ...
