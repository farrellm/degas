"""Interface shared by preprocessors."""

from pathlib import Path
from typing import Any, Protocol


class Preprocessor(Protocol):
    def run(self, model: Path, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Run on `image` with the model copied to `model`; returns the JSON answer."""
        ...

    def unload(self) -> None: ...
