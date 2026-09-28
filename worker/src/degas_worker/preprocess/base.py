"""Interface shared by preprocessors."""

import base64
import io
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


class Preprocessor(Protocol):
    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Run on `image` with its model (if it has one) copied to `model`; returns the JSON
        answer."""
        ...

    def unload(self) -> None: ...


def trace(image: Image.Image) -> dict[str, Any]:
    """The `/preprocess` answer for a control image: `{image: <base64 PNG>}`."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode()}
