"""Canny edges for an edges ControlNet: white lines on black."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from degas_worker.preprocess.base import trace


class Canny:
    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        with Image.open(image) as im:
            rgb = np.asarray(im.convert("RGB"))
        edges = cv2.Canny(rgb, int(params.get("low", 100)), int(params.get("high", 200)))
        return trace(Image.fromarray(edges))

    def unload(self) -> None:
        pass
