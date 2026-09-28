"""Depth Anything V2 (a transformers folder) for a depth ControlNet: near is white."""

import gc
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from degas_worker.preprocess.base import trace


class Depth:
    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.model: Any = None
        self.processor: Any = None

    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        if model is None:
            raise ValueError("Depth needs its model")
        self._load(model)
        with Image.open(image) as im:
            rgb = im.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            out = self.model(**inputs)
        depth = self.processor.post_process_depth_estimation(
            out, target_sizes=[(rgb.height, rgb.width)]
        )[0]["predicted_depth"].float()
        # The model predicts relative inverse depth, so the nearest point is the largest.
        lo, hi = depth.min(), depth.max()
        norm = ((depth - lo) / (hi - lo).clamp(min=1e-6) * 255).round().to(torch.uint8)
        return trace(Image.fromarray(norm.cpu().numpy()))

    def _load(self, model_dir: Path) -> None:
        if self.model is not None and self.model_dir == model_dir:
            return
        self.unload()
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(str(model_dir), torch_dtype=torch.float16)
            .to("cuda")
            .eval()
        )
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir))
        self.model_dir = model_dir

    def unload(self) -> None:
        if self.model is None:
            return
        self.model = None
        self.processor = None
        self.model_dir = None
        gc.collect()
        torch.cuda.empty_cache()
