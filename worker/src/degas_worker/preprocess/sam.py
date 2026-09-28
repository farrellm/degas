"""SAM 3 selection for the mask editor: taps (Sam3Tracker) or a description (Sam3).

Only one of the two models is resident at a time. The tracker's image embedding is cached
per image, so after the first tap on an image each tap only runs the mask decoder.
"""

import gc
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

from degas_worker import masks

EMBEDDINGS_KEPT = 4
THRESHOLD = 0.5


class Sam3:
    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.kind: str | None = None  # "tracker" or "text"
        self.model: Any = None
        self.processor: Any = None
        self.embeddings: OrderedDict[str, Any] = OrderedDict()
        self.can_embed = True

    def run(self, model: Path, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        with Image.open(image) as im:
            rgb = im.convert("RGB")
        points = params.get("points") or []
        text = params.get("text") or ""
        with torch.inference_mode():
            if text:
                found, scores = self._text(model, rgb, text, points)
            else:
                found, scores = self._points(model, rgb, image.name, points)
        return masks.candidates(found, scores)

    def _load(self, model_dir: Path, kind: str) -> None:
        if self.model is not None and self.model_dir == model_dir and self.kind == kind:
            return
        self.unload()
        model_cls, processor_cls = (
            (Sam3TrackerModel, Sam3TrackerProcessor)
            if kind == "tracker"
            else (Sam3Model, Sam3Processor)
        )
        self.model = model_cls.from_pretrained(str(model_dir)).to("cuda").eval()
        self.processor = processor_cls.from_pretrained(str(model_dir))
        self.model_dir = model_dir
        self.kind = kind

    def _points(
        self, model_dir: Path, image: Image.Image, key: str, points: list[dict[str, Any]]
    ) -> tuple[list[Image.Image], list[float]]:
        self._load(model_dir, "tracker")
        inputs = self.processor(
            images=image,
            input_points=[[[[p["x"], p["y"]] for p in points]]],
            input_labels=[[[1 if p.get("include", True) else 0 for p in points]]],
            return_tensors="pt",
        ).to("cuda")
        out = self._track(inputs, key)
        found = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])
        stack = found[0][0]  # (candidates, H, W)
        scores = out.iou_scores[0, 0].float().cpu().tolist()
        return [_image(m) for m in stack], scores

    def _track(self, inputs: Any, key: str) -> Any:
        """Run the tracker, reusing the image's embedding when this transformers allows it."""
        if self.can_embed:
            try:
                embedding = self.embeddings.get(key)
                if embedding is None:
                    embedding = self.model.get_image_embeddings(inputs["pixel_values"])
                    self.embeddings[key] = embedding
                    while len(self.embeddings) > EMBEDDINGS_KEPT:
                        self.embeddings.popitem(last=False)
                self.embeddings.move_to_end(key)
                return self.model(
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    image_embeddings=embedding,
                    multimask_output=True,
                )
            except (AttributeError, TypeError):
                self.can_embed = False
                self.embeddings.clear()
        return self.model(**inputs, multimask_output=True)

    def _text(
        self, model_dir: Path, image: Image.Image, text: str, points: list[dict[str, Any]]
    ) -> tuple[list[Image.Image], list[float]]:
        """Every match for the description, as one selection. Points pick among matches:
        keep those under an included point (if any), drop those under an excluded one."""
        self._load(model_dir, "text")
        inputs = self.processor(images=image, text=text, return_tensors="pt").to("cuda")
        out = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            out,
            threshold=THRESHOLD,
            mask_threshold=THRESHOLD,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        instances = [m.cpu().numpy().astype(bool) for m in result["masks"]]
        scores = [float(s) for s in result["scores"]]

        def under(m: Any, p: dict[str, Any]) -> bool:
            x, y = int(p["x"]), int(p["y"])
            return 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and bool(m[y, x])

        include = [p for p in points if p.get("include", True)]
        exclude = [p for p in points if not p.get("include", True)]
        kept = [
            i
            for i, m in enumerate(instances)
            if (not include or any(under(m, p) for p in include))
            and not any(under(m, p) for p in exclude)
        ]
        if not kept:
            return [], []
        union = np.logical_or.reduce([instances[i] for i in kept])
        return [_image(union)], [max(scores[i] for i in kept)]

    def unload(self) -> None:
        if self.model is None:
            return
        self.model = None
        self.processor = None
        self.model_dir = None
        self.kind = None
        self.embeddings.clear()
        gc.collect()
        torch.cuda.empty_cache()


def _image(mask: Any) -> Image.Image:
    array = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    return Image.fromarray((array > 0).astype(np.uint8) * 255)
