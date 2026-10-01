"""Faces for FaceID image prompts: InsightFace's detector and recognizer, run on onnxruntime.

`preprocessors/insightface/` holds two models from InsightFace's `buffalo_l` pack:
`det_10g.onnx` (SCRFD, which finds faces and their eyes, nose and mouth corners) and
`w600k_r50.onnx` (ArcFace, a 512-number identity). The pre- and post-processing follow
insightface's `model_zoo/scrfd.py`, `arcface_onnx.py` and `utils/face_align.py` (MIT); the
`insightface` package itself needs a compiler, so it isn't used. InsightFace's pretrained
models are for non-commercial research use.
"""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from degas_worker import deps, faces

DETECTOR = "det_10g.onnx"
RECOGNIZER = "w600k_r50.onnx"
# SCRFD misses a face that fills the frame, so a picture with none found is tried again
# smaller (as ComfyUI_IPAdapter_plus does), which shrinks the face against the anchors.
DETECT_SIZES = (640, 576, 512, 448, 384, 320)
STRIDES = (8, 16, 32)
ANCHORS = 2  # per location
SCORE = 0.5
OVERLAP = 0.4

Array = Any  # numpy arrays


@dataclass(frozen=True)
class Face:
    box: faces.Box
    score: float
    points: list[faces.Point]  # eyes, nose tip, mouth corners


class FaceAnalyzer:
    """Finds faces in RGB images, and turns one into an identity or an aligned crop."""

    def __init__(self, model_dir: Path) -> None:
        for name in (DETECTOR, RECOGNIZER):
            if not (model_dir / name).exists():
                raise ValueError(f"FaceID needs {name} in preprocessors/insightface/")
        ort = deps.ensure("onnxruntime", "onnxruntime-gpu")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.model_dir = model_dir
        self.detector = ort.InferenceSession(str(model_dir / DETECTOR), providers=providers)
        self.recognizer = ort.InferenceSession(str(model_dir / RECOGNIZER), providers=providers)

    def detect(self, rgb: Array) -> list[Face]:
        """Faces in the image, best first."""
        for size in DETECT_SIZES:
            if found := self._detect(rgb, size):
                return found
        return []

    def _detect(self, rgb: Array, size: int) -> list[Face]:
        """Faces found with the image fitted into a `size` px square."""
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        resized = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        name = self.detector.get_inputs()[0].name
        outputs = self.detector.run(None, {name: blob})
        return decode(outputs, scale, size)

    def identity(self, rgb: Array, face: Face) -> Array:
        """The face's ArcFace identity: 512 numbers of unit length."""
        crop = align(rgb, face, 112)
        blob = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        name = self.recognizer.get_inputs()[0].name
        embedding = self.recognizer.run(None, {name: blob})[0][0]
        return embedding / np.linalg.norm(embedding)


def decode(outputs: list[Array], scale: float, size: int) -> list[Face]:
    """SCRFD's outputs (scores, box distances and point offsets for each stride) for a `size`
    px input, as faces in image pixels, overlapping ones merged."""
    n = len(STRIDES)
    batched = outputs[0].ndim == 3
    boxes: list[faces.Box] = []
    scores: list[float] = []
    points: list[list[faces.Point]] = []
    for i, stride in enumerate(STRIDES):
        score, dist, kps = (outputs[i + k * n] for k in range(3))
        if batched:
            score, dist, kps = score[0], dist[0], kps[0]
        side = size // stride
        ys, xs = np.mgrid[:side, :side]
        centres = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
        centres = np.repeat(centres, ANCHORS, axis=0).astype(np.float32)
        dist, kps = dist * stride, kps * stride
        for j in np.where(score[:, 0] >= SCORE)[0]:
            cx, cy = centres[j]
            d = dist[j]
            boxes.append(
                (
                    float(cx - d[0]) / scale,
                    float(cy - d[1]) / scale,
                    float(cx + d[2]) / scale,
                    float(cy + d[3]) / scale,
                )
            )
            scores.append(float(score[j, 0]))
            k = kps[j]
            points.append(
                [(float(cx + k[2 * p]) / scale, float(cy + k[2 * p + 1]) / scale) for p in range(5)]
            )
    return [Face(boxes[i], scores[i], points[i]) for i in faces.nms(boxes, scores, OVERLAP)]


def align(rgb: Array, face: Face, size: int) -> Array:
    """A `size` px crop with the face's eyes, nose and mouth where ArcFace expects them."""
    (a, b, c), (d, e, f) = faces.similarity(face.points, faces.template(size))
    matrix = np.array([[a, b, c], [d, e, f]], dtype=np.float32)
    return cv2.warpAffine(rgb, matrix, (size, size), borderValue=0.0)


def main_face(analyzer: FaceAnalyzer, rgb: Array) -> Face | None:
    """The biggest face in the picture: the one a portrait is of."""
    found = analyzer.detect(rgb)
    at = faces.largest([f.box for f in found])
    return None if at is None else found[at]


class FacePreprocessor:
    """`/preprocess` `face`: the aligned crop of the picture's main face, which is what a
    FaceID image prompt reads, and how many faces were found."""

    def __init__(self) -> None:
        self.analyzer: FaceAnalyzer | None = None

    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        if model is None:
            raise ValueError("Finding faces needs InsightFace")
        if self.analyzer is None or self.analyzer.model_dir != model:
            self.analyzer = FaceAnalyzer(model)
        with Image.open(image) as im:
            rgb = np.asarray(im.convert("RGB"))
        found = self.analyzer.detect(rgb)
        at = faces.largest([f.box for f in found])
        if at is None:
            raise ValueError("No face found in this picture")
        crop = Image.fromarray(align(rgb, found[at], 224))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode(), "faces": len(found)}

    def unload(self) -> None:
        self.analyzer = None
