"""DWPose for a pose ControlNet: whole-body keypoints drawn as an OpenPose skeleton.

`preprocessors/dwpose/` holds the two ONNX models from yzd-v/DWPose: `yolox_l.onnx` (finds
people) and `dw-ll_ucoco_384.onnx` (133 keypoints per person). Pre- and post-processing and
the drawing follow controlnet_aux's DWPose annotator (Apache-2.0), so poses look the way
OpenPose ControlNets were trained on. onnxruntime-gpu is installed on first use; without a
usable CUDA provider it runs on the CPU, which takes about a second.
"""

import colorsys
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from degas_worker import deps
from degas_worker.preprocess.base import trace

DETECTOR = "yolox_l.onnx"
POSE = "dw-ll_ucoco_384.onnx"
DETECT_SIZE = (640, 640)
VISIBLE = 0.3
EPS = 0.01
DRAW_SIDE = 512

Array = Any  # numpy arrays; numpy's stubs make these noisy to annotate precisely


class DwPose:
    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.detector: Any = None
        self.pose: Any = None

    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        if model is None:
            raise ValueError("Pose needs DWPose")
        self._load(model)
        with Image.open(image) as im:
            rgb = np.asarray(im.convert("RGB"))
        boxes = detect(self.detector, rgb)
        keypoints, scores = estimate(self.pose, boxes, rgb)
        # Drawn at 512 px on the short side, like the annotator the ControlNets were trained
        # on, so limbs keep their thickness relative to the picture; then scaled to the image.
        h, w = rgb.shape[:2]
        k = DRAW_SIDE / min(h, w)
        canvas = draw(keypoints * k, scores, round(h * k), round(w * k))
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_CUBIC)
        return trace(Image.fromarray(canvas))

    def _load(self, model_dir: Path) -> None:
        if self.detector is not None and self.model_dir == model_dir:
            return
        for name in (DETECTOR, POSE):
            if not (model_dir / name).exists():
                raise ValueError(f"DWPose needs {name} in preprocessors/dwpose/")
        ort = deps.ensure("onnxruntime", "onnxruntime-gpu")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.detector = ort.InferenceSession(str(model_dir / DETECTOR), providers=providers)
        self.pose = ort.InferenceSession(str(model_dir / POSE), providers=providers)
        self.model_dir = model_dir

    def unload(self) -> None:
        self.detector = None
        self.pose = None
        self.model_dir = None


# -- detection (YOLOX) ---------------------------------------------------------------------


def detect(session: Any, image: Array) -> Array:
    """Boxes `[[x1, y1, x2, y2]]` around the people in an RGB image."""
    h, w = DETECT_SIZE
    ratio = min(h / image.shape[0], w / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None]
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    pred = _grid_decode(out, DETECT_SIZE)[0]
    boxes = np.empty_like(pred[:, :4])
    boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2
    boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
    boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2
    boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2
    boxes /= ratio
    scores = pred[:, 4:5] * pred[:, 5:]
    person = scores[:, 0]  # class 0 is "person"
    keep = person > 0.1
    boxes, person = boxes[keep], person[keep]
    kept = _nms(boxes, person, 0.45)
    boxes, person = boxes[kept], person[kept]
    return boxes[person > VISIBLE]


def _grid_decode(outputs: Array, size: tuple[int, int]) -> Array:
    grids, strides = [], []
    for stride in (8, 16, 32):
        hs, ws = size[0] // stride, size[1] // stride
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides.append(np.full((*grid.shape[:2], 1), stride))
    grid = np.concatenate(grids, 1)
    stride = np.concatenate(strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grid) * stride
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * stride
    return outputs


def _nms(boxes: Array, scores: Array, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


# -- keypoints (RTMPose-style SimCC) ---------------------------------------------------------

MEAN = np.array([123.675, 116.28, 103.53])
STD = np.array([58.395, 57.12, 57.375])


def estimate(session: Any, boxes: Array, image: Array) -> tuple[Array, Array]:
    """Keypoints `(people, 133, 2)` in pixels and their scores `(people, 133)`."""
    h, w = session.get_inputs()[0].shape[2:]
    size = (int(w), int(h))
    if len(boxes) == 0:
        boxes = np.array([[0, 0, image.shape[1], image.shape[0]]], dtype=np.float32)
    all_points, all_scores = [], []
    for box in boxes:
        center = np.array([box[0] + box[2], box[1] + box[3]]) * 0.5
        scale = np.array([box[2] - box[0], box[3] - box[1]]) * 1.25
        scale = _fix_aspect(scale, size[0] / size[1])
        warp = _warp_matrix(center, scale, size)
        crop = cv2.warpAffine(image, warp, size, flags=cv2.INTER_LINEAR)
        crop = ((crop - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        names = [o.name for o in session.get_outputs()]
        simcc_x, simcc_y = session.run(names, {session.get_inputs()[0].name: crop})
        points, scores = _simcc_maximum(simcc_x, simcc_y)
        points = points / 2.0  # SimCC split ratio
        points = points / np.array(size) * scale + center - scale / 2
        all_points.append(points[0])
        all_scores.append(scores[0])
    return np.array(all_points), np.array(all_scores)


def _fix_aspect(scale: Array, aspect: float) -> Array:
    w, h = scale
    return np.array([w, w / aspect]) if w > h * aspect else np.array([h * aspect, h])


def _third_point(a: Array, b: Array) -> Array:
    d = a - b
    return b + np.array([-d[1], d[0]])


def _warp_matrix(center: Array, scale: Array, size: tuple[int, int]) -> Array:
    dst_w, dst_h = size
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, scale[0] * -0.5])
    src[2] = _third_point(src[0], src[1])
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + np.array([0.0, dst_w * -0.5])
    dst[2] = _third_point(dst[0], dst[1])
    return cv2.getAffineTransform(src, dst)


def _simcc_maximum(simcc_x: Array, simcc_y: Array) -> tuple[Array, Array]:
    n, k, _ = simcc_x.shape
    sx, sy = simcc_x.reshape(n * k, -1), simcc_y.reshape(n * k, -1)
    locs = np.stack((np.argmax(sx, axis=1), np.argmax(sy, axis=1)), -1).astype(np.float32)
    vx, vy = np.amax(sx, axis=1), np.amax(sy, axis=1)
    vals = np.minimum(vx, vy)
    locs[vals <= 0.0] = -1
    return locs.reshape(n, k, 2), vals.reshape(n, k)


# -- drawing (OpenPose format) -------------------------------------------------------------

# COCO-WholeBody body order → OpenPose's 18 body points (with a neck between the shoulders).
MMPOSE = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
OPENPOSE = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
LIMBS = [
    (2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11),
    (2, 12), (12, 13), (13, 14), (2, 1), (1, 15), (15, 17), (1, 16), (16, 18),
]  # fmt: skip
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]  # fmt: skip
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
    (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19),
    (19, 20),
]  # fmt: skip


def draw(keypoints: Array, scores: Array, height: int, width: int) -> Array:
    """An RGB OpenPose image: body limbs and joints, hands, and face points on black."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(keypoints) == 0:
        return canvas
    info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    neck = np.mean(info[:, [5, 6]], axis=1)
    neck[:, 2] = np.logical_and(info[:, 5, 2] > VISIBLE, info[:, 6, 2] > VISIBLE)
    info = np.insert(info, 17, neck, axis=1)
    info[:, OPENPOSE] = info[:, MMPOSE]
    points, visible = info[..., :2].copy(), info[..., 2] > VISIBLE
    points[~visible] = -1

    canvas = _draw_body(canvas, points, visible)
    _draw_hands(canvas, np.concatenate([points[:, 92:113], points[:, 113:134]]))
    for face in points[:, 24:92]:
        for x, y in face:
            if x > EPS and y > EPS:
                cv2.circle(canvas, (int(x), int(y)), 3, (255, 255, 255), thickness=-1)
    return canvas


def _draw_body(canvas: Array, points: Array, visible: Array) -> Array:
    for i, (a, b) in enumerate(LIMBS):
        for person in range(len(points)):
            if not (visible[person, a - 1] and visible[person, b - 1]):
                continue
            (x1, y1), (x2, y2) = points[person, a - 1], points[person, b - 1]
            length = math.hypot(x1 - x2, y1 - y2)
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            polygon = cv2.ellipse2Poly(
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                (int(length / 2), 4),
                int(angle),
                0,
                360,
                1,
            )
            cv2.fillConvexPoly(canvas, polygon, COLORS[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for person in range(len(points)):
            if visible[person, i]:
                x, y = points[person, i]
                cv2.circle(canvas, (int(x), int(y)), 4, COLORS[i], thickness=-1)
    return canvas


def _draw_hands(canvas: Array, hands: Array) -> None:
    for hand in hands:
        for n, (a, b) in enumerate(HAND_EDGES):
            (x1, y1), (x2, y2) = hand[a], hand[b]
            if min(x1, y1, x2, y2) > EPS:
                r, g, bl = colorsys.hsv_to_rgb(n / len(HAND_EDGES), 1.0, 1.0)
                color = (int(r * 255), int(g * 255), int(bl * 255))
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        for x, y in hand:
            if x > EPS and y > EPS:
                cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)
