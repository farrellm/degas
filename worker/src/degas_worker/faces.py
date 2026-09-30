"""Face geometry for FaceID image prompts: non-maximum suppression over detections, and the
five-point alignment InsightFace's recognizer was trained on.

Plain Python, no numpy, so it can be tested without the GPU dependencies. The detector and
recognizer themselves are in `preprocess/face.py`.
"""

from collections.abc import Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2
Point = tuple[float, float]
Affine = tuple[tuple[float, float, float], tuple[float, float, float]]

# Where ArcFace expects the eyes, nose tip and mouth corners in a 112 px crop
# (insightface.utils.face_align.arcface_dst).
ARCFACE_DST: tuple[Point, ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def template(size: int) -> list[Point]:
    """ARCFACE_DST for a `size` px crop, as `face_align.norm_crop` scales it."""
    if size % 112 == 0:
        ratio, dx = size / 112, 0.0
    else:
        ratio = size / 128
        dx = 8 * ratio
    return [(x * ratio + dx, y * ratio) for x, y in ARCFACE_DST]


def similarity(src: Sequence[Point], dst: Sequence[Point]) -> Affine:
    """The rotation, uniform scale and translation that best maps `src` onto `dst` (least
    squares; the same as Umeyama's estimate, which `norm_crop` uses)."""
    n = len(src)
    if n != len(dst) or n < 2:
        raise ValueError("similarity needs two or more matching points")
    sx = sum(p[0] for p in src) / n
    sy = sum(p[1] for p in src) / n
    dx = sum(p[0] for p in dst) / n
    dy = sum(p[1] for p in dst) / n
    norm = dot = cross = 0.0
    for (px, py), (qx, qy) in zip(src, dst, strict=True):
        x, y, u, v = px - sx, py - sy, qx - dx, qy - dy
        norm += x * x + y * y
        dot += x * u + y * v
        cross += x * v - y * u
    if norm == 0:
        raise ValueError("the points coincide")
    a, b = dot / norm, cross / norm
    return ((a, -b, dx - (a * sx - b * sy)), (b, a, dy - (b * sx + a * sy)))


def iou(p: Box, q: Box) -> float:
    """Overlap of two boxes, counting pixels inclusively as InsightFace's NMS does."""
    w = min(p[2], q[2]) - max(p[0], q[0]) + 1
    h = min(p[3], q[3]) - max(p[1], q[1]) + 1
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area = (p[2] - p[0] + 1) * (p[3] - p[1] + 1) + (q[2] - q[0] + 1) * (q[3] - q[1] + 1)
    return inter / (area - inter)


def nms(boxes: Sequence[Box], scores: Sequence[float], threshold: float = 0.4) -> list[int]:
    """Indices of the boxes kept, best first: each drops the lower-scoring boxes it overlaps
    by more than `threshold`."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) <= threshold for j in keep):
            keep.append(i)
    return keep


def largest(boxes: Sequence[Box]) -> int | None:
    """The index of the biggest box: the face a portrait is of."""
    if not boxes:
        return None
    return max(
        range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
    )
