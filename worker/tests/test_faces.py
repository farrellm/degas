import math

import pytest

from degas_worker.faces import ARCFACE_DST, iou, largest, nms, similarity, template


def apply(m, p):  # type: ignore[no-untyped-def]
    (a, b, c), (d, e, f) = m
    return (a * p[0] + b * p[1] + c, d * p[0] + e * p[1] + f)


def test_similarity_recovers_a_rotation_scale_and_shift() -> None:
    angle, k, shift = math.radians(20), 1.7, (30.0, -12.0)
    cos, sin = math.cos(angle) * k, math.sin(angle) * k
    src = list(ARCFACE_DST)
    dst = [(cos * x - sin * y + shift[0], sin * x + cos * y + shift[1]) for x, y in src]
    m = similarity(src, dst)
    for p, q in zip(src, dst, strict=True):
        assert apply(m, p) == pytest.approx(q)


def test_a_face_lands_on_the_arcface_template() -> None:
    # A face at twice the template's size, offset: aligning it to 224 px maps it exactly.
    face = [(2 * x + 100, 2 * y + 40) for x, y in ARCFACE_DST]
    m = similarity(face, template(224))
    got = [c for p in face for c in apply(m, p)]
    assert got == pytest.approx([c for p in template(224) for c in p])
    assert template(224)[0] == pytest.approx((76.5892, 103.3926))
    # Sizes that aren't a multiple of 112 use insightface's 128 px layout, shifted 8 px.
    assert template(128)[0] == pytest.approx((38.2946 + 8, 51.6963))


def test_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = [(0, 0, 100, 100), (5, 5, 105, 105), (200, 200, 260, 260)]
    assert nms(boxes, [0.8, 0.9, 0.7]) == [1, 2]
    assert iou(boxes[0], boxes[2]) == 0


def test_the_main_face_is_the_biggest() -> None:
    assert largest([(0, 0, 10, 10), (0, 0, 50, 40), (0, 0, 30, 30)]) == 1
    assert largest([]) is None
