import base64
import io

from PIL import Image

from degas_worker import masks


def test_outpaint_canvas_masks_the_margins_and_the_seam() -> None:
    source = Image.new("RGB", (100, 80), (200, 10, 10))
    canvas, mask = masks.outpaint_canvas(source, {"x": 50, "y": 0, "w": 100, "h": 80}, (200, 80), 8)
    assert canvas.size == mask.size == (200, 80)
    assert canvas.getpixel((100, 40)) == (200, 10, 10)
    # Kept: the source minus an 8 px seam on its left and right (both face a margin),
    # but not along the top and bottom, which are the canvas's own edges.
    assert mask.getbbox() == (0, 0, 200, 80)
    kept = Image.eval(mask, lambda v: 255 - v).getbbox()
    assert kept == (58, 0, 142, 80)


def test_composite_keeps_the_original_outside_the_mask() -> None:
    original = Image.new("RGB", (4, 1), (0, 0, 0))
    generated = Image.new("RGB", (4, 1), (255, 255, 255))
    mask = Image.new("L", (4, 1), 0)
    mask.putpixel((3, 0), 255)
    out = masks.composite(original, generated, mask)
    assert [out.getpixel((x, 0)) for x in range(4)] == [(0, 0, 0)] * 3 + [(255, 255, 255)]


def test_candidates_run_small_to_large_and_skip_empty_masks() -> None:
    assert masks.order_candidates([30, 10, 20], [0.2, 0.3, 0.9]) == ([1, 2, 0], 1)
    sizes = [(0, 0, 4, 4), (0, 0, 1, 1), None]
    found = []
    for box in sizes:
        m = Image.new("L", (4, 4), 0)
        if box:
            m.paste(255, box)
        found.append(m)
    body = masks.candidates(found, [0.9, 0.5, 0.99])
    assert body["chosen"] == 1
    areas = []
    for c in body["candidates"]:
        with Image.open(io.BytesIO(base64.b64decode(c["mask"]))) as im:
            areas.append(sum(im.histogram()[128:]))
    assert areas == [1, 16]
