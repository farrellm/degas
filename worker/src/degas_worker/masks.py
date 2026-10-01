"""Mask and canvas helpers for inpaint and outpaint (Pillow only, no torch).

Masks are single-channel: white is redrawn, black is kept, grey is blended.
"""

import base64
import io
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from degas_worker.spec import Place

# The blurred copy of the source that fills an outpaint's margins before denoising.
FILL_BLUR = 48


def blur(mask: Image.Image, radius: int) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else mask


def outpaint_canvas(
    source: Image.Image, place: Place, size: tuple[int, int], blend: int
) -> tuple[Image.Image, Image.Image]:
    """The canvas an outpaint starts from, and its mask.

    `source` (already at `place`'s size) sits at `place` on a `size` canvas. The margins
    are filled with a blurred stretch of the source, which only matters below full strength.
    The mask covers the margins and reaches `blend` px into the source along the edges that
    face a margin, so the seam is redrawn too.
    """
    w, h = size
    x, y, pw, ph = place["x"], place["y"], place["w"], place["h"]
    canvas = source.convert("RGB").resize(size).filter(ImageFilter.GaussianBlur(FILL_BLUR))
    canvas.paste(source.convert("RGB"), (x, y))
    left = x + (blend if x > 0 else 0)
    top = y + (blend if y > 0 else 0)
    right = x + pw - (blend if x + pw < w else 0)
    bottom = y + ph - (blend if y + ph < h else 0)
    mask = Image.new("L", size, 255)
    if right > left and bottom > top:
        ImageDraw.Draw(mask).rectangle((left, top, right - 1, bottom - 1), fill=0)
    return canvas, mask


def composite(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """Keep the original outside the mask, so what wasn't redrawn skips the VAE round trip."""
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), mask.convert("L"))


def order_candidates(areas: list[int], scores: list[float]) -> tuple[list[int], int]:
    """Candidate masks from smallest to largest, and the position of the best-scoring one."""
    order = sorted(range(len(areas)), key=lambda i: areas[i])
    best = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
    return order, order.index(best) if order else 0


def encode(mask: Image.Image) -> str:
    buf = io.BytesIO()
    mask.convert("L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def candidates(masks: list[Image.Image], scores: list[float]) -> dict[str, Any]:
    """The `/preprocess` answer for a selection: candidates by size, and which to show first."""
    areas = [sum(m.convert("L").histogram()[128:]) for m in masks]
    keep = [i for i, a in enumerate(areas) if a > 0]
    order, chosen = order_candidates([areas[i] for i in keep], [scores[i] for i in keep])
    return {
        "candidates": [{"mask": encode(masks[keep[i]]), "score": scores[keep[i]]} for i in order],
        "chosen": chosen,
    }
