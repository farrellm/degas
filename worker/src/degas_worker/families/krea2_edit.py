"""Krea 2 edits: where the reference images go, and the prompt that sees them (torch-free).

Krea 2 has no edit model. conradlocke's edit LoRAs (General Edit, Identity Edit) were trained
with two paths, ported here from comfyui-krea2edit (Apache-2.0, lbouaraba/comfyui-krea2edit
2.x, which mirrors ai-toolkit's `predict_velocity_edit`):

- Appearance: each image is VAE-encoded and its clean tokens go before the noisy target's,
  `[text | image 1 | image 2 | target]`, told apart only by the RoPE frame axis (images 1, 2…,
  the target 0). An image is fitted inside the output at its own aspect ratio (`fit`) and
  centred on the target's grid (`place`). Target-to-image attention can be strengthened
  (`REF_BOOST`).
- Semantics: the prompt is encoded by Qwen3-VL together with the images (`grounded_prompt`),
  each at most `GROUNDING_PX` on its long side. With CFG the negative is the empty prompt with
  the same images, which is what the LoRAs were trained with as the unconditional.
"""

from dataclasses import dataclass
from typing import Literal

# The output's pixels per transformer token: the VAE shrinks 8x and the transformer packs 2x2.
TOKEN_PX = 16
# The long side the text encoder sees an image at (trained with 384-768 px jitter).
GROUNDING_PX = 768
# How near the output's aspect ratio an image must be to be cropped to fill it exactly.
CROP_TOLERANCE = 0.08

Placement = Literal["aligned", "beside"]
EditKind = Literal["general", "identity"]

SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
PREFIX = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
VISION = "<|vision_start|><|image_pad|><|vision_end|>"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True)
class Fit:
    """How an image becomes reference tokens: crop it to `crop` (left, top, right, bottom),
    resize to `size` (width, height, multiples of `TOKEN_PX`)."""

    crop: tuple[int, int, int, int]
    size: tuple[int, int]

    @property
    def grid(self) -> tuple[int, int]:
        """(rows, columns) of tokens."""
        return self.size[1] // TOKEN_PX, self.size[0] // TOKEN_PX


def fit(src_w: int, src_h: int, out_w: int, out_h: int) -> Fit:
    """Fit an image inside the output at the output's pixel density, as the LoRAs were trained.

    Near the output's aspect ratio, the image is centre-cropped to it and fills the output
    exactly; a margin of a token or two would be filled with repeated edge content. Otherwise
    it is scaled to fit inside, each side floored to a multiple of 16 (and no more than the
    output's), and centre-cropped to the floored shape first so nothing is squashed.
    """
    scale = min(out_h / src_h, out_w / src_w)
    tol = 1 - CROP_TOLERANCE
    if src_h * scale >= out_h * tol and src_w * scale >= out_w * tol:
        cover = max(out_h / src_h, out_w / src_w)
        ch, cw = min(src_h, round(out_h / cover)), min(src_w, round(out_w / cover))
        return Fit(_centred(src_w, src_h, cw, ch), (out_w, out_h))
    nh = min(max(TOKEN_PX, int(src_h * scale) // TOKEN_PX * TOKEN_PX), _floor(out_h))
    nw = min(max(TOKEN_PX, int(src_w * scale) // TOKEN_PX * TOKEN_PX), _floor(out_w))
    ch = min(src_h, max(1, round(nh / scale)))
    cw = min(src_w, max(1, round(nw / scale)))
    return Fit(_centred(src_w, src_h, cw, ch), (nw, nh))


def place(
    grids: list[tuple[int, int]], target: tuple[int, int], placements: list[Placement]
) -> list[tuple[float, float]]:
    """Each image's (row, column) offset on the target's token grid.

    `aligned` centres the image on the target (fractional offsets are exact in RoPE), which is
    how the LoRAs were trained. `beside` puts it to the right of the target and of any image
    placed beside before it, at the same rows, so its layout doesn't pin the output's.
    """
    rows, cols = target
    right = float(cols)
    out: list[tuple[float, float]] = []
    for (gh, gw), how in zip(grids, placements, strict=True):
        top, left = max(0.0, (rows - gh) / 2), max(0.0, (cols - gw) / 2)
        if how == "beside":
            left = right
            right = left + gw
        out.append((top, left))
    return out


def placements(kind: EditKind, count: int) -> list[Placement]:
    """With Identity's two images, the scene (the source) is aligned and the person beside, as
    conradlocke suggests for putting a person into a scene. A single image is aligned."""
    if kind == "identity" and count == 2:
        return ["aligned", "beside"]
    return ["aligned"] * count


def ref_boosts(kind: EditKind, count: int) -> list[float]:
    """How much more each image is attended to (conradlocke's settings): 4 for General Edit;
    2 for Identity's person (the last image), 1 for its scene."""
    if kind == "general":
        return [4.0] * count
    return [1.0] * (count - 1) + [2.0]


def grounded_prompt(prompt: str, images: int) -> str:
    """The text encoder's input: the images' vision tokens, then the prompt, in the Qwen chat
    template with Krea's system prompt."""
    return PREFIX + VISION * images + prompt + SUFFIX


def grounding_size(w: int, h: int) -> tuple[int, int]:
    """The size an image goes to the text encoder at: at most `GROUNDING_PX` on its long side."""
    if max(w, h) <= GROUNDING_PX:
        return w, h
    s = GROUNDING_PX / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))


def _floor(n: int) -> int:
    return max(TOKEN_PX, n // TOKEN_PX * TOKEN_PX)


def _centred(w: int, h: int, cw: int, ch: int) -> tuple[int, int, int, int]:
    left, top = (w - cw) // 2, (h - ch) // 2
    return left, top, left + cw, top + ch
