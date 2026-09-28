"""ControlNet bookkeeping for runners: pipeline arguments and area masks.

Kept free of torch and diffusers so it can be tested without a GPU.
"""

from typing import Any

from PIL import Image


def control_kwargs(
    units: list[dict[str, Any]], images: list[Image.Image], mode: str
) -> dict[str, Any]:
    """The pipeline arguments for ControlNet units, in unit order.

    One unit takes plain values and several take lists (diffusers wraps a list of ControlNets
    in a `MultiControlNetModel`). Text-to-image calls the control image `image`; the
    image-to-image and inpaint pipelines use `image` for the source, so it is `control_image`.
    """
    single = len(units) == 1
    scales = [float(u["scale"]) for u in units]
    return {
        "image" if mode == "t2i" else "control_image": images[0] if single else images,
        "controlnet_conditioning_scale": scales[0] if single else scales,
        "control_guidance_start": [float(u["start"]) for u in units],
        "control_guidance_end": [float(u["end"]) for u in units],
    }


def crop_areas(
    areas: list[Image.Image | None], box: tuple[int, int, int, int], size: tuple[int, int]
) -> list[Image.Image | None]:
    """Crop area masks the way *Around the mask* crops the image, and scale them back to the
    size the pipeline redraws at."""
    return [a.crop(box).resize(size, Image.Resampling.BILINEAR) if a else None for a in areas]
