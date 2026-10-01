"""SDXL image prompts (IP-Adapter): the pictures' embeddings, FaceID faces and areas.

One adapter per unit is loaded into the shared UNet with the CLIP image encoder they read
pictures with (see `SdxlRunner._load_image_prompts`). Each unit's scale names the blocks its
purpose uses and is set again between steps for its step range.
"""

from typing import Any

import numpy as np
import torch
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models.embeddings import IPAdapterFaceIDPlusImageProjection
from PIL import Image

from degas_worker.families import ip_adapter
from degas_worker.families.base import RunContext
from degas_worker.families.control import crop_areas
from degas_worker.spec import ImagePromptUnit


def picture_embeds(
    pipe: Any, prompts: list[ImagePromptUnit], ctx: RunContext, cfg: bool, faces: Any
) -> list[Any]:
    """Each unit's pictures encoded once for the whole batch (diffusers would encode them
    again for every image).

    A FaceID unit reads each picture's main face: its InsightFace identity goes in as the
    unit's embeddings, and CLIP reads the aligned face crop, which FaceID Plus takes as
    its projection's `clip_embeds`.
    """
    pictures: list[list[Image.Image]] = []
    identities: dict[int, Any] = {}
    for n, unit in enumerate(prompts):
        unit_pictures = []
        for ref in unit["images"]:
            with Image.open(ctx.blob(ref)) as im:
                unit_pictures.append(im.convert("RGB"))
        if ip_adapter.is_faceid(unit["adapter"]["path"]):
            identities[n], unit_pictures = faces_of(faces, unit_pictures, n + 1)
        pictures.append(unit_pictures)
    device = pipe._execution_device
    with torch.no_grad():
        embeds = list(pipe.prepare_ip_adapter_image_embeds(pictures, None, device, 1, cfg))
    layers = pipe.unet.encoder_hid_proj.image_projection_layers
    for n, ids in identities.items():
        unit = prompts[n]
        layer = layers[n]
        if isinstance(layer, IPAdapterFaceIDPlusImageProjection):
            layer.clip_embeds = embeds[n].to(device, torch.float16)
            layer.shortcut = ip_adapter.has_shortcut(unit["adapter"]["path"])
            layer.shortcut_scale = float(unit.get("structure", 1.0))
        found = torch.from_numpy(ids)[None].to(device, torch.float16)  # 1, pictures, 512
        embeds[n] = torch.cat([torch.zeros_like(found), found]) if cfg else found
    return embeds


def faces_of(faces: Any, pictures: list[Image.Image], unit: int) -> tuple[Any, list[Image.Image]]:
    """Each picture's main face: its identities (pictures by 512) and aligned crops."""
    from degas_worker.preprocess.face import align, main_face  # noqa: PLC0415 - cv2

    if faces is None:
        raise ValueError("FaceID needs InsightFace in preprocessors/insightface/")
    ids, crops = [], []
    for k, picture in enumerate(pictures, 1):
        rgb = np.asarray(picture)
        face = main_face(faces, rgb)
        if face is None:
            raise ValueError(f"Image prompt {unit}: no face found in picture {k}")
        ids.append(faces.identity(rgb, face))
        crops.append(Image.fromarray(align(rgb, face, 224)))
    return np.stack(ids).astype(np.float32), crops


def unit_areas(
    prompts: list[ImagePromptUnit],
    ctx: RunContext,
    size: tuple[int, int],
    box: tuple[int, int, int, int] | None,
) -> list[Any] | None:
    """Each unit's area as diffusers takes it (one mask per picture), None for a unit
    without one; None when no unit has an area."""
    areas: list[Image.Image | None] = []
    for unit in prompts:
        if unit.get("mask"):
            with Image.open(ctx.blob(unit["mask"])) as im:
                areas.append(im.convert("L"))
        else:
            areas.append(None)
    if not any(areas):
        return None
    if box:
        areas = crop_areas(areas, box, size)
    processor = IPAdapterMaskProcessor()
    out: list[Any] = []
    for unit, area in zip(prompts, areas, strict=True):
        if area is None:
            out.append(None)
            continue
        mask = processor.preprocess([area], height=size[1], width=size[0])
        out.append(mask.reshape(1, 1, *mask.shape[-2:]).repeat(1, len(unit["images"]), 1, 1))
    return out
