"""SDXL ControlNets: loading them, their inputs, and limiting one to an area.

Regional ControlNet: a unit with an area mask has its ControlNet's residuals multiplied by
the mask, downsampled to each residual's size, so it only guides that area.
"""

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from diffusers import ControlNetModel
from PIL import Image

from degas_worker.families.base import RunContext
from degas_worker.spec import ControlUnit


def load_controlnet(path: str, local: Path, *, offload: bool) -> Any:
    """Load a ControlNet (a diffusers folder or a single file), onto the GPU unless the
    pipeline is offloaded."""
    if is_union(local):
        raise ValueError(f"{path} is a union ControlNet, which Degas can't use yet")
    try:
        if local.is_dir():
            net = ControlNetModel.from_pretrained(str(local), torch_dtype=torch.float16)
        else:
            net = ControlNetModel.from_single_file(str(local), torch_dtype=torch.float16)
    except Exception as e:
        raise ValueError(f"Could not load ControlNet {path}: {e}") from e
    return net if offload else net.to("cuda")


def control_inputs(
    units: list[ControlUnit], ctx: RunContext
) -> tuple[list[Image.Image], list[Image.Image | None]]:
    """Each unit's control image, and its area mask if it has one (both at output size)."""
    images: list[Image.Image] = []
    areas: list[Image.Image | None] = []
    for unit in units:
        with Image.open(ctx.blob(unit["image"])) as im:
            images.append(im.convert("RGB"))
        if unit.get("mask"):
            with Image.open(ctx.blob(unit["mask"])) as im:
                areas.append(im.convert("L"))
        else:
            areas.append(None)
    return images, areas


def is_union(local: Path) -> bool:
    """Whether a ControlNet is a union model (one net for many control types)."""
    if local.is_dir():
        try:
            config = json.loads((local / "config.json").read_text())
        except (OSError, ValueError):
            return False
        return bool(config.get("_class_name") == "ControlNetUnionModel")
    if local.suffix != ".safetensors":
        return False
    from safetensors import safe_open  # noqa: PLC0415 - only needed here

    with safe_open(str(local), framework="pt") as f:
        return any(k.startswith(("task_embedding", "control_type_proj")) for k in f.keys())  # noqa: SIM118 - not a dict


@contextlib.contextmanager
def limit_to_areas(units: list[tuple[Any, Image.Image]]) -> Iterator[None]:
    """While the pipeline runs, multiply each ControlNet's residuals by its area mask.

    `forward` is replaced on the instance rather than the model being wrapped in another
    module, so the pipelines' `isinstance` checks keep passing. accelerate's offload hook also
    lives on the instance's `forward`; the masked one calls through it.
    """
    restore: list[tuple[Any, Any]] = []
    try:
        for net, area in units:
            restore.append((net, net.__dict__.get("forward")))
            net.forward = _masked(net.forward, area)
        yield
    finally:
        for net, forward in reversed(restore):
            if forward is None:
                del net.forward
            else:
                net.forward = forward


def _masked(forward: Any, area: Image.Image) -> Any:
    mask = torch.from_numpy(np.asarray(area, dtype=np.float32) / 255.0)[None, None]
    sized: dict[tuple[Any, ...], Any] = {}

    def fit(residual: Any) -> Any:
        key = (tuple(residual.shape[-2:]), residual.device, residual.dtype)
        if key not in sized:
            m = F.interpolate(mask.to(residual.device), size=residual.shape[-2:], mode="area")
            sized[key] = m.to(residual.dtype)
        return residual * sized[key]

    def masked_forward(*args: Any, **kwargs: Any) -> Any:
        out = forward(*args, **kwargs)
        if isinstance(out, tuple):  # return_dict=False, as the pipelines call it
            return ([fit(d) for d in out[0]], fit(out[1]), *out[2:])
        out.down_block_res_samples = [fit(d) for d in out.down_block_res_samples]
        out.mid_block_res_sample = fit(out.mid_block_res_sample)
        return out

    return masked_forward
