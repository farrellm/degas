"""GPU information for /health (torch is optional so the worker runs without a GPU)."""

from importlib import metadata
from typing import Any


def versions() -> dict[str, str | None]:
    """Versions of the libraries that decide what a seed produces."""
    out: dict[str, str | None] = {}
    for name in ("diffusers", "torch"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_info() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415 - slow import, only when asked
    except ImportError:
        return {"gpu": None, "vram_free": None, "vram_total": None}
    if not torch.cuda.is_available():
        return {"gpu": None, "vram_free": None, "vram_total": None}
    free, total = torch.cuda.mem_get_info()
    return {"gpu": torch.cuda.get_device_name(0), "vram_free": free, "vram_total": total}
