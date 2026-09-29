"""Removes the 2 px lattice the Qwen-Image VAE leaves on decoded images (design §4.3).

Ported from ComfyUI-DeGrid (github.com/lunaaispace-eng/ComfyUI-DeGrid, `degrid_core.py`,
Apache-2.0), keeping only the automatic mode. A separable notch at the Nyquist frequency
extracts the 2 px component; each image's correction is clamped to a limit taken from its own
amplitude, so real edges pass through. An image whose four 2x2 sublattices agree has no grid,
and is returned untouched.
"""

import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from PIL import Image

# 9-tap alternating binomial: response sin^8(w/2), exactly 1 at a 2 px period and 0 at DC.
_KERNEL = (1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0)
_NORM = 256.0
_PAD = 4
# A raw Qwen VAE lattice is 1-5/255 peak to peak; below this the image is clean.
_NEGLIGIBLE = 0.5 / 255.0
_MAX_SAMPLES = 1_000_000


def degrid(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    x = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    x = x.view(rgb.height, rgb.width, 3).permute(2, 0, 1)[None].float() / 255.0
    corr = _extract_grid(x)
    if _lattice_amp(corr) < _NEGLIGIBLE:
        return image
    limit = _auto_limit(corr)
    out = (x - corr.clamp(-limit, limit)).clamp(0.0, 1.0)
    pixels = (out[0].permute(1, 2, 0) * 255.0).round().to(torch.uint8).contiguous()
    cleaned = Image.frombytes("RGB", rgb.size, pixels.numpy().tobytes())
    if image.mode == "RGBA":  # the VAE decodes RGBA: keep the alpha as it came
        cleaned.putalpha(image.getchannel("A"))
    return cleaned


def _extract_grid(x: torch.Tensor) -> torch.Tensor:
    """The 2 px component of `x` ([1, C, H, W]): Bx + By - Bxy."""
    _b, c, h, w = x.shape
    if h <= 2 * _PAD or w <= 2 * _PAD:
        return torch.zeros_like(x)
    k = torch.tensor(_KERNEL, dtype=x.dtype) / _NORM
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    bx = F.conv2d(F.pad(x, (_PAD, _PAD, 0, 0), mode="reflect"), kx, groups=c)
    by = F.conv2d(F.pad(x, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    bxy = F.conv2d(F.pad(bx, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    return bx + by - bxy


def _lattice_amp(corr: torch.Tensor) -> float:
    """Peak-to-peak spread of the four 2x2 sublattices' means, in the loudest channel.

    The VAE's grid has one phase across the frame, so it survives the averaging; real
    detail that falls in the notch band averages away.
    """
    h, w = corr.shape[2] // 2 * 2, corr.shape[3] // 2 * 2
    if h < 2 or w < 2:
        return 0.0
    c = corr[:, :, :h, :w]
    m = torch.stack([c[:, :, i::2, j::2].mean(dim=(2, 3)) for i in (0, 1) for j in (0, 1)], -1)
    return float((m.amax(-1) - m.amin(-1)).amax())


def _auto_limit(corr: torch.Tensor) -> float:
    """Three times the 75th percentile of |corr| (smooth areas dominate), within bounds."""
    flat = corr.abs().reshape(-1)
    if flat.numel() > _MAX_SAMPLES:
        flat = flat[:: flat.numel() // _MAX_SAMPLES + 1]
    q = float(torch.quantile(flat, 0.75))
    return min(max(q * 3.0, 0.004), 0.05)
