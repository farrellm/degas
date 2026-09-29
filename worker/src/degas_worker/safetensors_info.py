"""Reads a `.safetensors` header without loading any weights (or importing torch)."""

import json
import struct
from collections import Counter
from pathlib import Path

FLOAT8 = ("F8_E4M3", "F8_E5M2")


def tensor_dtypes(path: Path) -> dict[str, str]:
    """Tensor name → safetensors dtype (`BF16`, `F8_E4M3`, …)."""
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return {name: info["dtype"] for name, info in header.items() if name != "__metadata__"}


def stored_float8(path: Path) -> str | None:
    """The float8 dtype a checkpoint's transformer blocks are stored in, if they are.

    Only the blocks count: an all-in-one file also holds text encoders and a VAE, and fp8
    checkpoints keep their norms and embeddings in higher precision.
    """
    dtypes = Counter(
        dtype for name, dtype in tensor_dtypes(path).items() if "blocks." in name
    ).most_common(1)
    if dtypes and dtypes[0][0] in FLOAT8:
        return dtypes[0][0]
    return None
