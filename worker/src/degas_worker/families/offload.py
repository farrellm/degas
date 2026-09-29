"""Whether a pipeline's weights fit on the GPU, shared by the runners that load big models."""

from typing import Any

import torch

# Offload to the CPU when the loaded weights take more than this share of the GPU's memory,
# leaving the rest for activations and the VAE decode.
OFFLOAD_ABOVE = 0.7


def loaded_bytes(pipe: Any) -> int:
    total = 0
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            total += sum(p.numel() * p.element_size() for p in component.parameters())
    return total


def place(pipe: Any) -> bool:
    """Move the pipeline to the GPU, or offload it if it wouldn't fit. Returns True if offloaded."""
    _free, total = torch.cuda.mem_get_info()
    if loaded_bytes(pipe) > total * OFFLOAD_ABOVE:
        pipe.enable_model_cpu_offload()
        return True
    pipe.to("cuda")
    return False
