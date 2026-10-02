"""Whether a pipeline's weights fit on the GPU, shared by the runners that load big models."""

from typing import Any

import torch

# Offload to the CPU when the loaded weights take more than this share of the GPU's memory,
# leaving the rest for activations and the VAE decode.
OFFLOAD_ABOVE = 0.7


def module_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def loaded_bytes(pipe: Any) -> int:
    return sum(
        module_bytes(component)
        for component in pipe.components.values()
        if isinstance(component, torch.nn.Module)
    )


def is_offloaded(pipe: Any) -> bool:
    """Whether `enable_model_cpu_offload` is in effect: its hook is on the pipeline's models."""
    return any(
        hasattr(component, "_hf_hook")
        for component in pipe.components.values()
        if isinstance(component, torch.nn.Module)
    )


def place(pipe: Any) -> bool:
    """Move the pipeline to the GPU, or offload it if it wouldn't fit. Returns True if offloaded."""
    _free, total = torch.cuda.mem_get_info()
    if loaded_bytes(pipe) > total * OFFLOAD_ABOVE:
        pipe.enable_model_cpu_offload()
        return True
    pipe.to("cuda")
    return False
