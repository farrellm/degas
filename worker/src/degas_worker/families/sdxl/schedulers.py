"""SDXL's samplers and noise schedules."""

from typing import Any

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
)

# Keep in sync with the server descriptor (degas/families/sdxl.py).
SCHEDULERS: dict[str, tuple[Any, dict[str, Any]]] = {
    "euler": (EulerDiscreteScheduler, {}),
    "euler_a": (EulerAncestralDiscreteScheduler, {}),
    "dpmpp_2m": (DPMSolverMultistepScheduler, {}),
    "dpmpp_2m_sde": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "dpmpp_3m_sde": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 3},
    ),
    "ddim": (DDIMScheduler, {}),
    "unipc": (UniPCMultistepScheduler, {}),
}

# Noise schedules, for the scheduler classes that take them.
SCHEDULES: dict[str, dict[str, Any]] = {
    "default": {},
    "karras": {"use_karras_sigmas": True},
    "exponential": {"use_exponential_sigmas": True},
}
_SCHEDULED = (EulerDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler)


def make_scheduler(config: Any, name: str, schedule: str) -> Any:
    """The sampler `name` with noise schedule `schedule`, from a checkpoint's scheduler
    config."""
    try:
        cls, kwargs = SCHEDULERS[name]
        sigmas = SCHEDULES[schedule]
    except KeyError as e:
        raise ValueError(f"Unknown scheduler or schedule {e.args[0]!r}") from None
    if issubclass(cls, _SCHEDULED):
        kwargs = {**kwargs, **sigmas}
    return cls.from_config(config, **kwargs)
