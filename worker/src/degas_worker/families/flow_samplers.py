"""Samplers for flow-match pipelines (Qwen-Image 2.1, Krea 2): diffusers' Euler, or ER-SDE.

Both run on `FlowMatchEulerDiscreteScheduler`'s sigmas, so the pipeline's shift and any
sigma spacing (`use_beta_sigmas`) apply to either.
"""

from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)

from degas_worker.families import er_sde
from degas_worker.families.runtime import seeded

# Keep in sync with the server (degas/families/schema.py, FLOW_SAMPLERS).
SAMPLERS = ("euler", "er_sde")


def flow_scheduler(config: Any, sampler: str, **extra: Any) -> Any:
    """The sampler `sampler`, from the pipeline's scheduler config with `extra` on top."""
    if sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler {sampler!r}")
    cls = ErSdeScheduler if sampler == "er_sde" else FlowMatchEulerDiscreteScheduler
    return cls.from_config(config, **extra)


def seeded_noise(scheduler: Any, seed: int) -> Any:
    """The pipeline's generator for `seed`. An ER-SDE scheduler draws its noise from it too,
    after the initial latents."""
    generator = seeded(seed)
    if isinstance(scheduler, ErSdeScheduler):
        scheduler.generator = generator
    return generator


class ErSdeScheduler(FlowMatchEulerDiscreteScheduler):  # type: ignore[misc]
    """ComfyUI's `er_sde` sampler on the flow-match schedule (`er_sde.py` has the math): the
    same sigmas as Euler, with a third-order step that adds fresh noise each step. The noise is
    drawn on the CPU from `generator`, so a seed gives the same image on every GPU."""

    # Neither `__init__` nor `set_timesteps` is overridden: diffusers reads their signatures
    # (`from_config` for the config's keys, `retrieve_timesteps` for a `sigmas` parameter).
    # The weights are worked out when `step` first sees a new schedule instead.
    generator: torch.Generator | None = None
    _for_sigmas: Any = None
    _coefficients: tuple[er_sde.Step, ...] = ()
    _history: tuple[torch.Tensor, ...] = ()  # the last x0, then its divided difference

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        *args: Any,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        if self._for_sigmas is not self.sigmas:  # `set_timesteps` made a new schedule
            self._for_sigmas = self.sigmas
            self._coefficients = tuple(er_sde.coefficients([float(s) for s in self.sigmas]))
            self._history = ()
        if self.step_index is None:
            self._init_step_index(timestep)
        coef = self._coefficients[self.step_index]
        x = sample.to(torch.float32)
        x0 = x - float(self.sigmas[self.step_index]) * model_output.to(torch.float32)
        prev = coef.sample * x + coef.denoised * x0
        history: tuple[torch.Tensor, ...] = (x0,)
        if coef.d1:
            dd = (x0 - self._history[0]) / coef.d1_over
            prev = prev + coef.d1 * dd
            history = (x0, dd)
            if coef.d2:
                prev = prev + coef.d2 * (dd - self._history[1]) / coef.d2_over
        if coef.noise:
            noise = torch.randn(x.shape, generator=self.generator, dtype=torch.float32)
            prev = prev + coef.noise * noise.to(x.device)
        self._history = history
        self._step_index += 1
        prev = prev.to(model_output.dtype)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev) if return_dict else (prev,)
