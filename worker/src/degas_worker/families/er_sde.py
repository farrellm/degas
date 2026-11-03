"""ER-SDE (Cui et al., "Elucidating the solution space of extended reverse-time SDEs for
diffusion models", arXiv:2309.06169) for flow-matching models, torch-free.

A port of ComfyUI's `sample_er_sde` (VP ER-SDE-Solver-3, `comfy/k_diffusion/sampling.py`),
which follows the paper's code (QinpengCui/ER-SDE-Solver). For a flow model the sample at
sigma is `(1 - sigma) * x0 + sigma * noise`, so alpha = 1 - sigma, and the solver works in
lambda = sigma / alpha. Each step is linear in the sample, the step's denoised prediction
(x0), its first and second divided differences in lambda, and fresh noise. `coefficients`
gives those weights for a sigma schedule; the runner applies them to tensors.
"""

import math
from dataclasses import dataclass, replace

# ComfyUI's defaults.
INTEGRATION_POINTS = 200
MAX_STAGE = 3
# A flow schedule starts at sigma 1, where lambda is infinite; ComfyUI moves it this far in.
FIRST_SIGMA_OFFSET = 1e-4


@dataclass(frozen=True)
class Step:
    sample: float
    denoised: float
    # On the first divided difference of x0 in lambda, (x0_i - x0_{i-1}) / d1_over; 0 if unused.
    d1: float = 0.0
    d1_over: float = 1.0
    # On the second, (dd_i - dd_{i-1}) / d2_over; 0 if unused.
    d2: float = 0.0
    d2_over: float = 1.0
    noise: float = 0.0


def noise_scaler(lam: float) -> float:
    """ComfyUI's default ER-SDE noise scale function."""
    return lam * (math.exp(lam**0.3) + 10.0)


def coefficients(sigmas: list[float], s_noise: float = 1.0) -> list[Step]:
    """One `Step` per step of a schedule that ends at sigma 0 (len(sigmas) - 1 of them)."""
    sigmas = list(sigmas)
    if len(sigmas) > 1 and sigmas[0] >= 1:
        sigmas[0] = 1 - FIRST_SIGMA_OFFSET
    lams = [s / (1 - s) for s in sigmas]
    steps = []
    for i in range(len(sigmas) - 1):
        if sigmas[i + 1] == 0:
            steps.append(Step(sample=0.0, denoised=1.0))
            continue
        stage = min(MAX_STAGE, i + 1)
        lam_s, lam_t = lams[i], lams[i + 1]
        alpha_s, alpha_t = 1 - sigmas[i], 1 - sigmas[i + 1]
        r = noise_scaler(lam_t) / noise_scaler(lam_s)
        step = Step(sample=alpha_t / alpha_s * r, denoised=alpha_t * (1 - r))
        if stage >= 2:
            dt = lam_t - lam_s
            width = -dt / INTEGRATION_POINTS
            points = [lam_t + k * width for k in range(INTEGRATION_POINTS)]
            s = sum(1 / noise_scaler(p) for p in points) * width
            step = replace(
                step,
                d1=alpha_t * (dt + s * noise_scaler(lam_t)),
                d1_over=lam_s - lams[i - 1],
            )
            if stage >= 3:
                s_u = sum((p - lam_s) / noise_scaler(p) for p in points) * width
                step = replace(
                    step,
                    d2=alpha_t * (dt**2 / 2 + s_u * noise_scaler(lam_t)),
                    d2_over=(lam_s - lams[i - 2]) / 2,
                )
        if s_noise > 0:
            var = lam_t**2 - lam_s**2 * r**2
            step = replace(step, noise=alpha_t * s_noise * math.sqrt(var) if var > 0 else 0.0)
        steps.append(step)
    return steps
