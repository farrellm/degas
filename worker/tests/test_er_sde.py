import math

import pytest

from degas_worker.families.er_sde import FIRST_SIGMA_OFFSET, coefficients

SIGMAS = [1.0, 0.95, 0.85, 0.7, 0.5, 0.3, 0.15, 0.05, 0.0]


def test_one_step_per_interval_and_the_last_returns_x0() -> None:
    steps = coefficients(SIGMAS)
    assert len(steps) == len(SIGMAS) - 1
    last = steps[-1]
    assert (last.sample, last.denoised, last.d1, last.d2, last.noise) == (0, 1, 0, 0, 0)


def test_stages_rise_to_three_as_history_builds() -> None:
    steps = coefficients(SIGMAS)
    assert (steps[0].d1, steps[0].d2) == (0, 0)
    assert steps[1].d1 != 0
    assert steps[1].d2 == 0
    assert all(s.d1 != 0 and s.d2 != 0 for s in steps[2:-1])


def test_a_clean_sample_stays_on_the_flow_path() -> None:
    """With x = (1 - sigma) * x0 and no noise, the first-order step lands on (1 - sigma') * x0."""
    sigmas = [1 - FIRST_SIGMA_OFFSET, *SIGMAS[1:]]
    for i, step in enumerate(coefficients(SIGMAS)[:-1]):
        alpha_s, alpha_t = 1 - sigmas[i], 1 - sigmas[i + 1]
        assert step.sample * alpha_s + step.denoised == pytest.approx(alpha_t)


def test_noise_keeps_the_marginal_at_the_next_sigma() -> None:
    """The sample's carried noise plus the fresh noise has std sigma'."""
    sigmas = [1 - FIRST_SIGMA_OFFSET, *SIGMAS[1:]]
    for i, step in enumerate(coefficients(SIGMAS)[:-1]):
        std = math.hypot(step.sample * sigmas[i], step.noise)
        assert std == pytest.approx(sigmas[i + 1], rel=1e-6)


def test_no_noise_without_s_noise() -> None:
    assert all(s.noise == 0 for s in coefficients(SIGMAS, s_noise=0))


def test_divided_differences_are_over_lambda_gaps() -> None:
    lams = [s / (1 - s) for s in [1 - FIRST_SIGMA_OFFSET, *SIGMAS[1:-1]]]
    steps = coefficients(SIGMAS)
    assert steps[2].d1_over == pytest.approx(lams[2] - lams[1])
    assert steps[2].d2_over == pytest.approx((lams[2] - lams[0]) / 2)
