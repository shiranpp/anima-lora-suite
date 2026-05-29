"""
Standalone re-implementation of the RES (Refined Exponential Solver) sampler
family that RES4LYF's ``ClownsharKSampler`` is built on.

    Reference: https://github.com/ClownsharkBatwing/RES4LYF
               beta/phi_functions.py   (the φ_j remainder series)
               rk_method_beta.py       (calculate_res_2m_step)

RES4LYF's node is welded to ComfyUI's model-management / guider system; importing
it would drag all of ComfyUI back in — exactly the fragility we're avoiding. So
this module re-implements the *math* against a plain denoiser callable, with no
ComfyUI / RES4LYF import. It is pure ``torch`` and unit-testable on CPU.

A "denoiser" is any callable ``denoise(x, sigma) -> denoised`` that, given a
noisy latent ``x`` at noise level ``sigma`` (a per-batch tensor), returns the
model's prediction of the clean latent. CFG, conditioning, and LoRA application
all live *inside* that callable (see ``backends.py``); the sampler only walks the
sigma schedule.

Samplers implemented:
    euler            — 1st order, deterministic exponential Euler
    euler_ancestral  — 1st order + ancestral (eta) noise
    res_2s           — 2nd order single-step (exponential midpoint), RES_2S
    res_2m           — 2nd order multistep (the ClownsharKSampler default), RES_2M

``eta`` controls how much noise is re-injected each step (the SDE knob); it is
applied to every sampler, matching ClownsharKSampler where ``eta`` defaults 0.5.
"""

import math
from typing import Callable, List, Optional

import torch

SAMPLER_NAMES: List[str] = ["res_2m", "res_2s", "euler", "euler_ancestral"]

Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --- φ functions (exponential-integrator coefficients) ----------------------
# φ_j(z) = (e^z - Σ_{k=0}^{j-1} z^k/k!) / z^j   — the remainder series from
# RES4LYF/beta/phi_functions.py::_phi. Scalar version; h is small so float64
# keeps it stable near z -> 0.

def _phi(j: int, z: float) -> float:
    if abs(z) < 1e-7:
        # limit: φ_j(0) = 1/j!
        return 1.0 / math.factorial(j)
    remainder = sum((z ** k) / math.factorial(k) for k in range(j))
    return (math.exp(z) - remainder) / (z ** j)


# --- ancestral (eta) noise split --------------------------------------------

def _ancestral_step(sigma_from: float, sigma_to: float, eta: float):
    """Split ``sigma_to`` into (sigma_down, sigma_up) for ancestral sampling.

    Matches k-diffusion's ``get_ancestral_step`` (RES4LYF 'hard' gaussian mode).
    ``sigma_down`` is where the deterministic step lands; ``sigma_up`` is the
    amount of fresh noise added afterwards.
    """
    if eta <= 0.0 or sigma_to <= 0.0:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * math.sqrt(sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2),
    )
    sigma_down = math.sqrt(max(sigma_to ** 2 - sigma_up ** 2, 0.0))
    return sigma_down, sigma_up


def _noise_like(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


# --- the sampler -------------------------------------------------------------

@torch.no_grad()
def sample(
    denoise: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    sampler: str = "res_2m",
    eta: float = 0.5,
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Integrate the probability-flow ODE/SDE from ``sigmas[0]`` down to 0.

    Args:
        denoise:  callable ``(x, sigma_batch) -> denoised``.
        x:        initial latent, already scaled to ``sigmas[0]`` noise level.
        sigmas:   1-D tensor of ``steps + 1`` descending sigmas ending in 0.
        sampler:  one of ``SAMPLER_NAMES``.
        eta:      ancestral/SDE noise amount (ClownsharKSampler default 0.5).
        s_noise:  scales the injected ancestral noise (default 1.0).
        generator: seeded RNG for reproducible ancestral noise.
        callback: optional ``(step_index, total_steps)`` progress hook.

    Returns the final denoised latent.
    """
    sampler = (sampler or "res_2m").lower()
    if sampler not in SAMPLER_NAMES:
        raise ValueError(f"unknown sampler {sampler!r}; choose from {SAMPLER_NAMES}")

    s_in = x.new_ones((x.shape[0],))
    n = len(sigmas) - 1
    old_denoised: Optional[torch.Tensor] = None

    for i in range(n):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        if callback is not None:
            callback(i, n)

        denoised = denoise(x, sigma * s_in)

        # Every sampler re-injects noise per the eta knob (the SDE behaviour).
        sigma_down, sigma_up = _ancestral_step(sigma, sigma_next, eta)

        if sigma_next <= 0.0:
            # Last step: jump straight to the clean prediction.
            x = denoised
        elif sampler in ("euler", "euler_ancestral"):
            d = (x - denoised) / sigma
            x = x + d * (sigma_down - sigma)
        elif sampler == "res_2s":
            x = _res_2s_step(denoise, x, denoised, sigma, sigma_down, s_in)
        else:  # res_2m
            sigma_prev = float(sigmas[i - 1]) if i > 0 else None
            x = _res_2m_step(x, denoised, old_denoised, sigma, sigma_down, sigma_prev)

        # Ancestral noise injection (skipped on the final clean step).
        if sigma_up > 0.0 and sigma_next > 0.0:
            x = x + _noise_like(x, generator) * (sigma_up * s_noise)

        old_denoised = denoised

    return x


def _res_2m_step(x, denoised, old_denoised, sigma, sigma_down, sigma_prev):
    """RES_2M — 2nd-order multistep exponential update.

    Exponential-integrator form of the data-prediction ODE in log-sigma time:
        x_{n+1} = e^{-h} x_n - (e^{-h} - 1) · d̂
    where ``h = -log(sigma_down/sigma)`` and ``d̂`` is the denoised estimate,
    refined by the previous step's denoised via the φ-derived 2M correction
    (equivalent to RES4LYF's ``calculate_res_2m_step`` with c2 = -h_prev/h).
    """
    h = -math.log(sigma_down / sigma)
    e = math.exp(-h)  # == sigma_down / sigma
    if old_denoised is None or sigma_prev is None:
        # Bootstrap: 1st-order exponential Euler (RES_1).
        return e * x - (e - 1.0) * denoised

    h_prev = -math.log(sigma / sigma_prev)
    c2 = -h_prev / h
    # 2M b-coefficients from the φ functions (φ evaluated at -h).
    phi1 = _phi(1, -h)
    phi2 = _phi(2, -h)
    b2 = phi2 / c2
    b1 = phi1 - b2
    # h·(b1·eps1 + b2·eps2) with eps = denoised - x, in exponential form this
    # collapses to the familiar DPM++/RES 2M corrector:
    denoised_d = (b1 * denoised + b2 * old_denoised) / (b1 + b2)
    return e * x - (e - 1.0) * denoised_d


def _res_2s_step(denoise, x, denoised, sigma, sigma_down, s_in):
    """RES_2S — 2nd-order single-step exponential midpoint method."""
    h = -math.log(sigma_down / sigma)
    # Midpoint in log-sigma time.
    sigma_mid = sigma * math.exp(-0.5 * h)
    e_half = math.exp(-0.5 * h)
    x_2 = e_half * x - (e_half - 1.0) * denoised
    denoised_2 = denoise(x_2, sigma_mid * s_in)
    e = math.exp(-h)
    return e * x - (e - 1.0) * denoised_2
