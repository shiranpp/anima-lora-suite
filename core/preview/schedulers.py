"""
Sigma schedules for the standalone sampler.

These produce a 1-D tensor of ``steps + 1`` sigmas, descending from ``sigma_max``
to ``0`` (the trailing zero marks the final denoised step). They mirror the
schedules offered by RES4LYF / ComfyUI; ``beta57`` (RES4LYF's default) needs
SciPy's incomplete-beta function which we don't depend on, so it falls back to
``karras`` — visually very close for a quick preview.
"""

import math
from typing import List

import torch

SCHEDULER_NAMES: List[str] = ["karras", "exponential", "linear", "simple", "beta57"]


def _append_zero(sigmas: torch.Tensor) -> torch.Tensor:
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0,
           device="cpu") -> torch.Tensor:
    """Karras et al. (2022) schedule — the most common default."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return _append_zero(sigmas)


def exponential(n: int, sigma_min: float, sigma_max: float, device="cpu") -> torch.Tensor:
    """Exponential (log-linear) spacing in sigma."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return _append_zero(sigmas)


def linear(n: int, sigma_min: float, sigma_max: float, device="cpu") -> torch.Tensor:
    """Plain linear spacing in sigma (ComfyUI's 'normal'-ish)."""
    sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    return _append_zero(sigmas)


def simple(n: int, sigma_min: float, sigma_max: float, device="cpu") -> torch.Tensor:
    """ComfyUI 'simple' — uniform in normalized index, here approximated by linear."""
    return linear(n, sigma_min, sigma_max, device=device)


def get_sigmas(name: str, steps: int, sigma_min: float, sigma_max: float,
               device="cpu") -> torch.Tensor:
    """Dispatch by name. Unknown / beta57 -> karras (no SciPy dep)."""
    name = (name or "karras").lower()
    if name == "exponential":
        return exponential(steps, sigma_min, sigma_max, device)
    if name == "linear":
        return linear(steps, sigma_min, sigma_max, device)
    if name == "simple":
        return simple(steps, sigma_min, sigma_max, device)
    # "karras", "beta57" (fallback), or anything unrecognised
    return karras(steps, sigma_min, sigma_max, device=device)
