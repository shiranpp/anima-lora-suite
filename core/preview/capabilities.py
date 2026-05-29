"""
Capability probe for the preview feature.

The UI calls this (via ``/api/preview/capabilities``) to decide what to show:
whether real Anima generation is possible, or — since there is no CPU fallback —
why preview is unavailable. Cheap and side-effect free; safe to call on startup.
"""

from typing import Optional

import torch

from .backends import AnimaModelBackend, ModelPaths
from .sampler import SAMPLER_NAMES
from .schedulers import SCHEDULER_NAMES


def preview_capabilities(paths: Optional[dict] = None) -> dict:
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else "CPU"

    mp = ModelPaths.from_dict(paths)
    real_ok, real_reason = AnimaModelBackend.deps_ok()
    missing_models = mp.missing()

    can_real = real_ok and not missing_models
    if can_real:
        active, why = "anima", f"real Anima generation ready on {device_name}"
    elif not real_ok:
        active, why = "unavailable", real_reason
    else:
        active, why = "unavailable", "set model paths: " + "; ".join(missing_models)

    return {
        "cuda": cuda,
        "device": device_name,
        "torch": torch.__version__,
        "real_backend_available": can_real,
        "active_backend": active,
        "reason": why,
        "missing_models": missing_models,
        "samplers": SAMPLER_NAMES,
        "schedulers": SCHEDULER_NAMES,
    }
