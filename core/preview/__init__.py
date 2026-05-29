"""
Anima LoRA Editor — live preview package.

Standalone, in-process sample-image generation so you can *see* the effect of a
LoRA edit without launching ComfyUI or any external server. The sampler is a
faithful, ComfyUI-independent re-implementation of the RES (Refined Exponential
Solver) family used by RES4LYF's ``ClownsharKSampler``
(https://github.com/ClownsharkBatwing/RES4LYF) — see ``sampler.py``.

One model backend is provided (see ``backends.py``):

* ``AnimaModelBackend`` — loads the real Anima DiT + VAE + Qwen3 text encoder
  from configurable paths and samples on the GPU. Lit up by ``setup_preview``.
  This is GPU-only: with no CUDA / Anima weights, ``generate_preview`` raises
  instead of falling back — there is no CPU stand-in.

Public surface used by ``app.py``:

    from core.preview import (
        PreviewConfig,
        generate_preview,        # -> PreviewResult (png_bytes + meta)
        preview_capabilities,    # -> dict for the UI to show what's available
        SAMPLER_NAMES,
        SCHEDULER_NAMES,
    )
"""

from .pipeline import PreviewConfig, PreviewResult, generate_preview
from .capabilities import preview_capabilities
from .sampler import SAMPLER_NAMES
from .schedulers import SCHEDULER_NAMES

__all__ = [
    "PreviewConfig",
    "PreviewResult",
    "generate_preview",
    "preview_capabilities",
    "SAMPLER_NAMES",
    "SCHEDULER_NAMES",
]
