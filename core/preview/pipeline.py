"""
Preview pipeline: edited LoRA + prompt -> PNG.

Glues the pieces together:
    load the real Anima GPU backend (error if unavailable)
      -> apply the *current* edit to the LoRA in memory
      -> encode prompt / negative
      -> build the initial noise latent (seeded)
      -> run the vendored ClownsharKSampler RES solver
      -> VAE-decode to RGB
      -> encode PNG

The loaded backend is cached across calls (loading a 2 B DiT is expensive); the
cache key is the model-path set so changing paths reloads.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from ..editor import EditConfig, edit_lora, load_lora_state_dict
from .backends import AnimaBackend, AnimaModelBackend, ModelPaths
from .capabilities import preview_capabilities
from .pngio import encode_png
from .sampler import SAMPLER_NAMES, sample
from .schedulers import get_sigmas


@dataclass
class PreviewConfig:
    prompt: str = "a serene anime portrait, soft lighting, detailed eyes"
    negative: str = "blurry, low quality, deformed"
    steps: int = 20
    cfg: float = 5.5
    seed: int = 0
    sampler: str = "res_2m"
    scheduler: str = "karras"
    eta: float = 0.5
    width: int = 512
    height: int = 512
    # Hi-res fix: render at width×height, then latent-upscale by ``upscale`` and
    # run a short refine pass to reach the final size. ``upscale`` <= 1 disables
    # it (the sample is returned at the base size, unchanged). ``hires_denoise``
    # is how much noise the refine pass re-injects (0 = no change, 1 = re-render).
    upscale: float = 1.0
    hires_steps: int = 12
    hires_denoise: float = 0.5
    # Source LoRA + the edit to apply before previewing (mirrors /api/edit).
    # Single-LoRA fields, kept for back-compat with older callers.
    lora_path: str = ""
    edit: dict = field(default_factory=dict)
    # Multi-layer stack: each {"lora_path": str, "edit": dict}. When present this
    # takes precedence over the single ``lora_path``/``edit`` pair and the preview
    # reflects the *combination* of every layer.
    loras: list = field(default_factory=list)
    # Real-backend model files.
    model_paths: dict = field(default_factory=dict)

    def clamp(self) -> "PreviewConfig":
        self.steps = max(1, min(int(self.steps), 100))
        self.width = max(64, min(int(self.width), 2048))
        self.height = max(64, min(int(self.height), 2048))
        # The latent is the pixel size / 8 (VAE downscale), and the DiT then
        # patchifies that latent in 2x2 blocks — so the latent side must be even
        # and the pixel side a multiple of 16, not just 8. (A /8-but-odd-/16 size
        # like 504 -> latent 63 trips the DiT's patch-divisibility assert.)
        self.width -= self.width % 16
        self.height -= self.height % 16
        # Hi-res fix knobs. Cap the *final* long edge at 2048 so a large upscale
        # factor on an already-big base can't blow up VRAM on the decode.
        self.upscale = max(1.0, min(float(self.upscale), 4.0))
        self.hires_steps = max(1, min(int(self.hires_steps), 100))
        self.hires_denoise = max(0.05, min(float(self.hires_denoise), 1.0))
        long_edge = max(self.width, self.height)
        if long_edge * self.upscale > 2048:
            self.upscale = max(1.0, 2048.0 / long_edge)
        if self.sampler not in SAMPLER_NAMES:
            self.sampler = "res_2m"
        return self

    @property
    def final_width(self) -> int:
        # /16 so the upscaled latent stays patch-divisible (see clamp()).
        return (int(self.width * self.upscale) // 16) * 16

    @property
    def final_height(self) -> int:
        return (int(self.height * self.upscale) // 16) * 16


@dataclass
class PreviewResult:
    png_bytes: bytes
    meta: dict

    def data_uri(self) -> str:
        b64 = base64.b64encode(self.png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"


# --- backend cache -----------------------------------------------------------

_CACHE: Dict[str, AnimaBackend] = {}


def _build_edit_config(edit: dict) -> Optional[EditConfig]:
    if not edit:
        return None
    enabled = edit.get("enabled_blocks")
    return EditConfig(
        enabled_blocks=set(int(b) for b in enabled) if enabled is not None else set(range(28)),
        block_strengths={int(k): float(v) for k, v in (edit.get("block_strengths") or {}).items()},
        llm_adapter_enabled=bool(edit.get("llm_adapter_enabled", True)),
        llm_adapter_strength=float(edit.get("llm_adapter_strength", 1.0)),
        other_enabled=bool(edit.get("other_enabled", True)),
        other_strength=float(edit.get("other_strength", 1.0)),
        global_strength=float(edit.get("global_strength", 1.0)),
    )


def _layer_specs(cfg: PreviewConfig):
    """Normalize the config into a list of (lora_path, edit) layer specs.

    Prefers the multi-layer ``loras`` stack; falls back to the single
    ``lora_path``/``edit`` pair. Layers without a path are dropped.
    """
    specs = []
    for spec in (cfg.loras or []):
        path = (spec.get("lora_path") or "").strip()
        if path:
            specs.append((path, spec.get("edit") or {}))
    if not specs and cfg.lora_path:
        specs.append((cfg.lora_path, cfg.edit or {}))
    return specs


def _edited_state_dict(cfg: PreviewConfig) -> Optional[Dict[str, torch.Tensor]]:
    """Load a single LoRA and apply its edit, in memory (no file written)."""
    specs = _layer_specs(cfg)
    if not specs:
        return None
    return _edit_one(*specs[0])


def _edit_one(lora_path: str, edit: dict) -> Dict[str, torch.Tensor]:
    sd = load_lora_state_dict(lora_path)
    edit_cfg = _build_edit_config(edit)
    if edit_cfg is None:
        return sd
    new_sd, _ = edit_lora(sd, edit_cfg)
    return new_sd


def _edited_state_dicts(cfg: PreviewConfig):
    """Load every layer and apply its edit — one edited state_dict per layer."""
    return [_edit_one(path, edit) for path, edit in _layer_specs(cfg)]


def try_get_backend(model_paths: Optional[dict]) -> Optional[AnimaBackend]:
    """Return a loaded (cached) Anima backend, or ``None`` if it can't be
    brought up (no CUDA, models not set, load error). Never raises — callers
    that have a cheaper fallback (e.g. keyword attribution) use this instead of
    ``_select_backend``."""
    caps = preview_capabilities(model_paths)
    if not caps["real_backend_available"]:
        return None
    mp = ModelPaths.from_dict(model_paths)
    key = f"anima::{mp.dit}|{mp.vae}|{mp.text_encoder}"
    be = _CACHE.get(key)
    if be is None:
        try:
            be = AnimaModelBackend(mp)
            be.load()
        except Exception:
            return None
        _CACHE[key] = be
    return be


def _select_backend(cfg: PreviewConfig) -> AnimaBackend:
    caps = preview_capabilities(cfg.model_paths)
    if not caps["real_backend_available"]:
        # GPU-only: no CPU stand-in. Surface why so the UI can guide the user.
        raise RuntimeError(caps["reason"])

    mp = ModelPaths.from_dict(cfg.model_paths)
    key = f"anima::{mp.dit}|{mp.vae}|{mp.text_encoder}"
    be = _CACHE.get(key)
    if be is None:
        be = AnimaModelBackend(mp)
        be.load()
        _CACHE[key] = be
    return be


def _upscale_latent(latent: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Bicubic-resample a latent to (target_h, target_w) latent cells.

    The Anima latent is 5-D ``(B, C, T=1, h, w)`` and ``F.interpolate`` only does
    bicubic on 4-D tensors, so we fold the (singleton) temporal axis into the
    channel axis for the resize and fold it back out. Plain 4-D latents pass
    straight through. Done in float32 — interpolating a clean latent is cheap and
    the refine pass denoises away any resampling softness.
    """
    import torch.nn.functional as F

    if latent.ndim == 5:
        b, c, t, h, w = latent.shape
        x = latent.reshape(b, c * t, h, w).float()
        x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
        return x.reshape(b, c, t, target_h, target_w).to(latent.dtype)
    if latent.ndim == 4:
        x = F.interpolate(latent.float(), size=(target_h, target_w), mode="bicubic", align_corners=False)
        return x.to(latent.dtype)
    return latent


def _hires_sigmas(backend, cfg, sigma_min, sigma_max, device):
    """Truncated sigma schedule for the refine pass.

    Builds the full ``hires_steps`` schedule, then keeps only the tail starting
    at the sigma matching ``hires_denoise`` — lower denoise = start later (less
    re-noising) = fewer effective steps, so detail is added without redrawing.
    """
    full = backend.make_sigmas(cfg.scheduler, cfg.hires_steps, sigma_min, sigma_max, device)
    if full is None:
        full = get_sigmas(cfg.scheduler, cfg.hires_steps, sigma_min, sigma_max, device=device)
    start = sigma_min + (sigma_max - sigma_min) * cfg.hires_denoise
    idx = 0
    for i in range(len(full)):
        if float(full[i]) <= start:
            idx = i
            break
    sub = full[idx:]
    if sub.numel() < 2:  # always leave at least one real step
        sub = full[-2:]
    return sub


def _hires_fix(backend, cfg, latent, denoise, gen, device, sigma_min, sigma_max):
    """Latent-upscale ``latent`` to the final size and refine it.

    Returns ``(latent, out_w, out_h)``. The refine pass reuses the *same*
    denoiser (prompt/CFG/LoRA already baked in), so the upscaled image keeps the
    exact look of the base sample, only sharper and at native resolution.
    """
    th, tw = cfg.final_height // 8, cfg.final_width // 8
    latent_up = _upscale_latent(latent, th, tw)

    sub = _hires_sigmas(backend, cfg, sigma_min, sigma_max, device)
    noise = torch.randn(latent_up.shape, generator=gen, device=device, dtype=latent_up.dtype)
    x = backend.renoise(latent_up, float(sub[0]), noise)

    latent = sample(denoise, x, sub, sampler=cfg.sampler, eta=cfg.eta, generator=gen)
    return latent, cfg.final_width, cfg.final_height


def generate_preview(cfg: PreviewConfig) -> PreviewResult:
    cfg = cfg.clamp()
    t0 = time.time()

    backend = _select_backend(cfg)

    # Apply the edited LoRA stack (empty if no paths -> base model preview). The
    # backend merges them all, so the sample reflects the combination of layers.
    edited = _edited_state_dicts(cfg)
    backend.set_lora(edited)

    # Seeded RNG for reproducible noise + ancestral steps. The real backend
    # requires CUDA (enforced in _select_backend), so generation runs on GPU.
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed) & 0x7FFFFFFF)

    backend.begin(cfg.seed)

    sigma_min, sigma_max = backend.sigma_range()
    sigmas = backend.make_sigmas(cfg.scheduler, cfg.steps, sigma_min, sigma_max, device)
    if sigmas is None:
        sigmas = get_sigmas(cfg.scheduler, cfg.steps, sigma_min, sigma_max, device=device)

    shape = backend.latent_shape(cfg.width, cfg.height, batch=1)
    x = torch.randn(shape, generator=gen, device=device) * float(sigmas[0])

    cond = backend.encode(cfg.prompt, cfg.negative)
    denoise = backend.denoiser(cond, cfg.cfg)

    latent = sample(
        denoise, x, sigmas,
        sampler=cfg.sampler, eta=cfg.eta, generator=gen,
    )

    # Optional hi-res fix: latent-upscale + short refine pass for a sharp,
    # native-resolution result (skipped when upscale <= 1).
    out_w, out_h = cfg.width, cfg.height
    if cfg.upscale > 1.0:
        latent, out_w, out_h = _hires_fix(
            backend, cfg, latent, denoise, gen, device, sigma_min, sigma_max,
        )

    img = backend.decode(latent)
    png = encode_png(np.ascontiguousarray(img))

    meta = {
        "backend": backend.name,
        "is_real": backend.is_real,
        "sampler": cfg.sampler,
        "scheduler": cfg.scheduler,
        "steps": cfg.steps,
        "cfg": cfg.cfg,
        "eta": cfg.eta,
        "seed": cfg.seed,
        "width": out_w,
        "height": out_h,
        "base_width": cfg.width,
        "base_height": cfg.height,
        "upscale": round(cfg.upscale, 3),
        "hires_steps": cfg.hires_steps if cfg.upscale > 1.0 else 0,
        "hires_denoise": cfg.hires_denoise if cfg.upscale > 1.0 else 0.0,
        "device": device,
        "elapsed_s": round(time.time() - t0, 3),
        "lora_applied": len(edited) > 0,
        "lora_count": len(edited),
    }
    return PreviewResult(png_bytes=png, meta=meta)
