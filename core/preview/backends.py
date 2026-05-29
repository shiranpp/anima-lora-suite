"""
Model backends for the live preview.

A backend turns a text prompt + (optionally) an edited LoRA into a *denoiser*
the sampler can drive, plus a VAE-style ``decode`` that turns the final latent
into an RGB image. The sampler in ``sampler.py`` knows nothing about Anima,
CFG, or VAEs — all of that lives here, behind a small protocol:

    AnimaModelBackend  — loads the Anima DiT + Qwen-Image VAE + Qwen3 text
                         encoder from the vendored ``core.anima`` package and
                         samples on CUDA. Requires the generation extras
                         (``setup_preview``) and the model files. GPU only —
                         there is no CPU fallback.

It implements the ``AnimaBackend`` surface.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch


Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def embed_cross_context(dit, tok, enc, qwen3, prompt: str, device):
    """Encode ``prompt`` into the DiT cross-attention context.

    Tokenizes with the vendored Anima strategy, runs the Qwen3 text encoder,
    then bridges through the DiT's ``llm_adapter`` (Qwen3 -> DiT). Returns the
    ``[B, seq, ctx]`` tensor the DiT cross-attention consumes — the same object
    the denoiser is conditioned on, and the projection target the keyword
    attribution measures. Shared by the GPU backend and the CPU fallback
    encoder so both produce an identical context for a given prompt.
    """
    tokens = tok.tokenize(prompt)
    pe, am, ids, t5am = enc.encode_tokens(tok, [qwen3], tokens)
    if isinstance(pe, np.ndarray):
        pe = torch.from_numpy(pe).unsqueeze(0)
        am = torch.from_numpy(am).unsqueeze(0)
        ids = torch.from_numpy(ids).unsqueeze(0)
        t5am = torch.from_numpy(t5am).unsqueeze(0)
    pe = pe.to(device, dtype=dit.t_embedding_norm.weight.dtype)
    am = am.to(device)
    ids = ids.to(device, dtype=torch.long)
    t5am = t5am.to(device)
    if getattr(dit, "use_llm_adapter", False) and hasattr(dit, "llm_adapter"):
        cross = dit.llm_adapter(
            source_hidden_states=pe, target_input_ids=ids,
            target_attention_mask=t5am, source_attention_mask=am,
        )
        cross[~t5am.bool()] = 0
    else:
        cross = pe
    return cross


class AnimaBackend:
    """Interface every preview backend implements."""

    name: str = "base"
    is_real: bool = False

    def load(self) -> None:
        """Heavy one-time init (load models). Idempotent."""

    def begin(self, seed: int) -> None:
        """Called once per generation with the run seed (default: no-op)."""

    def sigma_range(self) -> Tuple[float, float]:
        """(sigma_min, sigma_max) for the schedule."""
        return (0.03, 14.6)

    def make_sigmas(self, scheduler, steps, sigma_min, sigma_max, device):
        """Optional backend-specific sigma schedule. Return None to let the
        pipeline use the named scheduler from ``schedulers.get_sigmas``."""
        return None

    def latent_shape(self, width: int, height: int, batch: int = 1) -> Tuple[int, ...]:
        raise NotImplementedError

    def renoise(self, x0: torch.Tensor, sigma: float, noise: torch.Tensor) -> torch.Tensor:
        """Add noise to a clean latent ``x0`` up to noise level ``sigma``.

        Used by the hi-res fix to bring an upscaled clean latent back to a
        partially-noised state the sampler can refine from. The default is the
        k-diffusion variance-exploding form ``x0 + sigma·noise`` (the inverse of
        the default ``denoised = x - sigma·d`` step); flow backends override it.
        """
        return x0 + float(sigma) * noise

    def set_lora(self, state_dict: Optional[Dict[str, torch.Tensor]]) -> None:
        """Apply (or clear) the edited LoRA before sampling."""

    def encode(self, prompt: str, negative: str = ""):
        """Return an opaque conditioning object passed back to ``denoiser``."""
        raise NotImplementedError

    def denoiser(self, cond, cfg: float) -> Denoiser:
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """Latent -> uint8 (H, W, 3) RGB image."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real backend — loads Anima via the vendored core.anima package and samples on CUDA.
# ---------------------------------------------------------------------------

def _clean_path(value: Optional[str]) -> str:
    """Normalize a user-entered path.

    Strips whitespace and the surrounding quotes Windows Explorer's "Copy as
    path" adds (``"G:\\...\\model.safetensors"``). Without this the path *looks*
    set but ``os.path.exists`` fails on the literal quotes, so it gets reported
    as a missing model — the classic "I set the path but it says set model paths".
    """
    p = (value or "").strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"'):
        p = p[1:-1].strip()
    return os.path.expanduser(p)


@dataclass
class ModelPaths:
    dit: str = ""
    vae: str = ""
    text_encoder: str = ""   # Qwen3 (dir or file)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ModelPaths":
        d = d or {}
        return cls(
            dit=_clean_path(d.get("dit")),
            vae=_clean_path(d.get("vae")),
            text_encoder=_clean_path(d.get("text_encoder")),
        )

    def missing(self):
        out = []
        for label, p in (("DiT", self.dit), ("VAE", self.vae), ("Qwen3 text encoder", self.text_encoder)):
            if not p:
                out.append(f"{label} (not set)")
            elif not os.path.exists(p):
                out.append(f"{label} ({p} not found)")
        return out


class AnimaModelBackend(AnimaBackend):
    """Real Anima generation on GPU — standalone, no ComfyUI.

    Loads the Anima DiT + WanVAE + Qwen3 text encoder using the vendored Anima
    model code in ``core.anima``, merges the *edited* LoRA in memory
    (``core.anima.lora_anima``), encodes the prompt with the vendored Anima
    tokenize/encode strategies, then samples with our ``ClownsharKSampler``
    RES solver.

    Anima is a **rectified-flow** model: the DiT predicts a velocity ``v`` and
    the canonical sampler integrates ``x += v·dt`` over sigmas 1→0. Our RES
    solver works on a *denoiser* (x0 prediction), and the two are related by
    ``denoised = x − σ·v`` — which makes the flow ODE identical to the denoiser
    ODE the solver already integrates (and makes ``euler`` reduce exactly to
    the canonical Euler step, a handy correctness check).

    The vendored Anima modules are imported lazily, so the editor still
    *starts* on a CPU-only box — but a preview request there fails fast with a
    clear error instead of returning anything.
    """

    name = "anima"
    is_real = True

    def __init__(self, paths: ModelPaths, device: str = "cuda"):
        self.paths = paths
        self.device = device
        self._dit = None
        self._vae = None
        self._vae_scale = None
        self._qwen3 = None
        self._tok = None
        self._enc = None
        self._dtype = None
        self._lora_anima = None   # vendored networks.lora_anima module
        self._merged = []         # [(temp_lora_path, multiplier), ...] currently merged
        self._loaded = False

    # -- availability ------------------------------------------------------
    @staticmethod
    def deps_ok() -> Tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, (
                f"CUDA not available in this Python (torch {torch.__version__}). "
                "Anima preview needs a CUDA GPU — install a CUDA build of torch "
                "for your driver from https://pytorch.org/get-started/locally/, "
                "then re-run setup_preview."
            )
        return True, "ok"

    def load(self) -> None:
        if self._loaded:
            return
        ok, why = self.deps_ok()
        if not ok:
            raise RuntimeError(why)
        missing = self.paths.missing()
        if missing:
            raise RuntimeError("missing model files: " + "; ".join(missing))

        from ..anima import anima_utils as au, strategy_anima as sa, lora_anima
        self._lora_anima = lora_anima

        self._dtype = torch.bfloat16
        self._qwen3, _ = au.load_qwen3_text_encoder(self.paths.text_encoder, dtype=self._dtype, device="cpu")
        self._qwen3.eval().to(self.device)
        self._dit = au.load_anima_dit(self.paths.dit, dtype=self._dtype, device="cpu")
        self._dit.eval().to(self.device)
        self._vae, _, _, self._vae_scale = au.load_anima_vae(self.paths.vae, dtype=torch.bfloat16, device="cpu")
        self._vae.eval().to(self.device)
        self._vae_scale = [t.to(self.device) for t in self._vae_scale]

        self._tok = sa.AnimaTokenizeStrategy(qwen3_path=self.paths.text_encoder, qwen3_max_length=1024)
        self._enc = sa.AnimaTextEncodingStrategy()
        self._loaded = True

    # -- LoRA merge / unmerge (vendored convention) ------------------------
    def _apply_lora_file(self, lora_file: str, multiplier: float):
        lora_anima = self._lora_anima
        net, sd = lora_anima.create_network_from_weights(
            multiplier=multiplier, file=lora_file,
            ae=self._vae, text_encoders=[self._qwen3], unet=self._dit,
            for_inference=True,
        )
        net.merge_to([self._qwen3], self._dit, sd, self._dtype, self.device)
        del net, sd
        torch.cuda.empty_cache()

    def set_lora(self, state_dicts):
        """Bake an arbitrary *stack* of edited LoRAs into the DiT.

        Accepts a single state_dict (back-compat) or a list of them; each is
        merged in turn with multiplier 1.0, so the previewed model reflects the
        *combination* of every layer. Whatever was baked in by the previous call
        is unmerged first (in reverse order, the inverse of how it was applied).
        """
        # Normalize to a list; a bare dict is the single-LoRA case.
        if state_dicts is None:
            state_dicts = []
        elif isinstance(state_dicts, dict):
            state_dicts = [state_dicts]

        # Unmerge whatever is currently baked in (reverse order), then merge the
        # new stack on top of the clean weights.
        if self._merged:
            for prev_file, prev_mul in reversed(self._merged):
                try:
                    self._apply_lora_file(prev_file, -prev_mul)
                finally:
                    try:
                        os.remove(prev_file)
                    except OSError:
                        pass
            self._merged = []

        sds = [sd for sd in state_dicts if sd]
        if not sds:
            return
        import tempfile
        from ..editor import save_lora_state_dict
        merged = []
        for sd in sds:
            fd, tmp = tempfile.mkstemp(suffix=".safetensors", prefix="anima_preview_lora_")
            os.close(fd)
            save_lora_state_dict(sd, tmp)
            self._apply_lora_file(tmp, 1.0)
            merged.append((tmp, 1.0))
        self._merged = merged

    @contextlib.contextmanager
    def temporarily_merged(self, state_dict):
        """Merge a LoRA for the duration of the block, then restore the touched
        weights **bit-exactly** and leave ``set_lora`` bookkeeping untouched.

        ``set_lora`` unmerges with a negative multiplier, which is *lossy* in
        bf16 — fine for a one-shot preview, but it would slowly degrade the
        shared DiT if a feature merged/unmerged on it repeatedly (e.g. keyword
        attribution). Here we snapshot the exact ``weight`` of every module
        ``merge_to`` will modify, then ``copy_`` the snapshot back, so the model
        the preview reuses is byte-identical afterwards. The snapshot lives on
        CPU to avoid a VRAM spike.
        """
        if not state_dict:
            yield
            return
        import tempfile
        from ..editor import save_lora_state_dict
        lora_anima = self._lora_anima
        fd, tmp = tempfile.mkstemp(suffix=".safetensors", prefix="anima_validate_lora_")
        os.close(fd)
        save_lora_state_dict(state_dict, tmp)
        saved = []
        try:
            net, sd = lora_anima.create_network_from_weights(
                multiplier=1.0, file=tmp,
                ae=self._vae, text_encoders=[self._qwen3], unet=self._dit,
                for_inference=True,
            )
            # Snapshot every weight merge_to is about to overwrite, *before* merging.
            for lora in list(net.text_encoder_loras) + list(net.unet_loras):
                ref = getattr(lora, "org_module_ref", None)
                org = ref[0] if ref else getattr(lora, "org_module", None)
                w = getattr(org, "weight", None)
                if w is not None:
                    saved.append((w, w.detach().to("cpu", copy=True)))
            net.merge_to([self._qwen3], self._dit, sd, self._dtype, self.device)
            del net, sd
            torch.cuda.empty_cache()
            yield
        finally:
            with torch.no_grad():
                for param, clone in saved:
                    param.data.copy_(clone)
            try:
                os.remove(tmp)
            except OSError:
                pass
            torch.cuda.empty_cache()

    # -- generation surface ------------------------------------------------
    def sigma_range(self):
        return (1e-3, 1.0)  # rectified flow: sigma 1 (noise) -> 0 (clean)

    def make_sigmas(self, scheduler, steps, sigma_min, sigma_max, device):
        # Default flow schedule: linspace 1 -> 0 (matches the canonical Anima schedule).
        if (scheduler or "").lower() in ("karras", "exponential"):
            from .schedulers import get_sigmas
            return get_sigmas(scheduler, steps, 1e-3, 1.0, device=device)
        return torch.linspace(1.0, 0.0, steps + 1, device=device)

    def latent_shape(self, width, height, batch=1):
        # Anima latent: (B, 16, T=1, H/8, W/8). H/W already multiples of 8.
        return (batch, 16, 1, height // 8, width // 8)

    def renoise(self, x0, sigma, noise):
        # Rectified flow forward process: x_sigma = (1-sigma)·x0 + sigma·noise,
        # the exact inverse of this backend's ``denoised = x - sigma·v`` step
        # (sigma 1 -> pure noise, sigma 0 -> clean), so the refine pass starts
        # from a state the denoiser interprets correctly.
        s = float(sigma)
        return (1.0 - s) * x0 + s * noise

    def _embed(self, prompt: str):
        return embed_cross_context(self._dit, self._tok, self._enc, self._qwen3,
                                   prompt, self.device)

    def encode(self, prompt, negative=""):
        return {"pos": self._embed(prompt or ""), "neg": self._embed(negative or "")}

    def denoiser(self, cond, cfg) -> Denoiser:
        use_cfg = cfg > 1.0
        pos, neg = cond["pos"], cond["neg"]
        mdtype = self._dit.t_embedding_norm.weight.dtype

        def denoise(x, sigma):
            xb = x.to(mdtype)
            t = sigma.to(self.device, dtype=mdtype)         # (B,)
            _, _, _, lh, lw = xb.shape
            pad = torch.zeros(xb.shape[0], 1, lh, lw, dtype=mdtype, device=self.device)
            if use_cfg:
                xd = torch.cat([xb, xb], dim=0)
                td = torch.cat([t, t], dim=0)
                crossd = torch.cat([pos, neg], dim=0)
                padd = torch.cat([pad, pad], dim=0)
                v = self._dit(xd, td, crossd, padding_mask=padd)
                vp, vn = v.chunk(2)
                v = vn + cfg * (vp - vn)
            else:
                v = self._dit(xb, t, pos, padding_mask=pad)
            # velocity -> denoised (x0):  x0 = x - sigma * v
            s = sigma.to(x.device, dtype=x.dtype).view(-1, 1, 1, 1, 1)
            return x - s * v.to(x.dtype)

        return denoise

    def decode(self, latent):
        vdev = next(self._vae.parameters()).device
        vdtype = next(self._vae.parameters()).dtype
        decoded = self._vae.decode(latent.to(vdev, dtype=vdtype), self._vae_scale)
        image = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0)[0]
        if image.ndim == 4:          # (C, T, H, W) -> drop temporal dim
            image = image[:, 0, :, :]
        return (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
