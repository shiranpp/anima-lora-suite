"""
Keyword attribution — "validation prompt".

Where ``analyzer.analyze_lora`` answers *"which blocks does this LoRA change the
most overall?"* (prompt-free, just weight magnitudes), this module answers the
keyword-conditioned question *"which blocks does THIS keyword light up?"*.

LoRA weights aren't indexed by keyword — a concept is a distributed change
across all 28 blocks — so the only honest way to attribute a keyword is to push
it through the model and measure something prompt-dependent. Three tiers, most
to least faithful:

  • activation-delta (faithful, needs the GPU backend)
        Encode the keyword, run ONE DiT forward pass with the edited LoRA merged
        and once with it removed, using the *same* base conditioning so the
        per-block delta isolates the block-weight change. Each Anima ``Block`` is
        residual (``x_out = x_in + Δ``); we score ``‖Δ_on − Δ_off‖`` per block.

  • cross-attn projection (cheaper fallback)
        No sampling. Project the keyword's cross-attention context ``E`` through
        each block's ``cross_attn`` k/v LoRA: ``‖up @ (down @ Eᵀ)‖``. Only
        attributes the *text-conditioning* pathway (misses self-attn / MLP), so
        it undercounts — surfaced in the returned ``note``.

  • static (no model available)
        Falls back to the prompt-free ``analyze_lora`` score, clearly labelled.

The result dict mirrors ``analyze_lora`` so the UI paints it with the same
meter code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from .detect import extract_block_info
from .presets import ANIMA_NUM_BLOCKS

# Attribution runs at a modest resolution / single timestep — we want a stable
# per-block signal, not a pretty image, and this keeps VRAM + latency low.
VALIDATE_SIZE = 256
VALIDATE_SIGMA = 0.5


# ─── shared helpers ──────────────────────────────────────────────────────────

def _norm(t: torch.Tensor) -> float:
    """Frobenius norm as a float, in fp32 for stable bf16/fp16 results."""
    return float(torch.linalg.norm(t.detach().to(torch.float32).flatten()).item())


def _pack(block_totals: Dict[int, float], llm_total: float, other_total: float) -> dict:
    """Shape raw per-component scores like ``analyze_lora`` for the UI."""
    max_score = max(block_totals.values()) if block_totals else 0.0
    block_norm: Dict[int, int] = {}
    for i in range(ANIMA_NUM_BLOCKS):
        raw = block_totals.get(i, 0.0)
        block_norm[i] = int(round(100 * raw / max_score)) if max_score > 0 else 0
    return {
        "block_scores": {i: round(block_totals.get(i, 0.0), 4) for i in range(ANIMA_NUM_BLOCKS)},
        "block_norm": block_norm,
        "llm_adapter_score": round(llm_total, 4),
        "other_score": round(other_total, 4),
        "max_score": round(max_score, 4),
    }


# ─── tier 1: activation-delta (faithful) ─────────────────────────────────────

def _other_children(dit) -> List[Tuple[str, "torch.nn.Module"]]:
    """Direct DiT submodules that are neither the block stack nor the adapter —
    embedders, time embed, final layer. Their activation delta is the 'other'
    weights' keyword-conditioned effect (usually ~0; most LoRAs don't touch
    them)."""
    out = []
    for name, mod in dit.named_children():
        if name in ("blocks", "llm_adapter"):
            continue
        out.append((name, mod))
    return out


def _run_capture(backend, dit, cond, device, size, sigma, seed):
    """One DiT forward pass; returns per-block residual contributions (x_out −
    x_in) and 'other' submodule outputs, all on CPU/fp32. The latent is seeded
    so two passes differ only by the merged weights."""
    pre: Dict[int, torch.Tensor] = {}
    contrib: Dict[int, torch.Tensor] = {}
    other_out: Dict[str, torch.Tensor] = {}
    handles = []

    def pre_hook(i):
        def h(_mod, args):
            if args:
                pre[i] = args[0].detach()
        return h

    def post_hook(i):
        def h(_mod, _args, out):
            x_out = out[0] if isinstance(out, (tuple, list)) else out
            x_in = pre.get(i)
            c = (x_out.detach() - x_in) if x_in is not None else x_out.detach()
            contrib[i] = c.to("cpu", torch.float32)
        return h

    def other_hook(name):
        def h(_mod, _args, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                other_out[name] = t.detach().to("cpu", torch.float32)
        return h

    for i, blk in enumerate(dit.blocks):
        handles.append(blk.register_forward_pre_hook(pre_hook(i)))
        handles.append(blk.register_forward_hook(post_hook(i)))
    for name, mod in _other_children(dit):
        handles.append(mod.register_forward_hook(other_hook(name)))

    try:
        gen = torch.Generator(device=device).manual_seed(int(seed) & 0x7FFFFFFF)
        shape = backend.latent_shape(size, size, batch=1)
        x = torch.randn(shape, generator=gen, device=device)
        denoise = backend.denoiser(cond, 1.0)  # cfg<=1 → positive cond only
        s = torch.full((shape[0],), float(sigma), device=device)
        denoise(x, s)
    finally:
        for h in handles:
            h.remove()
    return contrib, other_out


def attribute_activation(
    backend,
    keyword: str,
    edited_sd: Dict[str, torch.Tensor],
    *,
    seed: int = 0,
    size: int = VALIDATE_SIZE,
    sigma: float = VALIDATE_SIGMA,
) -> dict:
    """Faithful keyword attribution via the loaded GPU backend.

    Runs the keyword through the DiT with the edited LoRA applied vs. not, and
    measures the per-block change in residual contribution. The LoRA is applied
    via ``backend.temporarily_merged`` (snapshot + bit-exact restore), so this
    NEVER degrades the shared model the preview reuses — it measures the LoRA's
    marginal effect on whatever state the model is currently in.
    """
    dit = backend._dit
    device = backend.device

    # 1. current weights (not mutated): keyword conditioning + reference pass.
    with torch.no_grad():
        cond_base = backend.encode(keyword, "")
        cross_off = cond_base["pos"]
        off_contrib, off_other = _run_capture(backend, dit, cond_base, device, size, sigma, seed)

    # 2. edited LoRA applied: SAME base conditioning so block deltas isolate the
    #    block-weight change (the adapter's effect is scored separately below).
    #    The context manager restores the touched weights exactly on exit.
    with backend.temporarily_merged(edited_sd):
        with torch.no_grad():
            cross_on = backend.encode(keyword, "")["pos"]
            on_contrib, on_other = _run_capture(backend, dit, cond_base, device, size, sigma, seed)

    block_totals: Dict[int, float] = {}
    for i in on_contrib:
        if i in off_contrib and on_contrib[i].shape == off_contrib[i].shape:
            block_totals[i] = _norm(on_contrib[i] - off_contrib[i])

    # LLMAdapter: how much the LoRA shifts the keyword's cross context.
    llm_total = (
        _norm(cross_on - cross_off)
        if torch.is_tensor(cross_on) and cross_on.shape == cross_off.shape
        else 0.0
    )
    # Other: delta over embedders / final layer (approximate — see _other_children).
    other_total = sum(
        _norm(on_other[k] - off_other[k])
        for k in on_other
        if k in off_other and on_other[k].shape == off_other[k].shape
    )
    return _pack(block_totals, llm_total, other_total)


# ─── tier 2: cross-attention projection (fallback) ───────────────────────────

def _is_cross_kv(key: str) -> bool:
    kl = key.lower()
    is_cross = ("cross_attn" in kl) or ("cross" in kl and "attn" in kl)
    is_kv = any(s in kl for s in ("k_proj", "v_proj", "to_k", "to_v"))
    return is_cross and is_kv


_DOWN_SUFFIXES = ("lora_down.weight", "lora_a.weight")


def _down_base(key: str) -> Optional[str]:
    """If ``key`` is a 'down'/'A' LoRA tensor, return its prefix (incl. trailing
    '.'); else None. Case-insensitive so kohya and diffusers names both match."""
    kl = key.lower()
    for suf in _DOWN_SUFFIXES:
        if kl.endswith(suf):
            return key[: len(key) - len(suf)]
    return None


def _cross_attn_pairs(state_dict: Dict[str, torch.Tensor]) -> Dict[int, list]:
    """Group cross-attn k/v LoRA pairs by block index.

    Returns ``{block: [(down, up, scale), ...]}`` where ``scale`` folds in the
    ``alpha/rank`` factor when an ``.alpha`` is present (matching the true
    contribution magnitude)."""
    out: Dict[int, list] = defaultdict(list)
    for key in state_dict:
        if not _is_cross_kv(key):
            continue
        base = _down_base(key)
        if base is None:
            continue
        # Pair with the matching 'up'/'B' tensor, preserving original casing.
        up = next((base + s for s in ("lora_up.weight", "lora_B.weight", "lora_b.weight")
                   if (base + s) in state_dict), None)
        if up is None:
            continue
        down_t, up_t = state_dict[key], state_dict[up]
        rank = down_t.shape[0]
        scale = 1.0
        alpha_key = base + "alpha"
        if alpha_key in state_dict:
            try:
                scale = float(state_dict[alpha_key].item()) / float(rank)
            except Exception:
                scale = 1.0
        tag, n = extract_block_info(key)
        if tag == "block":
            out[n].append((down_t, up_t, scale))
    return out


def attribute_cross_attn(state_dict: Dict[str, torch.Tensor], cross_emb: torch.Tensor) -> dict:
    """Project the keyword's cross context through each block's cross-attn k/v
    LoRA. Cheap (no sampling); text-pathway only."""
    E = cross_emb
    if torch.is_tensor(E) and E.dim() == 3:
        E = E[0]
    # State-dict tensors are CPU/fp32; the backend may hand us a CUDA/bf16 E.
    # Project on CPU (E is tiny) so the matmul never crosses devices/dtypes.
    E = E.detach().to("cpu", torch.float32)
    ctx = E.shape[-1]

    block_totals: Dict[int, float] = {}
    pairs = _cross_attn_pairs(state_dict)
    for n, items in pairs.items():
        total = 0.0
        for down, up, scale in items:
            d = down.detach().to(torch.float32)
            u = up.detach().to(torch.float32)
            if d.shape[-1] != ctx:  # dim mismatch (unexpected adapter shape) — skip
                continue
            proj = u @ (d @ E.t())   # [out, rank] @ ([rank, ctx] @ [ctx, seq]) = [out, seq]
            total += _norm(proj) * scale
        if total > 0:
            block_totals[n] = total
    return _pack(block_totals, 0.0, 0.0)


# ─── CPU cross-context encoder (for the no-GPU fallback) ─────────────────────

class CpuCrossContext:
    """Loads just enough Anima on CPU (Qwen3 + DiT's llm_adapter) to turn a
    keyword into its cross-attention context, for the cross-attn fallback when
    no CUDA backend is available. The 2B DiT load is RAM-heavy but there's no
    sampling, so the actual attribution stays cheap."""

    def __init__(self, model_paths):
        from .preview.backends import ModelPaths
        self.mp = model_paths if isinstance(model_paths, ModelPaths) else ModelPaths.from_dict(model_paths)
        self._dit = None
        self._qwen3 = None
        self._tok = None
        self._enc = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        missing = self.mp.missing()
        if missing:
            raise RuntimeError("missing model files: " + "; ".join(missing))
        from .anima import anima_utils as au, strategy_anima as sa
        dtype = torch.float32  # CPU: fp32 for numeric stability
        self._qwen3, _ = au.load_qwen3_text_encoder(self.mp.text_encoder, dtype=dtype, device="cpu")
        self._qwen3.eval()
        self._dit = au.load_anima_dit(self.mp.dit, dtype=dtype, device="cpu")
        self._dit.eval()
        self._tok = sa.AnimaTokenizeStrategy(qwen3_path=self.mp.text_encoder, qwen3_max_length=1024)
        self._enc = sa.AnimaTextEncodingStrategy()
        self._loaded = True

    def cross_context(self, keyword: str) -> torch.Tensor:
        from .preview.backends import embed_cross_context
        with torch.no_grad():
            return embed_cross_context(self._dit, self._tok, self._enc, self._qwen3, keyword, "cpu")


# Cache the CPU encoder per model-path set — loading the DiT is expensive.
_CPU_CACHE: Dict[str, CpuCrossContext] = {}


def cpu_cross_context(model_paths) -> CpuCrossContext:
    from .preview.backends import ModelPaths
    mp = model_paths if isinstance(model_paths, ModelPaths) else ModelPaths.from_dict(model_paths)
    key = f"{mp.dit}|{mp.text_encoder}"
    enc = _CPU_CACHE.get(key)
    if enc is None:
        enc = CpuCrossContext(mp)
        enc.load()
        _CPU_CACHE[key] = enc
    return enc
