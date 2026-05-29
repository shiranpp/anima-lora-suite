"""
LoRA size reduction.

Three independent levers, applied in this order when combined:

  1. SVD rank reduction — recompute each low-rank pair (ΔW = up·down·scale) at a
     smaller rank. Reduces the *number* of weights. Quality cost is tunable via a
     fixed target rank or an energy-retention fraction.
  2. dtype downcast — store float tensors at lower precision (fp16/bf16 = ½ of
     fp32; fp8 = ¼). Reduces *bytes per weight*. fp16/bf16 is ~lossless for
     LoRAs; fp8 is lossy but small.

Both reuse the LoRA-pair bookkeeping from :mod:`core.editor` so the same kohya
and diffusers key layouts are understood. Non-pair tensors (bias/diff/embeds)
are downcast but never SVD'd, and convolutional (non-2D) pairs are left intact.
"""

from typing import Dict, List, Optional, Tuple

import torch

from .editor import _module_table, _pair_scale


# --- dtype handling ----------------------------------------------------------

# UI/API name -> torch dtype. fp8 entries are validated at call time because
# older torch builds lack them.
DTYPE_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}
for _fp8_name in ("float8_e4m3fn", "float8_e5m2"):
    _dt = getattr(torch, _fp8_name, None)
    if _dt is not None:
        DTYPE_ALIASES[_fp8_name] = _dt
        DTYPE_ALIASES[_fp8_name.replace("float8", "fp8")] = _dt  # fp8_e4m3fn alias


def resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Map a UI/API dtype name to a torch dtype. None/"keep"/"" -> None (no-op)."""
    if not name or name.lower() in ("keep", "same", "none", "original"):
        return None
    key = name.lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(
            f"unknown dtype {name!r} (have: {sorted(set(DTYPE_ALIASES))})"
        )
    return DTYPE_ALIASES[key]


def _is_fp8(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)


# --- reporting ---------------------------------------------------------------

def state_dict_nbytes(state_dict: Dict[str, torch.Tensor]) -> int:
    """Total tensor payload in bytes — a close estimate of the .safetensors size
    (safetensors adds only a small JSON header)."""
    return sum(t.numel() * t.element_size() for t in state_dict.values())


def dtype_histogram(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """{dtype_name: tensor_count} so the UI can show the current precision."""
    hist: Dict[str, int] = {}
    for t in state_dict.values():
        name = str(t.dtype).replace("torch.", "")
        hist[name] = hist.get(name, 0) + 1
    return hist


def size_profile(state_dict: Dict[str, torch.Tensor]) -> dict:
    """Compact data letting a client project the saved size at any dtype/rank.

    Splits the payload into three buckets:

      * ``pairs_by_rank`` — {rank: total (down+up) element count} over the 2-D
        low-rank pairs SVD can shrink. Reducing a rank-r₀ pair to rank r scales
        its element count by r/r₀, so the client can size any target rank exactly.
      * ``fixed_float_numel`` — float elements that downcasting shrinks but SVD
        doesn't (non-pair weights, conv pairs, half-pairs).
      * ``fixed_bytes`` — bytes neither lever touches (``.alpha`` scalars, which
        we keep full-precision, plus any integer/bool tensors).

    ``elem_size`` is the current bytes-per-element of the float weights.
    """
    tbl = _module_table(state_dict)
    pair_keys = set()
    pairs_by_rank: Dict[int, int] = {}
    for entry in tbl.values():
        if not (entry.get("down") and entry.get("up")):
            continue
        dkey, down = entry["down"]
        ukey, up = entry["up"]
        if down.ndim == 2 and up.ndim == 2:
            r = int(down.shape[0])
            pair_keys.add(dkey)
            pair_keys.add(ukey)
            pairs_by_rank[r] = pairs_by_rank.get(r, 0) + down.numel() + up.numel()

    fixed_float_numel = 0
    fixed_bytes = 0
    elem_size = None
    for key, t in state_dict.items():
        if t.is_floating_point() and not key.lower().endswith(".alpha"):
            if elem_size is None:
                elem_size = t.element_size()
            if key in pair_keys:
                continue  # already accounted for in pairs_by_rank
            fixed_float_numel += t.numel()
        else:
            fixed_bytes += t.numel() * t.element_size()

    return {
        "elem_size": elem_size or 4,
        "fixed_float_numel": fixed_float_numel,
        "fixed_bytes": fixed_bytes,
        "pairs_by_rank": {str(r): n for r, n in sorted(pairs_by_rank.items())},
    }


def dominant_float_dtype(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """The float dtype carrying the most bytes (what downcasting actually shrinks)."""
    by_bytes: Dict[str, int] = {}
    for t in state_dict.values():
        if t.is_floating_point():
            name = str(t.dtype).replace("torch.", "")
            by_bytes[name] = by_bytes.get(name, 0) + t.numel() * t.element_size()
    if not by_bytes:
        return None
    return max(by_bytes, key=by_bytes.get)


# --- downcast ----------------------------------------------------------------

def downcast_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Cast every floating-point tensor to ``dtype``.

    ``.alpha`` scalars are left untouched (they're tiny and want full precision
    so the loader's alpha/rank factor stays exact). Integer/bool tensors pass
    through. fp8 has no direct cast path from some dtypes, so we always route
    through fp32 first.
    """
    out: Dict[str, torch.Tensor] = {}
    for key, t in state_dict.items():
        if not t.is_floating_point() or key.lower().endswith(".alpha"):
            out[key] = t
            continue
        if t.dtype == dtype:
            out[key] = t
            continue
        if _is_fp8(dtype):
            out[key] = t.to(torch.float32).to(dtype)
        else:
            out[key] = t.to(dtype)
    return out


# --- SVD rank reduction ------------------------------------------------------

def _target_rank(s: torch.Tensor, rank: Optional[int], energy: Optional[float]) -> int:
    """Pick how many singular values to keep.

    ``rank`` caps it directly; ``energy`` (0–1) keeps the smallest prefix whose
    squared singular values reach that fraction of the total. The result is
    clamped to [1, len(s)] and never *increases* the rank.
    """
    full = s.numel()
    keep = full
    if energy is not None and 0 < energy < 1:
        power = torch.cumsum(s ** 2, dim=0)
        total = power[-1]
        # smallest k with retained energy >= target
        keep = int(torch.searchsorted(power, energy * total).item()) + 1
    if rank is not None:
        keep = min(keep, int(rank))
    return max(1, min(keep, full))


def svd_reduce_state_dict(
    state_dict: Dict[str, torch.Tensor],
    rank: Optional[int] = None,
    energy: Optional[float] = None,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Re-express each 2-D low-rank pair at a smaller rank via truncated SVD.

    For a pair (down ∈ r×in, up ∈ out×r) with scale s = alpha/rank, the update is
    ΔW = up @ (down·s). We SVD ΔW = U·Σ·Vᵀ, keep the top-k components, and emit::

        down' = √Σ_k · Vᵀ_k          (k × in)
        up'   = U_k · √Σ_k           (out × k)
        alpha = k                    (so the loader's alpha/rank factor is 1.0)

    Pairs already at/under the target rank are left byte-for-byte unchanged.
    Non-2D pairs (conv) and non-pair tensors pass through and are flagged.
    """
    if rank is None and energy is None:
        return dict(state_dict), {"pairs_reduced": 0, "pairs_total": 0,
                                  "rank_before": 0, "rank_after": 0, "skipped": []}

    tbl = _module_table(state_dict)
    consumed = set()
    out: Dict[str, torch.Tensor] = {}
    reduced = 0
    pairs_total = 0
    rank_before_sum = 0
    rank_after_sum = 0
    skipped: List[str] = []

    for base, entry in tbl.items():
        if not (entry.get("down") and entry.get("up")):
            continue  # half a pair — leave for passthrough below
        pairs_total += 1
        dkey, down = entry["down"]
        ukey, up = entry["up"]
        akey = entry["alpha"][0] if entry.get("alpha") else base + ".alpha"
        consumed.update({dkey, ukey})
        if entry.get("alpha"):
            consumed.add(entry["alpha"][0])

        if down.ndim != 2 or up.ndim != 2:
            skipped.append(base)
            out[dkey] = down
            out[ukey] = up
            if entry.get("alpha"):
                out[akey] = entry["alpha"][1]
            continue

        r0 = down.shape[0]
        rank_before_sum += r0
        if rank is not None and r0 <= rank:
            # already small enough — keep exactly as-is
            rank_after_sum += r0
            out[dkey] = down
            out[ukey] = up
            if entry.get("alpha"):
                out[akey] = entry["alpha"][1]
            continue

        scale = _pair_scale(entry)
        delta = (up.to(torch.float32) @ (down.to(torch.float32) * scale))  # out × in
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        k = _target_rank(S, rank, energy)
        sqrt_s = torch.sqrt(S[:k])
        down_new = (sqrt_s.unsqueeze(1) * Vh[:k])      # k × in
        up_new = (U[:, :k] * sqrt_s.unsqueeze(0))      # out × k

        out[dkey] = down_new.to(down.dtype)
        out[ukey] = up_new.to(up.dtype)
        adtype = entry["alpha"][1].dtype if entry.get("alpha") else torch.float32
        out[akey] = torch.tensor(float(k), dtype=adtype)
        rank_after_sum += k
        if k < r0:
            reduced += 1

    # Everything not part of a reduced pair (bias/diff/embeds, half-pairs).
    for key, t in state_dict.items():
        if key not in consumed and key not in out:
            out[key] = t

    info = {
        "pairs_total": pairs_total,
        "pairs_reduced": reduced,
        "rank_before": rank_before_sum,
        "rank_after": rank_after_sum,
        "skipped": skipped,
    }
    return out, info


# --- orchestration -----------------------------------------------------------

def compress_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[str] = None,
    svd_rank: Optional[int] = None,
    svd_energy: Optional[float] = None,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Apply SVD reduction then dtype downcast. Any lever may be None (skipped).

    Returns (new_sd, info) where info carries before/after byte estimates and the
    per-lever details, suitable for showing in the UI.
    """
    size_before = state_dict_nbytes(state_dict)
    dtype_before = dominant_float_dtype(state_dict)
    sd = state_dict
    svd_info = None

    if svd_rank is not None or svd_energy is not None:
        sd, svd_info = svd_reduce_state_dict(sd, rank=svd_rank, energy=svd_energy)

    target = resolve_dtype(dtype)
    if target is not None:
        if _is_fp8(target) and not _torch_supports_fp8(target):
            raise ValueError(
                f"this torch build can't store {dtype} — upgrade torch or pick fp16/bf16"
            )
        sd = downcast_state_dict(sd, target)

    size_after = state_dict_nbytes(sd)
    info = {
        "size_before": size_before,
        "size_after": size_after,
        "dtype_before": dtype_before,
        "dtype_after": dominant_float_dtype(sd),
        "ratio": (size_after / size_before) if size_before else 1.0,
        "svd": svd_info,
    }
    return sd, info


def _torch_supports_fp8(dtype: torch.dtype) -> bool:
    """Round-trip a tiny tensor to confirm this build can actually store the dtype."""
    try:
        _ = torch.zeros(1, dtype=torch.float32).to(dtype).cpu().contiguous()
        return True
    except (RuntimeError, TypeError):
        return False
