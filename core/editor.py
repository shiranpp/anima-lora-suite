"""
Core LoRA editing logic.

Takes a state_dict and an EditConfig (which blocks are enabled, per-block
strength, LLMAdapter toggle, other_weights toggle, global strength) and
returns a new filtered/scaled state_dict ready to save.

This is the standalone equivalent of ComfyUI's `comfy.sd.load_lora_for_models`
filtering step, but writing to a file instead of applying to a live model.
"""

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file

from .detect import extract_block_info, summarize_keys, detect_architecture
from .presets import ANIMA_NUM_BLOCKS


# --- IO helpers --------------------------------------------------------------

def load_lora_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a .safetensors or .pt LoRA file into a flat state_dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA file not found: {path}")
    if path.lower().endswith(".safetensors"):
        return load_file(path)
    return torch.load(path, map_location="cpu")


def save_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    out_path: str,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save a state_dict to .safetensors, preserving (and extending) metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    # safetensors requires all values be CPU tensors
    cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(cpu_sd, out_path, metadata=metadata or {})


# --- Edit configuration ------------------------------------------------------

@dataclass
class EditConfig:
    """How to transform the LoRA. All fields have safe defaults."""
    enabled_blocks: set = field(default_factory=lambda: set(range(ANIMA_NUM_BLOCKS)))
    # Per-block strength multiplier (1.0 = unchanged). Missing = 1.0.
    block_strengths: Dict[int, float] = field(default_factory=dict)
    # Apply LLMAdapter weights at all?
    llm_adapter_enabled: bool = True
    llm_adapter_strength: float = 1.0
    # Apply non-block, non-adapter weights (embeddings, time, finals, ...)?
    other_enabled: bool = True
    other_strength: float = 1.0
    # Global multiplier baked into the saved tensors (1.0 = leave for runtime)
    global_strength: float = 1.0


# --- Main editor -------------------------------------------------------------

def edit_lora(
    state_dict: Dict[str, torch.Tensor],
    config: EditConfig,
) -> tuple[Dict[str, torch.Tensor], dict]:
    """
    Apply an EditConfig to a LoRA state_dict.

    Returns (new_state_dict, info) where info is a small dict suitable for
    showing in the UI: counts, which blocks were kept/dropped, etc.

    Important detail about *scaling*: LoRAs decompose as W = up @ down * (alpha/rank).
    Multiplying *either* up or down by k scales the contribution by k. To avoid
    scaling each contribution by k^2 we apply the scale to lora_up / lora_B only,
    and leave lora_down / lora_A and .alpha untouched. Bias-style 1D tensors get
    scaled directly.
    """
    new_sd: Dict[str, torch.Tensor] = {}
    kept_blocks = set()
    dropped_blocks = set()
    kept_llm = False
    kept_other = 0

    g = config.global_strength

    for key, tensor in state_dict.items():
        tag, n = extract_block_info(key)

        # Decide whether this key is enabled and at what local strength
        if tag == "block":
            if n not in config.enabled_blocks:
                dropped_blocks.add(n)
                continue
            local = config.block_strengths.get(n, 1.0)
            kept_blocks.add(n)
        elif tag == "llm_adapter":
            if not config.llm_adapter_enabled:
                continue
            local = config.llm_adapter_strength
            kept_llm = True
        else:  # other
            if not config.other_enabled:
                continue
            local = config.other_strength
            kept_other += 1

        # Effective scale = global * local. Only applied to "up"/"B"/scalar
        # tensors to avoid double-counting on the low-rank product.
        effective = g * local
        if effective != 1.0:
            tensor = _scaled(key, tensor, effective)

        new_sd[key] = tensor

    # Make sure dropped_blocks doesn't double-count blocks that were never in the LoRA
    summary = summarize_keys(list(state_dict.keys()))
    present = set(summary["blocks_present"])
    dropped_blocks = dropped_blocks & present

    info = {
        "original_tensor_count": len(state_dict),
        "output_tensor_count": len(new_sd),
        "blocks_present_in_lora": sorted(present),
        "blocks_kept": sorted(kept_blocks),
        "blocks_dropped": sorted(dropped_blocks),
        "llm_adapter_kept": kept_llm,
        "llm_adapter_in_lora": summary["has_llm_adapter"],
        "other_tensors_kept": kept_other,
        "other_tensors_in_lora": summary["other_count"],
        "global_strength": g,
    }
    return new_sd, info


def _scaled(key: str, tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale only the half of the LoRA pair that won't double-count.

    Convention: scale lora_up / lora_B / scalar tensors. Leave lora_down / lora_A
    and explicit .alpha untouched.
    """
    kl = key.lower()
    is_down_side = kl.endswith("lora_down.weight") or kl.endswith("lora_a.weight")
    is_alpha = kl.endswith(".alpha")
    if is_down_side or is_alpha:
        return tensor
    return tensor * scale


# --- Merging multiple LoRAs into one -----------------------------------------
# The editor lets you stack several LoRAs as "layers"; saving writes a *single*
# file containing their combined effect. Two LoRAs touching the same module add
# their low-rank updates:  ΔW = up₁·down₁·s₁ + up₂·down₂·s₂  (sᵢ = alphaᵢ/rankᵢ).
# That sum is itself a low-rank update of combined rank r₁+r₂, expressed by
# concatenating the ups side-by-side and the downs stacked:
#       up = [up₁ | up₂]   (cat on the rank dim, dim=1)
#       down = [down₁; down₂]   (cat on the rank dim, dim=0)
# To keep a single uniform alpha we fold each layer's scale sᵢ into its down rows
# and emit alpha = combined_rank, so the loader's alpha/rank factor is 1.0.

# (down-suffix, up-suffix, names a LoRA pair). Both kohya and diffusers layouts.
_LORA_DOWN_SUFFIXES = (".lora_down.weight", ".lora_a.weight")
_LORA_UP_SUFFIXES = (".lora_up.weight", ".lora_b.weight")


def _split_lora_key(key: str):
    """Map a key to (module_base, role) where role ∈ {down, up, alpha} or None."""
    kl = key.lower()
    for suf in _LORA_DOWN_SUFFIXES:
        if kl.endswith(suf):
            return key[: -len(suf)], "down"
    for suf in _LORA_UP_SUFFIXES:
        if kl.endswith(suf):
            return key[: -len(suf)], "up"
    if kl.endswith(".alpha"):
        return key[: -len(".alpha")], "alpha"
    return key, None


def _module_table(state_dict: Dict[str, torch.Tensor]) -> "OrderedDict":
    """base -> {role: (original_key, tensor)} for every LoRA pair in the dict."""
    tbl: "OrderedDict" = OrderedDict()
    for key, t in state_dict.items():
        base, role = _split_lora_key(key)
        if role is None:
            continue
        tbl.setdefault(base, {})[role] = (key, t)
    return tbl


def _pair_scale(entry) -> float:
    """alpha/rank for a LoRA pair (rank = down.shape[0]); defaults to 1.0."""
    down = entry["down"][1]
    rank = down.shape[0] if down.ndim >= 1 else 1
    if entry.get("alpha") is not None:
        alpha = float(entry["alpha"][1].reshape(-1)[0].item())
    else:
        alpha = float(rank)
    return (alpha / rank) if rank else 1.0


def merge_loras(state_dicts: List[Dict[str, torch.Tensor]]) -> tuple[Dict[str, torch.Tensor], dict]:
    """Combine several (already-edited) LoRA state_dicts into one.

    Modules touched by a single layer pass through untouched (exact). Modules
    touched by two or more are rank-concatenated so the merged file reproduces
    the *sum* of their updates. Non-pair tensors (rare bias/diff weights) are
    summed when shapes match, else last-wins. Returns (merged_sd, info).
    """
    sources = [sd for sd in state_dicts if sd]
    if not sources:
        return {}, {"sources": 0, "modules_total": 0, "modules_concatenated": 0,
                    "collisions": []}
    if len(sources) == 1:
        sd = dict(sources[0])
        return sd, {"sources": 1, "modules_total": len(_module_table(sd)),
                    "modules_concatenated": 0, "collisions": []}

    tables = [_module_table(sd) for sd in sources]

    # Stable, first-seen order of every module base across all layers.
    all_bases, seen = [], set()
    for tbl in tables:
        for base in tbl:
            if base not in seen:
                seen.add(base)
                all_bases.append(base)

    out: Dict[str, torch.Tensor] = {}
    concatenated = 0
    collisions: List[str] = []

    for base in all_bases:
        present = [tbl[base] for tbl in tables if base in tbl]
        full = [e for e in present if e.get("down") and e.get("up")]

        # 0 or 1 contributing pair, or a partial/odd module: pass entries through.
        if len(full) <= 1:
            src = full[0] if full else present[0]
            for role in ("down", "up", "alpha"):
                if src.get(role):
                    k, t = src[role]
                    out[k] = t
            continue

        # Only true 2-D (linear) low-rank pairs can be rank-concatenated. Anything
        # else (e.g. conv) we can't combine safely — keep the first and flag it.
        linear = all(e["down"][1].ndim == 2 and e["up"][1].ndim == 2 for e in full)
        if not linear:
            src = full[0]
            for role in ("down", "up", "alpha"):
                if src.get(role):
                    k, t = src[role]
                    out[k] = t
            collisions.append(base)
            continue

        downs, ups = [], []
        for e in full:
            scale = _pair_scale(e)
            downs.append(e["down"][1].to(torch.float32) * scale)
            ups.append(e["up"][1].to(torch.float32))
        down_cat = torch.cat(downs, dim=0)        # stack rows  -> (Σrᵢ, in)
        up_cat = torch.cat(ups, dim=1)            # widen cols   -> (out, Σrᵢ)
        comb_rank = down_cat.shape[0]

        ent0 = full[0]
        dkey, d0 = ent0["down"]
        ukey, u0 = ent0["up"]
        out[dkey] = down_cat.to(d0.dtype)
        out[ukey] = up_cat.to(u0.dtype)
        # alpha == rank so the loader's alpha/rank factor is 1.0 (scale already folded in).
        akey = ent0["alpha"][0] if ent0.get("alpha") else base + ".alpha"
        adtype = ent0["alpha"][1].dtype if ent0.get("alpha") else torch.float32
        out[akey] = torch.tensor(float(comb_rank), dtype=adtype)
        concatenated += 1

    # Non-pair leftovers (no down/up/alpha role): bias/diff-style tensors.
    leftovers: "OrderedDict" = OrderedDict()
    for sd in sources:
        for key, t in sd.items():
            if _split_lora_key(key)[1] is None:
                leftovers.setdefault(key, []).append(t)
    for key, tensors in leftovers.items():
        if len(tensors) == 1:
            out[key] = tensors[0]
            continue
        try:
            acc = tensors[0].to(torch.float32).clone()
            for t in tensors[1:]:
                acc = acc + t.to(torch.float32)
            out[key] = acc.to(tensors[0].dtype)
        except (RuntimeError, TypeError):
            out[key] = tensors[-1]
            collisions.append(key)

    info = {
        "sources": len(sources),
        "modules_total": len(all_bases),
        "modules_concatenated": concatenated,
        "collisions": collisions,
    }
    return out, info


# --- Convenience one-shot ----------------------------------------------------

def edit_lora_file(
    in_path: str,
    out_path: str,
    config: EditConfig,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> dict:
    """Load -> edit -> save in one call. Returns the info dict."""
    sd = load_lora_state_dict(in_path)
    arch = detect_architecture(list(sd.keys()))
    new_sd, info = edit_lora(sd, config)
    info["detected_architecture"] = arch
    info["input_path"] = in_path
    info["output_path"] = out_path

    metadata = {
        "anima_lora_editor": "1",
        "anima_lora_editor.global_strength": str(config.global_strength),
        "anima_lora_editor.enabled_blocks": ",".join(str(b) for b in sorted(config.enabled_blocks)),
        "anima_lora_editor.llm_adapter_enabled": str(config.llm_adapter_enabled).lower(),
        "anima_lora_editor.other_enabled": str(config.other_enabled).lower(),
        "anima_lora_editor.source_architecture": arch,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    save_lora_state_dict(new_sd, out_path, metadata=metadata)
    return info
