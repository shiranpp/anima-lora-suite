"""
Per-block impact analyzer.

Computes a relative "impact score" for each AnimaBlock by measuring the
Frobenius norm of the low-rank update (up @ down) divided across the
tensors in that block. This is the same idea used by the ComfyUI Realtime
LoRA analyzer — it doesn't require running inference, just inspecting the
weights.

The score is normalized to 0..100 across blocks so the UI can color-code
checkboxes by relative importance.
"""

from collections import defaultdict
from typing import Dict, List

import torch

from .detect import extract_block_info
from .presets import ANIMA_NUM_BLOCKS


def _tensor_score(t: torch.Tensor) -> float:
    """Frobenius norm of a tensor as a float."""
    # Cast to fp32 for stable norm on bf16/fp16 LoRAs
    return float(torch.linalg.norm(t.detach().to(torch.float32).flatten()).item())


def analyze_lora(state_dict: Dict[str, torch.Tensor]) -> dict:
    """
    Compute per-block, LLMAdapter, and other-weights impact scores.

    Returns a dict shaped for the UI:
        {
            "block_scores":   {0: 12.3, 1: 4.1, ...},   # raw Frobenius totals
            "block_norm":     {0: 100, 1: 33, ...},     # normalized 0..100
            "llm_adapter_score": 8.7,
            "other_score": 1.2,
            "max_score": 12.3,
        }
    """
    block_totals: Dict[int, float] = defaultdict(float)
    llm_total = 0.0
    other_total = 0.0

    for key, tensor in state_dict.items():
        # Use only "up"/"B" tensors as a proxy for contribution magnitude.
        # This matches what the ComfyUI analyzer does — focusing on the
        # output side of the low-rank product gives a cleaner signal than
        # mixing up and down halves.
        kl = key.lower()
        if not (kl.endswith("lora_up.weight") or kl.endswith("lora_b.weight")):
            continue

        score = _tensor_score(tensor)
        tag, n = extract_block_info(key)
        if tag == "block":
            block_totals[n] += score
        elif tag == "llm_adapter":
            llm_total += score
        else:
            other_total += score

    # Normalize block scores to 0..100
    if block_totals:
        max_score = max(block_totals.values())
    else:
        max_score = 0.0

    block_norm: Dict[int, int] = {}
    for i in range(ANIMA_NUM_BLOCKS):
        raw = block_totals.get(i, 0.0)
        if max_score > 0:
            block_norm[i] = int(round(100 * raw / max_score))
        else:
            block_norm[i] = 0

    return {
        "block_scores": {i: round(block_totals.get(i, 0.0), 4) for i in range(ANIMA_NUM_BLOCKS)},
        "block_norm": block_norm,
        "llm_adapter_score": round(llm_total, 4),
        "other_score": round(other_total, 4),
        "max_score": round(max_score, 4),
    }


def impact_color(norm_score: int) -> str:
    """Map a 0..100 impact score to a CSS color band, matching the UI legend."""
    if norm_score >= 70:
        return "high"      # sakura pink (most important)
    if norm_score >= 40:
        return "medium"    # gold
    if norm_score >= 10:
        return "low"       # muted purple
    return "negligible"    # dim
