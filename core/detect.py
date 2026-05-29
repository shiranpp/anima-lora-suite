"""
Architecture detection and block-id extraction for Anima LoRAs.

Anima is a 2B DiT model derived from NVIDIA Cosmos-Predict2-2B-Text2Image,
with 28 AnimaBlock transformer layers (indices 0-27) plus an LLMAdapter
that bridges the Qwen3-0.6B text encoder to the DiT.

LoRAs trained by the sd-scripts-based Anima trainers (kohya-style) typically
produce keys like:
    lora_unet_blocks_<N>_<submodule>_<...>.lora_down.weight
    lora_unet_blocks_<N>_<submodule>_<...>.lora_up.weight
    lora_unet_blocks_<N>_<submodule>_<...>.alpha
    lora_unet_llm_adapter_<...>.lora_down.weight

Diffusers-style training (e.g. duongve's diffusion-pipe fork) may produce:
    transformer.blocks.<N>.<...>.lora_A.weight
    transformer.blocks.<N>.<...>.lora_B.weight

This module tries to handle both, and exposes a single tagger that returns
('block', N) for transformer block N, ('llm_adapter', None) for the bridge,
or ('other', None) for anything else.
"""

import re
from typing import Tuple, Optional, List


# --- Anima-specific key patterns --------------------------------------------

# kohya/sd-scripts format: lora_unet_blocks_<N>_...
_ANIMA_KOHYA_BLOCK_RE = re.compile(r"lora_unet_blocks_(\d+)[_\.]", re.IGNORECASE)

# diffusers/PEFT format: transformer.blocks.<N>. or just blocks.<N>.
_ANIMA_DIFFUSERS_BLOCK_RE = re.compile(r"(?:transformer\.)?blocks\.(\d+)\.", re.IGNORECASE)

# Some trainers (Musubi-style) use lora_unet_transformer_blocks_<N>_
_ANIMA_TRANSFORMER_BLOCK_RE = re.compile(r"lora_unet_transformer_blocks_(\d+)[_\.]", re.IGNORECASE)
_ANIMA_DIFFUSERS_TBLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.", re.IGNORECASE)

# LLMAdapter — the Qwen3 -> DiT bridge unique to Anima
_ANIMA_LLM_ADAPTER_RE = re.compile(r"(llm_adapter|llmadapter|llm\.adapter)", re.IGNORECASE)


def extract_block_info(key: str) -> Tuple[str, Optional[int]]:
    """
    Classify a single LoRA key.

    Returns:
        ('block', N)        — belongs to AnimaBlock N (0..27)
        ('llm_adapter', None) — belongs to the LLMAdapter
        ('other', None)     — embeddings, final layers, time embeds, etc.
    """
    for regex in (
        _ANIMA_KOHYA_BLOCK_RE,
        _ANIMA_TRANSFORMER_BLOCK_RE,
        _ANIMA_DIFFUSERS_TBLOCK_RE,
        _ANIMA_DIFFUSERS_BLOCK_RE,
    ):
        m = regex.search(key)
        if m:
            return ("block", int(m.group(1)))

    if _ANIMA_LLM_ADAPTER_RE.search(key):
        return ("llm_adapter", None)

    return ("other", None)


def detect_architecture(keys: List[str]) -> str:
    """
    Best-effort guess at the source architecture of a LoRA, used to warn the
    user if they load something that is clearly not an Anima LoRA.

    Returns one of: 'ANIMA', 'FLUX', 'SDXL', 'SD15', 'QWEN_IMAGE', 'ZIMAGE',
                    'WAN', 'UNKNOWN'.
    """
    keys_lower = [k.lower() for k in keys]
    joined = " ".join(keys_lower)

    # Architectures with very distinctive markers — check first so we don't
    # mis-classify them as Anima just because they happen to contain 'blocks'.
    if "img_mlp" in joined or "txt_mlp" in joined or "img_mod" in joined:
        return "QWEN_IMAGE"
    if "double_blocks" in joined or "single_blocks" in joined:
        return "FLUX"
    if "single_transformer_blocks" in joined:
        return "FLUX"
    if "lora_te1_" in joined or "lora_te2_" in joined:
        return "SDXL"

    # Cosmos/Anima-style: bare 'blocks.N' or 'lora_unet_blocks_N' WITHOUT the
    # transformer_blocks variant, and presence of LLMAdapter is a strong signal.
    has_llm_adapter = bool(_ANIMA_LLM_ADAPTER_RE.search(joined))
    has_anima_blocks = any(
        _ANIMA_KOHYA_BLOCK_RE.search(k) or _ANIMA_DIFFUSERS_BLOCK_RE.search(k)
        for k in keys_lower
    )

    if has_llm_adapter:
        return "ANIMA"
    if has_anima_blocks:
        # Could be Anima OR generic "blocks.N" — count blocks and see if it
        # fits the 28-block Anima/Cosmos profile.
        block_ids = set()
        for k in keys_lower:
            tag, n = extract_block_info(k)
            if tag == "block":
                block_ids.add(n)
        if block_ids and max(block_ids) <= 27 and min(block_ids) >= 0:
            return "ANIMA"

    # Older Wan/Z-Image style fallbacks (just for the warning, not used otherwise)
    if any("self_attn" in k or "cross_attn" in k for k in keys_lower) and "blocks" in joined:
        return "WAN"

    return "UNKNOWN"


def summarize_keys(keys: List[str]) -> dict:
    """
    Produce a quick summary of which Anima components a LoRA touches.
    Used by the UI to show "this LoRA has X blocks, LLMAdapter weights, etc."
    """
    blocks_present = set()
    has_llm_adapter = False
    other_count = 0

    for k in keys:
        tag, n = extract_block_info(k)
        if tag == "block":
            blocks_present.add(n)
        elif tag == "llm_adapter":
            has_llm_adapter = True
        else:
            other_count += 1

    return {
        "total_keys": len(keys),
        "blocks_present": sorted(blocks_present),
        "num_blocks": len(blocks_present),
        "has_llm_adapter": has_llm_adapter,
        "other_count": other_count,
    }
