"""Anima LoRA Editor — core package."""

from .editor import edit_lora, load_lora_state_dict, save_lora_state_dict, merge_loras
from .detect import detect_architecture, extract_block_info
from .presets import ANIMA_PRESETS, ANIMA_NUM_BLOCKS
from .analyzer import analyze_lora
from .compress import (
    compress_state_dict,
    state_dict_nbytes,
    dtype_histogram,
    dominant_float_dtype,
    resolve_dtype,
    size_profile,
)

__all__ = [
    "edit_lora",
    "load_lora_state_dict",
    "save_lora_state_dict",
    "merge_loras",
    "detect_architecture",
    "extract_block_info",
    "analyze_lora",
    "compress_state_dict",
    "state_dict_nbytes",
    "dtype_histogram",
    "dominant_float_dtype",
    "resolve_dtype",
    "ANIMA_PRESETS",
    "ANIMA_NUM_BLOCKS",
]
