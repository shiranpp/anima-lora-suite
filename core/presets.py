"""
Block presets for the Anima architecture (28 AnimaBlock layers, 0-27).

These follow the same naming/spirit as the ComfyUI Realtime LoRA presets
(All / Late / Mid / Early / Evens / Odds, etc) but with Anima-appropriate ranges.

Empirical heuristics for DiT-style models (Cosmos / Anima / Z-Image / Qwen):
- Early blocks (0-9):   coarse structure, broad composition
- Mid blocks  (10-18):  identity, anatomy, content
- Late blocks (19-27):  style, color, fine detail
"""

ANIMA_NUM_BLOCKS = 28

ANIMA_PRESETS = {
    "All Blocks":            {i: 1 for i in range(28)},
    "All Off":               {},
    "Character": dict(enumerate([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.2,0.2,0.2,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2,0.2])),
    "Style":   dict(enumerate([0.9,0.9,0.9,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.6,0.6,0.6,0.5,0.5])),
}


def preset_block_set(name: str) -> set:
    """Return the set of enabled block indices for a preset name."""
    if name not in ANIMA_PRESETS:
        raise KeyError(f"Unknown preset: {name!r}")
    val = ANIMA_PRESETS[name]
    return set() if val is None else set(val)
