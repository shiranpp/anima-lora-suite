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
    "All Blocks":            set(range(28)),
    "All Off":               set(),
    "Late Only (19-27)":     set(range(19, 28)),
    "Mid-Late (14-27)":      set(range(14, 28)),
    "Skip Early (10-27)":    set(range(10, 28)),
    "Mid Only (10-18)":      set(range(10, 19)),
    "Early Only (0-9)":      set(range(10)),
    "Peak Impact (16-24)":   set(range(16, 25)),
    "Face Priority (14-22)": set(range(14, 23)),
    "Style Focus (22-27)":   set(range(22, 28)),
    "Evens Only":            set(range(0, 28, 2)),
    "Odds Only":             set(range(1, 28, 2)),
    "Custom":                None,  # use individual toggles
}


def preset_block_set(name: str) -> set:
    """Return the set of enabled block indices for a preset name."""
    if name not in ANIMA_PRESETS:
        raise KeyError(f"Unknown preset: {name!r}")
    val = ANIMA_PRESETS[name]
    return set() if val is None else set(val)
