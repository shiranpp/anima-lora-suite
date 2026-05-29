"""
Minimal PNG encoder built on the standard library (``zlib`` + ``struct``).

The base editor deliberately avoids heavy runtime deps, so rather than pull in
Pillow just to write the preview image we encode the PNG by hand. Input is a
uint8 array shaped ``(H, W, 3)`` (RGB) or ``(H, W, 4)`` (RGBA).
"""

import struct
import zlib

import numpy as np


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, C)`` uint8 array (C in {3, 4}) as PNG bytes."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3|4) uint8, got {rgb.shape} {rgb.dtype}")

    h, w, c = rgb.shape
    color_type = 2 if c == 3 else 6  # 2 = truecolour, 6 = truecolour+alpha

    # Each scanline is prefixed with a filter-type byte (0 = none). Build the
    # whole raw block vectorised: prepend a zero column, then flatten.
    rows = rgb.reshape(h, w * c)
    filtered = np.concatenate([np.zeros((h, 1), dtype=np.uint8), rows], axis=1)
    raw = filtered.tobytes()
    compressed = zlib.compress(raw, level=6)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (
        signature
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
