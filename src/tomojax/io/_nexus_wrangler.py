"""Reusable NeXus wrangler preprocessing primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def constant_dark_field(value: float, shape: tuple[int, int]) -> NDArray[np.float32]:
    if not np.isfinite(value):
        raise ValueError("dark field override must be finite")
    return np.full(shape, float(value), dtype=np.float32)


__all__ = [
    "constant_dark_field",
]
