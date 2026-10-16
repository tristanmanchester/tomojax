"""Array-only detector operations, shared by scans and correction execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import jax


def mean_detector_pixels[T: (np.ndarray, jax.Array)](data: T, factor: int) -> T:
    """Average full detector blocks in FP32, dropping incomplete edge blocks.

    ``data`` is ``(views, rows, columns)`` and ``factor`` is positive. The
    array's backend is retained; physical detector geometry belongs to the
    caller. Cast before averaging, as when storing corrected FP32 pixels.
    """
    data = data.astype(np.float32, copy=False)
    views, nv, nu = data.shape
    rows, cols = nv // factor, nu // factor
    if isinstance(data, np.ndarray) and factor == 2 and data.flags.c_contiguous:
        # Preserve the column-pair then row-pair FP32 reduction. NumPy's
        # generic two-axis reduction cannot vectorize these short blocks well.
        pixels = data[:, : rows * 2, : cols * 2]
        result = pixels[:, ::2, ::2] + pixels[:, ::2, 1::2]
        result += pixels[:, 1::2, ::2] + pixels[:, 1::2, 1::2]
        result *= 0.25
        return cast("T", result)
    blocks = data[:, : rows * factor, : cols * factor].reshape(views, rows, factor, cols, factor)
    return cast("T", blocks.mean(axis=(2, 4)))
