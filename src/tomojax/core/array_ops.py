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
    if isinstance(data, np.ndarray) and 1 < factor < 8 and data.strides[-1] == data.itemsize:
        # NumPy reduces contiguous columns first, then rows; inner reductions
        # shorter than eight are sequential. Vectorize across output pixels,
        # preserving that FP32 order even for view-strided ASTRA imports.
        # Larger or column-strided blocks can use a different reduction order.
        pixels = data[:, : rows * factor, : cols * factor]
        if factor == 2:
            result = pixels[:, ::2, ::2] + pixels[:, ::2, 1::2]
            result += pixels[:, 1::2, ::2] + pixels[:, 1::2, 1::2]
        else:

            def sum_columns(row: int) -> np.ndarray:
                total = pixels[:, row::factor, ::factor].copy()
                for col in range(1, factor):
                    total += pixels[:, row::factor, col::factor]
                return total

            result = sum_columns(0)
            for row in range(1, factor):
                result += sum_columns(row)
        result /= factor * factor
        return cast("T", result)
    blocks = data[:, : rows * factor, : cols * factor].reshape(views, rows, factor, cols, factor)
    return cast("T", blocks.mean(axis=(2, 4)))
