"""Reusable NeXus wrangler preprocessing primitives."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from tomojax._typed_arrays import (
    equal_to_int,
    mean_float32,
    numpy_float32_array,
    numpy_float64_array,
    numpy_int64_array,
    scalar_float,
    scalar_int,
    shape2,
    shape3,
)
from tomojax.io._preprocess import flat_dark_to_absorption


def constant_dark_field(value: float, shape: tuple[int, int]) -> NDArray[np.float32]:
    if not np.isfinite(value):
        raise ValueError("dark field override must be finite")
    return np.full(shape, float(value), dtype=np.float32)


def flat_dark_correct_frames_to_absorption(
    data: NDArray[np.float32],
    image_key: NDArray[np.integer[Any]],
    min_intensity: float = 1e-6,
    assume_dark_field: float | None = None,
) -> NDArray[np.float32]:
    """Convert raw frames with NXtomo image_key labels into absorption projections."""
    data_f32 = numpy_float32_array(data)
    image_key_i64 = numpy_int64_array(image_key)
    is_proj = equal_to_int(image_key_i64, 0)
    is_flat = equal_to_int(image_key_i64, 1)
    is_dark = equal_to_int(image_key_i64, 2)

    proj = data_f32[is_proj]
    flats = data_f32[is_flat]
    darks = data_f32[is_dark]

    if flats.size == 0:
        raise RuntimeError("No flat fields found (image_key==1). Cannot normalise.")
    if darks.size == 0:
        if assume_dark_field is None:
            raise ValueError(
                "No dark fields found (image_key==2); pass --assume-dark-field VALUE "
                "to use an explicit constant dark field"
            )
        darks = constant_dark_field(assume_dark_field, shape2(data_f32))[None, ...]

    absorption = numpy_float32_array(
        cast(
            "object",
            flat_dark_to_absorption(
                proj.astype(np.float32, copy=False),
                flats.astype(np.float32, copy=False),
                darks.astype(np.float32, copy=False),
                min_intensity=float(min_intensity),
            ),
        )
    )
    absorption[~np.isfinite(absorption)] = 0.0
    return absorption


def spatial_bin(
    arr: NDArray[np.float32],
    bin_y: int = 1,
    bin_x: int = 1,
) -> NDArray[np.float32]:
    """Bin the last two spatial dimensions by mean, cropping remainders."""
    arr = numpy_float32_array(arr)
    bin_y = int(max(1, bin_y))
    bin_x = int(max(1, bin_x))
    if bin_y == 1 and bin_x == 1:
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 3:
        n, ny, nx = shape3(arr)
        ny_c = (ny // bin_y) * bin_y
        nx_c = (nx // bin_x) * bin_x
        if ny_c != ny or nx_c != nx:
            arr = arr[:, :ny_c, :nx_c]
        arr = arr.reshape(n, ny_c // bin_y, bin_y, nx_c // bin_x, bin_x)
        return mean_float32(arr, axis=(2, 4))
    if arr.ndim == 2:
        ny, nx = shape2(arr)
        ny_c = (ny // bin_y) * bin_y
        nx_c = (nx // bin_x) * bin_x
        if ny_c != ny or nx_c != nx:
            arr = arr[:ny_c, :nx_c]
        arr = arr.reshape(ny_c // bin_y, bin_y, nx_c // bin_x, bin_x)
        return mean_float32(arr, axis=(1, 3))
    raise ValueError(f"Unsupported array ndim for binning: {arr.ndim}")


def pad_to_multiples(
    arr: NDArray[np.float32],
    mult_y: int | None,
    mult_x: int | None,
    mode: str = "edge",
) -> NDArray[np.float32]:
    """Symmetrically pad last two spatial dimensions to requested multiples."""
    arr = numpy_float32_array(arr)
    if mult_y is None and mult_x is None:
        return arr
    if arr.ndim not in (2, 3):
        raise ValueError("Padding expects 2D or 3D arrays")

    ny, nx = shape2(arr)
    target_ny = ny
    target_nx = nx
    if mult_y and int(mult_y) > 1:
        m = int(mult_y)
        r = target_ny % m
        if r != 0:
            target_ny += m - r
    if mult_x and int(mult_x) > 1:
        m = int(mult_x)
        r = target_nx % m
        if r != 0:
            target_nx += m - r
    pad_y = max(0, target_ny - ny)
    pad_x = max(0, target_nx - nx)
    if pad_y == 0 and pad_x == 0:
        return arr
    py0 = pad_y // 2
    py1 = pad_y - py0
    px0 = pad_x // 2
    px1 = pad_x - px0
    pad_width = [(0, 0), (py0, py1), (px0, px1)] if arr.ndim == 3 else [(py0, py1), (px0, px1)]
    return numpy_float32_array(np.pad(arr, pad_width, mode=mode))  # type: ignore[reportCallIssue, reportArgumentType]


def summarize_angles(angles_deg: NDArray[np.float64]) -> dict[str, float | int | bool]:
    angles_deg = numpy_float64_array(angles_deg)
    out: dict[str, float | int | bool] = {
        "start_deg": (
            scalar_float(cast("object", angles_deg[0]))
            if scalar_int(angles_deg.size) > 0
            else 0.0
        ),
        "count": scalar_int(angles_deg.size),
        "endpoint": False,
    }
    if scalar_int(angles_deg.size) >= 2:
        out["step_deg"] = float(np.median(np.diff(angles_deg)))
    else:
        out["step_deg"] = 0.0
    return out


def volume_chunks(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Clamp placeholder-volume chunks so small grids remain writable."""
    target = (16, 16, 32)
    return (
        min(int(shape[0]), target[0]),
        min(int(shape[1]), target[1]),
        min(int(shape[2]), target[2]),
    )


__all__ = [
    "constant_dark_field",
    "flat_dark_correct_frames_to_absorption",
    "pad_to_multiples",
    "spatial_bin",
    "summarize_angles",
    "volume_chunks",
]
