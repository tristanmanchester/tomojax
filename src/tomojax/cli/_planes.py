"""Resolve preview planes on the original sampling grid."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from tomojax.errors import InvalidParameter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tomojax.cli.operations import Window

READ_BYTES = 16 << 20


def display_sampling(
    roi: list[tuple[int, int]], strides: tuple[int, ...]
) -> tuple[list[int], tuple[str, ...]]:
    """Bound display reads to 512 samples per axis on the requested sampling grid."""
    factors = tuple(
        ceil(len(range(lo, hi, step)) / 512) for (lo, hi), step in zip(roi, strides, strict=True)
    )
    steps = [step * factor for step, factor in zip(strides, factors, strict=True)]
    label = f"stride {strides}, display subsampling {factors}"
    return steps, (label,) if max(*strides, *factors) > 1 else ()


def plane_windows(
    axes: Sequence[str],
    shape: tuple[int, ...],
    mode: str,
    requested_axis: str | None,
    index: int | None,
    roi: Sequence[tuple[int, int]],
    strides: tuple[int, ...],
) -> Iterator[tuple[int | None, int, Sequence[tuple[int, int]], Sequence[int], tuple[str, ...]]]:
    if len(axes) == 2 and mode == "planes":
        yield None, 0, roi, strides, ()
        return
    if len(axes) != 3:
        raise ValueError(f"{mode} needs a three-dimensional array, got {axes}")
    if requested_axis is not None:
        axis_names = [requested_axis]
    elif mode == "sinogram":
        axis_names = ["v" if "v" in axes else axes[1]]
    elif mode == "projection" or "view" in axes:
        axis_names = ["view" if "view" in axes else axes[0]]
    else:
        axis_names = axes if index is None else [axes[0]]

    for axis in axis_names:
        if axis not in axes:
            raise ValueError(f"axis must be one of {axes}")
        d = axes.index(axis)
        selected = list(roi)
        lo, hi = selected[d]
        if not 0 <= lo < hi <= shape[d]:
            raise ValueError(f"roi outside {shape}")
        indices = range(lo, hi, strides[d])
        at = indices[len(indices) // 2] if index is None else index
        if not lo <= at < hi:
            raise ValueError(f"index {at} outside {axis} interval [{lo}, {hi})")
        selected[d] = (at, at + 1)
        stride, sampling = display_sampling(selected, strides)
        yield d, at, selected, stride, sampling


def window(shape: tuple[int, ...], request: Window) -> tuple[range, ...]:
    roi = request.roi if request.roi is not None else tuple((0, n) for n in shape)
    stride = request.stride if request.stride is not None else (1,) * len(shape)
    if len(stride) != len(shape) or any(s < 1 for s in stride):
        raise InvalidParameter("stride", "supply one positive stride per axis")
    if len(roi) != len(shape) or any(
        not 0 <= a < b <= n for (a, b), n in zip(roi, shape, strict=False)
    ):
        raise InvalidParameter("roi", f"expected nonempty half-open intervals inside {shape}")
    return tuple(range(a, b, s) for (a, b), s in zip(roi, stride, strict=True))
