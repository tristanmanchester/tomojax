"""Resolve preview planes on the original sampling grid."""

from math import ceil


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


def plane_windows(axes, shape, mode, requested_axis, index, roi, strides):
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
