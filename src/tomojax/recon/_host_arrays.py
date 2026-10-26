"""Validation shared by reconstructions that keep inputs and outputs in host memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
import mmap
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax


@dataclass(frozen=True)
class ProjectionRows:
    """A zero-extended detector-row window, read only for requested view batches."""

    projections: np.ndarray
    start: int
    rows: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """The window's projection shape, without reading the input storage."""
        return self.projections.shape[0], self.rows, self.projections.shape[2]

    def __getitem__(self, views: slice) -> np.ndarray:
        batch = self.projections[views]
        lo, hi = max(0, self.start), min(batch.shape[1], self.start + self.rows)
        if lo == self.start and hi == self.start + self.rows:
            return np.ascontiguousarray(batch[:, lo:hi], dtype=np.float32)
        data = np.zeros((batch.shape[0], self.rows, batch.shape[2]), dtype=np.float32)
        if hi > lo:
            data[:, lo - self.start : hi - self.start] = batch[:, lo:hi]
        return data


def _mapped_storage(array: np.ndarray) -> np.memmap | None:
    """Find mapped ownership through NumPy and buffer-protocol views, without reading pixels."""
    current: object = array
    while isinstance(current, np.ndarray | memoryview):
        if isinstance(current, np.memmap):
            return current
        current = current.obj if isinstance(current, memoryview) else current.base
    return None


def empty_host_array(shape: tuple[int, int, int], order: Literal["C", "F"]) -> np.ndarray:
    """Own demand-paged host storage independently of the NumPy allocation arena.

    The ndarray retains the anonymous mapping through its base, including when
    only a view survives. There is no file, process-wide allocator change or
    eager huge-page advice; pages are populated as slabs are written.
    """
    storage = mmap.mmap(
        -1,
        math.prod(shape) * np.dtype(np.float32).itemsize,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
    )
    return np.ndarray(shape, dtype=np.float32, buffer=storage, order=order)


def validate_host_arrays(
    projections: np.ndarray | jax.Array,
    out: np.ndarray | None,
    shape: tuple[int, int, int],
    context: str = "fbp_host",
    *,
    device_input: bool = False,
    order: Literal["C", "F"] = "C",
    allocate: Callable[[tuple[int, int, int], Literal["C", "F"]], np.ndarray] | None = None,
) -> np.ndarray:
    """Check host input/output arrays and allocate the output volume if needed.

    ``device_input`` also accepts a JAX array, which cannot overlap host storage.
    ``order`` selects storage only when allocating; caller-owned output is kept.
    ``allocate`` can supply privately owned host storage after input validation.
    Mapped inputs and outputs must use separate files, including buffer views.
    """
    # Host-only inverses must not initialize the JAX import stack merely to
    # reject device arrays. Device callers already have JAX loaded.
    host_input = isinstance(projections, np.ndarray)
    valid = host_input
    if device_input and not host_input:
        import jax

        valid = isinstance(projections, jax.Array)
    if not valid or projections.dtype.kind not in "buif":
        kinds = "NumPy or JAX" if device_input else "NumPy"
        raise TypeError(f"{context}: projections must be a real {kinds} array")
    if out is None:
        return (
            np.empty(shape, dtype=np.float32, order=order)
            if allocate is None
            else allocate(shape, order)
        )
    if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.float32:
        raise ValueError(f"{context}: out must be a float32 NumPy array with the volume shape")
    if not out.flags.writeable:
        raise ValueError(f"{context}: out must be writable")
    if not host_input:
        return out
    if np.may_share_memory(projections, out):
        raise ValueError(f"{context}: input and output storage must not overlap")
    input_map, output_map = _mapped_storage(projections), _mapped_storage(out)
    if input_map is not None and output_map is not None:
        source, target = input_map.filename, output_map.filename
        if source is None or target is None or Path(source).samefile(target):
            raise ValueError(f"{context}: memory-mapped input and output require separate files")
    return out


def validate_host_views(array: np.ndarray, *, weights: np.ndarray | None = None) -> None:
    """Reject invalid active data before ordered solver callbacks can be poisoned.

    Unweighted contiguous arrays can be read in either storage order. Weighted
    values must instead keep the same logical order as their activity mask,
    even when the two arrays have different physical layouts.
    """
    if weights is None and (array.flags.c_contiguous or array.flags.f_contiguous):
        # A view, not a full-volume copy. In particular, do not pack F-order
        # host slabs into C order merely to check that their values are finite.
        values = array.ravel(order="K")
    else:
        values = array.reshape(-1) if array.flags.c_contiguous else array.flat
    active = (
        None
        if weights is None
        else weights.reshape(-1)
        if weights.flags.c_contiguous
        else weights.flat
    )
    for start in range(0, array.size, 2**20):
        chunk = values[start : start + 2**20]
        if active is not None:
            chunk = np.where(active[start : start + 2**20] > 0, chunk, 0)
        with np.errstate(over="ignore", invalid="ignore"):
            finite = np.isfinite(np.asarray(chunk, dtype=np.float32)).all()
        if not finite:
            raise ValueError("streamed projections must be finite in FP32")
