"""Validation shared by reconstructions that keep inputs and outputs in host memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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


def check_measurements(projections: np.ndarray | jax.Array, method: str) -> None:
    """Check measured line integrals on their own device or in bounded host blocks."""
    from tomojax.core.host_volume import check_rows
    from tomojax.errors import InvalidParameter

    poisson = method == "mlem"
    if isinstance(projections, np.ndarray):
        check_rows(projections, method, nonnegative=poisson)
    else:
        from ._solver_validation import valid_measurements

        if not bool(valid_measurements(projections, nonnegative=poisson)):
            domain = "finite and nonnegative" if poisson else "finite in FP32"
            raise InvalidParameter("source", f"{method}: projections must be {domain}")


def validate_host_arrays(
    projections: np.ndarray | jax.Array,
    out: np.ndarray | None,
    shape: tuple[int, int, int],
    context: str = "fbp_host",
    *,
    device_input: bool = False,
    copies: int = 1,
) -> np.ndarray:
    """Check host input/output arrays and allocate the output volume if needed.

    ``device_input`` also accepts a JAX array, which cannot overlap host storage.
    Without ``out`` the volume comes from :func:`tomojax.core.host_volume.host_array`:
    in memory, or a temporary file when ``copies`` volumes (the output and the
    work volumes a solver keeps beside it) exceed half of host memory. Caller-owned
    output is kept. Mapped inputs and outputs must use separate files, including
    buffer views.
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
        from tomojax.core.host_volume import host_array

        return host_array(shape, copies=copies)
    if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.float32:
        raise ValueError(f"{context}: out must be a float32 NumPy array with the volume shape")
    if not out.flags.writeable:
        raise ValueError(f"{context}: out must be writable")
    from tomojax.core.host_volume import mapped_storage
    from tomojax.core.input_files import check_output

    output_map = mapped_storage(out)
    if output_map is not None and output_map.filename is not None and output_map.mode != "c":
        check_output(Path(output_map.filename), field="out")
    if host_input and np.may_share_memory(projections, out):
        raise ValueError(f"{context}: input and output storage must not overlap")
    input_map = mapped_storage(projections)
    if input_map is not None and output_map is not None:
        from tomojax.core.host_volume import file_identity

        source, target = file_identity(input_map), file_identity(output_map)
        if source is None or target is None or source == target:
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
