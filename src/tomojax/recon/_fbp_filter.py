"""The FBP filter of detector rows continued past the detector's edges."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

from tomojax._parameters import DEFAULT_EXTERIOR, Exterior

from ._continuation import Continuation, continued_columns, parallel_axis_distance
from .filters import rfft_filter_array

if TYPE_CHECKING:
    import numpy as np

    from tomojax._parameters import Filter
    from tomojax.core.geometry import Detector


class RowFilter(NamedTuple):
    """The FBP filter of a block of detector rows.

    Rows are continued past the detector's edges before filtering (see
    :mod:`tomojax.recon._continuation`): ``radius2`` is each row's squared
    continuation radius, shape ``(rows, 1)``, and ``spectrum`` the ramp filter,
    whose transform is long enough that no column the filter detector reads
    sees the continued row wrap around.
    """

    spectrum: jnp.ndarray
    radius2: jnp.ndarray | None

    def rows(self, first: int, stop: int) -> RowFilter:
        """The filter of detector rows ``first:stop``; rows outside the detector are zero."""
        if self.radius2 is None:
            return self
        count = self.radius2.shape[0]
        lo, hi = max(0, first), min(count, stop)
        if hi <= lo:
            return self._replace(radius2=jnp.ones((stop - first, 1), jnp.float32))
        inside = self.radius2[lo:hi]
        return self._replace(
            radius2=jnp.pad(inside, ((lo - first, stop - hi), (0, 0)), mode="edge")
        )


def row_filter(
    filter: Filter,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    exterior: Exterior = DEFAULT_EXTERIOR,
) -> RowFilter:
    """The filter of the rows of a ``(views, nv, nu)`` stack, read over the filter ``detector``."""
    if exterior == "zero":
        return RowFilter(
            rfft_filter_array(filter, detector.nu, float(detector.du), jnp.float32), None
        )
    nu = projections.shape[-1]
    pad = continued_columns(nu)
    fits = Continuation(parallel_axis_distance(detector, nu, pad), nu, pad)
    radius2 = fits.stack_radius(projections)
    reach = (detector.nu - nu) // 2
    spectrum = rfft_filter_array(filter, nu + pad + reach, float(detector.du), jnp.float32)
    return RowFilter(spectrum, jnp.asarray(radius2))


def continued_rows(
    rows: jnp.ndarray, detector: Detector, radius2: jnp.ndarray | None
) -> jnp.ndarray:
    """Measured rows continued along u past the detector's edges."""
    if radius2 is None:
        return rows
    nu = rows.shape[-1]
    pad = continued_columns(nu)
    fits = Continuation(parallel_axis_distance(detector, nu, pad), nu, pad)
    before, after = fits.continued(partial(jnp.asarray, dtype=rows.dtype), rows, radius2)
    return jnp.concatenate([before, rows, after], axis=-1)


def detector_columns(filtered: jnp.ndarray, nu: int, detector: Detector, pad: int) -> jnp.ndarray:
    """The filter detector's columns of filtered continued rows of transform length n_fft."""
    n_fft = filtered.shape[-1]
    first = pad - (detector.nu - nu) // 2
    if first >= 0:
        return filtered[..., first : first + detector.nu]
    # A filter detector wider than the continuation reads the circular transform's tail.
    return jnp.concatenate(
        [filtered[..., n_fft + first :], filtered[..., : first + detector.nu]], axis=-1
    )


def ramp_rows(rows: jnp.ndarray, filter: RowFilter, detector: Detector) -> jnp.ndarray:
    """Continue measured rows, ramp-filter them and read the filter detector's columns."""
    spectrum = filter.spectrum
    n_fft = 2 * (spectrum.shape[0] - 1)
    continued = continued_rows(rows, detector, filter.radius2)
    transformed = jnp.fft.rfft(continued, n=n_fft, axis=-1) * spectrum
    return detector_columns(
        jnp.fft.irfft(transformed, n=n_fft, axis=-1),
        rows.shape[-1],
        detector,
        (continued.shape[-1] - rows.shape[-1]) // 2,
    )
