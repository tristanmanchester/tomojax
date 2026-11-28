"""The FBP filter of detector rows continued past the detector's edges."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from tomojax._parameters import DEFAULT_EXTERIOR, Exterior
from tomojax.geometry.coverage import conjugate_weights, full_parallel_turn, outer_edge

from ._continuation import Continuation, continued_columns, parallel_axis_distance
from ._row_filter import filter_rows, filtered_profiles
from .filters import get_fbp_filter_np, rfft_filter_array

if TYPE_CHECKING:
    from tomojax._parameters import Filter
    from tomojax.core.geometry import Detector, Geometry

    from .fbp import FBPConfig


def horizontal_projections(
    data: np.ndarray | jnp.ndarray,
    geometry: Geometry,
    detector: Detector,
    roll: float,
    cfg: FBPConfig,
) -> tuple[np.ndarray | jnp.ndarray, Detector, FBPConfig, bool]:
    """Return physical rows and whether their uniform-width ramp needs conjugate weights."""
    from ._detector_rows import DetectorRows

    asymmetric = detector.center[0] != 0 or roll % 90 != 0
    full_turn = asymmetric and cfg.scale is None and full_parallel_turn(geometry)
    if not roll:
        return data, detector, cfg, full_turn
    layout = DetectorRows(detector, roll, full_turn)
    rows = layout.parallel_band(data, 0, layout.horizontal.nv, native=0, exterior=cfg.exterior)
    panel = layout.filter_detector(cfg.exterior)
    if layout.quarter_turn:
        return rows, panel, cfg, full_turn
    # Each oblique chord already carries its own weights and fitted tails.
    return rows, panel, replace(cfg, exterior="zero"), False


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
    measured: jnp.ndarray
    profiles: jnp.ndarray | None

    redundancy: jnp.ndarray | None

    def rows(self, first: int, stop: int) -> RowFilter:
        """The filter of detector rows ``first:stop``; rows outside the detector are zero."""
        if self.radius2 is None:
            return self
        index = jnp.arange(first, stop).clip(0, self.radius2.shape[0] - 1)
        profiles = None if self.profiles is None else self.profiles[:, index]
        return self._replace(radius2=self.radius2[index], profiles=profiles)


def row_filter(
    filter: Filter,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    exterior: Exterior = DEFAULT_EXTERIOR,
    *,
    separable: bool = True,
    full_turn: bool = False,
) -> RowFilter:
    """The filter of the rows of a ``(views, nv, nu)`` stack, read over the filter ``detector``."""
    nu = projections.shape[-1]
    offset = detector.center[0] / detector.du
    edge = outer_edge(offset, full_turn=full_turn)
    redundancy = jnp.asarray(conjugate_weights(nu, offset)) if edge is not None else None
    if exterior == "zero":
        spectrum = rfft_filter_array(filter, detector.nu, float(detector.du), jnp.float32)
        return RowFilter(spectrum, None, spectrum, None, redundancy)
    pad = continued_columns(nu)
    distance = parallel_axis_distance(detector, nu, pad)
    radius2 = jnp.asarray(Continuation(distance, nu, pad, edge=edge).stack_radius(projections))
    reach = (detector.nu - nu) // 2
    wide, measured = _spectra(filter, nu, pad, reach, float(detector.du))
    spectrum = jnp.asarray(wide)
    profiles = (
        filtered_profiles(radius2, None, spectrum, tuple(distance.tolist()), reach, reach)
        if separable
        else None
    )
    return RowFilter(spectrum, radius2, jnp.asarray(measured), profiles, redundancy)


@lru_cache(maxsize=16)
def _spectra(
    filter: Filter, nu: int, pad: int, reach: int, du: float
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = get_fbp_filter_np(filter, nu + pad + reach, du, "float32")
    # Measured input and requested output need only these lags of the wide
    # filter. Keep its exact spatial weights, including for windowed ramps.
    length = max(64, 1 << (2 * (nu + reach) - 1).bit_length())
    measured = spectrum
    if 2 * (len(spectrum) - 1) > length:
        impulse = np.fft.irfft(spectrum.astype(np.float64))
        h = length // 2
        measured = np.fft.rfft(np.concatenate((impulse[:h], impulse[-h:]))).real.astype(np.float32)
    return spectrum, measured


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


def ramp_rows(rows: jnp.ndarray, filter: RowFilter, detector: Detector) -> jnp.ndarray:
    """Continue measured rows, ramp-filter them and read the filter detector's columns."""
    nu = rows.shape[-1]
    reach = (detector.nu - nu) // 2
    measured = rows if filter.redundancy is None else rows * filter.redundancy
    result = filter_rows(measured, filter.measured, nu, reach, reach)
    radius, profiles = filter.radius2, filter.profiles
    if radius is None or profiles is None:
        return result
    fits = Continuation(
        parallel_axis_distance(detector, nu, nu),
        nu,
        nu,
        edge=outer_edge(detector.center[0], full_turn=filter.redundancy is not None),
    )
    place = partial(jnp.asarray, dtype=rows.dtype)
    low, high = fits.amplitudes(place, rows[..., : fits.window], rows[..., -fits.window :], radius)
    return result + low * profiles[0] + high * profiles[1]
