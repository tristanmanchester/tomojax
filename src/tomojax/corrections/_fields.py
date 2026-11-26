"""Calibration moments and illumination fields in acquisition order."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import logging
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax.core.host_volume import host_array, release_planes

from ._batches import HOST_THREADS, bands, view_batches
from ._records import Correction, Json
from ._validation import positive_finite

if TYPE_CHECKING:
    from collections.abc import Iterator

    import jax

    from tomojax._data import FrameStack

type CalibrationFrames = np.ndarray | jax.Array | FrameStack

LOG = logging.getLogger(__name__)


class Fields:
    """Dark-subtracted illumination sets, their positions and the additive dark field.

    ``flats`` has shape ``(sets, rows, columns)`` in counts, or counts per second
    when exposures are supplied. ``usable`` marks pixels with measured illumination.
    """

    def __init__(
        self, flats: np.ndarray, positions: np.ndarray, dark: np.ndarray, usable: np.ndarray
    ) -> None:
        self.flats, self.positions, self.dark, self.usable = flats, positions, dark, usable

    def window(self, rows: slice) -> Fields:
        """Select detector rows after resolving illumination on the whole acquisition."""
        return Fields(self.flats[:, rows], self.positions, self.dark[rows], self.usable[rows])


def flat_fields(
    flats: CalibrationFrames | None,
    positions: np.ndarray | None,
    darks: CalibrationFrames | None,
    white_level: float | None,
    shape: tuple[int, int],
    epsilon: float,
    *,
    flat_scales: np.ndarray | None = None,
    dark_source: str | None = None,
) -> tuple[Fields, tuple[Correction, ...]]:
    """The mean flat of each set, the mean dark, and the record of them.

    A pixel is usable where the flat exceeds the dark by at least ``epsilon`` and
    three standard errors of the two calibration means. Elsewhere the ratio is
    noise: ``first_pass`` interpolates interior gaps along each detector row and
    takes pixels outside its measured span as no absorption.
    At least one pixel must have measured illumination: without one, neither
    a transmission ratio nor an interpolation is determined by the calibration.
    An entirely unresolved calibration raises an input error, not an empty scan.
    """
    if darks is None and flats is not None:
        LOG.warning("no dark fields: correcting with a dark field of zero")
    zero = np.zeros(shape)
    dark, dark_variance = (zero, zero) if darks is None else _moments(darks, shape, "darks")
    dark_frames = 0 if darks is None or dark_source is not None else len(darks)
    if flats is None:
        if white_level is None:
            raise ValueError(
                "no flat fields: pass flats=... (frames of the beam without the object, or "
                "the counts of an unattenuated pixel)"
            )
        level = positive_finite(white_level, "white_level")
        sets = np.zeros(1)
        moments = ((level, zero, 1.0),)
        settings: dict[str, Json] = {"white_level": level, "darks": dark_frames}
    else:
        check_stack(flats, shape, "flats")
        where = np.zeros(len(flats)) if positions is None else np.asarray(positions, np.float64)
        if where.shape != (len(flats),) or not np.isfinite(where).all():
            raise ValueError(
                f"flat_positions needs one finite position for each of the {len(flats)} flats"
            )
        sets = np.unique(where)

        moments = set_moments(flats, shape, where, sets, flat_scales, dark)
        settings = {"flats": len(flats), "darks": dark_frames, "flat_sets": len(sets)}
        if flat_scales is not None:
            settings["flat_exposure_times"] = (1 / flat_scales).tolist()
        if len(sets) > 1:
            settings["flat_positions"] = [float(p) for p in sets]
    if dark_source is not None:
        settings["dark_removed_by"] = dark_source
    light = host_array((len(sets), *shape))
    usable = np.ones(shape, bool)
    for i, (mean, variance, scale) in enumerate(moments):
        signal = mean - dark if flat_scales is None else mean
        floor = np.maximum(epsilon, 3 * np.sqrt(variance + dark_variance * scale**2))
        usable &= np.isfinite(signal) & (signal > floor)
        light[i] = signal
        release_planes(light, i, i + 1)
    if not np.any(usable):
        raise ValueError(
            "No detector pixels have resolved illumination above the dark field: "
            "supply illuminated flats=... and matching darks=... (CLI --flats/--darks), "
            "or include an illuminated detector region in the crop"
        )
    found: dict[str, Json] = {"pixels_without_light": int(np.count_nonzero(~usable))}
    fields = Fields(light, sets, dark.astype(np.float32), usable)
    return fields, (Correction("flat_dark", settings, found),)


def check_stack(frames: CalibrationFrames, shape: tuple[int, int], name: str) -> None:
    if len(frames.shape) != 3 or tuple(frames.shape[1:]) != shape or not len(frames):
        raise ValueError(
            f"{name} are {tuple(frames.shape)}; a nonempty stack of frames of {shape} is needed"
        )


def _moments(
    frames: CalibrationFrames,
    shape: tuple[int, int],
    name: str,
    indices: np.ndarray | None = None,
    *,
    scales: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The mean of frames and the variance of that mean, in float64 over bounded batches.

    Bands of detector rows accumulate on separate threads; each pixel's sums
    take the same operations in the same order as on one thread.
    """
    check_stack(frames, shape, name)
    count = len(frames) if indices is None else len(indices)
    batch = max(1, (32 << 20) // (8 * shape[0] * shape[1]))
    total, squares = np.zeros(shape), np.zeros(shape)
    rows_of = bands(shape[0], HOST_THREADS)

    def accumulate(raw: np.ndarray, which: slice | np.ndarray, rows: slice) -> None:
        values = np.asarray(raw[:, rows], np.float64)
        if scales is not None:
            below = 0 if offset is None else offset[rows]
            values = (values - below) * scales[which, None, None]
        total[rows] += values.sum(axis=0)
        squares[rows] += np.einsum("ijk,ijk->jk", values, values)

    with (
        ThreadPoolExecutor(len(rows_of), thread_name_prefix="tomojax-moments") as pool,
        closing(view_batches(frames, count, batch, indices)) as batches,
    ):
        for start, stop, raw in batches:
            progress.check_canceled()
            which = slice(start, stop) if indices is None else indices[start:stop]
            for done in [pool.submit(accumulate, raw, which, rows) for rows in rows_of]:
                done.result()
    mean = total / count
    if count == 1:
        return mean, np.zeros(shape)
    return mean, np.maximum(0.0, squares - total * mean) / (count * (count - 1))


def bracketing_sets(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The flat sets that bracket a batch of views, and each view's two indices among them."""
    sets, inverse = np.unique(np.stack((lo, hi)), return_inverse=True)
    a, b = inverse.reshape(2, -1).astype(np.int32)
    return sets, a, b


def set_moments(
    flats: CalibrationFrames,
    shape: tuple[int, int],
    where: np.ndarray,
    sets: np.ndarray,
    flat_scales: np.ndarray | None,
    dark: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, float | np.floating]]:
    for position in sets:
        selected = None if len(sets) == 1 else np.flatnonzero(where == position)
        light, variance = _moments(
            flats,
            shape,
            "flats",
            selected,
            scales=flat_scales,
            offset=dark if flat_scales is not None else None,
        )
        scale = 1.0 if flat_scales is None else np.mean(flat_scales[where == position])
        yield light, variance, scale
