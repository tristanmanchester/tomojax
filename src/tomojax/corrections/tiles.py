"""Default count correction on a sampled detector tile in acquisition coordinates."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from ._engine import (
    Fields,
    check_stack,
    correct_view_batches,
    exposure_scales,
    flat_fields,
    flat_weights,
)
from ._illumination import measured_neighbours


def check_calibration_layout(frames: Any) -> None:
    """Check calibration dimensions and exposure metadata before reading any counts."""
    from ._validation import positive_finite

    for name in ("flats", "darks"):
        if (stack := getattr(frames, name)) is not None:
            check_stack(stack, frames.counts.shape[1:], name)
    if frames.white_level is not None:
        positive_finite(frames.white_level, "white_level")
    exposure_scales(frames.exposure_times, frames.flat_exposure_times, frames.views, frames.flats)


def corrected_tile(
    frames: Any,
    ranges: tuple[range, range, range],
    read: Callable[[Any, tuple[range, range, range]], np.ndarray],
) -> np.ndarray:
    """Calibrate a sampled tile at the selected views' original positions."""
    views, rows, cols = ranges
    view_key = slice(views.start, views.stop, views.step)
    scales, flat_scales = exposure_scales(
        frames.exposure_times, frames.flat_exposure_times, frames.views, frames.flats
    )
    positions = frames.view_positions
    at = np.asarray(views) + 0.5 if positions is None else positions[view_key]

    def fields_for(columns: range) -> Fields:
        def field(stack: Any) -> np.ndarray | None:
            return None if stack is None else read(stack, (range(len(stack)), rows, columns))

        return flat_fields(
            field(frames.flats),
            frames.flat_positions,
            field(frames.darks),
            frames.white_level,
            (len(rows), len(columns)),
            1e-6,
            flat_scales=flat_scales,
            require_light=(len(rows), len(columns)) == frames.counts.shape[1:],
        )[0]

    fields = fields_for(cols)
    selected: np.ndarray | slice = slice(None)
    columns = None
    counts = read(frames.counts, ranges)
    if not fields.usable.all() and len(cols) < frames.counts.shape[2]:
        # Calibration locates the neighbours outside the tile. Only those additional
        # sample columns can contribute to interpolation on the requested rows.
        fields = fields_for(range(frames.counts.shape[2]))
        requested = np.asarray(cols)
        left, right = (
            np.asarray(side)[:, requested]
            for side in measured_neighbours(jnp.asarray(fields.usable))
        )
        interior = (left >= 0) & (right < fields.usable.shape[1])
        support = np.union1d(requested, np.concatenate((left[interior], right[interior])))
        extra = np.setdiff1d(support, requested, assume_unique=True)
        selected = np.searchsorted(support, requested)
        runs = np.split(extra, np.flatnonzero(np.diff(extra) != 1) + 1)
        pieces = [
            counts,
            *(
                read(frames.counts, (views, rows, range(int(run[0]), int(run[-1]) + 1)))
                for run in runs
                if run.size
            ),
        ]
        counts = np.concatenate(pieces, axis=2)[
            :, :, np.argsort(np.concatenate((requested, extra)))
        ]
        fields = Fields(
            fields.flats[:, :, support],
            fields.positions,
            fields.dark[:, support],
            fields.usable[:, support],
        )
        columns = jnp.asarray(support)
    weights = flat_weights(at, fields.positions, exposure_scales=scales[view_key])
    out = np.empty(counts.shape, np.float32)
    correct_view_batches(
        counts, out, fields, weights, len(views), epsilon=1e-6, take_log=True, columns=columns
    )
    return out[:, :, selected]
