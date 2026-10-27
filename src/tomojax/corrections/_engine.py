"""Correct counts in view batches and complete sinogram rows in bounded slabs.

Counts steps precede flat/dark calibration, then transmission steps precede
-log, and line-integral steps follow it. Whole-row steps use every view of a
row at once; other steps use batches of views. A pre-log row step stages
transmission, not line integrals, when the pipeline needs both kinds of batch.

A row-efficient source can fuse calibration, row corrections and log without
an intermediate host stack. Detector binning always averages the corrected
line integrals. Both paths materialise the final corrected host stack.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.array_ops import mean_detector_pixels

from ._records import Correction, Json
from ._steps import DOMAINS, Step

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging.getLogger(__name__)

# Host memory a batch of views (or slab of rows) may take, as float32.
_BATCH_BYTES = 256 * 2**20
_SIMPLE_FRAME_BYTES = 32 * 2**20


def correct_frames(
    counts: Any,
    *,
    flats: Any,
    flat_positions: np.ndarray | None,
    darks: Any,
    white_level: float | None,
    steps: Sequence[Step],
    epsilon: float,
    view_positions: np.ndarray | None = None,
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
    batch_views: int | None = None,
    binning: int = 1,
) -> Corrected:
    """Convert detector counts to line integrals and record each correction.

    ``flats`` are flat frames and ``flat_positions`` the number of views
    recorded before each (frames at one position are one set; None, one set);
    ``white_level`` stands in for flats on scanners that record it instead.
    ``darks`` default to zero. ``view_positions`` places each view among the
    flats (by default view ``i`` at ``i + 1/2``). ``binning`` averages blocks of
    ``binning`` x ``binning`` detector pixels of the corrected line integrals.

    Parameters
    ----------
    counts : array-like
        Detector counts with shape ``(views, rows, columns)``. May be a lazy source.
    flats : array-like or None
        Open-beam frames with shape ``(frames, rows, columns)``, or None when
        white_level supplies the open-beam count.
    flat_positions : numpy.ndarray or None
        Number of views recorded before each flat frame. Frames at the same
        position form one set; None puts all frames in one set.
    darks : array-like or None
        Dark frames with shape ``(frames, rows, columns)``; None uses zero dark.
    white_level : float or None
        Positive open-beam count used when flats is None.
    steps : sequence of Step
        Corrections, run by domain and in the supplied order within each domain.
    epsilon : float
        Positive transmission floor before the logarithm.
    view_positions : numpy.ndarray or None, optional
        Position of each view among the flats. Defaults to ``i + 0.5`` for view i.
    exposure_times : numpy.ndarray or float or None, optional
        Sample exposure durations in seconds, one per view or a common duration.
    flat_exposure_times : numpy.ndarray or float or None, optional
        Flat exposure durations in seconds. Supply both exposure arrays to compare
        dark-subtracted count rates rather than integrated counts.

    batch_views : int or None, optional
        Positive view-batch size. None selects a memory-bounded size automatically.
    binning : int, optional
        Positive detector-block side length for mean binning of the corrected line
        integrals. Default is 1; incomplete edge blocks are dropped.

    Returns
    -------
    Corrected
        Float32 host line integrals, correction records and kept input-view indices.

    Raises
    ------
    ValueError
        If calibration, positions, batching, binning or correction domains are invalid.
    TypeError
        If a correction is not a Step.

    Examples
    --------
    >>> import numpy as np
    >>> from tomojax.corrections import correct_frames
    >>> result = correct_frames(
    ...     np.full((2, 3, 4), 50),
    ...     flats=None,
    ...     flat_positions=None,
    ...     darks=None,
    ...     white_level=100,
    ...     steps=(),
    ...     epsilon=1e-6,
    ... )
    >>> result.projections.shape
    (2, 3, 4)
    """
    batch_views = _validated_batch_views(batch_views)
    epsilon = positive_finite(epsilon, "epsilon")
    if isinstance(binning, bool) or not isinstance(binning, int | np.integer) or binning < 1:
        raise ValueError(f"binning needs a whole factor of at least 1, not {binning!r}")
    binning = int(binning)
    views, rows, cols = (int(s) for s in counts.shape)
    ordered = ordered_steps(steps)
    view_scales, flat_scales = exposure_scales(exposure_times, flat_exposure_times, views, flats)
    fields, field_record = flat_fields(
        flats, flat_positions, darks, white_level, (rows, cols), epsilon, flat_scales=flat_scales
    )
    if flat_scales is not None and exposure_times is not None:
        field_record[0].settings["sample_exposure_range_s"] = [
            float(np.min(exposure_times)),
            float(np.max(exposure_times)),
        ]
    at = (
        np.arange(views) + 0.5 if view_positions is None else np.asarray(view_positions, np.float64)
    )
    if at.shape != (views,) or not np.isfinite(at).all():
        raise ValueError(f"view_positions needs one finite position for each of the {views} views")
    interpolation = flat_weights(at, fields.positions, exposure_scales=view_scales)
    line = [s for d, s in ordered if d == "line integrals"]
    head, tail = _split_at_break(line)
    counts_steps = tuple(s for d, s in ordered if d == "counts")
    transmission_steps = tuple(s for d, s in ordered if d == "transmission")
    transmission_head, transmission_tail = _split_at_break(list(transmission_steps))
    settings = {"epsilon": float(epsilon), "factor": binning}
    slab = _row_slab(counts, ordered, batch_views, binning)
    if slab is not None:
        out, tally = _correct_row_slabs(
            counts,
            fields,
            interpolation,
            tuple(line),
            slab=slab,
            transmission_steps=transmission_steps,
            **settings,
        )
        kept, tail_records, head = np.arange(views), (), line
    else:
        # Binned on the device while the batches come back, unless the tail needs full frames.
        factor = 1 if tail or transmission_tail else binning
        out = np.empty((views, rows // factor, cols // factor), np.float32)
        batch = batch_views or frame_batch_size(
            views, rows, cols, (*counts_steps, *transmission_steps, *head)
        )
        tally = correct_view_batches(
            counts, out, fields, interpolation, batch, counts_steps=counts_steps,
            transmission_steps=tuple(transmission_head),
            line_steps=() if transmission_tail else tuple(head),
            epsilon=float(epsilon), factor=factor, take_log=not transmission_tail,
        )  # fmt: skip
        if transmission_tail:
            out, _, _ = apply_steps(out, transmission_tail, batch_views)
            out, tally = log_batches(out, tuple(head), epsilon, batch_views)
        out, kept, tail_records = apply_steps(out, tail, batch_views)
        if (tail or transmission_tail) and binning > 1:
            out = _binned(out, binning, batch_views)
    found: dict[str, Json] = {
        "nonpositive_transmission": int(tally[0]),
        "nonfinite_set_to_zero": int(tally[1]),
    }
    log = Correction("log", {"epsilon": float(epsilon)}, found)
    records = tuple(
        [s.record() for s in counts_steps] + list(field_record)
        + [s.record() for s in transmission_steps] + [log]
        + [s.record() for s in head] + list(tail_records)
    )  # fmt: skip
    return Corrected(out, records, kept)


class Corrected(NamedTuple):
    """Corrected line integrals with records and input-view indices.

    Parameters
    ----------
    projections : numpy.ndarray
        Float32 corrected line integrals of shape ``(views, rows, columns)``.
    records : tuple of Correction
        Records in correction order.
    kept : numpy.ndarray
        Increasing indices of retained views in the original input stack.
    """

    projections: np.ndarray
    records: tuple[Correction, ...]
    kept: np.ndarray  # indices of the input views, increasing


def correct_projections(
    projections: Any, steps: Sequence[Step], *, batch_views: int | None = None
) -> Corrected:
    """Correct a line-integral stack and record the steps applied.

    Parameters
    ----------
    projections : array-like
        Line integrals with shape ``(views, rows, columns)``; copied to float32.
    steps : sequence of Step
        Corrections acting on line integrals, in the order to run them.
    batch_views : int or None, optional
        Positive view-batch size. None selects a memory-bounded size automatically.

    Returns
    -------
    Corrected
        Corrected host line integrals, correction records and kept input-view indices.

    Raises
    ------
    ValueError
        If a step does not act on line integrals or batch_views is invalid.
    TypeError
        If a correction is not a Step.

    Examples
    --------
    >>> import numpy as np
    >>> from tomojax.corrections import BeamHardening, correct_projections
    >>> result = correct_projections(np.ones((2, 3, 4)), [BeamHardening((1, 0.05))])
    >>> result.projections.shape
    (2, 3, 4)
    """
    batch_views = _validated_batch_views(batch_views)
    ordered = ordered_steps(steps)
    if any(d != "line integrals" for d, _ in ordered):
        wrong = [type(s).__name__ for d, s in ordered if d != "line integrals"]
        raise ValueError(
            f"{', '.join(wrong)} act on detector counts or transmission, which a Scan of line "
            "integrals no longer holds: load the frames with tomojax.load_frames and pass "
            "them to Frames.corrected"
        )
    out = np.array(projections, np.float32, copy=True)
    out, kept, records = apply_steps(out, [s for _, s in ordered], batch_views)
    return Corrected(out, records, kept)


def frame_batch_size(views: int, rows: int, cols: int, steps: Sequence[Step]) -> int:
    """Bound streaming batches while retaining room for workspace-heavy steps."""
    workspace = max((s.workspace_pixels(rows, cols) for s in steps), default=0)
    # Smaller simple-frame batches overlap file reads and transfers more cheaply
    # and leave less idle device-pool storage before the reconstruction starts.
    # Keep the existing workspace allowance for phase retrieval and resampling.
    budget = _BATCH_BYTES if workspace else _SIMPLE_FRAME_BYTES
    pixels = rows * cols + workspace
    return max(1, min(views, budget // (4 * pixels)))


def _validated_batch_views(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value <= 0:
        raise ValueError(f"batch_views must be a positive integer, not {value!r}")
    return int(value)


def positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, not {value!r}")
    return value


def ordered_steps(steps: Sequence[Step]) -> list[tuple[str, Step]]:
    """``steps`` by domain (counts, transmission, line integrals), in order within each."""
    for step in steps:
        if not isinstance(step, Step):
            raise TypeError(
                f"corrections take tomojax.corrections steps, not {type(step).__name__}"
            )
        if step.domain not in DOMAINS:
            raise ValueError(f"{type(step).__name__} has an unknown domain {step.domain!r}")
        if step.selects_views and step.domain != "line integrals":
            raise ValueError("view-selecting corrections act on line integrals only")
        if step.whole_rows and step.domain == "counts":
            raise ValueError("whole-row corrections act on transmission or line integrals")
    return [(d, s) for d in DOMAINS for s in steps if s.domain == d]


class Fields:
    """Dark-subtracted illumination sets, their positions and the additive dark field.

    ``flats`` has shape ``(sets, rows, columns)`` in counts, or counts per second
    when exposures are supplied. ``usable`` marks pixels with measured illumination.
    """

    def __init__(
        self, flats: np.ndarray, positions: np.ndarray, dark: np.ndarray, usable: np.ndarray
    ) -> None:
        self.flats, self.positions, self.dark, self.usable = flats, positions, dark, usable


def flat_fields(
    flats: Any,
    positions: np.ndarray | None,
    darks: Any,
    white_level: float | None,
    shape: tuple[int, int],
    epsilon: float,
    *,
    flat_scales: np.ndarray | None = None,
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
    dark_frames = 0 if darks is None else len(darks)
    if flats is None:
        if white_level is None:
            raise ValueError(
                "no flat fields: pass flats (frames of the beam without the object) or "
                "white_level (the counts of an unattenuated pixel) to tomojax.load_frames"
            )
        level = positive_finite(white_level, "white_level")
        stack, variance, sets = np.full((1, *shape), level), np.zeros((1, *shape)), np.zeros(1)
        settings: dict[str, Any] = {"white_level": level, "darks": dark_frames}
    else:
        _check_stack(flats, shape, "flats")
        where = np.zeros(len(flats)) if positions is None else np.asarray(positions, np.float64)
        if where.shape != (len(flats),) or not np.isfinite(where).all():
            raise ValueError(
                f"flat_positions needs one finite position for each of the {len(flats)} flats"
            )
        sets = np.unique(where)
        moments = [
            _moments(
                flats,
                shape,
                "flats",
                None if len(sets) == 1 else np.flatnonzero(where == p),
                scales=flat_scales,
                offset=dark if flat_scales is not None else None,
            )
            for p in sets
        ]
        stack, variance = (np.stack(part) for part in zip(*moments, strict=True))
        settings = {"flats": len(flats), "darks": dark_frames, "flat_sets": len(sets)}
        if flat_scales is not None:
            settings["flat_exposure_times"] = (1 / flat_scales).tolist()
        if len(sets) > 1:
            settings["flat_positions"] = [float(p) for p in sets]
    light = stack - dark if flat_scales is None else stack
    dark_scale = (
        np.ones(len(sets))
        if flat_scales is None
        else np.asarray([np.mean(flat_scales[where == p]) for p in sets])
    )
    floor = np.maximum(
        epsilon, 3 * np.sqrt(variance + dark_variance * dark_scale[:, None, None] ** 2)
    )
    usable = cast("np.ndarray", (np.isfinite(light) & (light > floor)).all(axis=0))
    if not np.any(usable):
        raise ValueError(
            "No detector pixels have resolved illumination above the dark field: "
            "supply illuminated flats=... and matching darks=... (CLI --flats/--darks), "
            "or include an illuminated detector region in the crop"
        )
    found: dict[str, Json] = {"pixels_without_light": int(np.count_nonzero(~usable))}
    fields = Fields(light.astype(np.float32), sets, dark.astype(np.float32), usable)
    return fields, (Correction("flat_dark", settings, found),)


def _check_stack(frames: Any, shape: tuple[int, int], name: str) -> None:
    if len(frames.shape) != 3 or tuple(frames.shape[1:]) != shape or not len(frames):
        raise ValueError(
            f"{name} are {tuple(frames.shape)}; a nonempty stack of frames of {shape} is needed"
        )


def _moments(
    frames: Any,
    shape: tuple[int, int],
    name: str,
    indices: np.ndarray | None = None,
    *,
    scales: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The mean of frames and the variance of that mean, in float64 over bounded batches."""
    _check_stack(frames, shape, name)
    count = len(frames) if indices is None else len(indices)
    batch = max(1, (32 << 20) // (8 * shape[0] * shape[1]))
    total, squares = np.zeros(shape), np.zeros(shape)
    for start in range(0, count, batch):
        which = slice(start, start + batch) if indices is None else indices[start : start + batch]
        values = np.asarray(frames[which], np.float64)
        if scales is not None:
            values = (values - (0 if offset is None else offset)) * scales[which, None, None]
        total += values.sum(axis=0)
        squares += np.einsum("ijk,ijk->jk", values, values)
    mean = total / count
    if count == 1:
        return mean, np.zeros(shape)
    return mean, np.maximum(0.0, squares - total * mean) / (count * (count - 1))


def exposure_scales(
    sample: np.ndarray | float | None, flat: np.ndarray | float | None, views: int, flats: Any
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert measured durations to count-rate factors; white levels stay in counts."""
    if flats is None or (sample is None and flat is None):
        return np.ones(views, np.float32), None
    if sample is None or flat is None:
        raise ValueError(
            "Exposure correction needs both exposure_times=... and flat_exposure_times=... "
            "in tomojax.load_frames"
        )
    rates = []
    for name, value, count in (
        ("exposure_times", sample, views),
        ("flat_exposure_times", flat, len(flats)),
    ):
        times = np.broadcast_to(np.asarray(value, np.float64), (count,))
        if not np.all(np.isfinite(times) & (times > 0)):
            raise ValueError(f"{name} must contain positive finite durations in seconds")
        rates.append(1 / times)
    return rates[0].astype(np.float32), rates[1]


def flat_weights(
    q: np.ndarray, positions: np.ndarray, *, exposure_scales: np.ndarray | None = None
) -> tuple[np.ndarray, ...]:
    """For each view at ``q``, the flat sets either side of it and the weight of the later one.

    ``positions`` are the views recorded before each set; views before the
    first set or after the last take that set alone.
    """
    hi = np.clip(np.searchsorted(positions, q, side="right"), 0, len(positions) - 1)
    lo = np.clip(hi - 1, 0, len(positions) - 1)
    span = positions[hi] - positions[lo]
    t = np.where(span > 0, (q - positions[lo]) / np.where(span > 0, span, 1), 0.0)
    t = np.where(q >= positions[-1], 1.0, np.clip(t, 0.0, 1.0)) if len(positions) > 1 else t
    scales = np.ones(len(q), np.float32) if exposure_scales is None else exposure_scales
    return lo.astype(np.int32), hi.astype(np.int32), t.astype(np.float32), scales


@partial(
    jax.jit,
    static_argnames=(
        "epsilon",
        "counts_steps",
        "transmission_steps",
        "line_steps",
        "factor",
        "fill_gaps",
        "take_log",
    ),
)
def first_pass(
    raw: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    t: jax.Array,
    exposure_scale: jax.Array,
    flats: jax.Array,
    dark: jax.Array,
    usable: jax.Array,
    *,
    epsilon: float,
    counts_steps: tuple[Step, ...] = (),
    transmission_steps: tuple[Step, ...] = (),
    line_steps: tuple[Step, ...] = (),
    factor: int = 1,
    fill_gaps: bool = True,
    take_log: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = raw.astype(jnp.float32)
    for step in counts_steps:
        x = step.apply(x)
    weight = t[:, None, None]
    flat = (1 - weight) * flats[lo] + weight * flats[hi]
    transmission = (x - dark) * exposure_scale[:, None, None] / jnp.maximum(flat, epsilon)
    # An unresolved calibration is not a measured transmission. Fill interior
    # gaps before any transmission step can spread the ratio of two noise values.
    transmission = jnp.where(usable, transmission, 1.0)
    if fill_gaps:
        transmission = _interpolate_unmeasured(transmission, usable)
    for step in transmission_steps:
        transmission = step.apply(transmission)
    if not take_log:
        return transmission, jnp.asarray(0), jnp.asarray(0)
    return _log_transmission(transmission, epsilon=epsilon, line_steps=line_steps, factor=factor)


@partial(jax.jit, static_argnames=("epsilon", "line_steps", "factor"))
def _log_transmission(
    transmission: jax.Array,
    *,
    epsilon: float,
    line_steps: tuple[Step, ...] = (),
    factor: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Floor and log transmission, then run the phase prefix and bin its result."""
    nonpositive = jnp.sum(transmission <= 0)
    p = -jnp.log(jnp.maximum(transmission, epsilon))
    for step in line_steps:
        p = step.apply(p)
    bad = ~jnp.isfinite(p)
    p = jnp.where(bad, 0.0, p)
    return (mean_detector_pixels(p, factor) if factor > 1 else p), nonpositive, jnp.sum(bad)


def _interpolate_unmeasured(transmission: jax.Array, usable: jax.Array) -> jax.Array:
    """Fill bracketed calibration gaps from the same view's measured neighbours.

    Only columns between two measured pixels are interpolated. A row with no
    measured pixels, and the unilluminated margins of any row, retain the neutral
    transmission supplied by calibration. Measured pixels, including low-gain
    ones, retain their calibrated transmission.
    """
    cols = transmission.shape[-1]
    column = jnp.arange(cols)
    left = jax.lax.associative_scan(jnp.maximum, jnp.where(usable, column, -1), axis=1)
    right = jax.lax.associative_scan(
        jnp.minimum, jnp.where(usable, column, cols), axis=1, reverse=True
    )
    lower = jnp.take_along_axis(transmission, jnp.clip(left, 0, cols - 1)[None], axis=2)
    upper = jnp.take_along_axis(transmission, jnp.clip(right, 0, cols - 1)[None], axis=2)
    weight = (column - left) / jnp.maximum(right - left, 1)
    filled = lower + weight[None] * (upper - lower)
    interior = ~usable & (left >= 0) & (right < cols)
    return jnp.where(interior[None], filled, transmission)


def correct_view_batches(
    counts: Any,
    out: np.ndarray,
    fields: Fields,
    interpolation: tuple[np.ndarray, ...],
    batch: int,
    **settings: Any,
) -> np.ndarray:
    """Run ``first_pass`` over batches of views into ``out``; the tally of what it found."""
    flats, dark, usable = (jax.device_put(a) for a in (fields.flats, fields.dark, fields.usable))
    lo, hi, t, exposure_scale = interpolation
    tally = np.zeros(2, np.int64)  # non-positive transmission, non-finite results
    fill_gaps = not bool(fields.usable.all())

    def collect(start: int, stop: int, result: jax.Array, *bad: jax.Array) -> None:
        out[start:stop] = np.asarray(result)
        tally[:] += [int(b) for b in bad]

    pending = None
    for start, stop, raw in _batches(counts, counts.shape[0], batch):
        done = first_pass(
            jnp.asarray(raw), lo[start:stop], hi[start:stop], t[start:stop],
            exposure_scale[start:stop], flats, dark, usable,
            fill_gaps=fill_gaps,
            **settings,
        )  # fmt: skip
        done[0].copy_to_host_async()
        if pending is not None:  # the last batch comes back while this one runs
            collect(*pending)
        pending = (start, stop, *done)
    if pending is not None:
        collect(*pending)
    return tally


def log_batches(
    out: np.ndarray, steps: tuple[Step, ...], epsilon: float, batch_views: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Replace a host transmission stack by line integrals, counting log findings."""
    views, rows, cols = out.shape
    batch = batch_views or frame_batch_size(views, rows, cols, steps)
    tally = np.zeros(2, np.int64)
    for first in range(0, views, batch):
        done, nonpositive, bad = _log_transmission(
            jnp.asarray(out[first : first + batch]), epsilon=epsilon, line_steps=steps
        )
        out[first : first + batch] = np.asarray(done)
        tally += (int(nonpositive), int(bad))
    return out, tally


def _row_slab(
    counts: Any, ordered: Sequence[tuple[str, Step]], batch_views: int | None, factor: int
) -> int | None:
    """Rows per slab to fuse raw-count correction with whole-row steps.

    No step may look across rows, and the source must read a window of rows
    without decoding whole frames. The slab is a multiple of ``factor`` rows
    when the corrected line integrals are also binned.
    """
    if batch_views is not None or not ordered:
        return None
    # At native sampling, per-view file opens can cost more than the saved
    # transfers. Keep the existing view path for sources without bulk reads.
    if (
        factor == 1
        and not isinstance(counts, np.ndarray)
        and not getattr(counts, "_bulk_row_reads", False)
    ):
        return None
    if any(d == "counts" or not s.whole_rows or s.selects_views for d, s in ordered):
        return None
    if not isinstance(counts, np.ndarray) and not getattr(counts, "_efficient_row_windows", False):
        return None
    views, rows, cols = counts.shape
    itemsize = max(4, np.dtype(counts.dtype).itemsize)
    slab = min(rows, _BATCH_BYTES // (2 * itemsize * views * cols)) // factor * factor
    return slab or None


def _correct_row_slabs(
    counts: Any,
    fields: Fields,
    interpolation: tuple[np.ndarray, ...],
    steps: tuple[Step, ...],
    *,
    slab: int,
    **settings: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct and optionally bin slabs of whole rows: the corrected stack and the tally."""
    views, rows, cols = counts.shape
    factor = settings["factor"]
    out = np.empty((views, rows // factor, cols // factor), np.float32)
    flats, dark, usable = (jax.device_put(a) for a in (fields.flats, fields.dark, fields.usable))
    lo, hi, t, exposure_scale = (jax.device_put(q) for q in interpolation)
    tally = np.zeros(2, np.int64)

    def collect(start: int, stop: int, result: jax.Array, *bad: jax.Array) -> None:
        out[:, start // factor : stop // factor] = np.asarray(result)
        tally[:] += [int(b) for b in bad]

    def read(start: int, stop: int) -> np.ndarray:
        frames = (
            counts[:, start:stop]
            if isinstance(counts, np.ndarray)
            else counts.window(slice(start, stop), slice(None))[:]
        )
        return np.asarray(frames).astype(frames.dtype.newbyteorder("="), copy=False)

    pending = None
    for start in range(0, rows - rows % factor, slab):
        stop = min(start + slab, rows - rows % factor)
        done = first_pass(
            jnp.asarray(read(start, stop)), lo, hi, t, exposure_scale,
            flats[:, start:stop], dark[start:stop],
            usable[start:stop], line_steps=steps,
            fill_gaps=not bool(fields.usable[start:stop].all()), **settings,
        )  # fmt: skip
        done[0].copy_to_host_async()
        if pending is not None:
            collect(*pending)
        pending = (start, stop, *done)
    if pending is not None:
        collect(*pending)
    return out, tally


def _binned(out: np.ndarray, factor: int, batch_views: int | None) -> np.ndarray:
    """The detector block means of ``out``, a batch of views at a time."""
    views, rows, cols = out.shape
    batch = batch_views or max(1, min(views, _BATCH_BYTES // (4 * rows * cols)))
    binned = np.empty((views, rows // factor, cols // factor), np.float32)
    for start in range(0, views, batch):
        binned[start : start + batch] = mean_detector_pixels(out[start : start + batch], factor)
    return binned


def _split_at_break(steps: list[Step]) -> tuple[list[Step], list[Step]]:
    """The steps before the first that needs whole rows or selects views, and the rest."""
    for i, step in enumerate(steps):
        if step.whole_rows or step.selects_views:
            return steps[:i], steps[i:]
    return steps, []


def apply_steps(
    out: np.ndarray, steps: Sequence[Step], batch_views: int | None
) -> tuple[np.ndarray, np.ndarray, tuple[Correction, ...]]:
    """``steps`` over ``out``: whole-row steps on slabs of rows, the others on views.

    A step that selects views drops the others from ``out``. Returns the
    result, the indices of the views kept, and each step's record.
    """
    kept = np.arange(out.shape[0])
    records: list[Correction] = []
    i = 0
    while i < len(steps):
        step = steps[i]
        views, rows, cols = out.shape
        if step.whole_rows:
            _whole_rows(out, step)
            records.append(step.record())
            i += 1
        elif step.selects_views:
            batch = batch_views or frame_batch_size(views, rows, cols, (step,))
            stats = np.concatenate([
                np.asarray(_statistic(step, jnp.asarray(out[v0 : v0 + batch])))
                for v0 in range(0, views, batch)
            ])  # fmt: skip
            keep, found = step.select(stats, kept)
            if not keep.any():
                raise ValueError(f"{type(step).__name__} would reject every view")
            if not keep.all():
                out, kept = out[keep], kept[keep]
            records.append(Correction(step.record().name, step.record().settings, found))
            i += 1
        else:
            j = i
            while j < len(steps) and not (steps[j].whole_rows or steps[j].selects_views):
                j += 1
            run = tuple(steps[i:j])
            batch = batch_views or frame_batch_size(views, rows, cols, run)
            for v0 in range(0, views, batch):
                out[v0 : v0 + batch] = np.asarray(_apply(run, jnp.asarray(out[v0 : v0 + batch])))
            records.extend(s.record() for s in run)
            i = j
    return out, kept, tuple(records)


def _whole_rows(out: np.ndarray, step: Step) -> None:
    """Run a whole-row step over ``out`` in place, on slabs of rows (one back as the next runs)."""
    views, rows, cols = out.shape
    slab = max(1, min(rows, _BATCH_BYTES // (8 * views * cols)))  # two slabs are in flight
    pending = None
    for r0 in range(0, rows, slab):
        done = _apply((step,), jnp.asarray(out[:, r0 : r0 + slab]))
        done.copy_to_host_async()
        if pending is not None:
            out[:, pending[0] : pending[0] + slab] = np.asarray(pending[1])
        pending = (r0, done)
    if pending is not None:
        out[:, pending[0] : pending[0] + slab] = np.asarray(pending[1])


@partial(jax.jit, static_argnames=("step",))
def _statistic(step: Step, batch: jax.Array) -> jax.Array:
    return step.statistic(batch)


@partial(jax.jit, static_argnames=("steps",))
def _apply(steps: tuple[Step, ...], batch: jax.Array) -> jax.Array:
    for step in steps:
        batch = step.apply(batch)
    return batch


def _batches(counts: Any, views: int, batch: int) -> Any:
    """``(start, stop, frames)`` for each batch of views, the next read while one is used."""
    starts = list(range(0, views, batch))

    def read(start: int) -> np.ndarray:
        frames = np.asarray(counts[start : min(start + batch, views)])
        # Device arrays use native byte order; leave ordinary batches as views.
        return frames.astype(frames.dtype.newbyteorder("="), copy=False)

    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read, starts[0]) if starts else None
        for k, start in enumerate(starts):
            assert pending is not None
            frames = pending.result()
            if k + 1 < len(starts):
                pending = reader.submit(read, starts[k + 1])
            yield start, min(start + batch, views), frames
