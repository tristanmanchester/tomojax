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

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing, nullcontext
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax import progress
from tomojax.core.array_ops import check_binning, mean_detector_pixels
from tomojax.core.host_volume import discard_failed_outputs, release_planes
from tomojax.core.registration import correction_images

from ._batches import HostCopies, copy_projections, correction_group, view_batches
from ._fields import Fields, batch_fields, flat_fields
from ._illumination import interpolate_unmeasured
from ._records import Correction, Json
from ._steps import DOMAINS, Step
from ._validation import check_scan_steps, positive_finite, validated_views_per_batch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
    views_per_batch: int | None = None,
    binning: int = 1,
    output_rows: tuple[int, int] | None = None,
    dark_source: str | None = None,
    linear_limit: tuple[float, str] | None = None,
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
    views_per_batch : int or None, optional
        Positive view-batch size. None selects a memory-bounded size automatically.
    binning : int, optional
        Positive detector-block side length for mean binning of the corrected line
        integrals. Default is 1; incomplete edge blocks are dropped.
    output_rows : tuple of int or None, optional
        Half-open interval in the binned detector. Calibration uses the original detector;
        the selected corrections determine which native rows must be read.
    dark_source : str or None, optional
        What removed the dark before the counts were recorded (a camera that
        subtracts it, say), kept in the record as ``dark_removed_by`` in place of
        a count of dark frames. Default None.
    linear_limit : tuple of (float, str) or None, optional
        The detector's stated linear range in counts and where it is stated. The
        record keeps the fraction of counts above it and the line integral where
        it ends, and a warning names both when any count lies above. Default None.

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
    progress.check_cancelled()
    views_per_batch = validated_views_per_batch(views_per_batch)
    epsilon = positive_finite(epsilon, "epsilon")
    views, rows, cols = (int(s) for s in counts.shape)
    check_binning(binning, rows, cols)
    binning = int(binning)
    band = slice(0, rows)
    selected = slice(None)
    if output_rows is not None:
        from ._rows import preview_window

        lower, upper = output_rows
        band, steps = preview_window(steps, (rows, cols), lower, upper, binning)
        selected = slice(lower - band.start // binning, upper - band.start // binning)
    ordered = ordered_steps(steps)
    view_scales, flat_scales = exposure_scales(exposure_times, flat_exposure_times, views, flats)
    fields, field_record = flat_fields(
        flats, flat_positions, darks, white_level, (rows, cols), epsilon, flat_scales=flat_scales,
        dark_source=dark_source,
    )  # fmt: skip
    acquisition_fields, acquisition_counts = fields, counts
    if band != slice(0, rows):
        counts = (
            counts.window(band, slice(None)) if hasattr(counts, "window") else counts[:, band, :]
        )
        fields = fields.window(band)
        rows = counts.shape[1]
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
    limit = np.float32(np.inf if linear_limit is None else linear_limit[0])
    settings = {"epsilon": float(epsilon), "factor": binning, "limit": limit}
    with discard_failed_outputs() as outputs:
        slab = _row_slab(counts, ordered, views_per_batch, binning)
        if slab is not None:
            out, tally = _correct_row_slabs(
                counts,
                fields,
                interpolation,
                tuple(line),
                slab=slab,
                allocate=outputs.allocate,
                transmission_steps=transmission_steps,
                **settings,
            )
            kept, tail_records, head = np.arange(views), (), line
        else:
            # Binned on the device while the batches come back, unless the tail needs full frames.
            factor = 1 if tail or transmission_tail else binning
            out = outputs.allocate((views, rows // factor, cols // factor))
            batch = views_per_batch or frame_batch_size(
                views, rows, cols, (*counts_steps, *transmission_steps, *head)
            )
            tally = correct_view_batches(
                counts, out, fields, interpolation, batch, counts_steps=counts_steps,
                transmission_steps=tuple(transmission_head),
                line_steps=() if transmission_tail else tuple(head),
                epsilon=float(epsilon), factor=factor, take_log=not transmission_tail, limit=limit,
            )  # fmt: skip
            if transmission_tail:
                out, _, _ = apply_steps(out, transmission_tail, views_per_batch)
                above = tally[2]
                out, tally = log_batches(out, tuple(head), epsilon, views_per_batch)
                tally = np.append(tally, above)
            out, kept, tail_records = apply_steps(out, tail, views_per_batch)
            if (tail or transmission_tail) and binning > 1:
                out = _binned(out, binning, views_per_batch, outputs.allocate)
        if linear_limit is not None:
            report_linear_limit(
                field_record[0],
                linear_limit,
                acquisition_fields,
                view_scales,
                float(tally[2]) / counts_size(counts),
            )
        found: dict[str, Json] = {"nonpositive_transmission": int(tally[0])}
        log = Correction("log", {"epsilon": float(epsilon)}, found)
        records = tuple(
            [s.record() for s in counts_steps] + list(field_record)
            + [s.record() for s in transmission_steps] + [log]
            + [s.record() for s in head] + list(tail_records)
        )  # fmt: skip
        if output_rows is not None:
            out = np.array(out[:, selected], copy=True)
        if progress.previewing():
            with progress.task("correction", "central sinograms", None, ""):
                progress.preview(
                    lambda: correction_images(
                        acquisition_counts,
                        out,
                        kept,
                        binning,
                        "counts",
                        first_output_row=0 if output_rows is None else output_rows[0],
                    ),
                    final=True,
                )
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
    projections: Any, steps: Sequence[Step], *, views_per_batch: int | None = None
) -> Corrected:
    """Correct a line-integral stack and record the steps applied.

    Parameters
    ----------
    projections : array-like
        Line integrals with shape ``(views, rows, columns)``; copied to float32.
    steps : sequence of Step
        Corrections acting on line integrals, in the order to run them.
    views_per_batch : int or None, optional
        Positive view-batch size. None selects a memory-bounded size automatically.

    Returns
    -------
    Corrected
        Corrected host line integrals, correction records and kept input-view indices.

    Raises
    ------
    ValueError
        If a step does not act on line integrals or views_per_batch is invalid.
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
    progress.check_cancelled()
    views_per_batch = validated_views_per_batch(views_per_batch)
    ordered = ordered_steps(steps)
    check_scan_steps(steps)
    out = copy_projections(projections, steps)
    out, kept, records = apply_steps(out, [s for _, s in ordered], views_per_batch)
    if steps and progress.previewing():
        with progress.task("correction", "central sinograms", None, ""):
            progress.preview(
                lambda: correction_images(projections, out, kept, 1, "line_integrals"), final=True
            )
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


def exposure_scales(
    sample: np.ndarray | float | None, flat: np.ndarray | float | None, views: int, flats: Any
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert measured durations to count-rate factors; white levels stay in counts."""
    if flats is None or (sample is None and flat is None):
        return np.ones(views, np.float32), None
    if sample is None or flat is None:
        raise ValueError(
            "Exposure correction needs both exposure_times=... and flat_exposure_times=..."
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
    columns: jax.Array | None = None,
    take_log: bool = True,
    limit: jax.Array | float = jnp.inf,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x = raw.astype(jnp.float32)
    # Recorded counts above the camera's stated linear range: counted, never changed.
    above = jnp.sum(x > limit)
    for step in counts_steps:
        x = step.apply(x)
    weight = t[:, None, None]
    flat = (1 - weight) * flats[lo] + weight * flats[hi]
    transmission = (x - dark) * exposure_scale[:, None, None] / jnp.maximum(flat, epsilon)
    # An unresolved calibration is not a measured transmission. Fill interior
    # gaps before any transmission step can spread the ratio of two noise values.
    transmission = jnp.where(usable, transmission, 1.0)
    if fill_gaps:
        transmission = interpolate_unmeasured(transmission, usable, columns)
    for step in transmission_steps:
        transmission = step.apply(transmission)
    if not take_log:
        return transmission, jnp.asarray(0), jnp.asarray(0), above
    return (
        *_log_transmission(transmission, epsilon=epsilon, line_steps=line_steps, factor=factor),
        above,
    )


@partial(jax.jit, static_argnames=("epsilon", "line_steps", "factor"))
def _log_transmission(
    transmission: jax.Array,
    *,
    epsilon: float,
    line_steps: tuple[Step, ...] = (),
    factor: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Floor and log transmission, then run the phase prefix and bin its result.

    Non-finite results are counted, not replaced: an unknown measurement is not
    zero absorption, and the batch loops reject them.
    """
    nonpositive = jnp.sum(transmission <= 0)
    p = -jnp.log(jnp.maximum(transmission, epsilon))
    for step in line_steps:
        p = step.apply(p)
    bad = jnp.sum(~jnp.isfinite(p))
    return (mean_detector_pixels(p, factor) if factor > 1 else p), nonpositive, bad


def correct_view_batches(
    counts: Any,
    out: np.ndarray,
    fields: Fields,
    interpolation: tuple[np.ndarray, ...],
    batch: int,
    **settings: Any,
) -> np.ndarray:
    """Run ``first_pass`` over batches of views into ``out``; the tally of what it found."""
    dark, usable = jax.device_put(fields.dark), jax.device_put(fields.usable)
    resident = jax.device_put(fields.flats) if fields.flats.nbytes <= _SIMPLE_FRAME_BYTES else None
    lo, hi, t, exposure_scale = interpolation
    tally = np.zeros(3, np.int64)  # non-positive transmission, non-finite, above linear limit
    fill_gaps = not bool(fields.usable.all())

    def collect(start: int, stop: int, result: jax.Array, *bad: jax.Array) -> None:
        progress.check_cancelled()
        out[start:stop] = copies.result(result)
        release_planes(out, start, stop)
        tally[:] += [int(b) for b in bad]
        reject_nonfinite(int(bad[1]) if bad else 0, start, stop)
        progress.step("view batches")

    views = counts.shape[0]
    load = progress.for_unit("views")
    parts = (
        *settings.get("counts_steps", ()),
        "flat_dark",
        *settings.get("transmission_steps", ()),
        *(("log",) if settings["take_log"] else ()),
        *settings.get("line_steps", ()),
        *(("binned",) if settings.get("factor", 1) > 1 else ()),
    )
    with (
        correction_group(parts, range(0, views, batch), "view batches"),
        HostCopies() as copies,
        closing(view_batches(counts, views, batch)) as batches,
    ):
        pending = None
        for start, stop, raw in batches:
            if load is not None:
                load.advance(stop, {})
            progress.check_cancelled()
            with progress.preparing("correct view batch", start):
                flats, a, b = batch_fields(fields, resident, lo[start:stop], hi[start:stop])
                done = first_pass(
                    jnp.asarray(raw), a, b, t[start:stop],
                    exposure_scale[start:stop], flats, dark, usable,
                    fill_gaps=fill_gaps,
                    **settings,
                )  # fmt: skip
            copies.start(done[0])
            if pending is not None:  # the last batch comes back while this one runs
                collect(*pending)
            pending = (start, stop, *done)
        if pending is not None:
            collect(*pending)
    return tally


def counts_size(counts: Any) -> int:
    """How many recorded counts a stack holds."""
    return int(np.prod(counts.shape))


def linear_limit_edge(limit: float, fields: Fields, view_scales: np.ndarray) -> float:
    """The median line integral of a count at ``limit``: -log((limit - dark) x scale / flat)."""
    from tomojax.core.host_volume import host_array

    values = host_array((len(fields.flats), int(fields.usable.sum())), dtype=np.float64)
    numerator = (limit - fields.dark[fields.usable]) * float(np.median(view_scales))
    for row, flat in zip(values, fields.flats, strict=True):
        ratio = numerator / flat[fields.usable].astype(np.float64)
        row[:] = -np.log(np.maximum(ratio, np.finfo(np.float64).tiny))
    return float(np.median(values.reshape(-1), overwrite_input=True))


def report_linear_limit(
    record: Correction,
    linear_limit: tuple[float, str],
    fields: Fields,
    view_scales: np.ndarray,
    fraction: float,
) -> None:
    """Record the camera's stated linear range and what lies above it; warn when anything does.

    The line integral where the range ends is the median over measured pixels of
    -log((limit - dark) x exposure scale / flat): smaller line integrals come from counts
    above the range, so they are unreliable, and saturated air reads as absorbing.
    """
    limit, source = linear_limit
    edge = linear_limit_edge(limit, fields, view_scales)
    record.settings["linear_limit"] = float(limit)
    record.settings["linear_limit_source"] = source
    record.found["counts_above_linear_limit"] = fraction
    record.found["linear_limit_line_integral"] = edge
    if fraction > 0:
        LOG.warning(
            "%.1f%% of the recorded counts lie above the detector's stated linear range "
            "(%g counts, %s): line integrals below about %.3f are unreliable, and saturated "
            "air reads as absorbing.",
            100 * fraction, limit, source, edge,
        )  # fmt: skip


def log_batches(
    out: np.ndarray, steps: tuple[Step, ...], epsilon: float, views_per_batch: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Replace a host transmission stack by line integrals, counting log findings."""
    views, rows, cols = out.shape
    batch = views_per_batch or frame_batch_size(views, rows, cols, steps)
    tally = np.zeros(2, np.int64)
    starts = range(0, views, batch)
    with correction_group(("log", *steps), starts, "view batches"):
        for first in starts:
            progress.check_cancelled()
            done, nonpositive, bad = _log_transmission(
                jnp.asarray(out[first : first + batch]), epsilon=epsilon, line_steps=steps
            )
            out[first : first + batch] = np.asarray(done)
            tally += (int(nonpositive), int(bad))
            reject_nonfinite(int(bad), first, min(first + batch, views))
            progress.step("view batches")
    return out, tally


def reject_nonfinite(count: int, start: int, stop: int) -> None:
    """Raise for NaN or infinite line integrals: they are not measurements of zero absorption."""
    if count:
        raise ValueError(
            f"{count} corrected line integrals in views {start}-{stop - 1} are not finite; "
            "check the raw counts, flats and darks, and any correction step for NaN or Inf"
        )


def _row_slab(
    counts: Any, ordered: Sequence[tuple[str, Step]], views_per_batch: int | None, factor: int
) -> int | None:
    """Rows per slab to fuse raw-count correction with whole-row steps.

    No step may look across rows, and the source must read a window of rows
    without decoding whole frames. The slab is a multiple of ``factor`` rows
    when the corrected line integrals are also binned.
    """
    if views_per_batch is not None or not ordered:
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
    allocate: Callable[[tuple[int, ...]], np.ndarray],
    **settings: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct and optionally bin slabs of whole rows: the corrected stack and the tally."""
    views, rows, cols = counts.shape
    factor = settings["factor"]
    out = allocate((views, rows // factor, cols // factor))
    dark, usable = jax.device_put(fields.dark), jax.device_put(fields.usable)
    flats = fields.flats
    lo, hi, t, exposure_scale = (jax.device_put(q) for q in interpolation)
    tally = np.zeros(3, np.int64)

    def collect(start: int, stop: int, result: jax.Array, *bad: jax.Array) -> None:
        progress.check_cancelled()
        out[:, start // factor : stop // factor] = copies.result(result)
        tally[:] += [int(b) for b in bad]
        reject_nonfinite(int(bad[1]), 0, views)
        progress.step("row batches")

    def read(start: int, stop: int) -> np.ndarray:
        frames = (
            counts[:, start:stop]
            if isinstance(counts, np.ndarray)
            else counts.window(slice(start, stop), slice(None))[:]
        )
        return np.asarray(frames).astype(frames.dtype.newbyteorder("="), copy=False)

    device = jax.config.jax_default_device
    cpu = (device.platform if device is not None else jax.default_backend()) == "cpu"

    def run(
        start: int, stop: int, raw: np.ndarray
    ) -> tuple[int, int, jax.Array, jax.Array, jax.Array, jax.Array]:
        with jax.default_device(device):
            done = first_pass(
                jnp.asarray(raw), lo, hi, t, exposure_scale,
                flats[:, start:stop], dark[start:stop],
                usable[start:stop], line_steps=steps,
                fill_gaps=not bool(fields.usable[start:stop].all()), **settings,
            )  # fmt: skip
            copies.start(done[0])
            if cpu:
                jax.block_until_ready(done)
            return start, stop, *done

    def now(
        function, *args: Any
    ) -> Future[tuple[int, int, jax.Array, jax.Array, jax.Array, jax.Array]]:
        done = Future()
        done.set_result(function(*args))
        return done

    pending = None
    # CPU calls need workers; accelerators already overlap queued work and host copies.
    starts = range(0, rows - rows % factor, slab)
    parts = (
        "flat_dark",
        *settings["transmission_steps"],
        "log",
        *steps,
        *(("binned",) if factor > 1 else ()),
    )
    load = progress.for_unit("views")
    with (
        correction_group(parts, starts, "row batches"),
        HostCopies() as copies,
        ThreadPoolExecutor(max_workers=2) if cpu else nullcontext() as pool,
    ):
        submit = now if pool is None else pool.submit
        for start in starts:
            progress.check_cancelled()
            stop = min(start + slab, rows - rows % factor)
            # Lazy sources may share caches: read serially, with at most two bands live.
            raw = read(start, stop)
            if load is not None and stop == rows - rows % factor:
                load.advance(views, {})
                progress.check_cancelled()
            with progress.preparing("correct detector rows", start):
                done = submit(run, start, stop, raw)

            if pending is not None:
                collect(*pending.result())
            pending = done
        if pending is not None:
            collect(*pending.result())
    return out, tally


def _binned(
    out: np.ndarray,
    factor: int,
    views_per_batch: int | None,
    allocate: Callable[[tuple[int, ...]], np.ndarray],
) -> np.ndarray:
    """The detector block means of ``out``, a batch of views at a time."""
    views, rows, cols = out.shape
    batch = views_per_batch or max(1, min(views, _BATCH_BYTES // (4 * rows * cols)))
    binned = allocate((views, rows // factor, cols // factor))
    starts = range(0, views, batch)
    with progress.task("correction", "binned", len(starts), "view batches"):
        progress.placed(("cpu:0",))
        for start in starts:
            progress.check_cancelled()
            binned[start : start + batch] = mean_detector_pixels(out[start : start + batch], factor)
            progress.step("view batches")
    return binned


def _split_at_break(steps: list[Step]) -> tuple[list[Step], list[Step]]:
    """The steps before the first that needs whole rows or selects views, and the rest."""
    for i, step in enumerate(steps):
        if step.whole_rows or step.selects_views:
            return steps[:i], steps[i:]
    return steps, []


def apply_steps(
    out: np.ndarray, steps: Sequence[Step], views_per_batch: int | None
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
            batch = views_per_batch or frame_batch_size(views, rows, cols, (step,))
            starts = range(0, views, batch)
            with correction_group((step,), starts, "view batches"):
                stats = []
                for v0 in starts:
                    progress.check_cancelled()
                    stats.append(np.asarray(_statistic(step, jnp.asarray(out[v0 : v0 + batch]))))
                    progress.step("view batches")
                keep, found = step.select(np.concatenate(stats), kept)
                if not keep.any():
                    raise ValueError(f"{type(step).__name__} would reject every view")
                if not keep.all():
                    out, kept = out[keep], kept[keep]
                    # Later steps planned from the geometry see only the retained views.
                    retained = np.flatnonzero(keep)
                    steps = (*steps[: i + 1], *(s.retained(retained) for s in steps[i + 1 :]))
                records.append(Correction(step.record().name, step.record().settings, found))
            i += 1
        else:
            j = i
            while j < len(steps) and not (steps[j].whole_rows or steps[j].selects_views):
                j += 1
            run = tuple(steps[i:j])
            batch = views_per_batch or frame_batch_size(views, rows, cols, run)
            starts = range(0, views, batch)
            with correction_group(run, starts, "view batches"):
                for v0 in starts:
                    progress.check_cancelled()
                    out[v0 : v0 + batch] = np.asarray(
                        _apply(run, jnp.asarray(out[v0 : v0 + batch]))
                    )
                    release_planes(out, v0, v0 + batch)
                    progress.step("view batches")
            records.extend(s.record() for s in run)
            i = j
    return out, kept, tuple(records)


def _whole_rows(out: np.ndarray, step: Step) -> None:
    """Correct row slabs with concurrent CPU calls or asynchronous accelerator transfers."""
    views, rows, cols = out.shape
    slab = max(1, min(rows, _BATCH_BYTES // (8 * views * cols)))  # two slabs are in flight
    device = jax.config.jax_default_device

    def run(r0: int) -> None:
        # JAX's placement context is thread-local; it does not follow a worker task.
        with jax.default_device(device):
            out[:, r0 : r0 + slab] = np.asarray(
                _apply((step,), jnp.asarray(out[:, r0 : r0 + slab]))
            )

    starts = range(0, rows, slab)
    with correction_group((step,), starts, "row batches"), HostCopies() as copies:
        if (device.platform if device is not None else jax.default_backend()) == "cpu":
            # Only integer offsets are queued. Each worker owns at most one slab's arrays.
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = None
                for r0 in starts:
                    progress.check_cancelled()
                    launched = pool.submit(run, r0)
                    if pending is not None:
                        pending.result()
                        progress.step("row batches")
                    pending = launched
                if pending is not None:
                    progress.check_cancelled()
                    pending.result()
                    progress.step("row batches")
            return
        # Keep accelerator work asynchronous: workers add workspace, not useful parallelism.
        pending = None
        for r0 in starts:
            progress.check_cancelled()
            done = _apply((step,), jnp.asarray(out[:, r0 : r0 + slab]))
            copies.start(done)
            if pending is not None:
                out[:, pending[0] : pending[0] + slab] = copies.result(pending[1])
                progress.step("row batches")
            pending = (r0, done)
        if pending is not None:
            progress.check_cancelled()
            out[:, pending[0] : pending[0] + slab] = copies.result(pending[1])
            progress.step("row batches")


@partial(jax.jit, static_argnames=("step",))
def _statistic(step: Step, batch: jax.Array) -> jax.Array:
    return step.statistic(batch)


@partial(jax.jit, static_argnames=("steps",))
def _apply(steps: tuple[Step, ...], batch: jax.Array) -> jax.Array:
    for step in steps:
        batch = step.apply(batch)
    return batch
