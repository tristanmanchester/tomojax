"""Running corrections: batches of views through the device, slabs of rows for whole-row steps.

Frames are corrected in one compiled pass over batches of views: the counts
steps, then ``(I - D) / (F - D)`` with each view's flat interpolated between
the flat sets around it, the transmission steps, ``-log``, and the line-integral
steps up to the first that needs whole detector rows. Such a step, and every
step after it, then runs over the result: a whole-row step on slabs of rows
(every view), the others on batches of views again. Batches are read from the
host (NumPy, memmap or a lazily read file) while the device works on the last.

When only whole-row steps follow the ``-log``, they can share the first pass
on slabs read from a row-efficient source. With ``binning`` the detector means
are taken on the device, so no unbinned stack reaches the host in that path.
Both paths materialise the final corrected host stack; fusion avoids the
intermediate host/device round trip, not the final stack's memory cost.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
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
    from collections.abc import Callable, Iterator, Sequence

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
    epsilon = _positive_finite(epsilon, "epsilon")
    if isinstance(binning, bool) or not isinstance(binning, int | np.integer) or binning < 1:
        raise ValueError(f"binning needs a whole factor of at least 1, not {binning!r}")
    binning = int(binning)
    views, rows, cols = (int(s) for s in counts.shape)
    ordered = _ordered(steps)
    fields, field_record = _flat_fields(
        flats, flat_positions, darks, white_level, (rows, cols), epsilon
    )
    at = (
        np.arange(views) + 0.5 if view_positions is None else np.asarray(view_positions, np.float64)
    )
    if at.shape != (views,) or not np.isfinite(at).all():
        raise ValueError(f"view_positions needs one finite position for each of the {views} views")
    interpolation = _interpolation(at, fields.positions)
    line = [s for d, s in ordered if d == "line integrals"]
    head, tail = _split_at_break(line)
    counts_steps = tuple(s for d, s in ordered if d == "counts")
    transmission_steps = tuple(s for d, s in ordered if d == "transmission")
    settings = {"epsilon": float(epsilon), "factor": binning}
    slab = _row_slab(counts, ordered, batch_views, binning)
    if slab is not None:
        out, tally = _correct_row_slabs(
            counts, fields, interpolation, tuple(line), slab=slab, **settings
        )
        kept, tail_records, head = np.arange(views), (), line
    else:
        # Binned on the device while the batches come back, unless the tail needs full frames.
        factor = 1 if tail else binning
        out = np.empty((views, rows // factor, cols // factor), np.float32)
        batch = batch_views or _frame_batch_size(
            views, rows, cols, (*counts_steps, *transmission_steps, *head)
        )
        tally = _correct_view_batches(
            counts, out, fields, interpolation, batch, counts_steps=counts_steps,
            transmission_steps=transmission_steps, line_steps=tuple(head),
            epsilon=float(epsilon), factor=factor,
        )  # fmt: skip
        out, kept, tail_records = _run_rest(out, tail, batch_views)
        if tail and binning > 1:
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
    ordered = _ordered(steps)
    if any(d != "line integrals" for d, _ in ordered):
        wrong = [type(s).__name__ for d, s in ordered if d != "line integrals"]
        raise ValueError(
            f"{', '.join(wrong)} act on detector counts or transmission, which a Scan of line "
            "integrals no longer holds: load the frames with tomojax.load_frames and pass "
            "them to Frames.corrected"
        )
    out = np.array(projections, np.float32, copy=True)
    out, kept, records = _run_rest(out, [s for _, s in ordered], batch_views)
    return Corrected(out, records, kept)


def _frame_batch_size(views: int, rows: int, cols: int, steps: Sequence[Step]) -> int:
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


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, not {value!r}")
    return value


def _ordered(steps: Sequence[Step]) -> list[tuple[str, Step]]:
    """``steps`` by domain (counts, transmission, line integrals), in order within each."""
    for step in steps:
        if not isinstance(step, Step):
            raise TypeError(
                f"corrections take tomojax.corrections steps, not {type(step).__name__}"
            )
        if step.domain not in DOMAINS:
            raise ValueError(f"{type(step).__name__} has an unknown domain {step.domain!r}")
        if (step.whole_rows or step.selects_views) and step.domain != "line integrals":
            raise ValueError(
                f"{type(step).__name__}: steps on whole rows or selecting views act on "
                "line integrals only"
            )
    return [(d, s) for d in DOMAINS for s in steps if s.domain == d]


class _Fields:
    """The flat sets (``(sets, rows, columns)``), their positions and the dark field.

    ``usable`` (``(rows, columns)``) marks the pixels where every set's flat stands clear
    of the dark.
    """

    def __init__(
        self, flats: np.ndarray, positions: np.ndarray, dark: np.ndarray, usable: np.ndarray
    ) -> None:
        self.flats, self.positions, self.dark, self.usable = flats, positions, dark, usable


def _flat_fields(
    flats: Any,
    positions: np.ndarray | None,
    darks: Any,
    white_level: float | None,
    shape: tuple[int, int],
    epsilon: float,
) -> tuple[_Fields, tuple[Correction, ...]]:
    """The mean flat of each set, the mean dark, and the record of them.

    A pixel is usable where the flat exceeds the dark by at least ``epsilon`` and
    three standard errors of the two calibration means: elsewhere (outside the
    beam, or dead) the ratio is noise, and ``_first_pass`` takes it as no absorption.
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
        level = _positive_finite(white_level, "white_level")
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
            _moments(flats, shape, "flats", None if len(sets) == 1 else np.flatnonzero(where == p))
            for p in sets
        ]
        stack, variance = (np.stack(part) for part in zip(*moments, strict=True))
        settings = {"flats": len(flats), "darks": dark_frames, "flat_sets": len(sets)}
        if len(sets) > 1:
            settings["flat_positions"] = [float(p) for p in sets]
    light = stack - dark
    floor = np.maximum(epsilon, 3 * np.sqrt(variance + dark_variance))
    usable = cast("np.ndarray", (np.isfinite(light) & (light > floor)).all(axis=0))
    found: dict[str, Json] = {"pixels_without_light": int(np.count_nonzero(~usable))}
    fields = _Fields(stack.astype(np.float32), sets, dark.astype(np.float32), usable)
    return fields, (Correction("flat_dark", settings, found),)


def _check_stack(frames: Any, shape: tuple[int, int], name: str) -> None:
    if len(frames.shape) != 3 or tuple(frames.shape[1:]) != shape or not len(frames):
        raise ValueError(
            f"{name} are {tuple(frames.shape)}; a nonempty stack of frames of {shape} is needed"
        )


def _moments(
    frames: Any, shape: tuple[int, int], name: str, indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The mean of frames and the variance of that mean, in float64 over bounded batches."""
    _check_stack(frames, shape, name)
    count = len(frames) if indices is None else len(indices)
    batch = max(1, (32 << 20) // (8 * shape[0] * shape[1]))
    total, squares = np.zeros(shape), np.zeros(shape)
    for start in range(0, count, batch):
        which = slice(start, start + batch) if indices is None else indices[start : start + batch]
        values = np.asarray(frames[which], np.float64)
        total += values.sum(axis=0)
        squares += np.einsum("ijk,ijk->jk", values, values)
    mean = total / count
    if count == 1:
        return mean, np.zeros(shape)
    return mean, np.maximum(0.0, squares - total * mean) / (count * (count - 1))


def _interpolation(q: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, ...]:
    """For each view at ``q``, the flat sets either side of it and the weight of the later one.

    ``positions`` are the views recorded before each set; views before the
    first set or after the last take that set alone.
    """
    hi = np.clip(np.searchsorted(positions, q, side="right"), 0, len(positions) - 1)
    lo = np.clip(hi - 1, 0, len(positions) - 1)
    span = positions[hi] - positions[lo]
    t = np.where(span > 0, (q - positions[lo]) / np.where(span > 0, span, 1), 0.0)
    t = np.where(q >= positions[-1], 1.0, np.clip(t, 0.0, 1.0)) if len(positions) > 1 else t
    return lo.astype(np.int32), hi.astype(np.int32), t.astype(np.float32)


@partial(
    jax.jit,
    static_argnames=("epsilon", "counts_steps", "transmission_steps", "line_steps", "factor"),
)
def _first_pass(
    raw: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    t: jax.Array,
    flats: jax.Array,
    dark: jax.Array,
    usable: jax.Array,
    *,
    epsilon: float,
    counts_steps: tuple[Step, ...] = (),
    transmission_steps: tuple[Step, ...] = (),
    line_steps: tuple[Step, ...] = (),
    factor: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = raw.astype(jnp.float32)
    for step in counts_steps:
        x = step.apply(x)
    weight = t[:, None, None]
    flat = (1 - weight) * flats[lo] + weight * flats[hi]
    transmission = (x - dark) / jnp.maximum(flat - dark, epsilon)
    # No light is no measurement: take no absorption, rather than spread the ratio
    # of two noise values through the steps that follow.
    transmission = jnp.where(usable, transmission, 1.0)
    for step in transmission_steps:
        transmission = step.apply(transmission)
    nonpositive = jnp.sum(transmission <= 0)
    p = -jnp.log(jnp.maximum(transmission, epsilon))
    for step in line_steps:
        p = step.apply(p)
    bad = ~jnp.isfinite(p)
    p = jnp.where(bad, 0.0, p)
    return (mean_detector_pixels(p, factor) if factor > 1 else p), nonpositive, jnp.sum(bad)


def _correct_view_batches(
    counts: Any,
    out: np.ndarray,
    fields: _Fields,
    interpolation: tuple[np.ndarray, ...],
    batch: int,
    **settings: Any,
) -> np.ndarray:
    """Run ``_first_pass`` over batches of views into ``out``; the tally of what it found."""
    flats, dark, usable = (jax.device_put(a) for a in (fields.flats, fields.dark, fields.usable))
    lo, hi, t = interpolation
    tally = np.zeros(2, np.int64)  # non-positive transmission, non-finite results

    def collect(start: int, stop: int, result: jax.Array, *bad: jax.Array) -> None:
        out[start:stop] = np.asarray(result)
        tally[:] += [int(b) for b in bad]

    pending = None
    for start, stop, raw in _batches(counts, counts.shape[0], batch):
        done = _first_pass(
            jnp.asarray(raw), lo[start:stop], hi[start:stop], t[start:stop], flats, dark, usable,
            **settings,
        )  # fmt: skip
        done[0].copy_to_host_async()
        if pending is not None:  # the last batch comes back while this one runs
            collect(*pending)
        pending = (start, stop, *done)
    if pending is not None:
        collect(*pending)
    return tally


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
    if any(d != "line integrals" or not s.whole_rows or s.selects_views for d, s in ordered):
        return None
    if not isinstance(counts, np.ndarray) and not getattr(counts, "_efficient_row_windows", False):
        return None
    views, rows, cols = counts.shape
    itemsize = max(4, np.dtype(counts.dtype).itemsize)
    slab = min(rows, _BATCH_BYTES // (2 * itemsize * views * cols)) // factor * factor
    return slab or None


def _correct_row_slabs(
    counts: Any,
    fields: _Fields,
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
    lo, hi, t = (jax.device_put(q) for q in interpolation)
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
        done = _first_pass(
            jnp.asarray(read(start, stop)), lo, hi, t, flats[:, start:stop], dark[start:stop],
            usable[start:stop], line_steps=steps, **settings,
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


def _run_rest(
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
            batch = batch_views or _frame_batch_size(views, rows, cols, (step,))
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
            batch = batch_views or _frame_batch_size(views, rows, cols, run)
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


@dataclass
class CorrectedRows:
    """Owned device projections and their records, completed only when requested."""

    projections: jax.Array
    flat: Correction
    steps: tuple[Step, ...]
    epsilon: float
    counts: tuple[jax.Array, jax.Array]
    raw: np.ndarray

    @property
    def records(self) -> tuple[Correction, ...]:
        """Correction records for these measured rows, including any sampling halo."""
        log = Correction(
            "log",
            {"epsilon": self.epsilon},
            {
                "nonpositive_transmission": int(self.counts[0]),
                "nonfinite_set_to_zero": int(self.counts[1]),
            },
        )
        return (self.flat, log, *(step.record() for step in self.steps))


class _RowReader:
    """One current raw block and one prefetched block; requests own their small copies."""

    def __init__(self, counts: Any) -> None:
        self.counts = counts
        views, rows, cols = counts.shape
        self.batch = max(1, min(rows, (1 << 30) // (views * cols * counts.dtype.itemsize)))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tomojax-raw-rows")
        self.current = self.future = None
        self.first = self.next_first = -1

    def _read(self, first: int) -> np.ndarray:
        stop = min(first + self.batch, self.counts.shape[1])
        return self.counts.window(slice(first, stop), slice(None))[:]

    def read(self, first: int, stop: int) -> np.ndarray:
        if isinstance(self.counts, np.ndarray):
            return np.ascontiguousarray(self.counts[:, first:stop])
        views, rows, cols = self.counts.shape
        output = np.empty((views, stop - first, cols), dtype=self.counts.dtype)
        at = first
        while at < stop:
            block = at // self.batch * self.batch
            if block != self.first:
                if block != self.next_first:
                    if self.future is not None and not self.future.cancel():
                        self.future.result()
                    self.future = self.pool.submit(self._read, block)
                assert self.future is not None
                self.current = self.future.result()
                self.first = block
                self.next_first = block + self.batch
                self.future = (
                    self.pool.submit(self._read, self.next_first)
                    if self.next_first < rows
                    else None
                )
            end = min(stop, block + self.batch)
            assert self.current is not None
            output[:, at - first : end - first] = self.current[:, at - block : end - block]
            at = end
        return output

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.current = self.future = None


@contextmanager
def corrected_row_source(
    counts: Any,
    *,
    flats: Any,
    flat_positions: np.ndarray | None,
    darks: Any,
    white_level: float | None,
    steps: Sequence[Step],
    epsilon: float = 1e-6,
    view_positions: np.ndarray | None = None,
) -> Iterator[Callable[[int, int], CorrectedRows]]:
    """Read corrected detector-row intervals without a materialized projection stack.

    Each request returns owned JAX projections, independent of later reads and
    context closure. Only whole-row line-integral steps without view selection
    are supported; operations requiring neighboring rows must run on full frames.
    Raw inputs must be NumPy arrays or support lazy detector windows. The source
    and calibration arrays must remain unchanged until this context closes.

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
    epsilon : float, optional
        Positive transmission floor before the logarithm. Default is 1e-6.
    view_positions : numpy.ndarray or None, optional
        Position of each view among the flats. Defaults to ``i + 0.5`` for view i.

    Yields
    ------
    callable
        Reader accepting a half-open row interval ``(first, stop)`` and returning
        corrected rows with their correction records. Use within a ``with`` block.

    Raises
    ------
    ValueError
        If steps require full frames or select views, or calibration or positions
        are invalid.
    TypeError
        If counts lack detector-window access or a correction is not a Step.
    """
    ordered = _ordered(steps)
    if any(
        domain != "line integrals" or not step.whole_rows or step.selects_views
        for domain, step in ordered
    ):
        raise ValueError(
            "streamed corrections require whole-row line-integral steps without view selection"
        )
    if not isinstance(counts, np.ndarray) and not hasattr(counts, "window"):
        raise TypeError("streamed counts must be NumPy arrays or support detector windows")
    epsilon = _positive_finite(epsilon, "epsilon")
    views, rows, cols = counts.shape
    fields, records = _flat_fields(flats, flat_positions, darks, white_level, (rows, cols), epsilon)
    at = (
        np.arange(views) + 0.5 if view_positions is None else np.asarray(view_positions, np.float64)
    )
    if at.shape != (views,) or not np.isfinite(at).all():
        raise ValueError(f"view_positions needs one finite position for each of the {views} views")
    interpolation = tuple(jax.device_put(value) for value in _interpolation(at, fields.positions))
    reader = _RowReader(counts)
    line_steps = tuple(steps)

    def correct(first: int, stop: int) -> CorrectedRows:
        raw = reader.read(first, stop)
        raw = raw.astype(raw.dtype.newbyteorder("="), copy=False)
        selected = (fields.flats[:, first:stop], fields.dark[first:stop], fields.usable[first:stop])
        done = _first_pass(
            jax.device_put(raw),
            *interpolation,
            *(jax.device_put(np.ascontiguousarray(value)) for value in selected),
            epsilon=epsilon,
            line_steps=line_steps,
        )
        flat = Correction(
            "flat_dark",
            records[0].settings,
            {"pixels_without_light": int(np.count_nonzero(~selected[2]))},
        )
        return CorrectedRows(done[0], flat, line_steps, epsilon, (done[1], done[2]), raw)

    try:
        yield correct
    finally:
        reader.close()
