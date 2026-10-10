"""Running corrections: batches of views through the device, slabs of rows for whole-row steps.

Frames are corrected in one compiled pass over batches of views: the counts
steps, then ``(I - D) / (F - D)`` with each view's flat interpolated between
the flat sets around it, the transmission steps, ``-log``, and the line-integral
steps up to the first that needs whole detector rows. Such a step, and every
step after it, then runs over the result: a whole-row step on slabs of rows
(every view), the others on batches of views again. Batches are read from the
host (NumPy, memmap or a lazily read file) while the device works on the last.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._records import Correction, Json
from ._steps import DOMAINS, Step

if TYPE_CHECKING:
    from collections.abc import Sequence

LOG = logging.getLogger(__name__)

# Host memory a batch of views (or slab of rows) may take, as float32.
_BATCH_BYTES = 256 * 2**20


def correct_frames(
    counts: Any,
    *,
    flats: np.ndarray | None,
    flat_positions: np.ndarray | None,
    darks: np.ndarray | None,
    white_level: float | None,
    steps: Sequence[Step],
    epsilon: float,
    view_positions: np.ndarray | None = None,
    batch_views: int | None = None,
) -> Corrected:
    """Line integrals from ``counts`` ``(views, rows, columns)``, and the records of how.

    ``flats`` are flat frames and ``flat_positions`` the number of views
    recorded before each (frames at one position are one set; None, one set);
    ``white_level`` stands in for flats on scanners that record it instead.
    ``darks`` default to zero. ``view_positions`` places each view among the
    flats (by default view ``i`` at ``i + 1/2``).
    """
    batch_views = _validated_batch_views(batch_views)
    epsilon = _positive_finite(epsilon, "epsilon")
    views, rows, cols = (int(s) for s in counts.shape)
    ordered = _ordered(steps)
    fields, field_record = _flat_fields(flats, flat_positions, darks, white_level, (rows, cols))
    at = (
        np.arange(views) + 0.5 if view_positions is None else np.asarray(view_positions, np.float64)
    )
    if at.shape != (views,) or not np.isfinite(at).all():
        raise ValueError(f"view_positions needs one finite position for each of the {views} views")
    lo, hi, t = _interpolation(at, fields.positions)
    line = [s for d, s in ordered if d == "line integrals"]
    head, tail = _split_at_break(line)
    counts_steps = tuple(s for d, s in ordered if d == "counts")
    transmission_steps = tuple(s for d, s in ordered if d == "transmission")
    out = np.empty((views, rows, cols), np.float32)
    flat_stack, dark = jax.device_put(fields.flats), jax.device_put(fields.dark)
    batch = batch_views or max(1, min(views, _BATCH_BYTES // (4 * rows * cols)))
    tally = np.zeros(2, np.int64)  # non-positive transmission, non-finite results

    def collect(start: int, stop: int, result: jax.Array, *bad: jax.Array) -> None:
        out[start:stop] = np.asarray(result)
        tally[:] += [int(b) for b in bad]

    pending = None
    for start, stop, raw in _batches(counts, views, batch):
        done = _first_pass(
            jnp.asarray(raw), lo[start:stop], hi[start:stop], t[start:stop], flat_stack, dark,
            epsilon=float(epsilon), counts_steps=counts_steps,
            transmission_steps=transmission_steps, line_steps=tuple(head),
        )  # fmt: skip
        done[0].copy_to_host_async()
        if pending is not None:  # the last batch comes back while this one runs
            collect(*pending)
        pending = (start, stop, *done)
    if pending is not None:
        collect(*pending)
    found: dict[str, Json] = {
        "nonpositive_transmission": int(tally[0]),
        "nonfinite_set_to_zero": int(tally[1]),
    }
    log = Correction("log", {"epsilon": float(epsilon)}, found)
    out, kept, tail_records = _run_rest(out, tail, batch_views)
    records = tuple(
        [s.record() for s in counts_steps] + list(field_record)
        + [s.record() for s in transmission_steps] + [log]
        + [s.record() for s in head] + list(tail_records)
    )  # fmt: skip
    return Corrected(out, records, kept)


class Corrected(NamedTuple):
    """Corrected line integrals, the records of each step, and the input views they keep."""

    projections: np.ndarray
    records: tuple[Correction, ...]
    kept: np.ndarray  # indices of the input views, increasing


def correct_projections(
    projections: Any, steps: Sequence[Step], *, batch_views: int | None = None
) -> Corrected:
    """``projections`` (line integrals) through ``steps``, and their records."""
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
    """The flat sets (``(sets, rows, columns)``), their positions and the dark field."""

    def __init__(self, flats: np.ndarray, positions: np.ndarray, dark: np.ndarray) -> None:
        self.flats, self.positions, self.dark = flats, positions, dark


def _flat_fields(
    flats: np.ndarray | None,
    positions: np.ndarray | None,
    darks: np.ndarray | None,
    white_level: float | None,
    shape: tuple[int, int],
) -> tuple[_Fields, tuple[Correction, ...]]:
    """The mean flat of each set, the mean dark, and the record of them."""
    if darks is None and flats is not None:
        LOG.warning("no dark fields: correcting with a dark field of zero")
    dark = np.zeros(shape, np.float32) if darks is None else _mean(darks, shape, "darks")
    dark_frames = 0 if darks is None else len(darks)
    if flats is None:
        if white_level is None:
            raise ValueError(
                "no flat fields: pass flats (frames of the beam without the object) or "
                "white_level (the counts of an unattenuated pixel) to tomojax.load_frames"
            )
        level = _positive_finite(white_level, "white_level")
        stack = np.full((1, *shape), level, np.float32)
        record = Correction("flat_dark", {"white_level": level, "darks": dark_frames})
        return _Fields(stack, np.zeros(1), dark), (record,)
    flats = np.asarray(flats)
    if flats.ndim != 3 or flats.shape[1:] != shape or not len(flats):
        raise ValueError(
            f"flats are {flats.shape}; a nonempty stack of frames of {shape} is needed"
        )
    where = np.zeros(len(flats)) if positions is None else np.asarray(positions, np.float64)
    if where.shape != (len(flats),) or not np.isfinite(where).all():
        raise ValueError(
            f"flat_positions needs one finite position for each of the {len(flats)} flats"
        )
    sets = np.unique(where)
    stack = np.stack([np.asarray(flats[where == p], np.float64).mean(axis=0) for p in sets])
    settings: dict[str, Any] = {"flats": len(flats), "darks": dark_frames, "flat_sets": len(sets)}
    if len(sets) > 1:
        settings["flat_positions"] = [float(p) for p in sets]
    lit = stack - dark
    found: dict[str, Json] = {"pixels_without_light": int(np.count_nonzero(lit <= 0))}
    return _Fields(stack.astype(np.float32), sets, dark), (
        Correction("flat_dark", settings, found),
    )


def _mean(frames: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[1:] != shape or not len(frames):
        raise ValueError(
            f"{name} are {frames.shape}; a nonempty stack of frames of {shape} is needed"
        )
    return np.asarray(frames, np.float64).mean(axis=0).astype(np.float32)


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


@partial(jax.jit, static_argnames=("epsilon", "counts_steps", "transmission_steps", "line_steps"))
def _first_pass(
    raw: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    t: jax.Array,
    flats: jax.Array,
    dark: jax.Array,
    *,
    epsilon: float,
    counts_steps: tuple[Step, ...],
    transmission_steps: tuple[Step, ...],
    line_steps: tuple[Step, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = raw.astype(jnp.float32)
    for step in counts_steps:
        x = step.apply(x)
    weight = t[:, None, None]
    flat = (1 - weight) * flats[lo] + weight * flats[hi]
    transmission = (x - dark) / jnp.maximum(flat - dark, epsilon)
    nonpositive = jnp.sum(transmission <= 0)
    for step in transmission_steps:
        transmission = step.apply(transmission)
    p = -jnp.log(jnp.maximum(transmission, epsilon))
    for step in line_steps:
        p = step.apply(p)
    bad = ~jnp.isfinite(p)
    return jnp.where(bad, 0.0, p), nonpositive, jnp.sum(bad)


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
        batch = batch_views or max(1, min(views, _BATCH_BYTES // (4 * rows * cols)))
        if step.selects_views:
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
            continue
        if step.whole_rows:
            slab = max(1, min(rows, _BATCH_BYTES // (4 * views * cols)))
            for r0 in range(0, rows, slab):
                part = jnp.asarray(out[:, r0 : r0 + slab])
                out[:, r0 : r0 + slab] = np.asarray(_apply((step,), part))
            records.append(step.record())
            i += 1
            continue
        j = i
        while j < len(steps) and not (steps[j].whole_rows or steps[j].selects_views):
            j += 1
        run = tuple(steps[i:j])
        for v0 in range(0, views, batch):
            out[v0 : v0 + batch] = np.asarray(_apply(run, jnp.asarray(out[v0 : v0 + batch])))
        records.extend(s.record() for s in run)
        i = j
    return out, kept, tuple(records)


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
        return np.asarray(counts[start : min(start + batch, views)])

    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read, starts[0]) if starts else None
        for k, start in enumerate(starts):
            assert pending is not None
            frames = pending.result()
            if k + 1 < len(starts):
                pending = reader.submit(read, starts[k + 1])
            yield start, min(start + batch, views), frames
