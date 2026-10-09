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
from typing import TYPE_CHECKING, Any

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
    batch_views: int | None = None,
) -> tuple[np.ndarray, tuple[Correction, ...]]:
    """Line integrals from ``counts`` ``(views, rows, columns)``, and the records of how.

    ``flats`` are flat frames and ``flat_positions`` the number of views
    recorded before each (frames at one position are one set; None, one set);
    ``white_level`` stands in for flats on scanners that record it instead.
    ``darks`` default to zero.
    """
    views, rows, cols = (int(s) for s in counts.shape)
    ordered = _ordered(steps)
    fields, field_record = _flat_fields(flats, flat_positions, darks, white_level, (rows, cols))
    lo, hi, t = _interpolation(views, fields.positions)
    head, tail = _split_at_whole_rows([s for d, s in ordered if d == "line integrals"])
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
    records = tuple(
        [s.record() for s in counts_steps] + list(field_record)
        + [s.record() for s in transmission_steps] + [log]
        + [s.record() for d, s in ordered if d == "line integrals"]
    )  # fmt: skip
    _run_rest(out, tail, batch_views)
    return out, records


def correct_projections(
    projections: Any, steps: Sequence[Step], *, batch_views: int | None = None
) -> tuple[np.ndarray, tuple[Correction, ...]]:
    """``projections`` (line integrals) through ``steps``, and their records."""
    ordered = _ordered(steps)
    if any(d != "line integrals" for d, _ in ordered):
        wrong = [type(s).__name__ for d, s in ordered if d != "line integrals"]
        raise ValueError(
            f"{', '.join(wrong)} act on detector counts or transmission, which a Scan of line "
            "integrals no longer holds: load the frames with tomojax.load_frames and pass "
            "them to Frames.corrected"
        )
    out = np.array(projections, np.float32, copy=True)
    _run_rest(out, [s for _, s in ordered], batch_views)
    return out, tuple(s.record() for _, s in ordered)


def _ordered(steps: Sequence[Step]) -> list[tuple[str, Step]]:
    """``steps`` by domain (counts, transmission, line integrals), in order within each."""
    for step in steps:
        if not isinstance(step, Step):
            raise TypeError(
                f"corrections take tomojax.corrections steps, not {type(step).__name__}"
            )
        if step.domain not in DOMAINS:
            raise ValueError(f"{type(step).__name__} has an unknown domain {step.domain!r}")
        if step.whole_rows and step.domain != "line integrals":
            raise ValueError(f"{type(step).__name__}: whole-row steps act on line integrals only")
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
        level = float(white_level)
        stack = np.full((1, *shape), level, np.float32)
        record = Correction("flat_dark", {"white_level": level, "darks": dark_frames})
        return _Fields(stack, np.zeros(1), dark), (record,)
    flats = np.asarray(flats)
    if flats.ndim != 3 or flats.shape[1:] != shape:
        raise ValueError(f"flats are {flats.shape}; frames of {shape} are needed")
    where = np.zeros(len(flats)) if positions is None else np.asarray(positions, np.float64)
    if where.shape != (len(flats),):
        raise ValueError(f"flat_positions has {where.shape[0]} entries for {len(flats)} flats")
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
    if frames.ndim != 3 or frames.shape[1:] != shape:
        raise ValueError(f"{name} are {frames.shape}; frames of {shape} are needed")
    return np.asarray(frames, np.float64).mean(axis=0).astype(np.float32)


def _interpolation(views: int, positions: np.ndarray) -> tuple[np.ndarray, ...]:
    """For each view, the flat sets either side of it and the weight of the later one.

    View i sits at i + 1/2 among the positions (views recorded before each set);
    views before the first set or after the last take that set alone.
    """
    q = np.arange(views) + 0.5
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


def _split_at_whole_rows(steps: list[Step]) -> tuple[list[Step], list[Step]]:
    """The steps before the first whole-row step, and the rest."""
    for i, step in enumerate(steps):
        if step.whole_rows:
            return steps[:i], steps[i:]
    return steps, []


def _run_rest(out: np.ndarray, steps: Sequence[Step], batch_views: int | None) -> None:
    """``steps`` over ``out`` in place: whole-row steps on slabs of rows, the others on views."""
    views, rows, cols = out.shape
    i = 0
    while i < len(steps):
        if steps[i].whole_rows:
            slab = max(1, min(rows, _BATCH_BYTES // (4 * views * cols)))
            for r0 in range(0, rows, slab):
                part = jnp.asarray(out[:, r0 : r0 + slab])
                out[:, r0 : r0 + slab] = np.asarray(_apply((steps[i],), part))
            i += 1
            continue
        j = i
        while j < len(steps) and not steps[j].whole_rows:
            j += 1
        run = tuple(steps[i:j])
        batch = batch_views or max(1, min(views, _BATCH_BYTES // (4 * rows * cols)))
        for v0 in range(0, views, batch):
            out[v0 : v0 + batch] = np.asarray(_apply(run, jnp.asarray(out[v0 : v0 + batch])))
        i = j


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
