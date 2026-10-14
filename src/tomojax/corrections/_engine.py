"""Running corrections: batches of views through the device, slabs of rows for whole-row steps.

Frames are corrected in one compiled pass over batches of views: the counts
steps, then ``(I - D) / (F - D)`` with each view's flat interpolated between
the flat sets around it, the transmission steps, ``-log``, and the line-integral
steps up to the first that needs whole detector rows or selects views. A first
selecting step's summaries are collected while each batch is on the device;
the remaining steps then run over the result: a whole-row step on slabs of rows
(every view), the others on batches of views again. Batches are read from the
host (NumPy, memmap or a lazily read file) while the device works on the last.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax._typed_arrays import commit_array, mapped_storage
from tomojax.backends import api as backend_runtime
from tomojax.core.array_ops import mean_detector_pixels

from ._downloads import cuda_device_id, release_device, stage_download
from ._records import Correction, Json
from ._steps import DOMAINS, Step

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

LOG = logging.getLogger(__name__)

# Memory budget for a source read or float32 correction batch (or slab of rows).
_BATCH_BYTES = 256 * 2**20


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
    out: np.ndarray | None = None,
    binning: int = 1,
) -> Corrected:
    """Line integrals from ``counts`` ``(views, rows, columns)``, and the records of how.

    ``flats`` are flat frames and ``flat_positions`` the number of views
    recorded before each (frames at one position are one set; None, one set);
    ``white_level`` stands in for flats on scanners that record it instead.
    ``darks`` default to zero. ``view_positions`` places each view among the
    flats (by default view ``i`` at ``i + 1/2``).
    ``out`` is separate writable float32 NumPy/memmap storage: original view
    capacity, detector shape reduced by ``binning``. Rejected views compact
    into its leading views; the caller owns and flushes this storage.
    """
    batch_views = _validated_batch_views(batch_views)
    epsilon = _positive_finite(epsilon, "epsilon")
    views, rows, cols = (int(s) for s in counts.shape)
    ordered = _ordered(steps)
    if binning < 1:
        raise ValueError(f"binning needs a factor of at least 1, not {binning}")
    shape = (views, rows // binning, cols // binning)
    _validate_out(out, shape, (counts, flats, darks))
    fields, field_record = _flat_fields(flats, flat_positions, darks, white_level, (rows, cols))
    lo, hi, t = _view_interpolation(views, view_positions, fields.positions)
    line = [s for d, s in ordered if d == "line integrals"]
    head, tail = _split_at_break(line)
    counts_steps = tuple(s for d, s in ordered if d == "counts")
    transmission_steps = tuple(s for d, s in ordered if d == "transmission")
    counts_ceiling = _dtype_ceiling(counts)
    slab = _raw_row_slab(counts, ordered, batch_views, binning)
    if slab is not None:
        output, tally, readings = _correct_rows_binned(
            counts, fields, (lo, hi, t), tuple(line),
            epsilon=epsilon, factor=binning, slab=slab, target=out, ceiling=counts_ceiling,
        )  # fmt: skip
        records = _frame_records(
            field_record, epsilon, tally, (), (), line, (), ceiling=(counts_ceiling, readings)
        )
        return Corrected(output, records, np.arange(views))
    reduce_first = not tail and binning > 1
    bin_after_summary = binning > 1 and len(tail) == 1 and tail[0].selects_views
    needs_intermediate = bool(tail) and binning > 1 and not bin_after_summary
    output = (
        out if out is not None and not needs_intermediate
        else np.empty((views, rows, cols) if needs_intermediate else shape, np.float32)
    )  # fmt: skip
    flat_stack, dark = jax.device_put(fields.flats), jax.device_put(fields.dark)
    flat_variance, dark_variance = (
        jax.device_put(fields.flat_variance),
        jax.device_put(fields.dark_variance),
    )
    selector = tail[0] if tail and tail[0].selects_views else None
    view_steps = (*counts_steps, *transmission_steps, *head)
    if selector is not None:
        view_steps = (*view_steps, selector)
    batch = batch_views or _frame_batch_size(counts, view_steps)
    tally = np.zeros(2, np.int64)  # non-positive transmission, non-finite results
    ceiling_readings = 0
    summaries: list[np.ndarray] = []

    def collect(
        done: tuple[jax.Array, ...],
        summary: Callable[[], np.ndarray] | None,
        download: Callable[[], None],
    ) -> None:
        download()
        tally[:] += [int(b) for b in done[1:]]
        if summary is not None:
            summaries.append(summary())
        release_device(done[0])

    def diagnose(raw: np.ndarray) -> None:
        nonlocal ceiling_readings
        if counts_ceiling is not None:
            ceiling_readings += _count_ceiling_readings(raw, counts_ceiling)

    pass_fn = partial(_first_pass_binned, factor=binning) if reduce_first else _first_pass
    with ThreadPoolExecutor(max_workers=1) as statistic_worker:
        pending = None
        for start, stop, raw in _batches(
            counts, views, batch, after_read=diagnose if counts_ceiling is not None else None
        ):
            done = pass_fn(
                jnp.asarray(raw), lo[start:stop], hi[start:stop], t[start:stop], flat_stack, dark,
                flat_variance, dark_variance,
                epsilon=float(epsilon), counts_steps=counts_steps,
                transmission_steps=transmission_steps, line_steps=tuple(head),
            )  # fmt: skip
            # The first selecting step needs only small per-view summaries. Use
            # this device batch instead of transferring the corrected stack back
            # to the device once more in _run_rest.
            summary = (
                None if selector is None else _stage_statistic(selector, done[0], statistic_worker)
            )
            if bin_after_summary:
                # Keep original-pixel statistics, then compact the owned reduced storage.
                done = (_apply_binned((), done[0], factor=binning), *done[1:])
            download = stage_download(done[0], output[start:stop])
            if pending is not None:  # the last batch comes back while this one runs
                collect(*pending)
            pending = (done, summary, download)
        if pending is not None:
            collect(*pending)
            # The tail starts from host storage. Do not retain the last batch,
            # its download closure or its raw host input throughout row-slab work.
            del pending, done, summary, download, raw
    for field in (flat_stack, dark, flat_variance, dark_variance):
        release_device(field)
    del flat_stack, dark, flat_variance, dark_variance, field
    statistics = np.concatenate(summaries) if summaries else None
    output, kept, tail_records = _run_rest_binned(
        output, tail, batch_views, 1 if bin_after_summary else binning,
        statistics=statistics, target=output if bin_after_summary else out,
    )  # fmt: skip
    records = _frame_records(
        field_record, epsilon, tally, counts_steps, transmission_steps, head, tail_records,
        ceiling=(counts_ceiling, ceiling_readings),
    )  # fmt: skip
    return Corrected(output, records, kept)


class Corrected(NamedTuple):
    """Corrected line integrals, the records of each step, and the input views they keep."""

    projections: np.ndarray
    records: tuple[Correction, ...]
    kept: np.ndarray  # indices of the input views, increasing


def _frame_records(
    fields: tuple[Correction, ...],
    epsilon: float,
    tally: np.ndarray,
    counts: Sequence[Step],
    transmission: Sequence[Step],
    line: Sequence[Step],
    tail: tuple[Correction, ...],
    *,
    ceiling: tuple[int | None, int] = (None, 0),
) -> tuple[Correction, ...]:
    """The domain-ordered chain, retaining original-pixel log tallies and selector findings."""
    found: dict[str, Json] = {
        "nonpositive_transmission": int(tally[0]),
        "nonfinite_set_to_zero": int(tally[1]),
    }
    log = _with_counts_ceiling(Correction("log", {"epsilon": float(epsilon)}, found), *ceiling)
    return (
        *(s.record() for s in counts), *fields,
        *(s.record() for s in transmission), log,
        *(s.record() for s in line), *tail,
    )  # fmt: skip


def correct_projections(
    projections: Any,
    steps: Sequence[Step],
    *,
    batch_views: int | None = None,
    out: np.ndarray | None = None,
) -> Corrected:
    """``projections`` (line integrals) through ``steps``, and their records.

    ``out`` supplies separate writable float32 NumPy/memmap storage with the
    input shape; rejected views compact into its leading views. The caller
    owns this storage and flushes it if needed. Device and lazy sources are
    copied in ``batch_views`` batches, or automatic source-dtype-aware batches,
    without materializing a second whole host stack.
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
    if out is not None:
        _validate_out(out, np.shape(projections), (projections,))
    supplied_out = out is not None
    line = [s for _, s in ordered]
    head, tail = _split_at_break(line)
    resident = (
        _resident_prefix(projections, tuple(head), batch_views, out=out)
        if isinstance(projections, jax.Array) and head
        else None
    )
    if resident is not None:
        out, kept, records = _run_rest(resident, tail, batch_views, keep_storage=supplied_out)
        records = tuple(s.record() for s in head) + records
    else:
        if out is None:
            out = np.array(projections, np.float32, copy=True)
        else:
            _copy_input(projections, out, batch_views)
        out, kept, records = _run_rest(out, line, batch_views, keep_storage=supplied_out)
    return Corrected(out, records, kept)


def _copy_input(projections: Any, out: np.ndarray, batch_views: int | None) -> None:
    """Fill validated host storage, releasing temporary reads before correction steps run."""
    if isinstance(projections, np.ndarray):
        np.copyto(out, projections, casting="unsafe")
        return
    # A device/lazy source must not allocate a second whole host stack just
    # to fill caller-owned storage. Bound its reads by the source dtype too.
    views, rows, cols = out.shape
    itemsize = max(4, np.dtype(projections.dtype).itemsize)
    batch = batch_views or max(1, min(views, _BATCH_BYTES // (itemsize * rows * cols)))
    for start, stop, part in _batches(projections, views, batch):
        np.copyto(out[start:stop], part, casting="unsafe")


def _validate_out(out: np.ndarray | None, shape: tuple[int, ...], sources: tuple[Any, ...]) -> None:
    """Validate caller-owned host storage before reading or overwriting any input."""
    if out is None:
        return
    if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.float32:
        raise ValueError("out must be a float32 NumPy array or memmap with the input shape")
    if not out.flags.writeable:
        raise ValueError("out must be writable")
    target = mapped_storage(out)
    if target is not None:
        _validate_lazy_storage(target, sources)
    for source in (array for item in sources for array in _host_arrays(item)):
        if np.may_share_memory(source, out):
            raise ValueError("correction input and output storage must not overlap")
        origin = mapped_storage(source)
        if target is not None and origin is not None:
            source_name, target_name = origin.filename, target.filename
            if (
                source_name is None
                or target_name is None
                or Path(source_name).samefile(target_name)
            ):
                raise ValueError(
                    "memory-mapped correction input and output require separate named files"
                )


def _validate_lazy_storage(target: np.memmap, sources: tuple[Any, ...]) -> None:
    """Ask format owners about acquisition files without opening their pixels."""
    for source in sources:
        shares_file = getattr(source, "_shares_file", None)
        if shares_file is not None and (
            target.filename is None or shares_file(Path(target.filename))
        ):
            raise ValueError("lazy correction input and mapped output require separate named files")


def _host_arrays(source: Any) -> tuple[np.ndarray, ...]:
    """Views of addressable host storage only; never download a GPU shard for validation."""
    if isinstance(source, np.ndarray):
        return (source,)
    if isinstance(source, jax.Array):
        return tuple(
            np.asarray(part.data)
            for part in source.addressable_shards
            if part.device.platform == "cpu"
        )
    return ()


def _resident_prefix(
    projections: jax.Array,
    steps: tuple[Step, ...],
    batch_views: int | None,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray | None:
    """Download corrected view batches when the input's devices have known headroom.

    ``out``, when given, is validated caller-owned storage to fill.
    """
    batch = _resident_batch_size(projections, steps, batch_views)
    if batch is None:
        return None
    projections = commit_array(projections)
    views = int(projections.shape[0])
    out = np.empty(projections.shape, np.float32) if out is None else out
    for start in range(0, views, batch):
        stop = min(start + batch, views)
        # Fuse slicing and conversion into the first correction; no separate
        # resident input tile or overlapping corrected outputs are retained.
        part = _resident_apply(projections, start, stop - start, steps)
        part.copy_to_host_async()
        out[start:stop] = np.asarray(part)
        del part
    return out


def _resident_batch_size(
    projections: jax.Array, steps: tuple[Step, ...], requested: int | None
) -> int | None:
    """Reserve half each producer's headroom; unknown or oversized requests stay host-first."""
    free = [
        backend_runtime.device_free_memory_bytes(device=device) for device in projections.devices()
    ]
    if any(value is None for value in free):
        return None
    budget = min(_BATCH_BYTES, min(value for value in free if value is not None) // 2)
    if budget < _frame_bytes(projections, steps):
        return None
    cap = _frame_batch_size(projections, steps, budget=budget)
    if requested is not None:
        return requested if min(requested, int(projections.shape[0])) <= cap else None
    return cap


@partial(jax.jit, static_argnames=("count", "steps"))
def _resident_apply(
    projections: jax.Array, start: int, count: int, steps: tuple[Step, ...]
) -> jax.Array:
    part = jax.lax.dynamic_slice_in_dim(projections, start, count, axis=0)
    return _apply(steps, part.astype(jnp.float32))


def _frame_batch_size(counts: Any, steps: Sequence[Step], *, budget: int | None = None) -> int:
    """Bound source reads and float32 work, including the largest step's FFT workspace."""
    limit = _BATCH_BYTES if budget is None else budget
    return max(1, min(int(counts.shape[0]), limit // _frame_bytes(counts, steps)))


def _frame_bytes(counts: Any, steps: Sequence[Step]) -> int:
    rows, cols = (int(s) for s in counts.shape[1:])
    pixels = rows * cols
    workspace = max((s.workspace_pixels(rows, cols) for s in steps), default=0)
    return max(np.dtype(counts.dtype).itemsize * pixels, 4 * (pixels + workspace))


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

    def __init__(
        self,
        flats: np.ndarray,
        positions: np.ndarray,
        dark: np.ndarray,
        flat_variance: np.ndarray,
        dark_variance: np.ndarray,
    ) -> None:
        self.flats, self.positions, self.dark = flats, positions, dark
        self.flat_variance, self.dark_variance = flat_variance, dark_variance


def _flat_fields(
    flats: Any,
    positions: np.ndarray | None,
    darks: Any,
    white_level: float | None,
    shape: tuple[int, int],
) -> tuple[_Fields, tuple[Correction, ...]]:
    """The mean flat of each set, the mean dark, and the record of them."""
    darks = darks if darks is None or hasattr(darks, "shape") else np.asarray(darks)
    if darks is None and flats is not None:
        LOG.warning("no dark fields: correcting with a dark field of zero")
    dark, dark_variance = (
        (np.zeros(shape, np.float64), np.zeros(shape, np.float64))
        if darks is None
        else _moments(darks, shape, "darks")
    )
    dark_frames = 0 if darks is None else len(darks)
    if flats is None:
        if white_level is None:
            raise ValueError(
                "no flat fields: pass flats (frames of the beam without the object) or "
                "white_level (the counts of an unattenuated pixel) to tomojax.load_frames"
            )
        level = _positive_finite(white_level, "white_level")
        stack = np.full((1, *shape), level, np.float32)
        record = _field_record(
            {"white_level": level, "darks": dark_frames}, stack - dark, dark_variance
        )
        record = _with_calibration_ceiling(record, darks, dark, "dark")
        return _Fields(
            stack,
            np.zeros(1),
            dark.astype(np.float32),
            np.zeros_like(stack),
            dark_variance.astype(np.float32),
        ), (record,)
    flats = flats if hasattr(flats, "shape") else np.asarray(flats)
    if len(flats.shape) != 3 or flats.shape[1:] != shape or not len(flats):
        raise ValueError(
            f"flats are {flats.shape}; a nonempty stack of frames of {shape} is needed"
        )
    where = np.zeros(len(flats)) if positions is None else np.asarray(positions, np.float64)
    if where.shape != (len(flats),) or not np.isfinite(where).all():
        raise ValueError(
            f"flat_positions needs one finite position for each of the {len(flats)} flats"
        )
    sets = np.unique(where)
    statistics = [
        _moments(
            flats, shape, "flats", indices=np.flatnonzero(where == p) if len(sets) > 1 else None
        )
        for p in sets
    ]
    stack = np.stack([mean for mean, _ in statistics])
    flat_variance = np.stack([variance for _, variance in statistics])
    settings: dict[str, Any] = {"flats": len(flats), "darks": dark_frames, "flat_sets": len(sets)}
    if len(sets) > 1:
        settings["flat_positions"] = [float(p) for p in sets]
    record = _field_record(settings, stack - dark, flat_variance + dark_variance)
    record = _with_calibration_ceiling(record, flats, stack, "flat")
    record = _with_calibration_ceiling(record, darks, dark, "dark")
    return _Fields(
        stack.astype(np.float32),
        sets,
        dark.astype(np.float32),
        flat_variance.astype(np.float32),
        dark_variance.astype(np.float32),
    ), (record,)


def _field_record(settings: dict[str, Any], light: np.ndarray, variance: np.ndarray) -> Correction:
    found: dict[str, Json] = {"pixels_without_light": int(np.count_nonzero(light <= 0))}
    unreliable = ~np.isfinite(light) | ~np.isfinite(variance) | (light <= 3 * np.sqrt(variance))
    if unreliable.any():
        settings["minimum_signal_snr"] = 3.0
        found["pixels_without_reliable_light"] = int(np.count_nonzero(unreliable))
    return Correction("flat_dark", settings, found)


def _with_calibration_ceiling(
    record: Correction, frames: Any, mean: np.ndarray, field: str
) -> Correction:
    """Report integer sets fixed at their storage ceiling, without rereading them."""
    ceiling = _dtype_ceiling(frames)
    if ceiling is None:
        return record
    pixels = int(np.count_nonzero(mean == ceiling))
    if not pixels:
        return record
    LOG.warning(
        "%s calibration: %s pixels at dtype ceiling %s; readings may be clipped",
        field,
        pixels,
        ceiling,
    )
    return Correction(
        record.name,
        {**record.settings, f"{field}_dtype_ceiling": ceiling},
        {**record.found, f"{field}_pixels_at_dtype_ceiling": pixels},
    )


def _dtype_ceiling(frames: Any) -> int | None:
    """The integer storage ceiling, never an inferred ADC limit for floating data."""
    dtype = getattr(frames, "dtype", None)
    return int(np.iinfo(dtype).max) if dtype is not None and np.dtype(dtype).kind in "iu" else None


def _count_ceiling_readings(raw: np.ndarray, ceiling: int) -> int:
    """Count already-read inputs in small groups of complete views."""
    batch = max(1, (32 << 20) // (int(raw.shape[1]) * int(raw.shape[2])))
    return sum(
        int(np.count_nonzero(raw[start : start + batch] == ceiling))
        for start in range(0, len(raw), batch)
    )


def _with_counts_ceiling(record: Correction, ceiling: int | None, readings: int) -> Correction:
    """Attach a raw-input diagnostic, without changing any corrected reading."""
    if not readings or ceiling is None:
        return record
    LOG.warning(
        "counts: %s readings at dtype ceiling %s; readings may be clipped", readings, ceiling
    )
    return Correction(
        record.name,
        {**record.settings, "counts_dtype_ceiling": ceiling},
        {**record.found, "counts_at_dtype_ceiling": readings},
    )


def _moments(
    frames: Any, shape: tuple[int, int], name: str, *, indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and its sampling variance, in FP64 and bounded host batches."""
    frames = commit_array(frames) if hasattr(frames, "shape") else np.asarray(frames)
    if len(frames.shape) != 3 or frames.shape[1:] != shape or not len(frames):
        raise ValueError(
            f"{name} are {frames.shape}; a nonempty stack of frames of {shape} is needed"
        )
    count = len(frames) if indices is None else len(indices)
    batch = max(1, (32 << 20) // (8 * shape[0] * shape[1]))
    total, squared = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    for start in range(0, count, batch):
        selected = (
            slice(start, start + batch) if indices is None else indices[start : start + batch]
        )
        values = np.asarray(frames[selected], np.float64)
        total += np.sum(values, axis=0)
        squared += np.einsum("ijk,ijk->jk", values, values)
    mean = total / count
    variance = (
        np.maximum(0.0, (squared - total * mean) / (count * (count - 1)))
        if count > 1
        else np.zeros(shape, np.float64)
    )
    return mean, variance


def _view_interpolation(
    views: int, at: np.ndarray | None, positions: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Validate acquisition positions and interpolate each view's calibration fields."""
    q = np.arange(views) + 0.5 if at is None else np.asarray(at, np.float64)
    if q.shape != (views,) or not np.isfinite(q).all():
        raise ValueError(f"view_positions needs one finite position for each of the {views} views")
    return _interpolation(q, positions)


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
    flat_variance: jax.Array,
    dark_variance: jax.Array,
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
    # An unused calibration must not contaminate an endpoint through 0 * NaN
    # (or 0 * Inf). Interior views still need both bracketing fields.
    flat = jnp.where(
        weight == 0, flats[lo],
        jnp.where(weight == 1, flats[hi], (1 - weight) * flats[lo] + weight * flats[hi]),
    )  # fmt: skip
    variance = jnp.where(
        weight == 0, flat_variance[lo],
        jnp.where(weight == 1, flat_variance[hi],
                  (1 - weight) ** 2 * flat_variance[lo] + weight**2 * flat_variance[hi]),
    ) + dark_variance  # fmt: skip
    light = flat - dark
    reliable = (
        jnp.isfinite(light) & jnp.isfinite(variance)
        & (light > jnp.maximum(epsilon, 3 * jnp.sqrt(variance)))
    )  # fmt: skip
    # A calibration indistinguishable from darkness is an unmeasured pixel,
    # not a denominator of epsilon. Use neutral transmission before spatial
    # steps, so an FFT cannot spread enormous counts across the whole image.
    transmission = jnp.where(reliable, (x - dark) / jnp.maximum(light, epsilon), 1.0)
    for step in transmission_steps:
        transmission = step.apply(transmission)
    nonpositive = jnp.sum(transmission <= 0)
    p = -jnp.log(jnp.maximum(transmission, epsilon))
    for step in line_steps:
        p = step.apply(p)
    bad = ~jnp.isfinite(p)
    return jnp.where(bad, 0.0, p), nonpositive, jnp.sum(bad)


@partial(
    jax.jit,
    static_argnames=("epsilon", "counts_steps", "transmission_steps", "line_steps", "factor"),
)
def _first_pass_binned(
    raw, lo, hi, t, flats, dark, flat_variance, dark_variance, *,
    epsilon, counts_steps, transmission_steps, line_steps, factor,
):  # fmt: skip
    done = _first_pass(
        raw,
        lo,
        hi,
        t,
        flats,
        dark,
        flat_variance,
        dark_variance,
        epsilon=epsilon,
        counts_steps=counts_steps,
        transmission_steps=transmission_steps,
        line_steps=line_steps,
    )
    return mean_detector_pixels(done[0], factor), *done[1:]


def _raw_row_slab(
    counts: Any, ordered: Sequence[tuple[str, Step]], batch_views: int | None, factor: int
) -> int | None:
    """A bounded aligned row group only when both the corrections and source permit it."""
    if batch_views is not None or factor == 1 or not ordered:
        return None
    if any(d != "line integrals" or not s.whole_rows or s.selects_views for d, s in ordered):
        return None
    # Do not reread/decompress whole frames for every row group. Discovery owns
    # the lazy-source capability; unknown readers and device arrays keep the old path.
    if not isinstance(counts, np.ndarray) and not getattr(counts, "_efficient_row_windows", False):
        return None
    views, rows, cols = counts.shape
    itemsize = max(4, np.dtype(counts.dtype).itemsize)
    cap = min(rows, _BATCH_BYTES // (2 * itemsize * views * cols))
    slab = (cap // factor) * factor
    return slab if slab >= factor else None


def _first_rows_binned(
    raw, lo, hi, t, flats, dark, flat_variance, dark_variance, *, epsilon, steps, factor
):
    """Normalize wide counts separately, keeping row corrections entirely on device.

    Fusing 64-bit raw normalization into the row reduction can need much more
    cold-compilation scratch than the two FP32 stages. Ordinary count dtypes
    retain the fused dispatch; neither route transfers unbinned pixels to host.
    """
    if raw.dtype.itemsize > 4:
        p, nonpositive, bad = _first_pass(
            raw, lo, hi, t, flats, dark, flat_variance, dark_variance, epsilon=epsilon,
            counts_steps=(), transmission_steps=(), line_steps=(),
        )  # fmt: skip
        return _rows_binned(p, steps=steps, factor=factor), nonpositive, bad
    return _fused_rows_binned(
        raw, lo, hi, t, flats, dark, flat_variance, dark_variance,
        epsilon=epsilon, steps=steps, factor=factor,
    )  # fmt: skip


@partial(jax.jit, static_argnames=("epsilon", "steps", "factor"))
def _fused_rows_binned(
    raw, lo, hi, t, flats, dark, flat_variance, dark_variance, *, epsilon, steps, factor
):
    p, nonpositive, bad = _first_pass(
        raw, lo, hi, t, flats, dark, flat_variance, dark_variance, epsilon=epsilon,
        counts_steps=(), transmission_steps=(), line_steps=(),
    )  # fmt: skip
    return _rows_binned(p, steps=steps, factor=factor), nonpositive, bad


@partial(jax.jit, static_argnames=("steps", "factor"))
def _rows_binned(p, *, steps, factor):
    """Each whole-row step retains its original FP32 storage barrier."""
    p = p.astype(jnp.float32)
    for step in steps:
        p = step.apply(p).astype(jnp.float32)
    return mean_detector_pixels(p, factor)


def _correct_rows_binned(
    counts: Any,
    fields: _Fields,
    interpolation: tuple[np.ndarray, ...],
    steps: tuple[Step, ...],
    *,
    epsilon: float,
    factor: int,
    slab: int,
    target: np.ndarray | None,
    ceiling: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream row-local raw corrections without allocating an original-sized host stack.

    Also returns the number of raw readings at the integer ``ceiling``.
    """
    views, rows, cols = counts.shape
    output = (
        np.empty((views, rows // factor, cols // factor), np.float32) if target is None else target
    )
    flats, dark = jax.device_put(fields.flats), jax.device_put(fields.dark)
    flat_variance, dark_variance = (
        jax.device_put(fields.flat_variance),
        jax.device_put(fields.dark_variance),
    )
    lo, hi, t = (jax.device_put(q) for q in interpolation)
    tally = np.zeros(2, np.int64)
    readings = 0

    def collect(start: int, end: int, done: tuple[jax.Array, ...]) -> None:
        output[:, start // factor : end // factor] = np.asarray(done[0])
        tally[:] += [int(n) for n in done[1:]]

    pending = None
    for start in range(0, rows, slab):
        end = min(start + slab, rows)
        raw = (
            counts[:, start:end, :]
            if isinstance(counts, np.ndarray)
            else counts.window(slice(start, end), slice(None))[:]
        )
        raw = np.asarray(raw)
        raw = raw.astype(raw.dtype.newbyteorder("="), copy=False)
        if ceiling is not None:
            readings += _count_ceiling_readings(raw, ceiling)
        done = _first_rows_binned(
            jnp.asarray(raw), lo, hi, t, flats[:, start:end], dark[start:end],
            flat_variance[:, start:end], dark_variance[start:end],
            epsilon=epsilon, steps=steps, factor=factor,
        )  # fmt: skip
        done[0].copy_to_host_async()
        if pending is not None:
            collect(*pending)
        pending = (start, end, done)
    if pending is not None:
        collect(*pending)
    return output, tally, readings


@partial(jax.jit, static_argnames=("steps", "factor"))
def _apply_binned(steps: tuple[Step, ...], batch: jax.Array, *, factor: int) -> jax.Array:
    return mean_detector_pixels(_apply(steps, batch), factor)


def _run_rest_binned(
    data: np.ndarray,
    steps: Sequence[Step],
    batch_views: int | None,
    factor: int,
    *,
    statistics: np.ndarray | None,
    target: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[Correction, ...]]:
    """Keep original-detector dependencies; reduce only after the final step."""
    if factor == 1 or not steps:
        return _run_rest(
            data, steps, batch_views, statistics=statistics, keep_storage=target is not None
        )
    views, rows, cols = data.shape
    cap = max(1, min(rows, _BATCH_BYTES // (8 * views * cols)))
    slab = (cap // factor) * factor
    if steps and steps[-1].whole_rows and slab >= factor:
        data, kept, records = _run_rest(data, steps[:-1], batch_views, statistics=statistics)
        views, rows, cols = data.shape
        output = (
            np.empty((views, rows // factor, cols // factor), np.float32)
            if target is None
            else target[:views]
        )
        _whole_rows_binned(data, steps[-1], slab, factor, output)
        return output, kept, (*records, steps[-1].record())
    # Selectors or a bin larger than a bounded row group retain the old path.
    data, kept, records = _run_rest(data, steps, batch_views, statistics=statistics)
    reduced = mean_detector_pixels(data, factor)
    if target is None:
        return reduced, kept, records
    output = target[: len(kept)]
    np.copyto(output, reduced)
    return output, kept, records


def _whole_rows_binned(
    data: np.ndarray, step: Step, slab: int, factor: int, output: np.ndarray
) -> None:
    """Use aligned, bounded input groups and download only complete pixel means."""
    stop = (data.shape[1] // factor) * factor
    pending = None
    for start in range(0, stop, slab):
        end = min(start + slab, stop)
        done = _apply_binned((step,), jnp.asarray(data[:, start:end]), factor=factor)
        done.copy_to_host_async()
        if pending is not None:
            a, b, result = pending
            output[:, a:b] = np.asarray(result)
            del result
        pending = (start // factor, end // factor, done)
    if pending is not None:
        a, b, result = pending
        output[:, a:b] = np.asarray(result)


def _split_at_break(steps: list[Step]) -> tuple[list[Step], list[Step]]:
    """The steps before the first that needs whole rows or selects views, and the rest."""
    for i, step in enumerate(steps):
        if step.whole_rows or step.selects_views:
            return steps[:i], steps[i:]
    return steps, []


def _run_rest(
    out: np.ndarray,
    steps: Sequence[Step],
    batch_views: int | None,
    *,
    statistics: np.ndarray | None = None,
    keep_storage: bool = False,
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
        if step.selects_views:
            batch = batch_views or _frame_batch_size(out, (step,))
            if i == 0 and statistics is not None:
                stats = statistics
            else:
                stats = np.concatenate([
                    np.asarray(_statistic(step, jnp.asarray(out[v0 : v0 + batch])))
                    for v0 in range(0, views, batch)
                ])  # fmt: skip
            keep, found = step.select(stats, kept)
            if not keep.any():
                raise ValueError(f"{type(step).__name__} would reject every view")
            if not keep.all():
                out = _compact(out, keep, batch) if keep_storage else out[keep]
                kept = kept[keep]
            records.append(Correction(step.record().name, step.record().settings, found))
            i += 1
            continue
        if step.whole_rows:
            # Two pending input/output slabs share the old one-slab budget.
            slab = max(1, min(rows, _BATCH_BYTES // (8 * views * cols)))
            _whole_rows(out, step, slab)
            records.append(step.record())
            i += 1
            continue
        j = i
        while j < len(steps) and not (steps[j].whole_rows or steps[j].selects_views):
            j += 1
        run = tuple(steps[i:j])
        batch = batch_views or _frame_batch_size(out, run)
        for v0 in range(0, views, batch):
            out[v0 : v0 + batch] = np.asarray(_apply(run, jnp.asarray(out[v0 : v0 + batch])))
        records.extend(s.record() for s in run)
        i = j
    return out, kept, tuple(records)


def _whole_rows(out: np.ndarray, step: Step, slab: int) -> None:
    """Overlap one slab's transfer with the next; disjoint rows can be written in place."""
    if out.shape[0] * out.shape[2] * out.dtype.itemsize > _BATCH_BYTES // 2:
        # A single row cannot be split across views: keep its original serial footprint.
        for r0 in range(out.shape[1]):
            out[:, r0 : r0 + 1] = np.asarray(_apply((step,), jnp.asarray(out[:, r0 : r0 + 1])))
        return
    pending = None
    for r0 in range(0, out.shape[1], slab):
        done = _apply((step,), jnp.asarray(out[:, r0 : r0 + slab]))
        done.copy_to_host_async()
        if pending is not None:
            previous, result = pending
            out[:, previous : previous + slab] = np.asarray(result)
            del result
        pending = (r0, done)
    if pending is not None:
        previous, result = pending
        out[:, previous : previous + slab] = np.asarray(result)


def _compact(out: np.ndarray, keep: np.ndarray, batch: int) -> np.ndarray:
    """Keep views in their original order using bounded copies into the leading storage."""
    written = 0
    for start in range(0, len(keep), batch):
        selected = out[start : start + batch][keep[start : start + batch]]
        out[written : written + len(selected)] = selected
        written += len(selected)
    return out[:written]


@partial(jax.jit, static_argnames=("step",))
def _statistic(step: Step, batch: jax.Array) -> jax.Array:
    return step.statistic(batch)


def _stage_statistic(
    step: Step, batch: jax.Array, worker: ThreadPoolExecutor
) -> Callable[[], np.ndarray]:
    """Collect a summary without blocking CUDA downloads on device-loop dispatch.

    Lower on the caller: its JAX configuration, mesh and captured arrays belong
    to the calculation. Only the compiled execution moves to the worker. The
    owner keeps each input until both its download and this collection finish.
    """
    if (
        cuda_device_id(batch) is None
        or jax.config.jax_disable_jit
        or jax.config.jax_debug_nans
        or jax.config.jax_debug_infs
    ):
        done = _statistic(step, batch)
        return lambda: np.asarray(done)
    compiled = _statistic.lower(step, batch).compile()
    (device,) = batch.devices()  # cuda_device_id has checked the single producer.
    free = backend_runtime.device_free_memory_bytes(device=device)
    workspace = compiled.memory_analysis()
    # Keep room for the next correction's inputs, outputs and temporaries.
    # The serial path remains available for tight or unknown allocator budgets.
    reserve = None if workspace is None else 2 * workspace.temp_size_in_bytes + 4 * batch.nbytes
    if reserve is None or free is None or free < reserve:
        done = compiled(batch)
        return lambda: np.asarray(done)
    future = worker.submit(compiled, batch)
    return lambda: np.asarray(future.result())


@partial(jax.jit, static_argnames=("steps",))
def _apply(steps: tuple[Step, ...], batch: jax.Array) -> jax.Array:
    for step in steps:
        batch = step.apply(batch)
    return batch


def _batches(
    counts: Any, views: int, batch: int, *, after_read: Callable[[np.ndarray], None] | None = None
) -> Any:
    """``(start, stop, frames)`` for each batch of views, the next read while one is used."""
    starts = list(range(0, views, batch))
    # The worker does not inherit default_device; preserve existing placement
    # before slicing rather than relocating an uncommitted source wholesale.
    counts = commit_array(counts)

    def read(start: int) -> np.ndarray:
        frames = np.asarray(counts[start : min(start + batch, views)])
        # Device arrays use native byte order; leave ordinary batches as views.
        frames = frames.astype(frames.dtype.newbyteorder("="), copy=False)
        if after_read is not None:
            after_read(frames)
        return frames

    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read, starts[0]) if starts else None
        for k, start in enumerate(starts):
            assert pending is not None
            frames = pending.result()
            if k + 1 < len(starts):
                pending = reader.submit(read, starts[k + 1])
            yield start, min(start + batch, views), frames
