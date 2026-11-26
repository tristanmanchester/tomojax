"""Owned corrected detector-row bands without a materialized projection stack."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.devices import current_device
from tomojax.core.host_transfer import upload_host
from tomojax.core.host_volume import host_available_memory_bytes

from ._engine import (
    Corrected,
    apply_steps,
    calibration,
    correct_view_batches,
    first_pass,
    frame_batch_size,
    linear_limit_edge,
    log_batches,
    ordered_steps,
    reject_nonfinite,
)
from ._fields import Fields
from ._hardening import BeamHardening
from ._records import Correction
from ._steps import RadialDistortion, Step, Zingers, apply_radial_window, radial_coordinates
from ._validation import positive_finite

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from tomojax._data import FrameStack


@dataclass(frozen=True)
class _RadialWindow(RadialDistortion):
    first: int

    def apply(self, batch: jax.Array) -> jax.Array:
        return apply_radial_window(self, batch, self.first)


def preview_window(
    steps: Sequence[Step], shape: tuple[int, int], lower: int, upper: int, factor: int
) -> tuple[slice, tuple[Step, ...]]:
    """Propagate complete native output blocks backwards across the ordered corrections.

    Parameters
    ----------
    steps : sequence of Step
        Corrections in their supplied order.
    shape : tuple of int
        Native detector rows and columns.
    lower, upper : int
        Half-open row bounds in the binned output.
    factor : int
        Native block size.

    Returns
    -------
    tuple
        Native input interval and its locally registered correction steps.
    """
    from ._validation import check_rows

    rows, cols = shape
    check_rows((lower, upper), rows, factor)
    first, stop = lower * factor, upper * factor
    needed_first, needed_stop = first, stop
    for _, step in reversed(ordered_steps(steps)):
        if step.whole_rows or isinstance(step, BeamHardening):
            continue
        if isinstance(step, Zingers):
            first, stop = max(0, first - step.size // 2), min(rows, stop + step.size // 2)
        elif isinstance(step, RadialDistortion):
            y, _ = radial_coordinates(step, first, stop, cols)
            y = jnp.clip(y, 0, rows - 1)
            first = int(jnp.floor(jnp.min(y)))
            stop = min(rows, int(jnp.floor(jnp.max(y))) + 2)
        else:
            # Selection statistics, propagation and undeclared transforms need their full input.
            first, stop = 0, rows
        needed_first, needed_stop = min(first, needed_first), max(stop, needed_stop)
    first = needed_first // factor * factor
    stop = min(rows, (needed_stop + factor - 1) // factor * factor)
    local = tuple(
        _RadialWindow(center=step.center, coefficients=step.coefficients, first=first)
        if first and isinstance(step, RadialDistortion)
        else step
        for step in steps
    )
    return slice(first, stop), local


@dataclass
class CorrectedRows:
    """Owned device projections and their records, completed only when requested.

    Parameters
    ----------
    projections : jax.Array
        Corrected line integrals of the requested rows, in (view, row, column) order.
    flat : Correction
        Record of the flat and dark normalization.
    steps : tuple of Step
        Corrections applied after normalization, in order.
    epsilon : float
        Floor applied to transmission before the logarithm.
    counts : tuple of jax.Array
        Device counts of nonpositive transmissions and of nonfinite results.
    raw : numpy.ndarray
        Raw detector counts read for these rows.
    """

    projections: jax.Array
    flat: Correction
    steps: tuple[Step, ...]
    epsilon: float
    counts: tuple[jax.Array, jax.Array]
    raw: np.ndarray

    @property
    def records(self) -> tuple[Correction, ...]:
        """Correction records for these measured rows, including any sampling halo.

        Returns
        -------
        tuple of Correction
            Flat normalization, transmission steps, logarithm and line-integral steps.

        Raises
        ------
        ValueError
            If any corrected value is not finite.
        """
        reject_nonfinite(int(self.counts[1]), 0, len(self.projections))
        log = Correction(
            "log", {"epsilon": self.epsilon}, {"nonpositive_transmission": int(self.counts[0])}
        )
        before = [step.record() for step in self.steps if step.domain == "transmission"]
        after = [step.record() for step in self.steps if step.domain == "line integrals"]
        return (self.flat, *before, log, *after)


# Each view's rows of a raw block are one positional read. File systems serve reads
# of a megabyte two to five times faster than the 130 KiB reads of 1 GiB blocks of an
# 8,000-view scan 2560 pixels wide; a parallel file system shows the most difference.
_RUN_BYTES = 1 << 20
_BLOCK_BYTES = 1 << 30


def _block_rows(shape: tuple[int, int, int], itemsize: int) -> int:
    """Rows per raw block: a megabyte per view, at least 1 GiB, within the host's room.

    At most three blocks are held: the current one, the next, being read, and
    the previous one while slabs still use it. They take at most an eighth of
    the available host memory, or 1 GiB each when that is less.
    """
    views, rows, cols = shape
    row = views * cols * itemsize
    size = max(_BLOCK_BYTES, views * _RUN_BYTES)
    room = host_available_memory_bytes()
    if room is not None:
        size = min(size, max(_BLOCK_BYTES, room // 24))
    return max(1, min(rows, size // row))


class _RowReader:
    """One current raw block and one prefetched block; requests are views of a block.

    The first block is read from the start, while the calibration is computed.
    """

    def __init__(self, counts: Any) -> None:
        self.counts = counts
        views, rows, cols = counts.shape
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tomojax-raw-rows")
        self.current: np.ndarray | None = None
        self.first, self.next_first, self.future = -1, 0, None
        self.batch: int = rows
        if isinstance(counts, np.ndarray):  # already in memory: one block, never read
            self.current, self.first = counts, 0
        else:
            self.batch = _block_rows((views, rows, cols), counts.dtype.itemsize)
            self.future = self.pool.submit(self._read, 0)

    def _read(self, first: int) -> np.ndarray:
        stop = min(first + self.batch, self.counts.shape[1])
        return self.counts.window(slice(first, stop), slice(None))[:]

    def read(self, first: int, stop: int) -> np.ndarray:
        """Rows ``first:stop`` of every view: a view of one block, or a copy across two."""
        rows = self.counts.shape[1]
        parts: list[np.ndarray] = []
        at = first
        while at < stop:
            block = at // self.batch * self.batch
            if block != self.first:
                if block != self.next_first:
                    if self.future is not None and not self.future.cancel():
                        _ = self.future.result()
                        del _
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
            parts.append(self.current[:, at - block : end - block])
            at = end
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.current = self.future = None


def slab_steps(steps: Sequence[Step]) -> list[tuple[str, Step]]:
    """Check which corrections can run independently on complete detector rows.

    Parameters
    ----------
    steps : sequence of Step
        Proposed raw corrections.

    Returns
    -------
    list of tuple
        Domain and step pairs in pipeline order.
    """
    from tomojax.errors import InvalidParameter

    for i, step in enumerate(steps):
        if step.domain == "counts" or not step.whole_rows or step.selects_views:
            raise InvalidParameter(
                f"corrections[{i}]",
                "streamed corrections require whole-row transmission or line-integral steps "
                "without view selection",
            )
    return ordered_steps(steps)


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
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
    host: bool = False,
    dark_source: str | None = None,
    linear_limit: tuple[float, str] | None = None,
) -> Iterator[Callable[[int, int], CorrectedRows | Corrected]]:
    """Read corrected detector-row intervals without a materialized projection stack.

    Each request returns owned projections, independent of later reads and
    context closure. Only whole-row transmission and line-integral steps without view
    selection are supported; operations requiring neighboring rows must run on full frames.
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
    exposure_times : numpy.ndarray or float or None, optional
        Sample exposure durations in seconds, one per view or a common duration.
    flat_exposure_times : numpy.ndarray or float or None, optional
        Flat exposure durations in seconds, one per flat or a common duration.
    host : bool, optional
        Return NumPy bands using bounded device view batches, instead of keeping
        the whole corrected band on the device. Default is False.
    dark_source : str or None, optional
        What removed the dark before the counts were recorded (a camera that
        subtracts it, say), kept in the record as ``dark_removed_by`` in place of
        a count of dark frames. Default None.
    linear_limit : tuple of (float, str) or None, optional
        The detector's stated linear range in counts and where it is stated. The
        record keeps the fraction of counts above it and the line integral where
        it ends, and a warning names both when any count lies above. Default None.

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
    ordered = slab_steps(steps)
    if not isinstance(counts, np.ndarray) and not hasattr(counts, "window"):
        raise TypeError("streamed counts must be NumPy arrays or support detector windows")
    epsilon = positive_finite(epsilon, "epsilon")
    views, rows, cols = counts.shape
    # The first raw rows are read while the calibration is computed.
    reader = None if host else _RowReader(counts)
    try:
        fields, records, interpolation, view_scales = calibration(
            views, (rows, cols), flats=flats, flat_positions=flat_positions, darks=darks,
            white_level=white_level, epsilon=epsilon, view_positions=view_positions,
            exposure_times=exposure_times, flat_exposure_times=flat_exposure_times,
            dark_source=dark_source,
        )  # fmt: skip
        limit = np.float32(np.inf if linear_limit is None else linear_limit[0])
        edge = None
        if linear_limit is not None:  # each band's record reports its counts above it
            edge = linear_limit_edge(linear_limit[0], fields, view_scales)
            records[0].settings["linear_limit"] = float(linear_limit[0])
            records[0].settings["linear_limit_source"] = linear_limit[1]
        device_interpolation = None if host else tuple(jax.device_put(v) for v in interpolation)
        steps = tuple(s for _, s in ordered)
        line_steps = tuple(s for domain, s in ordered if domain == "line integrals")
        transmission_steps = tuple(s for domain, s in ordered if domain == "transmission")

        def correct(first: int, stop: int) -> CorrectedRows | Corrected:
            if host:
                return _correct_host_rows(
                    counts,
                    fields,
                    interpolation,
                    records[0],
                    first,
                    stop,
                    steps,
                    epsilon,
                    limit,
                    edge,
                )
            assert reader is not None
            assert device_interpolation is not None
            raw = reader.read(first, stop)
            raw = raw.astype(raw.dtype.newbyteorder("="), copy=False)
            selected = (
                fields.flats[:, first:stop],
                fields.dark[first:stop],
                fields.usable[first:stop],
            )
            projections, nonpositive, nonfinite, above = first_pass(
                upload_host(raw, current_device()),
                *device_interpolation,
                *(jax.device_put(np.ascontiguousarray(value)) for value in selected),
                epsilon=epsilon,
                line_steps=line_steps,
                transmission_steps=transmission_steps,
                fill_gaps=not bool(selected[2].all()),
                limit=limit,
            )
            nonpositive.copy_to_host_async()
            nonfinite.copy_to_host_async()
            flat = Correction(
                "flat_dark",
                records[0].settings,
                {
                    "pixels_without_light": int(np.count_nonzero(~selected[2])),
                    **_above(above, raw.size, edge),
                },
            )
            return CorrectedRows(projections, flat, steps, epsilon, (nonpositive, nonfinite), raw)

        yield correct
    finally:
        if reader is not None:
            reader.close()


def _correct_host_rows(
    counts: np.ndarray | FrameStack,
    fields: Fields,
    interpolation: tuple[np.ndarray, ...],
    flat_record: Correction,
    first: int,
    stop: int,
    steps: tuple[Step, ...],
    epsilon: float,
    limit: np.float32,
    edge: float | None,
) -> Corrected:
    """Correct a host band with bounded raw reads and device workspaces."""
    rows = slice(first, stop)
    raw = counts[:, rows] if isinstance(counts, np.ndarray) else counts.window(rows, slice(None))
    selected = fields.window(rows)
    output = np.empty(raw.shape, np.float32)
    batch = frame_batch_size(raw.shape[0], raw.shape[1], raw.shape[2], ())
    transmission = tuple(s for s in steps if s.domain == "transmission")
    line = tuple(s for s in steps if s.domain == "line integrals")
    tally = correct_view_batches(
        raw, output, selected, interpolation, batch, epsilon=epsilon, take_log=not transmission,
        limit=limit,
    )  # fmt: skip
    above = int(tally[2])
    if transmission:
        output, _, _ = apply_steps(output, transmission, None)
        output, tally = log_batches(output, (), epsilon, None)
    output, kept, rest = apply_steps(output, line, None)
    flat = Correction(
        "flat_dark",
        flat_record.settings,
        {
            "pixels_without_light": int(np.count_nonzero(~selected.usable)),
            **_above(above, int(np.prod(raw.shape)), edge),
        },
    )
    log = Correction("log", {"epsilon": epsilon}, {"nonpositive_transmission": int(tally[0])})
    return Corrected(output, (flat, *(s.record() for s in transmission), log, *rest), kept)


def _above(count: int | jax.Array, size: int, edge: float | None) -> dict[str, float]:
    """A band's counts above the stated linear range, when the camera states one."""
    if edge is None:
        return {}
    return {"counts_above_linear_limit": int(count) / size, "linear_limit_line_integral": edge}
