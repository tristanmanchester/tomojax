"""Owned corrected detector-row bands without a materialized projection stack."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from ._engine import (
    Corrected,
    Fields,
    apply_steps,
    correct_view_batches,
    exposure_scales,
    first_pass,
    flat_fields,
    flat_weights,
    frame_batch_size,
    log_batches,
    ordered_steps,
    positive_finite,
)
from ._records import Correction
from ._steps import Step

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


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
        before = [step.record() for step in self.steps if step.domain == "transmission"]
        after = [step.record() for step in self.steps if step.domain == "line integrals"]
        return (self.flat, *before, log, *after)


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
    exposure_times: np.ndarray | float | None = None,
    flat_exposure_times: np.ndarray | float | None = None,
    host: bool = False,
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
    ordered = ordered_steps(steps)
    if any(
        domain == "counts" or not step.whole_rows or step.selects_views for domain, step in ordered
    ):
        raise ValueError(
            "streamed corrections require whole-row transmission or line-integral steps "
            "without view selection"
        )
    if not isinstance(counts, np.ndarray) and not hasattr(counts, "window"):
        raise TypeError("streamed counts must be NumPy arrays or support detector windows")
    epsilon = positive_finite(epsilon, "epsilon")
    views, rows, cols = counts.shape
    view_scales, flat_scales = exposure_scales(exposure_times, flat_exposure_times, views, flats)
    fields, records = flat_fields(
        flats, flat_positions, darks, white_level, (rows, cols), epsilon, flat_scales=flat_scales
    )
    if flat_scales is not None and exposure_times is not None:
        records[0].settings["sample_exposure_range_s"] = [
            float(np.min(exposure_times)),
            float(np.max(exposure_times)),
        ]
    at = (
        np.arange(views) + 0.5 if view_positions is None else np.asarray(view_positions, np.float64)
    )
    if at.shape != (views,) or not np.isfinite(at).all():
        raise ValueError(f"view_positions needs one finite position for each of the {views} views")
    interpolation = flat_weights(at, fields.positions, exposure_scales=view_scales)
    device_interpolation = None if host else tuple(jax.device_put(v) for v in interpolation)
    reader = None if host else _RowReader(counts)
    steps = tuple(s for _, s in ordered)
    line_steps = tuple(s for domain, s in ordered if domain == "line integrals")
    transmission_steps = tuple(s for domain, s in ordered if domain == "transmission")

    def correct(first: int, stop: int) -> CorrectedRows | Corrected:
        if host:
            return _correct_host_rows(
                counts, fields, interpolation, records[0], first, stop, steps, epsilon
            )
        assert reader is not None
        assert device_interpolation is not None
        raw = reader.read(first, stop)
        raw = raw.astype(raw.dtype.newbyteorder("="), copy=False)
        selected = (fields.flats[:, first:stop], fields.dark[first:stop], fields.usable[first:stop])
        done = first_pass(
            jax.device_put(raw),
            *device_interpolation,
            *(jax.device_put(np.ascontiguousarray(value)) for value in selected),
            epsilon=epsilon,
            line_steps=line_steps,
            transmission_steps=transmission_steps,
            fill_gaps=not bool(selected[2].all()),
        )
        flat = Correction(
            "flat_dark",
            records[0].settings,
            {"pixels_without_light": int(np.count_nonzero(~selected[2]))},
        )
        return CorrectedRows(done[0], flat, steps, epsilon, (done[1], done[2]), raw)

    try:
        yield correct
    finally:
        if reader is not None:
            reader.close()


def _correct_host_rows(
    counts: Any,
    fields: Fields,
    interpolation: tuple[np.ndarray, ...],
    flat_record: Correction,
    first: int,
    stop: int,
    steps: tuple[Step, ...],
    epsilon: float,
) -> Corrected:
    """Correct a host band with bounded raw reads and device workspaces."""
    rows = slice(first, stop)
    raw = counts[:, rows] if isinstance(counts, np.ndarray) else counts.window(rows, slice(None))
    selected = Fields(
        fields.flats[:, rows], fields.positions, fields.dark[rows], fields.usable[rows]
    )
    output = np.empty(raw.shape, np.float32)
    batch = frame_batch_size(raw.shape[0], raw.shape[1], raw.shape[2], ())
    transmission = tuple(s for s in steps if s.domain == "transmission")
    line = tuple(s for s in steps if s.domain == "line integrals")
    tally = correct_view_batches(
        raw, output, selected, interpolation, batch, epsilon=epsilon, take_log=not transmission
    )
    if transmission:
        output, _, _ = apply_steps(output, transmission, None)
        output, tally = log_batches(output, (), epsilon, None)
    output, kept, rest = apply_steps(output, line, None)
    flat = Correction(
        "flat_dark",
        flat_record.settings,
        {"pixels_without_light": int(np.count_nonzero(~selected.usable))},
    )
    log = Correction(
        "log",
        {"epsilon": epsilon},
        {
            "nonpositive_transmission": int(tally[0]),
            "nonfinite_set_to_zero": int(tally[1]),
        },
    )
    return Corrected(output, (flat, *(s.record() for s in transmission), log, *rest), kept)
