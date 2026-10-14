"""Full-support calibration statistics and view positions, accumulated once on host."""

import logging
from typing import Any

import numpy as np

from tomojax._typed_arrays import commit_array

from ._records import Correction, Json

LOG = logging.getLogger(__name__)


def positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, not {value!r}")
    return value


class Fields:
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


def flat_fields(
    flats: Any,
    positions: np.ndarray | None,
    darks: Any,
    white_level: float | None,
    shape: tuple[int, int],
) -> tuple[Fields, tuple[Correction, ...]]:
    """The mean flat of each set, the mean dark, and the record of them."""
    darks = darks if darks is None or hasattr(darks, "shape") else np.asarray(darks)
    if darks is None and flats is not None:
        LOG.warning("no dark fields: correcting with a dark field of zero")
    dark, dark_variance = (
        (np.zeros(shape, np.float64), np.zeros(shape, np.float64))
        if darks is None
        else moments(darks, shape, "darks")
    )
    dark_frames = 0 if darks is None else len(darks)
    if flats is None:
        if white_level is None:
            raise ValueError(
                "no flat fields: pass flats (frames of the beam without the object) or "
                "white_level (the counts of an unattenuated pixel) to tomojax.load_frames"
            )
        level = positive_finite(white_level, "white_level")
        stack = np.full((1, *shape), level, np.float32)
        record = field_record(
            {"white_level": level, "darks": dark_frames}, stack - dark, dark_variance
        )
        record = with_calibration_ceiling(record, darks, dark, "dark")
        return Fields(
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
        moments(
            flats, shape, "flats", indices=np.flatnonzero(where == p) if len(sets) > 1 else None
        )
        for p in sets
    ]
    stack = np.stack([mean for mean, _ in statistics])
    flat_variance = np.stack([variance for _, variance in statistics])
    settings: dict[str, Any] = {"flats": len(flats), "darks": dark_frames, "flat_sets": len(sets)}
    if len(sets) > 1:
        settings["flat_positions"] = [float(p) for p in sets]
    record = field_record(settings, stack - dark, flat_variance + dark_variance)
    record = with_calibration_ceiling(record, flats, stack, "flat")
    record = with_calibration_ceiling(record, darks, dark, "dark")
    return Fields(
        stack.astype(np.float32),
        sets,
        dark.astype(np.float32),
        flat_variance.astype(np.float32),
        dark_variance.astype(np.float32),
    ), (record,)


def field_record(settings: dict[str, Any], light: np.ndarray, variance: np.ndarray) -> Correction:
    found: dict[str, Json] = {"pixels_without_light": int(np.count_nonzero(light <= 0))}
    unreliable = ~np.isfinite(light) | ~np.isfinite(variance) | (light <= 3 * np.sqrt(variance))
    if unreliable.any():
        settings["minimum_signal_snr"] = 3.0
        found["pixels_without_reliable_light"] = int(np.count_nonzero(unreliable))
    return Correction("flat_dark", settings, found)


def with_calibration_ceiling(
    record: Correction, frames: Any, mean: np.ndarray, field: str
) -> Correction:
    """Report integer sets fixed at their storage ceiling, without rereading them."""
    ceiling = dtype_ceiling(frames)
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


def moments(
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


def view_interpolation(
    views: int, at: np.ndarray | None, positions: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Validate acquisition positions and interpolate each view's calibration fields."""
    q = np.arange(views) + 0.5 if at is None else np.asarray(at, np.float64)
    if q.shape != (views,) or not np.isfinite(q).all():
        raise ValueError(f"view_positions needs one finite position for each of the {views} views")
    return interpolation(q, positions)


def interpolation(q: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, ...]:
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


def dtype_ceiling(frames: Any) -> int | None:
    """The integer storage ceiling, never an inferred ADC limit for floating data."""
    dtype = getattr(frames, "dtype", None)
    return int(np.iinfo(dtype).max) if dtype is not None and np.dtype(dtype).kind in "iu" else None


def count_ceiling_readings(raw: np.ndarray, ceiling: int) -> int:
    """Count already-read inputs in small groups of complete views."""
    batch = max(1, (32 << 20) // (int(raw.shape[1]) * int(raw.shape[2])))
    return sum(
        int(np.count_nonzero(raw[start : start + batch] == ceiling))
        for start in range(0, len(raw), batch)
    )


def with_counts_ceiling(record: Correction, ceiling: int | None, readings: int) -> Correction:
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
