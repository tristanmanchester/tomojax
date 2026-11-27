"""Selection and value conversion for registered three-dimensional arrays."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import math
import operator
from typing import Literal

import msgspec
import numpy as np

from tomojax import progress
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Axis, Uniform

type Interval = slice | tuple[int | None, int | None] | None
type Crop = Sequence[Interval]
type ExportDtype = Literal["float32", "uint16", "uint8"]
type Region = tuple[slice, ...]


@dataclass(frozen=True)
class Source:
    """A registered array with a window reader pinned to its generation.

    Parameters
    ----------
    name : str
        Member name.
    shape : tuple of int
        Extents in stored axis order.
    dtype : numpy.dtype
        Source scalar type.
    axes : tuple of Axis
        Named coordinates in stored order.
    unit : str
        Unit of the array values.
    domain : str
        Quantity measured by each sample.
    read : callable
        Reads a rectangular region of the source.
    record : dict
        Metadata for the object that owns this array.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    axes: tuple[Axis, ...]
    unit: str
    domain: str
    read: Callable[[Region], np.ndarray]
    record: dict[str, object]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or len(self.axes) != 3:
            raise InvalidParameter("source", "export needs a three-dimensional member")


def interval(part: Interval, size: int) -> tuple[int, int]:
    if part is None:
        return 0, size
    if isinstance(part, slice):
        lo, hi = part.start, part.stop
        if part.step not in (None, 1):
            raise InvalidParameter("crop", "crop needs a unit step")
    else:
        if len(part) != 2:
            raise InvalidParameter("crop", "expected a (start, stop) pair")
        lo, hi = part
    lo = 0 if lo is None else operator.index(lo)
    hi = size if hi is None else operator.index(hi)
    if not 0 <= lo < hi <= size:
        raise InvalidParameter("crop", f"interval {lo}:{hi} is outside an axis of size {size}")
    return lo, hi


@dataclass(frozen=True)
class Selection:
    source: Source
    bounds: tuple[tuple[int, int], ...]
    factors: tuple[int, ...]
    order: tuple[int, ...]

    @classmethod
    def select(cls, source: Source, crop: Crop | None, binning: int, axis: str | None) -> Selection:
        if crop is not None and len(crop) != 3:
            raise InvalidParameter("crop", "supply one interval per stored axis")
        if isinstance(binning, bool) or not isinstance(binning, int) or binning < 1:
            raise InvalidParameter("binning", "binning must be a positive integer")
        factors = tuple(1 if a.name == "view" else binning for a in source.axes)
        bounds = tuple(
            interval(p, n) for p, n in zip(crop or (None,) * 3, source.shape, strict=True)
        )
        if any(hi - lo < f for (lo, hi), f in zip(bounds, factors, strict=True)):
            raise InvalidParameter("binning", "the selected interval must contain a whole bin")
        names = [a.name for a in source.axes]
        if axis is not None and axis not in names:
            raise InvalidParameter("axis", f"expected one of {', '.join(names)}")
        first = 0 if axis is None else names.index(axis)
        return cls(source, bounds, factors, (first, *(i for i in range(3) if i != first)))

    @property
    def stored_shape(self) -> tuple[int, ...]:
        return tuple((hi - lo) // f for (lo, hi), f in zip(self.bounds, self.factors, strict=True))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.stored_shape[i] for i in self.order)

    @property
    def axes(self) -> tuple[Axis, ...]:
        axes = []
        for i in self.order:
            axis = self.source.axes[i]
            coord = axis.coordinates
            if isinstance(coord, Uniform):
                start = self.bounds[i][0] + (self.factors[i] - 1) / 2
                coord = Uniform(
                    coord.origin + start * coord.spacing, coord.spacing * self.factors[i]
                )
                axis = msgspec.structs.replace(axis, coordinates=coord)
            axes.append(axis)
        return tuple(axes)

    @property
    def block_depth(self) -> int:
        # Reserve space for binning, quantization and the codec in the same 256 MiB budget.
        plane = math.prod(self.source.shape[1:]) * self.source.dtype.itemsize
        return max(1, (32 << 20) // (plane * self.factors[0]))

    def read(self, first: int, stop: int) -> np.ndarray:
        starts = [lo for lo, _ in self.bounds]
        dims = list(self.stored_shape)
        starts[0] += first * self.factors[0]
        dims[0] = stop - first
        region = tuple(
            slice(s, s + n * f) for s, n, f in zip(starts, dims, self.factors, strict=True)
        )
        block = self.source.read(region)
        if self.factors != (1, 1, 1):
            bins = tuple(x for pair in zip(dims, self.factors, strict=True) for x in pair)
            block = block.reshape(bins).mean(axis=(1, 3, 5), dtype=np.float32)
        return block

    def percentiles(self) -> tuple[float, float]:
        depth = self.stored_shape[0]
        samples = []
        for z in np.linspace(0, depth - 1, min(depth, 16)).round().astype(int):
            progress.check_canceled()
            plane = self.read(z, z + 1).ravel()
            sample = plane[:: max(1, math.ceil(plane.size / 2**20))]
            samples.append(sample[np.isfinite(sample)])
        data = np.concatenate(samples)
        if data.size:
            lo, hi = np.percentile(data, [0.1, 99.9])
            if lo < hi:
                return float(lo), float(hi)
        raise InvalidParameter("range", "no spread in the finite sample; supply range=(low, high)")


@dataclass(frozen=True)
class Values:
    dtype: str
    unit: str
    range: tuple[float, float] | None
    range_from: str | None
    scale: float
    offset: float

    @classmethod
    def resolve(
        cls, selection: Selection, dtype: str | None, bounds: Sequence[float] | None
    ) -> Values:
        source = selection.source
        if source.domain == "mask":
            if dtype is not None or bounds is not None or max(selection.factors) != 1:
                raise InvalidParameter("source", "a mask takes no dtype, range or binning")
            return cls("uint8", "1", None, None, 1.0, 0.0)
        dtype = "float32" if dtype is None else dtype
        if dtype == "float32":
            if bounds is not None:
                raise InvalidParameter("range", "range applies only to integer export")
            return cls(dtype, source.unit, None, None, 1.0, 0.0)
        if dtype not in ("uint8", "uint16"):
            raise InvalidParameter("dtype", "expected float32, uint16 or uint8")
        if bounds is None:
            low, high = selection.percentiles()
            origin = f"percentiles 0.1 and 99.9 of {min(selection.stored_shape[0], 16)} planes"
        else:
            if len(bounds) != 2:
                raise InvalidParameter("range", "expected two values (low, high)")
            low, high = map(float, bounds)
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise InvalidParameter("range", "range must be finite with low < high")
            origin = "given"
        return cls(dtype, source.unit, (low, high), origin, (high - low) / np.iinfo(dtype).max, low)


@dataclass
class Blocks:
    selection: Selection
    values: Values
    below: int = 0
    above: int = 0

    def __iter__(self) -> Iterator[tuple[Region, np.ndarray]]:
        selection, values = self.selection, self.values
        depth = selection.stored_shape[0]
        dtype = np.dtype(values.dtype).newbyteorder("<")
        stored_axis = selection.order.index(0)
        for start in range(0, depth, selection.block_depth):
            progress.check_canceled()
            end = min(depth, start + selection.block_depth)
            block = selection.read(start, end)
            if values.range is not None:
                low, high = values.range
                finite = np.isfinite(block)
                if not finite.all():
                    counts = np.count_nonzero(~finite, axis=(1, 2))
                    k = int(np.flatnonzero(counts)[0])
                    raise ValueError(
                        f"plane {start + k} has {counts[k]} non-finite values; export float32"
                    )
                self.below += int(np.count_nonzero(block < low))
                self.above += int(np.count_nonzero(block > high))
                scaled = block.astype(np.float64)
                np.clip(scaled, low, high, out=scaled)
                scaled -= low
                scaled /= values.scale
                np.rint(scaled, out=scaled)
                block = scaled
            elif selection.source.domain == "mask" and (block.min() < 0 or block.max() > 255):
                raise ValueError("mask labels must fit uint8 (0 to 255)")
            region = [slice(None)] * 3
            region[stored_axis] = slice(start, end)
            yield tuple(region), block.astype(dtype, copy=False).transpose(selection.order)
            for _ in range(end - start):
                progress.step("slices")
