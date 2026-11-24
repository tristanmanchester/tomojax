"""PNG images of registered planes, sinograms and pose tables, with labels.

A plane whose two axes are spatial and share a unit is resampled to square
physical pixels; any other image (a sinogram's view axis, say) keeps its samples.
Sizes are capped before anything is allocated. Grey levels come from explicit
limits, which several panels can share. Files are written to a fixed temporary
name beside the target and moved onto it once whole.
"""

from __future__ import annotations

from functools import cache
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tomojax.wire.arrays import Uniform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from tomojax.geometry import Grid
    from tomojax.progress import Sample
    from tomojax.wire.arrays import Axis

PLANE_SIDE = 2048  # the longest side of one plane's image
SHEET_SIDE = 1600  # the longest side of a contact sheet
_SMALLEST = 256  # smaller images are enlarged by a whole factor to at least this side
_STRIP = 22  # label height in pixels
_SPATIAL = frozenset({"volume", "detector"})

type Limits = tuple[float, float]


@cache
def _font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def limits(*images: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> Limits:
    """Grey-level limits shared by ``images``: percentiles of their finite values."""
    values = [np.asarray(image, np.float32) for image in images]
    finite = np.concatenate([v[np.isfinite(v)] for v in values]) if values else np.empty(0)
    if finite.size == 0:
        return (0.0, 1.0)
    low, high = (float(x) for x in np.percentile(finite, [lower, upper]))
    return (low, high) if high > low else (low, low + 1.0)


def to_uint8(image: np.ndarray, bounds: Limits) -> np.ndarray:
    """``image`` scaled to 0 to 255 between ``bounds``; non-finite values are black."""
    low, high = bounds
    data = np.asarray(image, np.float32)
    scaled = np.clip((data - low) / (high - low), 0.0, 1.0)
    return np.where(np.isfinite(scaled), np.round(scaled * 255), 0).astype(np.uint8)


def _pitch(axis: Axis) -> float | None:
    coordinates = axis.coordinates
    return abs(coordinates.spacing) if isinstance(coordinates, Uniform) else None


def display_shape(shape: tuple[int, int], axes: Sequence[Axis], side: int) -> tuple[int, int]:
    """The (rows, columns) to draw: square physical pixels where the axes allow, capped.

    Both axes spatial in one unit with known pitches: rows and columns are scaled so
    each pixel covers the same length. Otherwise the samples are kept. The result is
    then reduced by an integer factor until its longest side fits ``side``, or enlarged
    by one when it is shorter than 256 pixels.
    """
    rows, columns = shape
    first, second = axes
    pitches = (_pitch(first), _pitch(second))
    commensurate = (
        first.frame in _SPATIAL
        and second.frame in _SPATIAL
        and first.unit == second.unit
        and None not in pitches
        and all(p and p > 0 for p in pitches)
    )
    if commensurate:
        row_pitch, column_pitch = pitches
        assert row_pitch is not None
        assert column_pitch is not None
        finest = min(row_pitch, column_pitch)
        rows = max(1, round(rows * row_pitch / finest))
        columns = max(1, round(columns * column_pitch / finest))
    longest = max(rows, columns)
    if longest < min(_SMALLEST, side):  # a small volume, enlarged to be seen
        enlarge = min(_SMALLEST, side) // longest
        return rows * enlarge, columns * enlarge
    factor = max(1, math.ceil(longest / side))
    return max(1, rows // factor), max(1, columns // factor)


def resample(image: np.ndarray, shape: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    """``image`` (float) at ``shape``: bilinearly, or by nearest sample to keep rows sharp."""
    data = np.asarray(image, np.float32)
    if data.shape == shape:
        return data
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = Image.fromarray(data).resize((shape[1], shape[0]), method)
    return np.asarray(resized, np.float32)


def sample_image(sample: Sample, *, side: int = PLANE_SIDE) -> np.ndarray:
    """A registered 2-D sample as a float image ready for :func:`to_uint8`."""
    data = np.asarray(sample.data)
    if data.ndim != 2:
        raise ValueError(f"an image needs two axes, not {data.ndim}")
    shape = (int(data.shape[0]), int(data.shape[1]))
    return resample(data, display_shape(shape, sample.axes, side))


def labelled(image: np.ndarray, label: str) -> Image.Image:
    """A greyscale ``image`` (uint8) under a strip of white ``label`` text."""
    picture = Image.fromarray(np.asarray(image, np.uint8))
    width = max(picture.width, math.ceil(_font().getlength(label)) + 8)
    canvas = Image.new("L", (width, picture.height + _STRIP), 0)
    canvas.paste(picture, (0, _STRIP))
    ImageDraw.Draw(canvas).text((4, 3), label, fill=255, font=_font())
    return canvas


def sheet(panels: Sequence[Image.Image], *, columns: int) -> Image.Image:
    """``panels`` in a grid, ``columns`` wide, each cell the size of the largest."""
    if not panels:
        raise ValueError("a sheet needs at least one panel")
    width = max(p.width for p in panels)
    height = max(p.height for p in panels)
    columns = max(1, min(columns, len(panels)))
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("L", (columns * width, rows * height), 0)
    for i, panel in enumerate(panels):
        canvas.paste(panel, ((i % columns) * width, (i // columns) * height))
    return canvas


def outlined(panel: Image.Image) -> Image.Image:
    """``panel`` with a white frame, to pick it out in a sheet."""
    framed = panel.copy()
    ImageDraw.Draw(framed).rectangle(
        (0, 0, framed.width - 1, framed.height - 1), outline=255, width=3
    )
    return framed


def pose_plot(
    table: np.ndarray, names: Sequence[str], units: Sequence[str], title: str
) -> Image.Image:
    """Each pose column against the view index, one small panel per column."""
    data = np.asarray(table, np.float64)
    views = data.shape[0]
    width, height, margin = 320, 120, 8
    panels: list[Image.Image] = []
    for column, (name, unit) in enumerate(zip(names, units, strict=True)):
        values = data[:, column]
        finite = values[np.isfinite(values)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
        span = high - low or 1.0
        canvas = Image.new("L", (width, height + _STRIP), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 3), f"{name} ({unit})  {low:.4g} … {high:.4g}", fill=255, font=_font(12))
        draw.rectangle((0, _STRIP, width - 1, height + _STRIP - 1), outline=80)
        points = [
            (
                margin + (width - 2 * margin) * (i / max(views - 1, 1)),
                _STRIP + margin + (height - 2 * margin) * (1 - (v - low) / span),
            )
            for i, v in enumerate(values)
            if math.isfinite(v)
        ]
        if len(points) > 1:
            draw.line(points, fill=255, width=1)
        panels.append(canvas)
    grid = sheet(panels, columns=3)
    canvas = Image.new("L", (grid.width, grid.height + _STRIP), 0)
    canvas.paste(grid, (0, _STRIP))
    ImageDraw.Draw(canvas).text((4, 3), title, fill=255, font=_font())
    return canvas


def save_png(image: Image.Image, path: Path) -> Path:
    """Write ``image`` to ``path`` through ``.<name>.tmp`` beside it, moved on once whole."""
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as stream:
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def save_volume_plane(path: Path, volume: ArrayLike, grid: Grid, axis: str, label: str) -> Path:
    """Write the central ``axis`` plane of a reconstructed volume, labelled."""
    from tomojax.core.registration import planes

    sample = planes(np.asarray(volume), grid)[axis]
    image = sample_image(sample)
    return save_png(labelled(to_uint8(image, limits(image)), label), path)


__all__ = [
    "PLANE_SIDE",
    "SHEET_SIDE",
    "Limits",
    "display_shape",
    "labelled",
    "limits",
    "outlined",
    "pose_plot",
    "resample",
    "sample_image",
    "save_png",
    "save_volume_plane",
    "sheet",
    "to_uint8",
]
