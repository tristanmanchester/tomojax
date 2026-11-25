"""Marks and feature outlines placed in source array coordinates and drawn on rendered planes.

A mark's coordinates are array indices in the source's storage order: (z, y, x) for
volumes and (view, v, u) for projections, so the same mark lands on the same voxel in
every plane, ROI, stride and zoom. A position may instead be a pixel in an image an
earlier view returned, which that render's placements convert to array indices. A
plane shows a mark when the mark reaches within ``TOLERANCE`` index units of it along
the plane normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import floor, log10, sqrt
from typing import TYPE_CHECKING, Annotated, Literal

import msgspec
import numpy as np
from PIL import Image, ImageDraw

from tomojax.cli.images import Panel, compose, png
from tomojax.wire.arrays import Record

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

TOLERANCE = 2.0
PALETTE = {
    "yellow": (255, 214, 0),
    "cyan": (0, 229, 255),
    "magenta": (255, 64, 245),
    "orange": (255, 140, 0),
    "green": (110, 255, 60),
    "red": (255, 48, 48),
}
type Colour = Literal["yellow", "cyan", "magenta", "orange", "green", "red"]
Radius = Annotated[float, msgspec.Meta(gt=0)]


class PixelPosition(Record):
    """A pixel (x right, y down, from the top-left corner) of a returned image."""

    pixel: tuple[float, float]
    image: str


type Position = tuple[float, ...] | PixelPosition


class Mark(Record, kw_only=True):
    """Fields every mark shares: a short label drawn beside it and its colour."""

    label: str | None = None
    colour: Colour = "yellow"


class PointMark(Mark, tag="point", kw_only=True):
    """A dot at one array position."""

    at: Position


class TextMark(Mark, tag="text", kw_only=True):
    """A label at one array position, without a symbol."""

    at: Position


class ArrowMark(Mark, tag="arrow", kw_only=True):
    """An arrow whose head points at ``to``."""

    tail: Position = msgspec.field(name="from")
    to: Position


class BoxMark(Mark, tag="box", kw_only=True):
    """An axis-aligned box between two opposite corners, inclusive."""

    corners: tuple[Position, Position]


class CircleMark(Mark, tag="circle", kw_only=True):
    """A sphere of ``radius`` index units; each plane shows its cross-section."""

    at: Position
    radius: Radius


class PolylineMark(Mark, tag="polyline", kw_only=True):
    """An open path through two or more positions."""

    points: tuple[Position, ...]


class PolygonMark(Mark, tag="polygon", kw_only=True):
    """A closed outline through three or more positions."""

    points: tuple[Position, ...]


type AnyMark = PointMark | TextMark | ArrowMark | BoxMark | CircleMark | PolylineMark | PolygonMark


@dataclass(frozen=True)
class Frame:
    """Where a rendered plane sits in its source array.

    ``axes`` names every source dimension. ``rows`` and ``cols`` give the source
    dimension, first index and step of each displayed axis; ``normal`` the fixed
    dimension and its index, if any.
    """

    axes: tuple[str, ...]
    rows: tuple[int, int, int]
    cols: tuple[int, int, int]
    normal: tuple[int, int] | None


@dataclass(frozen=True)
class Cell:
    """The rectangle a plane's samples occupy in the returned image."""

    image: int
    x: float
    y: float
    sx: float
    sy: float

    def pixel(self, row: float, col: float) -> tuple[float, float]:
        """Image position of a sample-grid position, with samples centred on integers."""
        return self.x + (col + 0.5) * self.sx, self.y + (row + 0.5) * self.sy


@dataclass(frozen=True)
class Placement:
    """One plane of a returned image: where it sits in the source and in the image."""

    frame: Frame
    cell: Cell
    size: tuple[int, int]  # sample rows and columns

    def contains(self, x: float, y: float) -> bool:
        """Whether an image pixel falls on this plane."""
        c = self.cell
        return c.x <= x < c.x + self.size[1] * c.sx and c.y <= y < c.y + self.size[0] * c.sy

    def index(self, row: float, col: float) -> tuple[float, ...]:
        """Array indices of a sample-grid position on this plane."""
        f = self.frame
        position = [0.0] * len(f.axes)
        position[f.rows[0]] = f.rows[1] + row * f.rows[2]
        position[f.cols[0]] = f.cols[1] + col * f.cols[2]
        if f.normal is not None:
            position[f.normal[0]] = f.normal[1]
        return tuple(position)

    def located(self, x: float, y: float) -> tuple[float, ...]:
        """Array indices of an image pixel on this plane."""
        c = self.cell
        return self.index((y - c.y) / c.sy - 0.5, (x - c.x) / c.sx - 0.5)


def locate(
    marks: Sequence[AnyMark], renders: Callable[[str], Sequence[Placement]]
) -> tuple[AnyMark, ...]:
    """Convert pixel positions to array indices through the renders they refer to."""

    def index(position: Position, field: str) -> tuple[float, ...]:
        if not isinstance(position, PixelPosition):
            return position
        x, y = position.pixel
        plane = next((p for p in renders(position.image) if p.contains(x, y)), None)
        if plane is None:
            raise ValueError(f"{field}: pixel ({x:g}, {y:g}) is not on a plane of {position.image}")
        return tuple(round(v, 2) for v in plane.located(x, y))

    def one(mark: AnyMark, field: str) -> AnyMark:
        if isinstance(mark, ArrowMark):
            return msgspec.structs.replace(
                mark, tail=index(mark.tail, field), to=index(mark.to, field)
            )
        if isinstance(mark, BoxMark):
            corners = (index(mark.corners[0], field), index(mark.corners[1], field))
            return msgspec.structs.replace(mark, corners=corners)
        if isinstance(mark, PolylineMark | PolygonMark):
            return msgspec.structs.replace(mark, points=tuple(index(q, field) for q in mark.points))
        return msgspec.structs.replace(mark, at=index(mark.at, field))

    return tuple(one(mark, f"marks[{i}]") for i, mark in enumerate(marks))


def parse(marks: Sequence[object]) -> tuple[AnyMark, ...]:
    """Validate marks given as JSON-like dictionaries or as mark records."""
    return msgspec.convert(list(marks), type=tuple[AnyMark, ...], from_attributes=True)


def name(mark: AnyMark) -> str:
    """The mark's type, as written in its JSON."""
    return str(mark.__struct_config__.tag)


def positions(mark: AnyMark) -> tuple[Position, ...]:
    """The array positions a mark is defined by."""
    if isinstance(mark, ArrowMark):
        return (mark.tail, mark.to)
    if isinstance(mark, BoxMark):
        return mark.corners
    if isinstance(mark, PolylineMark | PolygonMark):
        return mark.points
    return (mark.at,)


def check(marks: Sequence[AnyMark], axes: Sequence[str]) -> None:
    """Reject marks whose coordinates do not match the source's axes."""
    for i, mark in enumerate(marks):
        least = 3 if isinstance(mark, PolygonMark) else 2 if isinstance(mark, PolylineMark) else 1
        points = positions(mark)
        if len(points) < least:
            raise ValueError(f"marks[{i}] needs at least {least} points")
        for point in points:
            if isinstance(point, PixelPosition):
                raise ValueError(f"marks[{i}] gives a pixel; here give array indices")
            if len(point) != len(axes):
                raise ValueError(
                    f"marks[{i}] has {len(point)} coordinates; give one per source axis "
                    f"({', '.join(axes)}) as array indices"
                )
        if isinstance(mark, TextMark) and not mark.label:
            raise ValueError(f"marks[{i}] is text and needs a label")


@dataclass(frozen=True)
class Shape:
    """A mark projected onto one plane's sample grid, or why it is not shown there."""

    kind: str
    points: tuple[tuple[float, float], ...]
    radius: tuple[float, float] | None = None
    distance: float | None = None
    reason: str | None = None


def indices(position: Position) -> tuple[float, ...]:
    """A position known to be in array indices (pixels are located first)."""
    assert not isinstance(position, PixelPosition), "locate pixel positions before projecting"
    return position


def project(mark: AnyMark, frame: Frame) -> Shape:
    """Map a mark onto one plane in sample-grid coordinates (row, col)."""
    kind = name(mark)
    (rd, r0, rs), (cd, c0, cs) = frame.rows, frame.cols

    def grid(point: Position) -> tuple[float, float]:
        point = indices(point)
        return ((point[rd] - r0) / rs, (point[cd] - c0) / cs)

    def away(point: Position) -> float:
        return 0.0 if frame.normal is None else indices(point)[frame.normal[0]] - frame.normal[1]

    if isinstance(mark, BoxMark):
        a, b = (indices(c) for c in mark.corners)
        if frame.normal is not None:
            d = frame.normal[0]
            lo, hi = sorted((a[d], b[d]))
            gap = max(lo - frame.normal[1], frame.normal[1] - hi, 0.0)
            if gap > TOLERANCE:
                return Shape(kind, (), distance=gap, reason="off_plane")
        return Shape(kind, (grid(a), grid(b)))
    if isinstance(mark, CircleMark):
        d = away(mark.at)
        if abs(d) > mark.radius + TOLERANCE:
            return Shape(kind, (), distance=abs(d), reason="off_plane")
        # The cross-section of the sphere, or its rim when the plane only grazes it.
        r = sqrt(max(mark.radius**2 - d**2, 0.0))
        return Shape(kind, (grid(mark.at),), radius=(r / rs, r / cs), distance=abs(d))
    anchor = mark.to if isinstance(mark, ArrowMark) else None
    points = positions(mark)
    checked = (anchor,) if anchor is not None else points
    worst = max(abs(away(p)) for p in checked)
    if worst > TOLERANCE:
        return Shape(kind, (), distance=worst, reason="off_plane")
    return Shape(kind, tuple(grid(p) for p in points), distance=worst)


def anchor(shape: Shape) -> tuple[float, float]:
    """The sample-grid position a mark's reported pixel and value refer to."""
    if shape.kind == "arrow":
        return shape.points[1]
    rows, cols = zip(*shape.points, strict=True)
    return (sum(rows) / len(rows), sum(cols) / len(cols))


def footprint(shape: Shape, size: tuple[int, int]) -> np.ndarray:
    """Samples covered by a shape on a (rows, cols) grid, for the value under it."""
    h, w = size
    mask = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    xy = [(c, r) for r, c in shape.points]
    if shape.kind == "box":
        (x0, y0), (x1, y1) = xy
        draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), fill=1)
    elif shape.kind == "circle" and shape.radius is not None:
        (x, y), (rr, rc) = xy[0], shape.radius
        draw.ellipse((x - rc, y - rr, x + rc, y + rr), fill=1)
    elif shape.kind == "polygon":
        draw.polygon(xy, fill=1)
    elif shape.kind == "polyline":
        draw.line(xy, fill=1)
    selected = np.array(mask, dtype=bool)
    if not selected.any():
        r, c = anchor(shape)
        i, j = round(r), round(c)
        if 0 <= i < h and 0 <= j < w:
            selected[i, j] = True
    return selected


def outline(mask: np.ndarray) -> np.ndarray:
    """The boundary samples of a boolean mask: set samples with an unset 4-neighbour."""
    padded = np.pad(mask, 1, constant_values=False)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:] & mask
    return mask & ~interior


def nice_step(span: float, count: int = 8) -> float:
    """The 1, 2 or 5 times power-of-ten step closest to ``count`` ticks over ``span``."""
    raw = max(span, 1.0) / count
    base = 10 ** floor(log10(raw))
    return min((m * base for m in (1, 2, 5, 10)), key=lambda step: abs(step - raw))


def ticks(start: int, step: int, n: int) -> list[tuple[float, int]]:
    """Index labels and their sample-grid positions along one displayed axis."""
    last = start + step * (n - 1)
    every = max(1, round(nice_step(last - start)))
    first = -(-start // every) * every
    return [((v - start) / step, v) for v in range(first, last + 1, every)]


def draw_ticks(draw: ImageDraw.ImageDraw, frame: Frame, cell: Cell, size: tuple[int, int]) -> None:
    """Faint index ticks with labels along the top and left image edges."""
    h, w = size
    grey = (150, 150, 150)
    left, top = cell.x, cell.y
    right, bottom = cell.x + w * cell.sx, cell.y + h * cell.sy
    for position, value in ticks(frame.cols[1], frame.cols[2], w):
        x, _ = cell.pixel(0, position)
        draw.line((x, top - 4, x, top - 1), fill=grey)
        draw.text((x, top - 5), str(value), fill=grey, anchor="mb")
    for position, value in ticks(frame.rows[1], frame.rows[2], h):
        _, y = cell.pixel(position, 0)
        draw.line((left - 4, y, left - 1, y), fill=grey)
        draw.text((left - 6, y), str(value), fill=grey, anchor="rm")
    draw.text((right + 3, top - 5), frame.axes[frame.cols[0]], fill=grey, anchor="lb")
    draw.text((left - 6, bottom + 3), frame.axes[frame.rows[0]], fill=grey, anchor="rt")


def draw_shape(draw: ImageDraw.ImageDraw, shape: Shape, cell: Cell, mark: AnyMark) -> None:
    """Draw a projected mark with a dark halo so it reads on any grey."""
    colour = PALETTE[mark.colour]
    xy = [cell.pixel(r, c) for r, c in shape.points]
    for fill, width in (((0, 0, 0), 4), (colour, 2)):
        if shape.kind == "point":
            (x, y) = xy[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=fill, width=width)
            draw.point((x, y), fill=fill)
        elif shape.kind == "arrow":
            (x0, y0), (x1, y1) = xy
            draw.line((x0, y0, x1, y1), fill=fill, width=width)
            length = max(np.hypot(x1 - x0, y1 - y0), 1e-9)
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            for side in (1, -1):
                hx = x1 - 9 * ux + side * 5 * uy
                hy = y1 - 9 * uy - side * 5 * ux
                draw.line((x1, y1, hx, hy), fill=fill, width=width)
        elif shape.kind == "box":
            (x0, y0), (x1, y1) = xy
            half_x, half_y = cell.sx / 2, cell.sy / 2
            draw.rectangle(
                (
                    min(x0, x1) - half_x,
                    min(y0, y1) - half_y,
                    max(x0, x1) + half_x,
                    max(y0, y1) + half_y,
                ),
                outline=fill,
                width=width,
            )
        elif shape.kind == "circle" and shape.radius is not None:
            (x, y), (rr, rc) = xy[0], shape.radius
            rx, ry = max(rc * cell.sx, 2), max(rr * cell.sy, 2)
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=fill, width=width)
        elif shape.kind in ("polyline", "polygon"):
            path = [*xy, xy[0]] if shape.kind == "polygon" else xy
            draw.line(path, fill=fill, width=width, joint="curve")


def draw_label(draw: ImageDraw.ImageDraw, shape: Shape, cell: Cell, mark: AnyMark) -> None:
    """Write a mark's label beside its anchor, or in place of a text mark's symbol."""
    colour = PALETTE[mark.colour]
    if mark.label:
        r, c = anchor(shape)
        x, y = cell.pixel(r, c)
        offset = 0 if shape.kind == "text" else 7
        draw.text(
            (x + offset, y - offset),
            mark.label,
            fill=colour,
            anchor="mm" if shape.kind == "text" else "lb",
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )


@dataclass(frozen=True)
class Overlay:
    """Marks, feature outlines and ticks for one rendered plane."""

    frame: Frame
    marks: tuple[AnyMark, ...] = ()
    outlines: tuple[tuple[np.ndarray, Colour], ...] = ()
    ticks: bool = True


def draw(
    image: Image.Image,
    overlay: Overlay,
    cell: Cell,
    pixels: np.ndarray,
    scale: tuple[float, float],
) -> list[dict[str, object]]:
    """Draw one plane's overlay and report where each mark landed and the value under it.

    A drawn mark's report places its value in the plane: ``percentile`` is the share
    of the plane's samples below it and ``display`` its position between the display
    limits (0 the lower, 1 the upper), so a mark on background reads as a miss.
    """
    finite = np.sort(np.asarray(pixels, float)[np.isfinite(pixels)])
    lo, hi = scale
    canvas = ImageDraw.Draw(image)
    size = pixels.shape[:2]
    for mask, colour in overlay.outlines:
        width, height = round(size[1] * cell.sx), round(size[0] * cell.sy)
        scaled = Image.fromarray(np.uint8(mask) * 255).resize(
            (max(width, 1), max(height, 1)), Image.Resampling.NEAREST
        )
        edge = outline(np.asarray(scaled) > 0)
        # Two pixels wide, so the line survives a client scaling the image down.
        edge[1:] |= edge[:-1]
        edge[:, 1:] |= edge[:, :-1].copy()
        edge = Image.fromarray(np.uint8(edge) * 255)
        x, y = round(cell.x), round(cell.y)
        image.paste(PALETTE[colour], (x, y, x + edge.width, y + edge.height), mask=edge)
    if overlay.ticks:
        draw_ticks(canvas, overlay.frame, cell, size)
    # Shapes are clipped to the plane; labels may run into the margin.
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shapes = ImageDraw.Draw(layer)
    reports = []
    for mark in overlay.marks:
        shape = project(mark, overlay.frame)
        if shape.reason is not None:
            distance = round(shape.distance or 0.0, 3)
            reports.append({"drawn": False, "reason": "off_plane", "distance": distance})
            continue
        row, col = anchor(shape)
        covered = footprint(shape, size)
        extended = shape.kind in ("box", "circle", "polyline", "polygon")
        inside = -0.5 <= row < size[0] - 0.5 and -0.5 <= col < size[1] - 0.5
        if not (covered.any() if extended else inside):
            reports.append({"drawn": False, "reason": "outside_roi"})
            continue
        draw_shape(shapes, shape, cell, mark)
        draw_label(canvas, shape, cell, mark)
        x, y = cell.pixel(row, col)
        values = pixels[covered]
        values = values[np.isfinite(values)]
        value = float(f"{values.mean():.6g}") if values.size else None
        report: dict[str, object] = {
            "drawn": True,
            "image": cell.image,
            "pixel": [round(x, 1), round(y, 1)],
            "mean" if extended else "value": value,
        }
        if value is not None and finite.size:
            below = np.searchsorted(finite, value) / finite.size
            report["percentile"] = round(100 * float(below), 1)
            report["display"] = round((value - lo) / (hi - lo), 2) if hi > lo else None
        reports.append(report)
    box = (
        round(cell.x),
        round(cell.y),
        round(cell.x + size[1] * cell.sx),
        round(cell.y + size[0] * cell.sy),
    )
    clipped = layer.crop(box)
    image.paste(clipped, box[:2], mask=clipped)
    return reports


def frame_of(
    axes: Sequence[str],
    window: Sequence[tuple[int, int, int]],
    normal: int | None,
) -> Frame:
    """Describe a rendered plane from the source window it samples.

    ``window`` holds one (start, stop, step) per source dimension and ``normal`` is the
    dimension held fixed, or None for a two-dimensional source.
    """
    shown = [d for d in range(len(axes)) if d != normal]
    rows, cols = shown
    return Frame(
        tuple(axes),
        (rows, window[rows][0], window[rows][2]),
        (cols, window[cols][0], window[cols][2]),
        None if normal is None else (normal, window[normal][0]),
    )


def draw_grid(image: Image.Image, every: int = 100) -> None:
    """Dotted lines every ``every`` image pixels, labelled in pixels, for aiming by pixel."""
    canvas = ImageDraw.Draw(image)
    colour = (90, 170, 255)
    width, height = image.size
    for x in range(every, width, every):
        canvas.point([(x, y) for y in range(0, height, 4)], fill=colour)
        canvas.text((x + 2, 2), str(x), fill=colour)
    for y in range(every, height, every):
        canvas.point([(x, y) for x in range(0, width, 4)], fill=colour)
        canvas.text((2, y + 2), str(y), fill=colour)


def annotate(
    panels: list[Panel],
    scales: list[tuple[float, float]],
    overlays: list[Overlay],
    marks: Sequence[AnyMark],
    *,
    grid: bool = False,
) -> tuple[list[bytes], list[dict[str, object]], list[list[Placement]]]:
    """Montage planes three to an image with their overlays, and report every mark.

    Each mark's report gives its positions in array indices and lists, per plane,
    whether it was drawn, its pixel in which returned image and the value under it.
    The placements say where each plane landed, for marks given later as pixels.
    """
    placed: list[list[dict[str, object]]] = [[] for _ in marks]
    layouts: list[list[Placement]] = []

    def decorate(
        picture: Image.Image, j: int, rect: tuple[float, float, float, float], first: int
    ) -> None:
        i = first + j
        cell = Cell(first // 3, *rect)
        layouts[-1].append(Placement(overlays[i].frame, cell, panels[i].pixels.shape[:2]))
        for m, report in enumerate(draw(picture, overlays[i], cell, panels[i].pixels, scales[i])):
            placed[m].append({"panel": i, **report})

    images = []
    for i in range(0, len(panels), 3):
        layouts.append([])
        picture = compose(panels[i : i + 3], scales[i : i + 3], partial(decorate, first=i))
        if grid:
            draw_grid(picture)
        images.append(png(picture))
    reports = [
        {
            "type": name(mark),
            **({"label": mark.label} if mark.label else {}),
            "positions": [[round(v, 2) for v in indices(q)] for q in positions(mark)],
            "panels": found,
        }
        for mark, found in zip(marks, placed, strict=True)
    ]
    return images, reports, layouts
