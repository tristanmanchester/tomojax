"""Render registered tomography planes with one explicit scale per comparison."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Panel:
    """A labelled 2D image with its pixel aspect ratio, physical unit and domain."""

    title: str
    pixels: np.ndarray
    labels: tuple[str, ...]
    unit: str
    aspect: float = 1.0
    domain: str = ""


def label_number(value: float) -> str:
    """Format panel annotations without displaying roundoff around zero."""
    return "0" if abs(value) < 1e-12 else f"{value:.5g}"


def limits(panels: list[Panel]) -> tuple[float, float]:
    """Choose shared intensity limits from the finite pixels of all panels."""
    finite = np.concatenate([p.pixels[np.isfinite(p.pixels)] for p in panels])
    if not finite.size:
        return 0, 1
    lo, hi = np.percentile(finite, (1, 99))
    if hi == lo:
        lo, hi = finite.min(), finite.max()
    return float(lo), float(hi)


def png(image: Image.Image) -> bytes:
    """Encode an image as PNG bytes for display."""
    stream = BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def raster(panel: Panel, lo: float, hi: float) -> Image.Image:
    """Map panel intensities to the display range before resizing."""
    data = panel.pixels.astype(float)
    scaled = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)
    grey = np.uint8(np.nan_to_num(np.clip(scaled, 0, 1)) * 255)
    return Image.fromarray(grey)


def thumbnail(image: Image.Image) -> bytes:
    """Fit an unlabelled image into a single MCP panel."""
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return png(image)


def montage(panels: list[Panel], scales: list[tuple[float, float]]) -> bytes:
    """Draw up to three planes side by side, mapping equal intensities to equal greys."""
    if len(panels) == 1 and not panels[0].labels:
        return thumbnail(raster(panels[0], *scales[0]))

    labels = [
        [
            line
            for label in (
                p.title,
                *p.labels,
                f"limits [{label_number(lo)}, {label_number(hi)}] {p.unit}",
            )
            for line in wrap(label, 49, max_lines=3, placeholder="...")
        ]
        for p, (lo, hi) in zip(panels, scales, strict=True)
    ]
    width, height = 328, 240
    image = Image.new("L", (width * len(panels), height + 18 * max(map(len, labels)) + 10), 28)
    draw = ImageDraw.Draw(image)
    for i, (panel, lines) in enumerate(zip(panels, labels, strict=True)):
        h, w = panel.pixels.shape[:2]
        scale = min((width - 12) / w, height / (h * panel.aspect))
        resized = raster(panel, *scales[i]).resize(
            (max(1, round(w * scale)), max(1, round(h * panel.aspect * scale))),
            Image.Resampling.NEAREST,
        )
        image.paste(
            resized, (i * width + (width - resized.width) // 2, (height - resized.height) // 2)
        )
        for j, label in enumerate(lines):
            draw.text((i * width + 6, height + 5 + j * 18), label, fill=248)
    return png(image)


def pose_plot(values: np.ndarray, columns: list[dict], views: np.ndarray, view_unit: str) -> bytes:
    """Six acquisition-ordered traces with labelled horizontal and vertical scales."""
    image = Image.new("RGB", (840, 438), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), "Per-view pose corrections", fill="black")
    start, stop = float(views.min()), float(views.max())
    for i, col in enumerate(columns):
        left, top = 12 + (i % 3) * 276, 36 + (i // 3) * 196
        series = values[:, i]
        lo, hi = float(series.min()), float(series.max())
        draw.text(
            (left, top),
            f"{col['name']}   [{label_number(lo)}, {label_number(hi)}] {col['unit']}",
            fill="black",
        )
        x = (views - start) / (stop - start) if stop > start else np.full(len(views), 0.5)
        y = (series - lo) / (hi - lo) if hi > lo else np.full(len(views), 0.5)
        box = (left + 2, top + 22, left + 256, top + 146)
        draw.rectangle(box, outline="#abb4bd")
        points = list(zip(box[0] + x * 254, box[3] - y * 124, strict=True))
        if len(points) > 1:
            draw.line(points, fill="#0b557d", width=2)
        else:
            draw.point(points[0], fill="#0b557d")
        draw.text(
            (left, top + 151),
            f"view [{label_number(start)}, {label_number(stop)}] {view_unit}",
            fill="black",
        )
    return png(image)
