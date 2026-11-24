"""Render registered tomography planes with one explicit scale per comparison."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Panel:
    title: str
    pixels: np.ndarray
    labels: tuple[str, ...]
    unit: str
    aspect: float = 1.0


def limits(panels: list[Panel]) -> tuple[float, float]:
    finite = np.concatenate([p.pixels[np.isfinite(p.pixels)] for p in panels])
    if not finite.size:
        return 0, 1
    lo, hi = np.percentile(finite, (1, 99))
    if hi == lo:
        lo, hi = finite.min(), finite.max()
    return float(lo), float(hi)


def png(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def montage(panels: list[Panel], intensity: tuple[float, float]) -> bytes:
    """Draw up to three planes side by side, mapping equal intensities to equal greys."""
    lo, hi = intensity
    labels = [
        [
            line
            for label in (p.title, *p.labels, f"limits [{lo:.5g}, {hi:.5g}] {p.unit}")
            for line in wrap(label, 49, max_lines=3, placeholder="...")
        ]
        for p in panels
    ]
    width, height = 328, 240
    image = Image.new("L", (width * len(panels), height + 18 * max(map(len, labels)) + 10), 28)
    draw = ImageDraw.Draw(image)
    for i, (panel, lines) in enumerate(zip(panels, labels, strict=True)):
        data = panel.pixels.astype(float)
        scaled = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)
        grey = np.uint8(np.nan_to_num(np.clip(scaled, 0, 1)) * 255)
        h, w = panel.pixels.shape[:2]
        scale = min((width - 12) / w, height / (h * panel.aspect))
        resized = Image.fromarray(grey).resize(
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
        draw.text((left, top), f"{col['name']}   [{lo:.4g}, {hi:.4g}] {col['unit']}", fill="black")
        x = (views - start) / (stop - start) if stop > start else np.full(len(views), 0.5)
        y = (series - lo) / (hi - lo) if hi > lo else np.full(len(views), 0.5)
        box = (left + 2, top + 22, left + 256, top + 146)
        draw.rectangle(box, outline="#abb4bd")
        points = list(zip(box[0] + x * 254, box[3] - y * 124, strict=True))
        if len(points) > 1:
            draw.line(points, fill="#0b557d", width=2)
        else:
            draw.point(points[0], fill="#0b557d")
        draw.text((left, top + 151), f"view [{start:.5g}, {stop:.5g}] {view_unit}", fill="black")
    return png(image)
