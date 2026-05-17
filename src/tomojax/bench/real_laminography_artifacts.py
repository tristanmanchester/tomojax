"""Artifact writers for real-laminography staged runs."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tomojax.bench._real_laminography_visuals import resize_nearest_2d, save_uint8_png, scale_uint8
from tomojax.bench.real_laminography_planning import (
    real_lamino_global_z_to_local_index,
    real_lamino_xy_at_global_z,
)

if TYPE_CHECKING:
    from pathlib import Path


def real_lamino_orthos_image(volume: np.ndarray, *, preview_local_z: int) -> np.ndarray:
    """Return the three-panel orthos image used by real-lamino stage artifacts."""
    vol = np.asarray(volume, dtype=np.float32)
    cx, cy = vol.shape[0] // 2, vol.shape[1] // 2
    z = int(np.clip(preview_local_z, 0, vol.shape[2] - 1))
    panels = [
        scale_uint8(vol[:, :, z].T),
        scale_uint8(vol[:, cy, :].T),
        scale_uint8(vol[cx, :, :].T),
    ]
    gap = 8
    h = max(panel.shape[0] for panel in panels)
    w = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((h, w), dtype=np.uint8)
    x = 0
    for panel in panels:
        y = (h - panel.shape[0]) // 2
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return canvas


def save_real_lamino_z_stack(
    path: Path,
    volume: np.ndarray,
    *,
    grid: Any,
    full_nz: int,
    z_range: tuple[int, int],
    max_cols: int,
) -> str:
    """Write a tiled z-stack preview and return the written path."""
    z0, z1 = z_range
    images: list[tuple[int, np.ndarray]] = []
    for z in range(int(z0), int(z1) + 1):
        local = real_lamino_global_z_to_local_index(z, full_nz=full_nz, grid=grid)
        if 0 <= local < np.asarray(volume).shape[2]:
            images.append(
                (z, real_lamino_xy_at_global_z(volume, grid=grid, full_nz=full_nz, global_z=z))
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        iio.imwrite(path, np.zeros((16, 16), dtype=np.uint8))
        return str(path)
    vals = np.concatenate([np.ravel(img[np.isfinite(img)]) for _, img in images])
    lo, hi = np.percentile(vals, [1.0, 99.0]) if vals.size else (0.0, 1.0)
    cols = max(1, min(int(max_cols), len(images)))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows), dpi=140)
    axes_arr = np.ravel(np.asarray(axes))
    for ax, (z, img) in zip(axes_arr, images, strict=False):
        ax.imshow(
            scale_uint8(img, lo=float(lo), hi=float(hi)), cmap="gray", interpolation="nearest"
        )
        ax.set_title(f"z={z}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def write_real_lamino_stage_products(
    *,
    stage_dir: Path,
    volume: np.ndarray,
    grid: Any,
    full_nz: int,
    preview_global_z: int,
    stack_z_range: tuple[int, int],
    snapshot_max_cols: int,
    input_reference: np.ndarray | None,
    fallback_reference: np.ndarray | None = None,
    suffix: str = "aligned",
) -> dict[str, str]:
    """Write the standard real-lamino stage image products.

    The returned keys and filenames intentionally preserve the historical
    script-level artifact contract.
    """
    local_z = real_lamino_global_z_to_local_index(preview_global_z, full_nz=full_nz, grid=grid)
    xy = real_lamino_xy_at_global_z(volume, grid=grid, full_nz=full_nz, global_z=preview_global_z)
    ref = input_reference
    if ref is None:
        ref = fallback_reference
    if ref is not None:
        ref = resize_nearest_2d(ref, xy.shape)
        diff = xy - ref
    else:
        diff = np.zeros_like(xy)
    artifacts = {
        "aligned_xy": save_uint8_png(
            stage_dir / f"{suffix}_xy_global_z{preview_global_z:03d}.png", xy
        ),
        "delta_xy": save_uint8_png(
            stage_dir / f"delta_xy_global_z{preview_global_z:03d}.png", diff
        ),
        "orthos": str(stage_dir / "orthos.png"),
        "z_stack": save_real_lamino_z_stack(
            stage_dir / f"z_stack_global_z{stack_z_range[0]:03d}_{stack_z_range[1]:03d}.png",
            volume,
            grid=grid,
            full_nz=full_nz,
            z_range=stack_z_range,
            max_cols=int(snapshot_max_cols),
        ),
    }
    iio.imwrite(stage_dir / "orthos.png", real_lamino_orthos_image(volume, preview_local_z=local_z))
    return artifacts


__all__ = [
    "real_lamino_orthos_image",
    "save_real_lamino_z_stack",
    "write_real_lamino_stage_products",
]
