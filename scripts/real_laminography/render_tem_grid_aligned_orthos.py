"""Render TEM-grid orthogonal slices after in-plane grid alignment.

The laminography runner's default orthos are axis-aligned in reconstruction
coordinates. For a TEM grid that is rotated in the XY plane, those XZ/YZ cuts
can miss the grid bars. This script estimates the grid's in-plane angle from
the final XY slice, rotates the reconstructed slab around z, and writes
grid-aligned orthos plus a small offset sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy import ndimage

from tomojax.bench import (
    center_crop as _center_crop,
    largest_centered_square_inside_rotated_frame as _largest_centered_square_inside_rotated_frame,
    load_volume_array as _load_volume,
    scale_uint8 as _scale_uint8,
)


def _projection_score(image: np.ndarray, *, trim: int) -> float:
    if trim > 0:
        image = image[trim:-trim, trim:-trim]
    arr = np.asarray(image, dtype=np.float32)
    arr = arr - float(np.nanmedian(arr))
    row_profile = np.nanmean(arr, axis=1)
    col_profile = np.nanmean(arr, axis=0)
    row_profile = row_profile - ndimage.gaussian_filter1d(row_profile, sigma=8.0)
    col_profile = col_profile - ndimage.gaussian_filter1d(col_profile, sigma=8.0)
    return float(np.nanvar(row_profile) + np.nanvar(col_profile))


def _estimate_grid_angle(
    xy: np.ndarray,
    *,
    angle_min: float,
    angle_max: float,
    coarse_step: float,
    fine_radius: float,
    fine_step: float,
    trim: int,
) -> tuple[float, list[dict[str, float]]]:
    coarse_angles = np.arange(angle_min, angle_max + 0.5 * coarse_step, coarse_step)
    coarse: list[tuple[float, float]] = []
    for angle in coarse_angles:
        rotated = ndimage.rotate(xy, float(angle), reshape=False, order=1, mode="nearest")
        coarse.append((float(angle), _projection_score(rotated, trim=trim)))

    best_coarse = max(coarse, key=lambda item: item[1])[0]
    fine_min = max(angle_min, best_coarse - fine_radius)
    fine_max = min(angle_max, best_coarse + fine_radius)
    fine_angles = np.arange(fine_min, fine_max + 0.5 * fine_step, fine_step)
    fine: list[tuple[float, float]] = []
    for angle in fine_angles:
        rotated = ndimage.rotate(xy, float(angle), reshape=False, order=1, mode="nearest")
        fine.append((float(angle), _projection_score(rotated, trim=trim)))

    best = max(fine, key=lambda item: item[1])[0]
    samples = [{"angle_deg": angle, "score": score} for angle, score in fine]
    return float(best), samples


def _orthos_from_yxz(
    vol_yxz: np.ndarray,
    *,
    z_index: int,
    y_index: int,
    x_index: int,
    lo: float,
    hi: float,
    xy_crop: tuple[slice, slice] | None = None,
    gap: int = 8,
) -> np.ndarray:
    z = int(np.clip(z_index, 0, vol_yxz.shape[2] - 1))
    y = int(np.clip(y_index, 0, vol_yxz.shape[0] - 1))
    x = int(np.clip(x_index, 0, vol_yxz.shape[1] - 1))
    xy = vol_yxz[:, :, z]
    if xy_crop is not None:
        xy = xy[xy_crop]
    panels = [
        _scale_uint8(xy, lo=lo, hi=hi),
        _scale_uint8(vol_yxz[y, :, :].T, lo=lo, hi=hi),
        _scale_uint8(vol_yxz[:, x, :].T, lo=lo, hi=hi),
    ]
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((height, width), dtype=np.uint8)
    cursor = 0
    for panel in panels:
        top = (height - panel.shape[0]) // 2
        canvas[top : top + panel.shape[0], cursor : cursor + panel.shape[1]] = panel
        cursor += panel.shape[1] + gap
    return canvas


def _offset_sweep(
    vol_yxz: np.ndarray,
    *,
    z_index: int,
    offsets: list[int],
    lo: float,
    hi: float,
) -> np.ndarray:
    cy, cx = vol_yxz.shape[0] // 2, vol_yxz.shape[1] // 2
    panels = []
    for offset in offsets:
        panels.append(_scale_uint8(vol_yxz[int(np.clip(cy + offset, 0, vol_yxz.shape[0] - 1)), :, :].T, lo=lo, hi=hi))
    for offset in offsets:
        panels.append(_scale_uint8(vol_yxz[:, int(np.clip(cx + offset, 0, vol_yxz.shape[1] - 1)), :].T, lo=lo, hi=hi))
    gap = 6
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels) + gap * (len(panels) - 1)
    canvas = np.zeros((height, width), dtype=np.uint8)
    cursor = 0
    for panel in panels:
        top = (height - panel.shape[0]) // 2
        canvas[top : top + panel.shape[0], cursor : cursor + panel.shape[1]] = panel
        cursor += panel.shape[1] + gap
    return canvas


def render(args: argparse.Namespace) -> dict[str, Any]:
    volume_xyz = _load_volume(Path(args.volume), key=args.key)
    vol_yxz = np.transpose(volume_xyz, (1, 0, 2))
    z_index = vol_yxz.shape[2] // 2 if args.z_index is None else int(args.z_index)
    xy = vol_yxz[:, :, int(np.clip(z_index, 0, vol_yxz.shape[2] - 1))]

    if args.angle_deg is None:
        angle_deg, angle_samples = _estimate_grid_angle(
            xy,
            angle_min=float(args.angle_min),
            angle_max=float(args.angle_max),
            coarse_step=float(args.coarse_step),
            fine_radius=float(args.fine_radius),
            fine_step=float(args.fine_step),
            trim=int(args.trim),
        )
    else:
        angle_deg = float(args.angle_deg)
        angle_samples = []

    rotated = ndimage.rotate(vol_yxz, angle_deg, axes=(1, 0), reshape=False, order=1, mode="nearest")
    finite = rotated[np.isfinite(rotated)]
    lo, hi = np.percentile(finite, [float(args.low_percentile), float(args.high_percentile)])
    cy, cx = rotated.shape[0] // 2, rotated.shape[1] // 2
    offsets = [int(item) for item in args.offsets.split(",") if item.strip()]
    xy_crop = None
    if args.xy_crop_size is not None:
        xy_crop = _center_crop((rotated.shape[0], rotated.shape[1]), size=int(args.xy_crop_size))
    elif args.crop_xy:
        xy_crop = _largest_centered_square_inside_rotated_frame(
            (rotated.shape[0], rotated.shape[1]),
            angle_deg=angle_deg,
            margin=int(args.crop_xy_margin),
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_dir / "grid_aligned_orthos.png",
        _orthos_from_yxz(
            rotated,
            z_index=z_index,
            y_index=cy,
            x_index=cx,
            lo=float(lo),
            hi=float(hi),
            xy_crop=xy_crop,
        ),
    )
    xy_image = rotated[:, :, z_index]
    if xy_crop is not None:
        xy_image = xy_image[xy_crop]
    iio.imwrite(out_dir / "grid_aligned_xy.png", _scale_uint8(xy_image, lo=float(lo), hi=float(hi)))
    iio.imwrite(out_dir / "grid_aligned_xz_center.png", _scale_uint8(rotated[cy, :, :].T, lo=float(lo), hi=float(hi)))
    iio.imwrite(out_dir / "grid_aligned_yz_center.png", _scale_uint8(rotated[:, cx, :].T, lo=float(lo), hi=float(hi)))
    iio.imwrite(out_dir / "grid_aligned_xz_yz_offset_sweep.png", _offset_sweep(rotated, z_index=z_index, offsets=offsets, lo=float(lo), hi=float(hi)))

    manifest: dict[str, Any] = {
        "input_volume": str(Path(args.volume).resolve()),
        "volume_key": args.key,
        "volume_shape_xyz": list(volume_xyz.shape),
        "z_index": int(z_index),
        "estimated_rotation_deg": float(angle_deg),
        "display_axis_order": "Y,X,Z",
        "center_yx_after_rotation": [int(cy), int(cx)],
        "xy_crop": None
        if xy_crop is None
        else {
            "rows": [int(xy_crop[0].start), int(xy_crop[0].stop)],
            "cols": [int(xy_crop[1].start), int(xy_crop[1].stop)],
            "margin": int(args.crop_xy_margin),
            "size": None if args.xy_crop_size is None else int(args.xy_crop_size),
        },
        "window_percentiles": [float(args.low_percentile), float(args.high_percentile)],
        "window_values": [float(lo), float(hi)],
        "offsets_px": offsets,
        "angle_score_samples": angle_samples,
        "outputs": {
            "orthos": str((out_dir / "grid_aligned_orthos.png").resolve()),
            "xy": str((out_dir / "grid_aligned_xy.png").resolve()),
            "xz_center": str((out_dir / "grid_aligned_xz_center.png").resolve()),
            "yz_center": str((out_dir / "grid_aligned_yz_center.png").resolve()),
            "offset_sweep": str((out_dir / "grid_aligned_xz_yz_offset_sweep.png").resolve()),
        },
    }
    (out_dir / "grid_aligned_orthos_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", required=True, help="Input .npz/.npy containing a 3D reconstructed volume")
    parser.add_argument("--key", default="x", help="NPZ key for the volume")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--z-index", type=int, default=None, help="Local z index to use for XY")
    parser.add_argument("--angle-deg", type=float, default=None, help="Manual display-space rotation angle")
    parser.add_argument("--angle-min", type=float, default=-12.0)
    parser.add_argument("--angle-max", type=float, default=12.0)
    parser.add_argument("--coarse-step", type=float, default=0.25)
    parser.add_argument("--fine-radius", type=float, default=0.5)
    parser.add_argument("--fine-step", type=float, default=0.05)
    parser.add_argument("--trim", type=int, default=24)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--offsets", default="-24,-16,-8,0,8,16,24")
    parser.add_argument("--crop-xy", action="store_true", help="Crop the XY panel to remove rotation padding")
    parser.add_argument("--crop-xy-margin", type=int, default=1)
    parser.add_argument("--xy-crop-size", type=int, default=None, help="Centered XY crop size in pixels")
    manifest = render(parser.parse_args())
    print(json.dumps({k: manifest[k] for k in ("estimated_rotation_deg", "z_index", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
