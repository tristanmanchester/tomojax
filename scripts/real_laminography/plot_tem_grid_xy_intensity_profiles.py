"""Plot horizontal and vertical intensity profiles through grid-aligned TEM XY slices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _load_volume(path: Path, *, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        volume = np.load(path)
    else:
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{path} does not contain key {key!r}; keys={data.files}")
            volume = data[key]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume from {path}, got {volume.shape}")
    return np.asarray(volume, dtype=np.float32)


def _center_crop(shape_yx: tuple[int, int], *, size: int) -> tuple[slice, slice]:
    crop = max(1, min(int(size), int(shape_yx[0]), int(shape_yx[1])))
    y0 = (int(shape_yx[0]) - crop) // 2
    x0 = (int(shape_yx[1]) - crop) // 2
    return slice(y0, y0 + crop), slice(x0, x0 + crop)


def _largest_centered_square_inside_rotated_frame(
    shape_yx: tuple[int, int],
    *,
    angle_deg: float,
    margin: int,
) -> tuple[slice, slice]:
    n = min(int(shape_yx[0]), int(shape_yx[1]))
    theta = abs(float(angle_deg)) % 90.0
    if theta > 45.0:
        theta = 90.0 - theta
    radians = np.deg2rad(theta)
    side = int(np.floor(n / (abs(np.cos(radians)) + abs(np.sin(radians)))))
    side = max(1, min(n, side - 2 * max(0, int(margin))))
    return _center_crop(shape_yx, size=side)


def _grid_aligned_xy(
    volume_xyz: np.ndarray,
    *,
    z_index: int,
    angle_deg: float,
    crop: tuple[slice, slice],
) -> np.ndarray:
    vol_yxz = np.transpose(volume_xyz, (1, 0, 2))
    rotated = ndimage.rotate(vol_yxz, angle_deg, axes=(1, 0), reshape=False, order=1, mode="nearest")
    z = int(np.clip(z_index, 0, rotated.shape[2] - 1))
    return np.asarray(rotated[:, :, z][crop], dtype=np.float32)


def _window_normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.asarray(np.clip((image - lo) / (hi - lo), 0.0, 1.0), dtype=np.float32)


def _write_csv(
    path: Path,
    *,
    final_h: np.ndarray,
    fbp_h: np.ndarray,
    final_v: np.ndarray,
    fbp_v: np.ndarray,
    final_h_norm: np.ndarray,
    fbp_h_norm: np.ndarray,
    final_v_norm: np.ndarray,
    fbp_v_norm: np.ndarray,
) -> None:
    n = max(len(final_h), len(final_v))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "final_horizontal",
                "fbp_horizontal",
                "final_vertical",
                "fbp_vertical",
                "final_horizontal_norm",
                "fbp_horizontal_norm",
                "final_vertical_norm",
                "fbp_vertical_norm",
            ],
        )
        writer.writeheader()
        for idx in range(n):
            writer.writerow(
                {
                    "index": idx,
                    "final_horizontal": float(final_h[idx]) if idx < len(final_h) else "",
                    "fbp_horizontal": float(fbp_h[idx]) if idx < len(fbp_h) else "",
                    "final_vertical": float(final_v[idx]) if idx < len(final_v) else "",
                    "fbp_vertical": float(fbp_v[idx]) if idx < len(fbp_v) else "",
                    "final_horizontal_norm": float(final_h_norm[idx]) if idx < len(final_h_norm) else "",
                    "fbp_horizontal_norm": float(fbp_h_norm[idx]) if idx < len(fbp_h_norm) else "",
                    "final_vertical_norm": float(final_v_norm[idx]) if idx < len(final_v_norm) else "",
                    "fbp_vertical_norm": float(fbp_v_norm[idx]) if idx < len(fbp_v_norm) else "",
                }
            )


def _draw_profile_panel(
    profiles: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    *,
    title: str,
    width: int = 560,
    height: int = 240,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 46, 16, 28, 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.text((left, 8), title, fill=(20, 20, 20))
    for frac in np.linspace(0.0, 1.0, 6):
        y = top + int(round((1.0 - frac) * plot_h))
        draw.line([(left, y), (left + plot_w, y)], fill=(225, 225, 225))
    for frac in np.linspace(0.0, 1.0, 5):
        x = left + int(round(frac * plot_w))
        draw.line([(x, top), (x, top + plot_h)], fill=(235, 235, 235))
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(40, 40, 40))
    draw.text((4, top - 4), "1.0", fill=(60, 60, 60))
    draw.text((8, top + plot_h - 8), "0.0", fill=(60, 60, 60))
    for profile_index, (label, values, color) in enumerate(profiles):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size < 2:
            continue
        xs = left + np.linspace(0, plot_w, arr.size)
        ys = top + (1.0 - np.clip(arr, 0.0, 1.0)) * plot_h
        points = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
        draw.line(points, fill=color, width=2)
        draw.text((left + 8, top + 10 + 16 * profile_index), label, fill=color)
    return image


def _stack_panels(panels: list[list[Image.Image]], *, gap: int = 12) -> Image.Image:
    row_widths = [sum(panel.width for panel in row) + gap * (len(row) - 1) for row in panels]
    row_heights = [max(panel.height for panel in row) for row in panels]
    width = max(row_widths)
    height = sum(row_heights) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row, row_h in zip(panels, row_heights, strict=True):
        x = 0
        for panel in row:
            canvas.paste(panel, (x, y))
            x += panel.width + gap
        y += row_h + gap
    return canvas


def render(args: argparse.Namespace) -> dict[str, Any]:
    final_volume = _load_volume(Path(args.final_volume), key=args.final_key)
    fbp_volume = _load_volume(Path(args.fbp_volume), key=args.fbp_key)
    if final_volume.shape != fbp_volume.shape:
        raise ValueError(f"Volume shapes differ: final={final_volume.shape}, fbp={fbp_volume.shape}")

    z_index = final_volume.shape[2] // 2 if args.z_index is None else int(args.z_index)
    crop = _largest_centered_square_inside_rotated_frame(
        (final_volume.shape[1], final_volume.shape[0]),
        angle_deg=float(args.angle_deg),
        margin=int(args.crop_margin),
    )
    final_xy = _grid_aligned_xy(final_volume, z_index=z_index, angle_deg=float(args.angle_deg), crop=crop)
    fbp_xy = _grid_aligned_xy(fbp_volume, z_index=z_index, angle_deg=float(args.angle_deg), crop=crop)

    row = final_xy.shape[0] // 2 if args.row is None else int(args.row)
    col = final_xy.shape[1] // 2 if args.col is None else int(args.col)
    row = int(np.clip(row, 0, final_xy.shape[0] - 1))
    col = int(np.clip(col, 0, final_xy.shape[1] - 1))

    final_h = final_xy[row, :]
    fbp_h = fbp_xy[row, :]
    final_v = final_xy[:, col]
    fbp_v = fbp_xy[:, col]
    final_norm = _window_normalize(final_xy)
    fbp_norm = _window_normalize(fbp_xy)
    final_h_norm = final_norm[row, :]
    fbp_h_norm = fbp_norm[row, :]
    final_v_norm = final_norm[:, col]
    fbp_v_norm = fbp_norm[:, col]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "grid_aligned_xy_center_line_profiles.csv"
    _write_csv(
        csv_path,
        final_h=final_h,
        fbp_h=fbp_h,
        final_v=final_v,
        fbp_v=fbp_v,
        final_h_norm=final_h_norm,
        fbp_h_norm=fbp_h_norm,
        final_v_norm=final_v_norm,
        fbp_v_norm=fbp_v_norm,
    )

    profile_grid_path = out_dir / "grid_aligned_xy_center_line_profiles.png"
    blue = (31, 119, 180)
    red = (214, 39, 40)
    panel_grid = _stack_panels(
        [
            [
                _draw_profile_panel([("Final aligned TV", final_h_norm, blue)], title="Horizontal center profile"),
                _draw_profile_panel([("Original FBP", fbp_h_norm, red)], title="Horizontal center profile"),
            ],
            [
                _draw_profile_panel([("Final aligned TV", final_v_norm, blue)], title="Vertical center profile"),
                _draw_profile_panel([("Original FBP", fbp_v_norm, red)], title="Vertical center profile"),
            ],
        ]
    )
    iio.imwrite(profile_grid_path, np.asarray(panel_grid))

    overlay_path = out_dir / "grid_aligned_xy_center_line_profiles_overlay.png"
    overlay_grid = _stack_panels(
        [
            [
                _draw_profile_panel(
                    [("Final aligned TV", final_h_norm, blue), ("Original FBP", fbp_h_norm, red)],
                    title="Horizontal center profile",
                ),
                _draw_profile_panel(
                    [("Final aligned TV", final_v_norm, blue), ("Original FBP", fbp_v_norm, red)],
                    title="Vertical center profile",
                ),
            ]
        ]
    )
    iio.imwrite(overlay_path, np.asarray(overlay_grid))

    manifest = {
        "final_volume": str(Path(args.final_volume).resolve()),
        "fbp_volume": str(Path(args.fbp_volume).resolve()),
        "volume_shape_xyz": list(final_volume.shape),
        "angle_deg": float(args.angle_deg),
        "z_index": int(z_index),
        "xy_crop": {
            "rows": [int(crop[0].start), int(crop[0].stop)],
            "cols": [int(crop[1].start), int(crop[1].stop)],
            "margin": int(args.crop_margin),
        },
        "profile_row_in_crop": int(row),
        "profile_col_in_crop": int(col),
        "outputs": {
            "profile_grid": str(profile_grid_path.resolve()),
            "profile_overlay": str(overlay_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    manifest_path = out_dir / "grid_aligned_xy_center_line_profiles_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-volume", required=True)
    parser.add_argument("--final-key", default="x")
    parser.add_argument("--fbp-volume", required=True)
    parser.add_argument("--fbp-key", default="x")
    parser.add_argument("--out", required=True)
    parser.add_argument("--angle-deg", type=float, default=10.6)
    parser.add_argument("--z-index", type=int, default=None)
    parser.add_argument("--crop-margin", type=int, default=1)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--col", type=int, default=None)
    manifest = render(parser.parse_args())
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
