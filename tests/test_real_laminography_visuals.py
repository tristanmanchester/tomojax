from __future__ import annotations

from pathlib import Path

import numpy as np

from tomojax.bench import (
    center_crop,
    grid_aligned_xy,
    largest_centered_square_inside_rotated_frame,
    load_volume_array,
    scale_uint8,
    window_normalize,
)


def test_real_lamino_visuals_load_crop_and_scale(tmp_path: Path) -> None:
    volume = np.arange(5 * 6 * 4, dtype=np.float32).reshape(5, 6, 4)
    npy = tmp_path / "volume.npy"
    npz = tmp_path / "volume.npz"
    np.save(npy, volume)
    np.savez(npz, x=volume)

    loaded_npy = load_volume_array(npy, key="x")
    loaded_npz = load_volume_array(npz, key="x")
    crop = center_crop((20, 12), size=8)
    rotated_crop = largest_centered_square_inside_rotated_frame(
        (20, 20),
        angle_deg=45.0,
        margin=1,
    )
    scaled = scale_uint8(np.asarray([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32), lo=0.0, hi=1.0)
    normalized = window_normalize(np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))

    np.testing.assert_array_equal(loaded_npy, volume)
    np.testing.assert_array_equal(loaded_npz, volume)
    assert crop == (slice(6, 14), slice(2, 10))
    assert rotated_crop[0].stop - rotated_crop[0].start == rotated_crop[1].stop - rotated_crop[1].start
    assert scaled.dtype == np.uint8
    assert scaled[0, 0] == 0
    assert scaled[1, 0] == 255
    assert np.isfinite(normalized).all()
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0


def test_real_lamino_visuals_grid_aligned_xy_respects_crop() -> None:
    volume = np.zeros((8, 10, 6), dtype=np.float32)
    volume[3:5, 4:6, 2] = 1.0
    crop = center_crop((10, 8), size=4)

    xy = grid_aligned_xy(volume, z_index=2, angle_deg=0.0, crop=crop)

    assert xy.shape == (4, 4)
    assert float(xy.max()) == 1.0
    assert float(xy.min()) == 0.0
