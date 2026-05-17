from __future__ import annotations

from pathlib import Path

import numpy as np

from tomojax.bench import (
    center_crop,
    grid_aligned_xy,
    largest_centered_square_inside_rotated_frame,
    load_volume_array,
    render_tem_grid_pose_artifacts,
    resize_nearest_2d,
    save_uint8_png,
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
    np.testing.assert_array_equal(
        resize_nearest_2d(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), (3, 3)),
        np.asarray([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 3.0, 4.0]], dtype=np.float32),
    )
    assert Path(save_uint8_png(tmp_path / "scaled.png", scaled)).exists()
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


def test_real_lamino_pose_visual_artifacts_write_manifest(tmp_path: Path) -> None:
    params_path = tmp_path / "params.csv"
    params_path.write_text(
        "view,dx,dz,phi_deg\n"
        "0,0.0,0.0,0.0\n"
        "1,1.0,-0.5,0.1\n"
        "2,-1.0,0.5,-0.1\n",
        encoding="utf-8",
    )

    manifest = render_tem_grid_pose_artifacts(params_path=params_path, out_dir=tmp_path / "out")

    assert manifest["n_views"] == 3
    assert manifest["summary"]["dx_min"] == -1.0
    assert Path(manifest["outputs"]["static_png"]).exists()
    assert Path(manifest["outputs"]["interactive_html"]).exists()
    assert Path(manifest["outputs"]["csv"]).exists()
    assert (tmp_path / "out" / "projection_pose_corrections_3d_manifest.json").exists()
