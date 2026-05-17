from pathlib import Path

import imageio.v3 as iio
import numpy as np

from tomojax.bench import (
    real_lamino_orthos_image,
    save_real_lamino_z_stack,
    scale_uint8,
    write_real_lamino_stage_products,
)
from tomojax.core.geometry import Grid


def test_real_lamino_orthos_image_preserves_panel_orientation() -> None:
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    orthos = real_lamino_orthos_image(volume, preview_local_z=1)

    assert orthos.shape == (4, 23)
    np.testing.assert_array_equal(orthos[0:3, 0:2], scale_uint8(volume[:, :, 1].T))
    np.testing.assert_array_equal(orthos[0:4, 10:12], scale_uint8(volume[:, 1, :].T))
    np.testing.assert_array_equal(orthos[0:4, 20:23], scale_uint8(volume[1, :, :].T))
    assert np.all(orthos[:, 2:10] == 0)
    assert np.all(orthos[:, 12:20] == 0)


def test_save_real_lamino_z_stack_writes_placeholder_for_empty_range(tmp_path: Path) -> None:
    volume = np.ones((2, 2, 2), dtype=np.float32)
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0, vol_origin=(0.0, 0.0, 0.0))
    path = tmp_path / "empty_stack.png"

    returned = save_real_lamino_z_stack(
        path,
        volume,
        grid=grid,
        full_nz=100,
        z_range=(90, 91),
        max_cols=2,
    )

    assert returned == str(path)
    assert path.exists()
    np.testing.assert_array_equal(iio.imread(path), np.zeros((16, 16), dtype=np.uint8))


def test_write_real_lamino_stage_products_returns_stable_keys_and_files(tmp_path: Path) -> None:
    volume = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    grid = Grid(nx=2, ny=3, nz=3, vx=1.0, vy=1.0, vz=1.0, vol_origin=(0.0, 0.0, -1.0))
    stage_dir = tmp_path / "stage"

    artifacts = write_real_lamino_stage_products(
        stage_dir=stage_dir,
        volume=volume,
        grid=grid,
        full_nz=3,
        preview_global_z=1,
        stack_z_range=(0, 2),
        snapshot_max_cols=2,
        input_reference=np.zeros((1, 1), dtype=np.float32),
        suffix="aligned",
    )

    assert artifacts == {
        "aligned_xy": str(stage_dir / "aligned_xy_global_z001.png"),
        "delta_xy": str(stage_dir / "delta_xy_global_z001.png"),
        "orthos": str(stage_dir / "orthos.png"),
        "z_stack": str(stage_dir / "z_stack_global_z000_002.png"),
    }
    for path in artifacts.values():
        assert Path(path).is_file()
    assert iio.imread(stage_dir / "orthos.png").shape == (3, 23)
