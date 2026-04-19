import h5py
import numpy as np
import pytest

from tomojax.core.geometry.base import Detector, Grid
from tomojax.data.io_hdf5 import NXTomoMetadata, convert, load_nxtomo, load_npz, save_npz
from tomojax.data.phantoms import random_cubes_spheres
from tomojax.utils import axes as axes_mod
from tomojax.utils.axes import infer_disk_axes
from tomojax.utils.fov import grid_from_detector_fov_cube


def test_infer_disk_axes_returns_none_without_grid_for_ambiguous_non_cubic_shapes():
    assert infer_disk_axes((200, 100, 50), grid=None) is None
    assert infer_disk_axes((50, 100, 200), grid=None) is None


def test_transpose_volume_raises_when_jax_array_support_is_incomplete(monkeypatch):
    monkeypatch.setattr(axes_mod, "_JAX_ARRAY_TYPES", (np.ndarray,))
    monkeypatch.setattr(axes_mod, "jnp", None)

    with pytest.raises(RuntimeError, match="jax.numpy is unavailable"):
        axes_mod.transpose_volume(np.zeros((2, 3, 4), dtype=np.float32), "xyz", "zyx")


def test_random_cubes_spheres_keeps_inscribed_fov_for_rotated_cube_seed_63():
    nx = ny = nz = 32
    vol = random_cubes_spheres(
        nx,
        ny,
        nz,
        n_cubes=1,
        n_spheres=0,
        min_size=8,
        max_size=8,
        min_value=1.0,
        max_value=1.0,
        seed=63,
        use_inscribed_fov=True,
    )
    coords = np.argwhere(vol > 0)
    assert coords.size > 0
    radii = np.sqrt((coords[:, 0] - nx / 2.0) ** 2 + (coords[:, 1] - ny / 2.0) ** 2)
    assert radii.max() <= min(nx, ny) / 2.0 + 1e-6


def test_grid_from_detector_fov_cube_respects_y_voxel_size_fit():
    grid = Grid(nx=20, ny=20, nz=20, vx=1.0, vy=2.0, vz=1.0)
    detector = Detector(nu=10, nv=10, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    roi = grid_from_detector_fov_cube(grid, detector)

    assert (roi.nx, roi.ny, roi.nz) == (4, 4, 4)


def test_grid_from_detector_fov_cube_does_not_keep_cubic_grid_when_y_exceeds_fov():
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=4.0, vz=1.0)
    detector = Detector(nu=10, nv=10, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    roi = grid_from_detector_fov_cube(grid, detector)

    assert (roi.nx, roi.ny, roi.nz) == (2, 2, 2)


def test_grid_from_detector_fov_cube_never_grows_beyond_input_y_extent():
    grid = Grid(nx=100, ny=10, nz=100, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=1000, nv=1000, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    roi = grid_from_detector_fov_cube(grid, detector)

    assert (roi.nx, roi.ny, roi.nz) == (10, 10, 10)


def test_load_npz_unwraps_dict_metadata_and_convert_roundtrips_to_nxs(tmp_path):
    npz_path = tmp_path / "sample.npz"
    nxs_path = tmp_path / "sample.nxs"

    projections = np.zeros((2, 3, 4), dtype=np.float32)
    grid = {"nx": 4, "ny": 4, "nz": 3, "vx": 1.0, "vy": 1.0, "vz": 1.0}
    detector = {"nu": 4, "nv": 3, "du": 1.0, "dv": 1.0, "det_center": [0.0, 0.0]}
    geometry_meta = {"tilt_deg": 35.0, "tilt_about": "x"}
    misalign_spec = {"rot_deg": 1.0, "trans_px": 2.0}

    save_npz(
        npz_path,
        projections=projections,
        thetas_deg=np.array([0.0, 90.0], dtype=np.float32),
        grid=grid,
        detector=detector,
        geometry_type="lamino",
        geometry_meta=geometry_meta,
        misalign_spec=misalign_spec,
    )

    loaded = load_npz(npz_path)
    assert isinstance(loaded["grid"], dict)
    assert isinstance(loaded["detector"], dict)
    assert isinstance(loaded["geometry_meta"], dict)
    assert isinstance(loaded["misalign_spec"], dict)

    convert(str(npz_path), str(nxs_path))
    meta = load_nxtomo(str(nxs_path))

    assert meta["grid"] == grid
    assert meta["detector"] == detector
    assert meta["geometry_meta"] == geometry_meta
    assert meta["misalign_spec"] == misalign_spec


def test_nxtomo_roundtrips_alignment_gauge_metadata(tmp_path):
    from tomojax.data.io_hdf5 import NXTomoMetadata, save_nxtomo

    nxs_path = tmp_path / "alignment_gauge.nxs"
    projections = np.zeros((2, 3, 4), dtype=np.float32)
    gauge = {
        "mode": "mean_translation",
        "dofs": ["dx", "dz"],
        "final": {"dx_mean_after": 0.0, "dz_mean_after": 0.0},
    }
    metadata = NXTomoMetadata(
        thetas_deg=np.array([0.0, 90.0], dtype=np.float32),
        detector={"nu": 4, "nv": 3, "du": 1.0, "dv": 1.0, "det_center": [0.0, 0.0]},
        align_params=np.zeros((2, 5), dtype=np.float32),
        align_gauge=gauge,
    )

    save_nxtomo(str(nxs_path), projections=projections, metadata=metadata)
    loaded = load_nxtomo(str(nxs_path))

    assert loaded.align_gauge == gauge
    assert loaded["align_gauge"] == gauge


def test_load_nxtomo_preserves_legacy_xyz_without_attr_or_grid(tmp_path):
    nxs_path = tmp_path / "legacy_no_attr_no_grid.nxs"

    projections = np.zeros((2, 3, 4), dtype=np.float32)
    volume_xyz = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)

    from tomojax.data.io_hdf5 import save_nxtomo

    save_nxtomo(
        str(nxs_path),
        projections,
        metadata=NXTomoMetadata(
            thetas_deg=np.array([0.0, 90.0], dtype=np.float32),
            detector={"nu": 4, "nv": 3, "du": 1.0, "dv": 1.0, "det_center": [0.0, 0.0]},
            geometry_type="parallel",
            volume=volume_xyz,
            volume_axes_order="xyz",
        ),
    )

    with h5py.File(nxs_path, "r+") as f:
        del f["/entry/processing/tomojax"].attrs["volume_axes_order"]

    meta = load_nxtomo(str(nxs_path))

    assert meta["volume_axes_order"] == "xyz"
    assert meta["disk_volume_axes_order"] == "xyz_legacy"
    assert meta["volume_axes_source"] == "heuristic"
    np.testing.assert_allclose(meta["volume"], volume_xyz)
