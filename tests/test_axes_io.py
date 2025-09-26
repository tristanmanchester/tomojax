import numpy as np
import h5py

from tomojax.data.io_hdf5 import save_nxtomo, load_nxtomo
from tomojax.utils.axes import DISK_VOLUME_AXES, INTERNAL_VOLUME_AXES


GRID_META = {"nx": 4, "ny": 3, "nz": 2, "vx": 1.0, "vy": 1.0, "vz": 1.0}
DET_META = {"nu": 6, "nv": 3, "du": 1.0, "dv": 1.0, "det_center": [0.0, 0.0]}


def _basic_projections():
    return np.zeros((5, DET_META["nv"], DET_META["nu"]), dtype=np.float32)


def test_save_volume_writes_zyx_and_attr(tmp_path):
    path = tmp_path / "sample_zyx.nxs"
    volume_xyz = np.arange(np.prod([GRID_META["nx"], GRID_META["ny"], GRID_META["nz"]]), dtype=np.float32).reshape(
        GRID_META["nx"], GRID_META["ny"], GRID_META["nz"]
    )

    save_nxtomo(
        path,
        projections=_basic_projections(),
        thetas_deg=np.linspace(0.0, 180.0, 5, dtype=np.float32),
        grid=GRID_META,
        detector=DET_META,
        geometry_type="parallel",
        volume=volume_xyz,
    )

    with h5py.File(path, "r") as f:
        vol_ds = f["/entry/processing/tomojax/volume"]
        assert vol_ds.shape == (GRID_META["nz"], GRID_META["ny"], GRID_META["nx"])
        attr = vol_ds.parent.attrs["volume_axes_order"]
        if isinstance(attr, bytes):
            attr = attr.decode("utf-8")
        elif isinstance(attr, np.ndarray):
            attr = attr.astype(str)[()]
        assert attr == DISK_VOLUME_AXES


def test_load_legacy_xyz_without_attr(tmp_path):
    path = tmp_path / "legacy_xyz.nxs"
    volume_xyz = np.arange(np.prod([GRID_META["nx"], GRID_META["ny"], GRID_META["nz"]]), dtype=np.float32).reshape(
        GRID_META["nx"], GRID_META["ny"], GRID_META["nz"]
    )
    save_nxtomo(
        path,
        projections=_basic_projections(),
        thetas_deg=np.linspace(0.0, 180.0, 5, dtype=np.float32),
        grid=GRID_META,
        detector=DET_META,
        geometry_type="parallel",
        volume=volume_xyz,
        volume_axes_order="xyz",
    )
    with h5py.File(path, "r+") as f:
        del f["/entry/processing/tomojax"].attrs["volume_axes_order"]

    meta = load_nxtomo(path)
    assert meta["volume_axes_order"] == INTERNAL_VOLUME_AXES
    assert meta["disk_volume_axes_order"] == "xyz_legacy"
    assert meta["volume_axes_source"] == "heuristic"
    assert meta["volume"].shape == volume_xyz.shape
    np.testing.assert_allclose(meta["volume"], volume_xyz)


def test_roundtrip_xyz_to_zyx_to_xyz(tmp_path):
    path = tmp_path / "roundtrip.nxs"
    volume_xyz = np.random.default_rng(42).normal(size=(GRID_META["nx"], GRID_META["ny"], GRID_META["nz"])).astype(np.float32)

    save_nxtomo(
        path,
        projections=_basic_projections(),
        thetas_deg=np.linspace(0.0, 180.0, 5, dtype=np.float32),
        grid=GRID_META,
        detector=DET_META,
        geometry_type="parallel",
        volume=volume_xyz,
    )

    meta = load_nxtomo(path)
    np.testing.assert_allclose(meta["volume"], volume_xyz)
    assert meta["volume_axes_order"] == INTERNAL_VOLUME_AXES
    assert meta["disk_volume_axes_order"] == DISK_VOLUME_AXES
    assert meta["volume_axes_source"] == "attr"


def test_no_change_for_projections_shape(tmp_path):
    path = tmp_path / "proj_shape.nxs"
    projections = _basic_projections()
    save_nxtomo(
        path,
        projections=projections,
        thetas_deg=np.linspace(0.0, 180.0, 5, dtype=np.float32),
        grid=GRID_META,
        detector=DET_META,
        geometry_type="parallel",
    )
    meta = load_nxtomo(path)
    assert meta["projections"].shape == projections.shape
