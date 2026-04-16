from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import jax.numpy as jnp
import pytest

from tomojax.data.geometry_meta import (
    build_geometry_from_meta,
    build_nominal_geometry_from_meta,
)
from tomojax.data.io_hdf5 import LoadedNXTomo
from tomojax.cli import recon as recon_cli
from tomojax.core.geometry import Grid, ParallelGeometry


def _parallel_meta(**updates):
    meta = {
        "geometry_type": "parallel",
        "thetas_deg": np.asarray([0.0, 45.0], dtype=np.float32),
        "detector": {
            "nu": 9,
            "nv": 7,
            "du": 1.25,
            "dv": 2.5,
            "det_center": [0.0, 0.0],
        },
        "grid": {
            "nx": 8,
            "ny": 10,
            "nz": 6,
            "vx": 0.8,
            "vy": 1.1,
            "vz": 1.4,
            "vol_origin": [1.0, 2.0, 3.0],
            "vol_center": [4.0, 5.0, 6.0],
        },
    }
    meta.update(updates)
    return meta


def test_build_geometry_from_meta_applies_saved_angle_offsets():
    meta = _parallel_meta(
        angle_offset_deg=np.asarray([2.0, -3.0], dtype=np.float32),
    )

    grid, detector, geom = build_geometry_from_meta(meta, apply_saved_alignment=True)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"] + meta["angle_offset_deg"],
    )

    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(1), dtype=np.float32),
        np.asarray(expected.pose_for_view(1), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_geometry_from_meta_skips_double_applying_saved_angle_offsets():
    meta = _parallel_meta(
        angle_offset_deg=np.asarray([2.0, -3.0], dtype=np.float32),
        misalign_spec={"schedule": "already-baked"},
    )

    grid, detector, geom = build_geometry_from_meta(meta, apply_saved_alignment=True)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"],
    )

    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(1), dtype=np.float32),
        np.asarray(expected.pose_for_view(1), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_geometry_from_meta_rejects_unsupported_geometry_types():
    meta = _parallel_meta(geometry_type="custom")

    with pytest.raises(ValueError, match="Unsupported geometry_type"):
        build_geometry_from_meta(meta)


def test_align_build_geometry_uses_grid_override_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, geom = build_nominal_geometry_from_meta(meta, grid_override=(11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)
    assert (grid.vx, grid.vy, grid.vz) == (1.25, 1.25, 2.5)
    expected = ParallelGeometry(grid=grid, detector=detector, thetas_deg=meta["thetas_deg"])
    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(1), dtype=np.float32),
        np.asarray(expected.pose_for_view(1), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_align_build_geometry_accepts_positional_grid_override_for_compatibility():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, _, _ = build_nominal_geometry_from_meta(meta, (11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)


def test_recon_build_geometry_infers_grid_from_detector_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, _ = build_nominal_geometry_from_meta(meta)

    assert (grid.nx, grid.ny, grid.nz) == (detector.nu, detector.nu, detector.nv)
    assert (grid.vx, grid.vy, grid.vz) == (detector.du, detector.du, detector.dv)


def test_build_geometry_from_meta_uses_volume_shape_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, _ = build_geometry_from_meta(meta, volume_shape=(11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)
    assert (grid.vx, grid.vy, grid.vz) == (detector.du, detector.du, detector.dv)


def test_recon_build_geometry_accepts_positional_grid_override_for_compatibility():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, _, _ = build_nominal_geometry_from_meta(meta, (11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)


def test_recon_build_geometry_preserves_grid_origin_and_center():
    meta = _parallel_meta()

    grid, _, _ = build_nominal_geometry_from_meta(meta)

    assert grid.vol_origin == (1.0, 2.0, 3.0)
    assert grid.vol_center == (4.0, 5.0, 6.0)


def test_recon_build_geometry_preserves_grid_origin_and_center_with_override():
    meta = _parallel_meta()

    grid, _, _ = build_nominal_geometry_from_meta(meta, (11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)
    assert grid.vol_origin == (1.0, 2.0, 3.0)
    assert grid.vol_center == (4.0, 5.0, 6.0)


def test_recon_build_geometry_preserves_full_grid_override_metadata():
    meta = _parallel_meta()
    override_grid = Grid(nx=11, ny=13, nz=15, vx=0.8, vy=1.1, vz=1.4)

    grid, _, geom = build_nominal_geometry_from_meta(meta, override_grid)

    assert grid == override_grid
    assert grid.vol_origin is None
    assert grid.vol_center is None
    assert geom.grid == override_grid


def test_recon_cli_grid_override_preserves_grid_origin_and_center(monkeypatch, tmp_path):
    meta = _parallel_meta(
        projections=np.zeros((2, 7, 9), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
        geometry_meta=None,
    )
    captured = {}

    monkeypatch.setattr(recon_cli, "load_nxtomo", lambda path: LoadedNXTomo.from_dataset(meta))
    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(recon_cli, "transfer_guard_context", lambda mode: nullcontext())
    monkeypatch.setattr(
        recon_cli,
        "fbp",
        lambda geom, recon_grid, detector, proj, **kwargs: jnp.zeros(
            (recon_grid.nx, recon_grid.ny, recon_grid.nz), dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        recon_cli,
        "save_nxtomo",
        lambda out, projections, **kwargs: captured.update(
            {"out": out, "projections": projections, **kwargs}
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(tmp_path / "input.nxs"),
            "--algo",
            "fbp",
            "--roi",
            "off",
            "--grid",
            "11",
            "13",
            "15",
            "--out",
            str(tmp_path / "out.nxs"),
        ],
    )

    recon_cli.main()

    grid = captured["metadata"].grid
    assert grid["nx"] == 11
    assert grid["ny"] == 13
    assert grid["nz"] == 15
    assert grid["vol_origin"] == [1.0, 2.0, 3.0]
    assert grid["vol_center"] == [4.0, 5.0, 6.0]


def test_recon_cli_grid_override_preserves_roi_centering(monkeypatch, tmp_path):
    meta = _parallel_meta(
        projections=np.zeros((2, 16, 16), dtype=np.float32),
        image_key=np.zeros((2,), dtype=np.int32),
        geometry_meta=None,
        detector={
            "nu": 16,
            "nv": 16,
            "du": 1.0,
            "dv": 1.0,
            "det_center": [0.0, 0.0],
        },
        grid={
            "nx": 32,
            "ny": 32,
            "nz": 16,
            "vx": 1.0,
            "vy": 1.0,
            "vz": 1.0,
            "vol_origin": [11.0, 12.0, 13.0],
            "vol_center": [1.0, 2.0, 3.0],
        },
    )
    captured = {}

    monkeypatch.setattr(recon_cli, "load_nxtomo", lambda path: LoadedNXTomo.from_dataset(meta))
    monkeypatch.setattr(recon_cli, "setup_logging", lambda: None)
    monkeypatch.setattr(recon_cli, "log_jax_env", lambda: None)
    monkeypatch.setattr(
        recon_cli,
        "transfer_guard_context",
        lambda mode: nullcontext(),
    )

    def _fbp_capture(geom, recon_grid, detector, proj, **kwargs):
        captured["runtime_grid"] = recon_grid
        captured["geom_grid"] = geom.grid
        return jnp.zeros(
            (recon_grid.nx, recon_grid.ny, recon_grid.nz),
            dtype=jnp.float32,
        )

    monkeypatch.setattr(recon_cli, "fbp", _fbp_capture)
    monkeypatch.setattr(
        recon_cli,
        "save_nxtomo",
        lambda out, projections, **kwargs: captured.update(
            {"out": out, "projections": projections, **kwargs}
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "recon",
            "--data",
            str(tmp_path / "input.nxs"),
            "--algo",
            "fbp",
            "--roi",
            "auto",
            "--grid",
            "20",
            "20",
            "12",
            "--out",
            str(tmp_path / "out.nxs"),
        ],
    )

    recon_cli.main()

    grid = captured["metadata"].grid
    assert grid["nx"] == 20
    assert grid["ny"] == 20
    assert grid["nz"] == 12
    assert "vol_origin" not in grid
    assert "vol_center" not in grid
    assert captured["runtime_grid"].vol_origin is None
    assert captured["runtime_grid"].vol_center is None
    assert captured["geom_grid"].vol_origin is None
    assert captured["geom_grid"].vol_center is None


def test_recon_build_geometry_keeps_nominal_geometry_for_saved_alignment_metadata():
    align_params = np.asarray(
        [[0.0, 0.0, 0.0, 1.25, -0.5], [0.1, -0.2, 0.3, 0.0, 0.25]],
        dtype=np.float32,
    )
    angle_offset = np.asarray([5.0, -3.0], dtype=np.float32)
    meta = _parallel_meta(align_params=align_params, angle_offset_deg=angle_offset)

    grid, detector, geom = build_nominal_geometry_from_meta(meta)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"],
    )

    for i in range(2):
        np.testing.assert_allclose(
            np.asarray(geom.pose_for_view(i), dtype=np.float32),
            np.asarray(expected.pose_for_view(i), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


def test_recon_build_geometry_skips_double_applying_baked_angle_offsets():
    angle_offset = np.asarray([7.0, -4.0], dtype=np.float32)
    meta = _parallel_meta(
        angle_offset_deg=angle_offset,
        misalign_spec={"kind": "scheduled"},
    )

    grid, detector, geom = build_nominal_geometry_from_meta(meta)
    expected = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"],
    )

    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(0), dtype=np.float32),
        np.asarray(expected.pose_for_view(0), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
