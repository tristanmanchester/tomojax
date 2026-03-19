from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from tomojax.align.parametrizations import se3_from_5d
from tomojax.cli.align import build_geometry as build_align_geometry
from tomojax.cli.recon import build_geometry as build_recon_geometry
from tomojax.core.geometry import ParallelGeometry


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


def test_align_build_geometry_uses_grid_override_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, geom = build_align_geometry(meta, grid_override=(11, 13, 15))

    assert (grid.nx, grid.ny, grid.nz) == (11, 13, 15)
    assert (grid.vx, grid.vy, grid.vz) == (1.25, 1.25, 2.5)
    expected = ParallelGeometry(grid=grid, detector=detector, thetas_deg=meta["thetas_deg"])
    np.testing.assert_allclose(
        np.asarray(geom.pose_for_view(1), dtype=np.float32),
        np.asarray(expected.pose_for_view(1), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_recon_build_geometry_infers_grid_from_detector_when_grid_metadata_missing():
    meta = _parallel_meta()
    meta.pop("grid")

    grid, detector, _ = build_recon_geometry(meta)

    assert (grid.nx, grid.ny, grid.nz) == (detector.nu, detector.nu, detector.nv)
    assert (grid.vx, grid.vy, grid.vz) == (detector.du, detector.du, detector.dv)



def test_recon_build_geometry_preserves_grid_origin_and_center():
    meta = _parallel_meta()

    grid, _, _ = build_recon_geometry(meta)

    assert grid.vol_origin == (1.0, 2.0, 3.0)
    assert grid.vol_center == (4.0, 5.0, 6.0)



def test_recon_build_geometry_applies_saved_alignment_and_angle_offsets():
    align_params = np.asarray(
        [[0.0, 0.0, 0.0, 1.25, -0.5], [0.1, -0.2, 0.3, 0.0, 0.25]],
        dtype=np.float32,
    )
    angle_offset = np.asarray([5.0, -3.0], dtype=np.float32)
    meta = _parallel_meta(align_params=align_params, angle_offset_deg=angle_offset)

    grid, detector, geom = build_recon_geometry(meta)
    base = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=meta["thetas_deg"] + angle_offset,
    )

    for i in range(2):
        expected = np.asarray(base.pose_for_view(i), dtype=np.float32) @ np.asarray(
            se3_from_5d(jnp.asarray(align_params[i], dtype=jnp.float32)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            np.asarray(geom.pose_for_view(i), dtype=np.float32),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )



def test_recon_build_geometry_skips_double_applying_baked_angle_offsets():
    angle_offset = np.asarray([7.0, -4.0], dtype=np.float32)
    meta = _parallel_meta(
        angle_offset_deg=angle_offset,
        misalign_spec={"kind": "scheduled"},
    )

    grid, detector, geom = build_recon_geometry(meta)
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
