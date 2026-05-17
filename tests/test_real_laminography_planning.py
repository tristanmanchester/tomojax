from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from tomojax.bench import (
    binned_pixel_scale,
    parse_shape3,
    pose_dx_dz_bounds,
    pose_phi_bounds,
    pose_polish_bounds,
    real_lamino_global_z_to_local_index,
    real_lamino_global_z_to_phys,
    real_lamino_grid_origin_z,
    real_lamino_local_z_to_global_index,
    real_lamino_xy_at_global_z,
    resolve_fixture_bin_factor,
    select_real_lamino_final_candidates,
    setup_det_u_bounds,
    validate_bin_factor,
    view_indices_for_smoke_shape,
)


def test_real_laminography_planning_resolves_smoke_binning() -> None:
    factor = resolve_fixture_bin_factor(
        projection_shape=(256, 256, 256),
        slab_nz=96,
        requested_bin_factor=2,
        smoke_shape=(64, 80, 70),
    )

    assert factor == 4


def test_real_laminography_planning_selects_deterministic_smoke_views() -> None:
    indices = view_indices_for_smoke_shape(10, (4, 64, 64))

    np.testing.assert_array_equal(indices, np.array([0, 3, 6, 9], dtype=np.int64))


def test_real_laminography_planning_scales_binned_bounds() -> None:
    args = SimpleNamespace(
        bin_factor=1,
        effective_bin_factor=4,
        pose_bounds_profile="reference_conservative",
    )

    assert binned_pixel_scale(args) == 0.25
    assert setup_det_u_bounds(args) == "det_u_px=-6:6"
    assert pose_phi_bounds(args) == "phi=-0.00872665:0.00872665"
    assert pose_dx_dz_bounds(args) == "dx=-2.5:2.5,dz=-2.5:2.5"
    assert pose_polish_bounds(args).endswith("dx=-2.5:2.5,dz=-2.5:2.5")


def test_real_laminography_planning_validates_bin_factor_and_shapes() -> None:
    assert validate_bin_factor("4") == 4
    assert parse_shape3("8x64x64") == (8, 64, 64)

    with pytest.raises(ValueError, match="integer >= 1"):
        validate_bin_factor(0)
    with pytest.raises(ValueError, match="diagnostic shape must be positive"):
        resolve_fixture_bin_factor(
            projection_shape=(8, 64, 64),
            slab_nz=32,
            requested_bin_factor=1,
            smoke_shape=(4, 0, 64),
        )
    with pytest.raises(argparse.ArgumentTypeError, match="projection shape dimensions must be positive"):
        parse_shape3("8x0x64")


def test_real_laminography_planning_selects_final_candidates() -> None:
    candidates = [
        {"label": "cor", "source_stage": "01_setup_geometry/01_cor"},
        {"label": "pose", "source_stage": "04_pose_polish"},
        {"label": "final", "source_stage": "05_final"},
    ]

    assert select_real_lamino_final_candidates(candidates, policy="all") == candidates
    assert select_real_lamino_final_candidates(candidates, policy="last-valid") == [
        candidates[-1]
    ]
    assert select_real_lamino_final_candidates(candidates, policy="setup_only") == [
        candidates[0]
    ]

    with pytest.raises(ValueError, match="final candidate policy"):
        select_real_lamino_final_candidates(candidates, policy="sharpest")


def test_real_laminography_planning_maps_global_and_local_z() -> None:
    grid = SimpleNamespace(nz=4, vz=2.0, vol_origin=None, vol_center=(0.0, 0.0, 1.0))

    assert real_lamino_grid_origin_z(grid) == -2.0
    assert real_lamino_global_z_to_phys(5, full_nz=8) == 1.5
    assert real_lamino_global_z_to_local_index(5, full_nz=8, grid=grid) == 2
    assert real_lamino_local_z_to_global_index(2, full_nz=8, grid=grid) == 6


def test_real_laminography_planning_extracts_xy_at_global_z_in_display_orientation() -> None:
    grid = SimpleNamespace(nz=3, vz=1.0, vol_origin=(0.0, 0.0, -1.0), vol_center=None)
    volume = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)

    xy = real_lamino_xy_at_global_z(volume, grid=grid, full_nz=3, global_z=1)

    np.testing.assert_array_equal(xy, volume[:, :, 1].T)
