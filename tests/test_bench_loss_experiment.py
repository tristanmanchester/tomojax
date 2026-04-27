from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tomojax.bench.loss_experiment import (
    make_misaligned_dataset,
    project_gt_with_estimated_poses,
)
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.data.io_hdf5 import NXTomoMetadata, load_nxtomo, save_nxtomo


def _write_tiny_gt(path: Path) -> tuple[Grid, Detector, np.ndarray]:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0.0, 180.0, 4, endpoint=False, dtype=np.float32)
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = 1.0
    save_nxtomo(
        str(path),
        projections=np.zeros((4, 4, 4), dtype=np.float32),
        metadata=NXTomoMetadata(
            thetas_deg=thetas,
            grid=grid,
            detector=detector,
            geometry_type="parallel",
            volume=volume,
            frame="sample",
        ),
    )
    return grid, detector, thetas


def test_make_misaligned_dataset_writes_projected_dataset_and_reuses_existing(
    tmp_path: Path,
) -> None:
    expdir = tmp_path / "experiment"
    gt_path = expdir / "gt.nxs"
    grid, detector, thetas = _write_tiny_gt(gt_path)

    mis_path = make_misaligned_dataset(
        str(expdir),
        str(gt_path),
        rot_deg=0.5,
        trans_px=0.25,
        seed=7,
    )
    first = load_nxtomo(mis_path)

    assert first.projections.shape == (4, 4, 4)
    assert first.projections.dtype == np.float32
    assert first.align_params is not None
    assert first.align_params.shape == (4, 5)
    assert np.any(np.abs(first.align_params) > 0.0)
    assert np.max(np.abs(first.align_params[:, :3])) <= np.deg2rad(0.5)
    assert np.max(np.abs(first.align_params[:, 3] / detector.du)) <= 0.25
    assert np.max(np.abs(first.align_params[:, 4] / detector.dv)) <= 0.25
    assert first.grid == grid.to_dict()
    assert first.detector == detector.to_dict()
    assert np.allclose(first.thetas_deg, thetas)
    assert first.geometry_type == "parallel"
    assert first.frame == "sample"

    reused_path = make_misaligned_dataset(
        str(expdir),
        str(gt_path),
        rot_deg=5.0,
        trans_px=5.0,
        seed=99,
    )
    reused = load_nxtomo(reused_path)

    assert reused_path == mis_path
    assert np.array_equal(reused.align_params, first.align_params)
    assert np.array_equal(reused.projections, first.projections)


def test_project_gt_with_estimated_poses_returns_projection_stack_shape_and_dtype() -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.linspace(0.0, 180.0, 4, endpoint=False, dtype=np.float32),
    )
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)
    params = np.zeros((4, 5), dtype=np.float32)

    projections = project_gt_with_estimated_poses(volume, grid, detector, geometry, params)

    assert projections.shape == (4, 4, 4)
    assert projections.dtype == jnp.float32
