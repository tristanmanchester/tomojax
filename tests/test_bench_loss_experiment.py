from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.bench.loss_experiment import (
    make_gt_dataset,
    make_misaligned_dataset,
    metrics_abs,
    metrics_relative,
    project_gt_with_estimated_poses,
)
from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.datasets import SimConfig
from tomojax.io import NXTomoMetadata, load_nxtomo, save_nxtomo


def _write_tiny_gt(
    path: Path,
    *,
    volume_value: float = 1.0,
) -> tuple[Grid, Detector, np.ndarray]:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0.0, 180.0, 4, endpoint=False, dtype=np.float32)
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1:3, 1:3, 1:3] = volume_value
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


def test_make_gt_dataset_reuses_existing_file_only_when_provenance_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expdir = tmp_path / "experiment"
    calls: list[SimConfig] = []

    def fake_simulate_to_file(cfg: SimConfig, out_path: str) -> str:
        calls.append(cfg)
        grid = Grid(nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, vx=1.0, vy=1.0, vz=1.0)
        detector = Detector(nu=cfg.nu, nv=cfg.nv, du=1.0, dv=1.0, det_center=(0.0, 0.0))
        volume = np.full((cfg.nx, cfg.ny, cfg.nz), float(cfg.seed), dtype=np.float32)
        save_nxtomo(
            out_path,
            projections=np.full(
                (cfg.n_views, cfg.nv, cfg.nu),
                float(cfg.seed),
                dtype=np.float32,
            ),
            metadata=NXTomoMetadata(
                thetas_deg=np.linspace(
                    0.0,
                    180.0,
                    cfg.n_views,
                    endpoint=False,
                    dtype=np.float32,
                ),
                grid=grid,
                detector=detector,
                geometry_type=cfg.geometry,
                volume=volume,
                frame="sample",
            ),
        )
        return out_path

    monkeypatch.setattr(
        "tomojax.bench.loss_experiment.simulate_to_file",
        fake_simulate_to_file,
    )

    kwargs = dict(nx=4, ny=4, nz=4, nu=4, nv=4, n_views=4, geometry="parallel", seed=7)
    gt_path = make_gt_dataset(str(expdir), **kwargs)
    first = load_nxtomo(gt_path)

    reused_path = make_gt_dataset(str(expdir), **kwargs)
    reused = load_nxtomo(reused_path)

    regenerated_path = make_gt_dataset(str(expdir), **{**kwargs, "seed": 9})
    regenerated = load_nxtomo(regenerated_path)

    assert len(calls) == 2
    assert reused_path == gt_path
    assert regenerated_path == gt_path
    assert np.array_equal(reused.projections, first.projections)
    assert np.all(regenerated.projections == 9.0)
    assert regenerated.geometry_meta is not None
    assert regenerated.geometry_meta["loss_experiment_provenance"]["seed"] == 9


def test_make_gt_dataset_regenerates_existing_file_without_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expdir = tmp_path / "experiment"
    gt_path = expdir / "gt.nxs"
    _write_tiny_gt(gt_path)
    calls = 0

    def fake_simulate_to_file(cfg: SimConfig, out_path: str) -> str:
        nonlocal calls
        calls += 1
        grid = Grid(nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, vx=1.0, vy=1.0, vz=1.0)
        detector = Detector(nu=cfg.nu, nv=cfg.nv, du=1.0, dv=1.0, det_center=(0.0, 0.0))
        save_nxtomo(
            out_path,
            projections=np.full((cfg.n_views, cfg.nv, cfg.nu), 3.0, dtype=np.float32),
            metadata=NXTomoMetadata(
                thetas_deg=np.linspace(
                    0.0,
                    180.0,
                    cfg.n_views,
                    endpoint=False,
                    dtype=np.float32,
                ),
                grid=grid,
                detector=detector,
                geometry_type=cfg.geometry,
                volume=np.full((cfg.nx, cfg.ny, cfg.nz), 3.0, dtype=np.float32),
                frame="sample",
            ),
        )
        return out_path

    monkeypatch.setattr(
        "tomojax.bench.loss_experiment.simulate_to_file",
        fake_simulate_to_file,
    )

    returned_path = make_gt_dataset(
        str(expdir),
        nx=4,
        ny=4,
        nz=4,
        nu=4,
        nv=4,
        n_views=4,
        geometry="parallel",
        seed=3,
    )
    loaded = load_nxtomo(returned_path)

    assert calls == 1
    assert np.all(loaded.projections == 3.0)
    assert loaded.geometry_meta is not None
    assert loaded.geometry_meta["loss_experiment_provenance"]["seed"] == 3


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
        rot_deg=0.5,
        trans_px=0.25,
        seed=7,
    )
    reused = load_nxtomo(reused_path)

    assert reused_path == mis_path
    assert np.array_equal(reused.align_params, first.align_params)
    assert np.array_equal(reused.projections, first.projections)
    assert reused.misalign_spec is not None
    provenance = reused.misalign_spec["loss_experiment_provenance"]
    assert provenance["rot_deg"] == 0.5
    assert provenance["trans_px"] == 0.25
    assert provenance["seed"] == 7
    assert "source_gt" in provenance


def test_make_misaligned_dataset_regenerates_when_misalignment_provenance_changes(
    tmp_path: Path,
) -> None:
    expdir = tmp_path / "experiment"
    gt_path = expdir / "gt.nxs"
    _write_tiny_gt(gt_path)

    mis_path = make_misaligned_dataset(
        str(expdir),
        str(gt_path),
        rot_deg=0.5,
        trans_px=0.25,
        seed=7,
    )
    first = load_nxtomo(mis_path)

    regenerated_path = make_misaligned_dataset(
        str(expdir),
        str(gt_path),
        rot_deg=0.5,
        trans_px=0.25,
        seed=99,
    )
    regenerated = load_nxtomo(regenerated_path)

    assert regenerated_path == mis_path
    assert not np.array_equal(regenerated.align_params, first.align_params)
    assert regenerated.misalign_spec is not None
    assert regenerated.misalign_spec["loss_experiment_provenance"]["seed"] == 99


def test_make_misaligned_dataset_regenerates_when_source_gt_changes(
    tmp_path: Path,
) -> None:
    expdir = tmp_path / "experiment"
    gt_path = expdir / "gt.nxs"
    _write_tiny_gt(gt_path, volume_value=1.0)

    mis_path = make_misaligned_dataset(
        str(expdir),
        str(gt_path),
        rot_deg=0.5,
        trans_px=0.25,
        seed=7,
    )
    first = load_nxtomo(mis_path)

    _write_tiny_gt(gt_path, volume_value=0.0)
    regenerated_path = make_misaligned_dataset(
        str(expdir),
        str(gt_path),
        rot_deg=0.5,
        trans_px=0.25,
        seed=7,
    )
    regenerated = load_nxtomo(regenerated_path)

    assert regenerated_path == mis_path
    assert np.array_equal(regenerated.align_params, first.align_params)
    assert not np.array_equal(regenerated.projections, first.projections)


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


@pytest.mark.parametrize("metric_fn", [metrics_abs, metrics_relative])
def test_metric_helpers_reject_mismatched_pose_parameter_shapes(metric_fn) -> None:
    params_true = np.zeros((4, 5), dtype=np.float32)
    params_est = np.zeros((3, 5), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        metric_fn(params_true, params_est, 1.0, 1.0)

    message = str(excinfo.value)
    assert "params_true.shape=(4, 5)" in message
    assert "params_est.shape=(3, 5)" in message
