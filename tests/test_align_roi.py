import logging

import pytest

import tomojax.align._pose_stage as pose_stage
import tomojax.cli.align as align_cli
import tomojax.cli.recon as recon_cli
from tomojax.cli.align import _resolve_recon_grid_and_mask
from tomojax.core.geometry import Detector, Grid


def test_grid_override_disables_cylindrical_mask():
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
        grid,
        detector,
        roi_mode="cyl",
        grid_override=(40, 24, 12),
    )

    assert (recon_grid.nx, recon_grid.ny, recon_grid.nz) == (40, 24, 12)
    assert apply_cyl_mask is False


def test_auto_roi_grid_override_preserves_centered_convention():
    grid = Grid(
        nx=32,
        ny=32,
        nz=16,
        vx=1.0,
        vy=1.0,
        vz=1.0,
        vol_origin=(10.0, 20.0, 30.0),
        vol_center=(1.0, 2.0, 3.0),
    )
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
        grid,
        detector,
        roi_mode="auto",
        grid_override=(20, 18, 12),
    )

    assert (recon_grid.nx, recon_grid.ny, recon_grid.nz) == (20, 18, 12)
    assert recon_grid.vol_origin is None
    assert recon_grid.vol_center is None
    assert apply_cyl_mask is False


def test_auto_roi_failure_warns_and_keeps_full_grid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    def fail_roi(*args: object, **kwargs: object) -> object:
        raise RuntimeError("bad detector geometry")

    monkeypatch.setattr(align_cli, "compute_roi", fail_roi)

    with caplog.at_level(logging.WARNING):
        recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
            grid,
            detector,
            roi_mode="auto",
            grid_override=None,
        )

    assert recon_grid is grid
    assert apply_cyl_mask is False
    assert "--roi=auto could not be applied" in caplog.text
    assert "bad detector geometry" in caplog.text


def test_requested_roi_failure_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    def fail_roi(*args: object, **kwargs: object) -> object:
        raise RuntimeError("bad detector geometry")

    monkeypatch.setattr(align_cli, "compute_roi", fail_roi)

    with pytest.raises(ValueError, match="--roi='cube'") as excinfo:
        _resolve_recon_grid_and_mask(
            grid,
            detector,
            roi_mode="cube",
            grid_override=None,
        )

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_recon_auto_roi_failure_warns_and_keeps_full_grid(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    def fail_roi(*args: object, **kwargs: object) -> object:
        raise RuntimeError("bad detector geometry")

    monkeypatch.setattr(recon_cli, "compute_roi", fail_roi)

    with caplog.at_level(logging.WARNING):
        recon_grid = recon_cli._resolve_recon_grid_for_cli(
            grid,
            detector,
            is_parallel=True,
            roi_mode="auto",
        )

    assert recon_grid is grid
    assert "--roi=auto could not be applied" in caplog.text
    assert "bad detector geometry" in caplog.text


def test_recon_requested_mask_failure_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    def fail_mask(*args: object, **kwargs: object) -> object:
        raise RuntimeError("mask failed")

    monkeypatch.setattr(recon_cli, "cylindrical_mask_xy", fail_mask)

    with pytest.raises(ValueError, match="--mask-vol='cyl'") as excinfo:
        recon_cli._resolve_volume_mask_for_cli(grid, detector, mask_vol="cyl")

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_alignment_requested_mask_failure_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = Grid(nx=32, ny=32, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    def fail_mask(*args: object, **kwargs: object) -> object:
        raise RuntimeError("mask failed")

    monkeypatch.setattr(pose_stage, "cylindrical_mask_xy", fail_mask)

    with pytest.raises(ValueError, match="mask_vol='cyl'") as excinfo:
        pose_stage._build_alignment_volume_mask(grid, detector, mask_vol="cyl")

    assert isinstance(excinfo.value.__cause__, RuntimeError)
