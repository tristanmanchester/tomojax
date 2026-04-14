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
