"""Minimal public-API tomography smoke example."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tomojax.forward import project_parallel_reference_arrays
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FistaConfig, fista_tv


def main() -> None:
    """Simulate a tiny cube phantom and reconstruct it with public facades."""
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    theta_deg = np.linspace(0.0, 180.0, 8, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=theta_deg)

    phantom = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    phantom = phantom.at[2:6, 2:6, 2:6].set(1.0)

    theta_rad = jnp.asarray(np.deg2rad(theta_deg), dtype=jnp.float32)
    zero_shift = jnp.zeros_like(theta_rad)
    projections = project_parallel_reference_arrays(
        phantom,
        theta_rad=theta_rad,
        dx_px=zero_shift,
        dz_px=zero_shift,
        detector_shape=(detector.nv, detector.nu),
    )

    volume, info = fista_tv(
        geometry,
        grid,
        detector,
        projections,
        config=FistaConfig(iters=8, lambda_tv=0.001, views_per_batch=1),
    )
    print(f"reconstruction_shape={tuple(volume.shape)}")
    print(f"loss={float(info['loss'][-1]):.6g}")


if __name__ == "__main__":
    main()
