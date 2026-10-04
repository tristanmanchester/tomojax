"""Reconstruct a synthetic object using matched public projection/solver APIs.

This is a usage example, not an independent accuracy benchmark: simulation and
reconstruction deliberately use the same Joseph discretization.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.datasets import shepp_logan_3d
from tomojax.forward import project_joseph
from tomojax.geometry import Detector, Grid, ParallelGeometry, stack_view_poses
from tomojax.recon import CGLSConfig, cgls


def reconstruct_example(
    *,
    size: int = 32,
    views: int = 60,
    iterations: int = 24,
    backend: Literal["jax", "pallas"] = "jax",
) -> tuple[np.ndarray, np.ndarray, dict[str, str | int | float]]:
    """Return reference, reconstruction, and full-volume error for a parallel scan."""
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = Detector(size, size, 1.0, 1.0)
    angles = np.linspace(0.0, 180.0, views, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid, detector, angles)
    truth = shepp_logan_3d(size, size, size)
    poses = stack_view_poses(geometry, views)
    projections = project_joseph(jnp.asarray(truth), poses, grid, detector, backend=backend)

    volume, info = cgls(
        geometry,
        grid,
        detector,
        projections,
        config=CGLSConfig(
            iters=iterations,
            projector_model="joseph",
            projector_backend=backend,
            views_per_batch=views,
        ),
    )
    volume = np.asarray(jax.device_get(volume))
    if not np.isfinite(volume).all():
        raise RuntimeError("The example produced a nonfinite reconstruction")
    error = np.linalg.norm(volume.astype(np.float64) - truth) / np.linalg.norm(truth)
    metrics = {
        "size": size,
        "views": views,
        "iteration_budget": iterations,
        "projector": "joseph_linear",
        "backend": backend,
        "device_platform": jax.default_backend(),
        "termination": str(info["termination"]),
        "volume_relative_l2": float(error),
    }
    return truth, volume, metrics


def main() -> None:
    """Run the small example on the available JAX device."""
    truth, volume, metrics = reconstruct_example()
    print(f"reference_shape={truth.shape} reconstruction_shape={volume.shape}")
    print(f"relative_l2={metrics['volume_relative_l2']:.6g}")
    print(f"termination={metrics['termination']}")


if __name__ == "__main__":
    main()
