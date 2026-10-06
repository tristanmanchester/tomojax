"""Simulate a parallel-beam scan of a Shepp-Logan phantom and reconstruct it.

This is a usage example, not an independent accuracy benchmark: simulation and
reconstruction deliberately use the same Joseph discretization.
"""

from __future__ import annotations

import jax
import numpy as np

import tomojax as tj
from tomojax.datasets import shepp_logan_3d


def reconstruct_example(
    *, size: int = 32, views: int = 60, iterations: int = 24
) -> tuple[np.ndarray, np.ndarray, dict[str, str | int | float]]:
    """Return reference, reconstruction, and full-volume error for a parallel scan."""
    grid = tj.Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = tj.Detector(size, size, 1.0, 1.0)
    angles = np.linspace(0.0, 180.0, views, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    truth = shepp_logan_3d(size, size, size)

    scan = tj.Scan(tj.project(geometry, truth), geometry)
    recon = tj.reconstruct(scan, "cgls", iterations=iterations)

    volume = np.asarray(recon.volume)
    if not np.isfinite(volume).all():
        raise RuntimeError("The example produced a nonfinite reconstruction")
    error = np.linalg.norm(volume.astype(np.float64) - truth) / np.linalg.norm(truth)
    metrics = {
        "size": size,
        "views": views,
        "iteration_budget": iterations,
        "method": recon.method,
        "device_platform": jax.default_backend(),
        "volume_relative_l2": float(error),
    }
    return truth, volume, metrics


def main() -> None:
    """Run the small example on the available JAX device."""
    truth, volume, metrics = reconstruct_example()
    print(f"reference_shape={truth.shape} reconstruction_shape={volume.shape}")
    print(f"relative_l2={metrics['volume_relative_l2']:.6g}")


if __name__ == "__main__":
    main()
