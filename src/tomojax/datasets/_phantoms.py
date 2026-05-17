"""Deterministic procedural phantoms for synthetic benchmark artifacts."""
# pyright: reportAny=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.data.phantoms import (
    blobs,
    cube,
    lamino_disk,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "blobs",
    "cube",
    "lamino_disk",
    "make_benchmark_phantom",
    "random_cubes_spheres",
    "rotated_centered_cube",
    "shepp_logan_3d",
    "sphere",
]


def make_benchmark_phantom(size: int, seed: int) -> NDArray[np.float32]:
    """Return a deterministic structured volume with asymmetric features."""
    rng = np.random.default_rng(seed)
    z, y, x = _grid(size)
    vol = np.zeros((size, size, size), dtype=np.float32)

    vol += 0.35 * _ellipsoid(x, y, z, center=(0.02, -0.04, 0.01), radii=(0.72, 0.58, 0.50))

    for _ in range(10):
        center = _triple(rng.uniform(-0.45, 0.45, size=3))
        radii = _triple(rng.uniform(0.05, 0.18, size=3))
        vol += np.float32(rng.uniform(0.05, 0.22)) * _ellipsoid(x, y, z, center=center, radii=radii)

    for _ in range(12):
        center = _triple(rng.uniform(-0.65, 0.65, size=3))
        radius = float(rng.uniform(0.015, 0.045))
        amp = np.float32(rng.uniform(0.20, 0.65))
        vol += amp * _gaussian_sphere(x, y, z, center=center, radius=radius)

    for center in ((0.48, -0.35, 0.30), (0.55, -0.28, 0.25), (0.43, -0.42, 0.38)):
        vol += np.float32(0.85) * _gaussian_sphere(x, y, z, center=center, radius=0.03)

    for _ in range(8):
        center = _triple(rng.uniform(-0.55, 0.55, size=3))
        radius = float(rng.uniform(0.03, 0.09))
        void = _gaussian_sphere(x, y, z, center=center, radius=radius) > 0.35
        vol[void] *= np.float32(rng.uniform(0.1, 0.5))

    texture = rng.normal(0.0, 1.0, size=vol.shape).astype(np.float32)
    support = vol > np.float32(0.02)
    vol += support * np.float32(0.015) * texture
    return np.clip(vol, 0.0, None).astype(np.float32)


def _triple(values: NDArray[np.float64]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _grid(size: int) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    axis = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)
    return np.meshgrid(axis, axis, axis, indexing="ij")


def _ellipsoid(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    *,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> NDArray[np.float32]:
    cx, cy, cz = center
    rx, ry, rz = radii
    values = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2
    return (values <= 1.0).astype(np.float32)


def _gaussian_sphere(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    z: NDArray[np.float32],
    *,
    center: tuple[float, float, float],
    radius: float,
) -> NDArray[np.float32]:
    cx, cy, cz = center
    dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return np.exp(-dist2 / np.float32(2.0 * radius * radius)).astype(np.float32)
