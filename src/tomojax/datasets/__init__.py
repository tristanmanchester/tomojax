"""Deterministic synthetic scans and phantoms for the ``tomojax simulate`` workflow."""

from __future__ import annotations

from tomojax.datasets._impl.artifacts import SimulationArtifacts
from tomojax.datasets._impl.detector import DetectorModel, DetectorSimulation
from tomojax.datasets._impl.phantoms import (
    blobs,
    cube,
    lamino_disk,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)
from tomojax.datasets._impl.simulate import (
    SimConfig,
    SimulatedData,
    make_phantom,
    simulate,
)

__all__ = [
    "DetectorModel",
    "DetectorSimulation",
    "SimConfig",
    "SimulatedData",
    "SimulationArtifacts",
    "blobs",
    "cube",
    "lamino_disk",
    "make_phantom",
    "random_cubes_spheres",
    "rotated_centered_cube",
    "shepp_logan_3d",
    "simulate",
    "sphere",
]
