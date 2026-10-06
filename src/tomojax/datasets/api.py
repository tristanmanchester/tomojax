"""Public API for deterministic synthetic datasets."""

from __future__ import annotations

from tomojax.datasets._phantoms import (
    blobs,
    cube,
    lamino_disk,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)
from tomojax.datasets._simulate import (
    SimConfig,
    SimMetadata,
    SimulatedData,
    SimulationArtefacts,
    make_phantom,
    simulate,
    simulate_to_file,
    validate_simulation_artefacts,
)

__all__ = [
    "SimConfig",
    "SimMetadata",
    "SimulatedData",
    "SimulationArtefacts",
    "blobs",
    "cube",
    "lamino_disk",
    "make_phantom",
    "random_cubes_spheres",
    "rotated_centered_cube",
    "shepp_logan_3d",
    "simulate",
    "simulate_to_file",
    "sphere",
    "validate_simulation_artefacts",
]
