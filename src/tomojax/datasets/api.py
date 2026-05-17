"""Public API for deterministic synthetic datasets."""

from __future__ import annotations

from tomojax.datasets._loader import (
    SyntheticArrayMetadata,
    SyntheticDatasetConsistency,
    SyntheticDatasetSidecars,
    load_synthetic_dataset_sidecars,
)
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
    LaminoGeometryMeta,
    SimConfig,
    SimMetadata,
    SimulatedData,
    SimulationArtefacts,
    apply_simulation_artefacts,
    make_phantom,
    simulate,
    simulate_to_file,
    validate_simulation_artefacts,
)

__all__ = [
    "LaminoGeometryMeta",
    "SimConfig",
    "SimMetadata",
    "SimulatedData",
    "SimulationArtefacts",
    "SyntheticArrayMetadata",
    "SyntheticDatasetConsistency",
    "SyntheticDatasetSidecars",
    "apply_simulation_artefacts",
    "blobs",
    "cube",
    "lamino_disk",
    "load_synthetic_dataset_sidecars",
    "make_phantom",
    "random_cubes_spheres",
    "rotated_centered_cube",
    "shepp_logan_3d",
    "simulate",
    "simulate_to_file",
    "sphere",
    "validate_simulation_artefacts",
]
