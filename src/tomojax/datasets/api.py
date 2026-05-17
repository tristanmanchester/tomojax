"""Public API for deterministic datasets."""

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
    lamino_disk_legacy,
    make_benchmark_phantom,
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
from tomojax.datasets._specs import SyntheticDatasetSpec, load_synthetic128_specs, synthetic128_spec
from tomojax.datasets._writer import SyntheticArtifactPaths, generate_synthetic_dataset

__all__ = [
    "LaminoGeometryMeta",
    "SimConfig",
    "SimMetadata",
    "SimulatedData",
    "SimulationArtefacts",
    "SyntheticArrayMetadata",
    "SyntheticArtifactPaths",
    "SyntheticDatasetConsistency",
    "SyntheticDatasetSidecars",
    "SyntheticDatasetSpec",
    "apply_simulation_artefacts",
    "blobs",
    "cube",
    "generate_synthetic_dataset",
    "lamino_disk",
    "lamino_disk_legacy",
    "load_synthetic128_specs",
    "load_synthetic_dataset_sidecars",
    "make_benchmark_phantom",
    "make_phantom",
    "random_cubes_spheres",
    "rotated_centered_cube",
    "shepp_logan_3d",
    "simulate",
    "simulate_to_file",
    "sphere",
    "synthetic128_spec",
    "validate_simulation_artefacts",
]
