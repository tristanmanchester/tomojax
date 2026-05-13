"""Dataset and benchmark fixture public facade."""

from __future__ import annotations

from tomojax.datasets.api import (
    SimConfig,
    SimulatedData,
    SimulationArtefacts,
    SyntheticArrayMetadata,
    SyntheticArtifactPaths,
    SyntheticDatasetSidecars,
    SyntheticDatasetSpec,
    generate_synthetic_dataset,
    load_synthetic128_specs,
    load_synthetic_dataset_sidecars,
    make_benchmark_phantom,
    simulate,
    simulate_to_file,
    synthetic128_spec,
    validate_simulation_artefacts,
)

__all__ = [
    "SimConfig",
    "SimulatedData",
    "SimulationArtefacts",
    "SyntheticArrayMetadata",
    "SyntheticArtifactPaths",
    "SyntheticDatasetSidecars",
    "SyntheticDatasetSpec",
    "generate_synthetic_dataset",
    "load_synthetic128_specs",
    "load_synthetic_dataset_sidecars",
    "make_benchmark_phantom",
    "simulate",
    "simulate_to_file",
    "synthetic128_spec",
    "validate_simulation_artefacts",
]
