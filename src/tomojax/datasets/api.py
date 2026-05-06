"""Public API for deterministic datasets."""

from __future__ import annotations

from tomojax.datasets._loader import SyntheticDatasetSidecars, load_synthetic_dataset_sidecars
from tomojax.datasets._phantoms import make_benchmark_phantom
from tomojax.datasets._specs import SyntheticDatasetSpec, load_synthetic128_specs, synthetic128_spec
from tomojax.datasets._writer import SyntheticArtifactPaths, generate_synthetic_dataset

__all__ = [
    "SyntheticArtifactPaths",
    "SyntheticDatasetSidecars",
    "SyntheticDatasetSpec",
    "generate_synthetic_dataset",
    "load_synthetic128_specs",
    "load_synthetic_dataset_sidecars",
    "make_benchmark_phantom",
    "synthetic128_spec",
]
