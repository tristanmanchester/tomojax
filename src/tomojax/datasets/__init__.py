"""Dataset and benchmark fixture public facade."""

from __future__ import annotations

from tomojax.datasets.api import (
    SyntheticArtifactPaths,
    SyntheticDatasetSidecars,
    SyntheticDatasetSpec,
    generate_synthetic_dataset,
    load_synthetic128_specs,
    load_synthetic_dataset_sidecars,
    make_benchmark_phantom,
    synthetic128_spec,
)

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
