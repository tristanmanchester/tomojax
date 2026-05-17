# tomojax.datasets

## Purpose

`tomojax.datasets` owns deterministic synthetic benchmark specifications,
phantom generation, dataset manifests, masks, and recovery tolerances.

The current implementation emits deterministic small synthetic artifacts for
fast development checks and can configure 128^3 benchmark artifacts from the v2
manifest. Its projection writer is a CPU reference projector used for synthetic
data generation, while production forward modelling belongs to
`tomojax.forward`.

## Public API

- `SyntheticArtifactPaths`
- `SyntheticArrayMetadata`
- `SyntheticDatasetSidecars`
- `SyntheticDatasetSpec`
- `SimConfig`
- `SimulatedData`
- `SimulationArtefacts`
- `generate_synthetic_dataset`
- `load_synthetic_dataset_sidecars`
- `load_synthetic128_specs`
- `make_benchmark_phantom`
- `make_phantom`
- `random_cubes_spheres`
- `simulate`
- `simulate_to_file`
- `synthetic128_spec`
- `validate_simulation_artefacts`

## Dependencies

Allowed future dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.motion`
- `tomojax.nuisance`
- `tomojax.forward`
- `tomojax.io`

Forbidden dependencies:

- private implementation files from other deep modules
- old staged alignment engines
- nondeterministic generation paths

## Invariants

- Every synthetic recovery path must be deterministic from a seed.
- Dataset manifests must include true and corrupted geometry metadata.
- Generated data should be written outside the source tree unless explicitly
  requested.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this facade exists and imports.
- `tests/test_synthetic_datasets.py` verifies manifest loading, deterministic
  phantom generation, and artifact emission.
