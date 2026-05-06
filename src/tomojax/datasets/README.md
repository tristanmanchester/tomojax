# tomojax.datasets

## Purpose

`tomojax.datasets` owns deterministic synthetic benchmark specifications,
phantom generation, dataset manifests, masks, and recovery tolerances.

The current implementation is a foundation writer: it emits deterministic
32^3 smoke artifacts and can configure 128^3 benchmark artifacts from the v2
manifest. Its projection writer is a CPU smoke projector, not the final
differentiable `tomojax.forward` reference path.

## Public API

- `SyntheticArtifactPaths`
- `SyntheticDatasetSpec`
- `generate_synthetic_dataset`
- `load_synthetic128_specs`
- `make_benchmark_phantom`
- `synthetic128_spec`

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
  phantom generation, and smoke artifact emission.
