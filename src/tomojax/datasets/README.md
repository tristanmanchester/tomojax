# tomojax.datasets

## Purpose

`tomojax.datasets` will own deterministic synthetic benchmark specifications,
phantom generation, dataset manifests, masks, and recovery tolerances.

This package is currently a v2 skeleton facade. It intentionally exposes no
public behavior until the synthetic benchmark foundation milestone defines
typed contracts.

## Public API

No public names are exported yet.

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

- `tests/test_v2_module_skeleton.py` verifies this skeleton facade exists and
  imports.
