# tomojax.forward

## Purpose

`tomojax.forward` will own differentiable projection, backprojection,
projection-domain residuals, and geometry-parameter reductions for the v2 JAX
reference path.

This package is currently a v2 skeleton facade. It intentionally exposes no
public behavior until the reference forward-model milestone defines typed
contracts.

## Public API

No public names are exported yet.

## Dependencies

Allowed future dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.motion`
- `tomojax.nuisance`
- `tomojax.backends`

Forbidden dependencies:

- private implementation files from other deep modules
- reconstruction or alignment solver orchestration
- Pallas fast paths as default behavior without JAX-reference equivalence tests

## Invariants

- The JAX reference implementation is the correctness oracle.
- Projection residuals must support masks and robust whitening.
- Backend fast paths must report provenance and compare against the reference
  path.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this skeleton facade exists and
  imports.
