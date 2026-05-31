# tomojax.nuisance

## Purpose

`tomojax.nuisance` handles acquisition-state effects that alter projections
without being geometry or motion: gain/offset, backgrounds, masks, noise
scales, stripe bias, and bad frames.

Current primitives are per-view gain/offset fitting and a low-frequency
background offset model. These model acquisition drift so geometry does not
need to absorb flat-field-like changes.

## Public API

- `BackgroundOffsetModel`
- `GainOffsetModel`
- `estimate_background_offset`
- `estimate_gain_offset`

## Dependencies

Allowed: `tomojax.core`, `tomojax.io`, JAX arrays.

Forbidden: private files from other modules, solvers, projectors.

## Invariants

- Nuisance state is separate from setup geometry and per-view pose.
- Public APIs must be typed.
- Provenance records must identify any nuisance correction applied.

## Tests

- `tests/test_product_surface.py` verifies the public module import surface.
- `tests/test_nuisance_models.py` verifies masked per-view gain/offset fitting,
  identity application, residual reduction for synthetic drift, and masked
  constant plus vertical-gradient background fitting.
