# tomojax.nuisance

## Purpose

`tomojax.nuisance` will own acquisition-state effects that alter measured
projections without being geometry or object motion: gain/offset, backgrounds,
masks, robust noise scales, stripe bias, bad frames, and related provenance.

The first Phase 8 public primitive is per-view gain/offset fitting. It models
affine intensity drift as acquisition state so geometry does not need to absorb
flat-field-like projection changes.

## Public API

- `GainOffsetModel`
- `estimate_gain_offset`

## Dependencies

Allowed dependencies:

- `tomojax.core`
- `tomojax.io`
- JAX arrays

Forbidden dependencies:

- private implementation files from other deep modules
- old generic utility modules
- solver or projector implementations

## Invariants

- Nuisance state is separate from setup geometry and per-view pose.
- Public APIs must be typed before export.
- Artifact/provenance records must identify any nuisance correction applied.

## Tests

- `tests/test_v2_module_skeleton.py` verifies this skeleton facade exists and
  imports.
- `tests/test_nuisance_gain_offset.py` verifies masked per-view fitting,
  identity application, and residual reduction for synthetic drift.
