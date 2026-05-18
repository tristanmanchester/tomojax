# tomojax.nuisance

## Purpose

`tomojax.nuisance` will own acquisition-state effects that alter measured
projections without being geometry or object motion: gain/offset, backgrounds,
masks, robust noise scales, stripe bias, bad frames, and related provenance.

The first Phase 8 public primitives are per-view gain/offset fitting and a
low-frequency background offset model. They model acquisition drift so geometry
does not need to absorb flat-field-like projection changes.

## Public API

- `BackgroundOffsetModel`
- `GainOffsetModel`
- `estimate_background_offset`
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
- `tests/test_nuisance_background.py` verifies masked constant plus
  vertical-gradient background fitting.
