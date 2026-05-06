# tomojax.nuisance

## Purpose

`tomojax.nuisance` will own acquisition-state effects that alter measured
projections without being geometry or object motion: gain/offset, backgrounds,
masks, robust noise scales, stripe bias, bad frames, and related provenance.

This package is currently a v2 skeleton facade. It intentionally exposes no
public behavior until the nuisance modelling milestone defines typed contracts.

## Public API

No public names are exported yet.

## Dependencies

Allowed future dependencies:

- `tomojax.core`
- `tomojax.io`

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
