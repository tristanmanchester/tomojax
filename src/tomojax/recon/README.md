# tomojax.recon

## Purpose

`tomojax.recon` owns volume reconstruction routines, schedules, priors, trace
rows, warm starts, and non-negativity/support constraints.

The current package still contains transitional reconstruction code. This
README and `api.py` define the public boundary while later milestones migrate
the default reconstruction step to FISTA / Huber-TV FISTA against the v2
forward model.

## Public API

- `FBPConfig`
- `FistaConfig`
- `ReferenceFISTAConfig`
- `ReferenceFISTAResult`
- `ReferenceFISTATraceRow`
- `Regulariser`
- `SPDHGConfig`
- `fbp`
- `fista_reconstruct_reference`
- `fista_tv`
- `reconstruct_average_reference`
- `spdhg_tv`
- `write_fista_trace_csv`

## Dependencies

Allowed dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.forward`
- `tomojax.backends`

Forbidden dependencies:

- alignment solver orchestration
- private implementation files from other deep modules
- generic utility modules

## Invariants

- The default v2 reconstruction step is FISTA / Huber-TV FISTA.
- Alignment must use stopped-gradient latent volumes.
- Reconstruction traces must be suitable for artifact reports.
- `fista_reconstruct_reference` is the first v2 JAX reference FISTA preview
  contract. It uses the current minimal reference projector.
- `reconstruct_average_reference` is a tiny smoke helper, not the final default
  reconstruction algorithm.

## Tests

- Existing reconstruction tests cover transitional behavior.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
