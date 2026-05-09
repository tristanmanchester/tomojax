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
- `ReferenceFISTADiagnosticArtifacts`
- `ReferenceFISTAResult`
- `ReferenceFISTASchedule`
- `ReferenceFISTAScheduleEntry`
- `ReferenceFISTATraceRow`
- `ReferenceReconstructionScheduleName`
- `Regulariser`
- `SPDHGConfig`
- `VolumeSupportKind`
- `centered_volume_support`
- `fbp`
- `fista_reconstruct_reference`
- `fista_tv`
- `reference_fista_diagnostic_artifacts`
- `reference_fista_schedule`
- `reconstruct_average_reference`
- `reconstruct_backprojection_reference`
- `spdhg_tv`
- `write_fista_trace_recomputed_csv`
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
- `fista_reconstruct_reference` is the v2 preview FISTA contract. It uses the
  core trilinear ray projector for residuals and the core explicit adjoint for
  data gradients, and can optionally project iterates into a nonnegative support
  mask.
- `centered_volume_support` creates deterministic cylindrical or spherical
  masks for synthetic reference diagnostics.
- `reference_fista_schedule` resolves the v2 preview/final schedule contract;
  full pyramid execution belongs to a later orchestration milestone.
- `reference_fista_diagnostic_artifacts` emits deterministic scalar-gradient,
  adjoint, detector-centre JVP/VJP, loss-normalisation, and trace-recompute
  payloads for diagnostic run artifacts.
- `reconstruct_average_reference` is a tiny smoke helper, not the final default
  reconstruction algorithm.
- `reconstruct_backprojection_reference` is a deterministic geometry-aware
  initializer using the core explicit adjoint for smoke and FISTA warm starts.

## Tests

- Existing reconstruction tests cover transitional behavior.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
