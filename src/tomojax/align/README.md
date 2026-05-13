# tomojax.align

## Purpose

`tomojax.align` owns alignment orchestration: alternating reconstruction and
geometry updates, continuation policy, observability reporting, gauge
canonicalisation, and solver traces.

This README and `api.py` define the public boundary. Diagnostic runners and
retained compatibility aliases remain internal/developer surfaces unless they
are explicitly re-exported below.

## Public API

- `AlignConfig`
- `AlignmentSmokeReport`
- `AlternatingAlignmentSolver`
- `AlternatingLevelSummary`
- `AlternatingSmokeConfig`
- `AlternatingSmokeResult`
- `ContinuationLevel`
- `ContinuationSchedule`
- `ContinuationScheduleName`
- `GeometryUpdateVolumeSource`
- `JointSchurDiagnostics`
- `JointSchurLMConfig`
- `JointSchurLMResult`
- `PoseOnlyLMConfig`
- `PoseOnlyLMResult`
- `SetupOnlyLMConfig`
- `SetupOnlyLMResult`
- `adapt_joint_schur_damping`
- `adapt_joint_schur_trust_radius`
- `align`
- `align_multires`
- `joint_schur_normal_eq_summary`
- `reference_continuation_schedule`
- `run_alternating_solver_smoke`
- `run_alignment_smoke`
- `schur_step_from_jacobian`
- `solve_joint_schur_lm`
- `solve_pose_only_lm`
- `solve_setup_only_lm`
- `write_joint_schur_normal_eq_summary`

## Dependencies

Allowed dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.motion`
- `tomojax.nuisance`
- `tomojax.forward`
- `tomojax.recon`
- `tomojax.verify`
- `tomojax.backends`
- `tomojax.io`

Forbidden dependencies:

- private implementation files from other deep modules
- generic utility modules
- new compatibility aliases outside the documented facade

## Invariants

- Default geometry optimisation must be gradient-first LM/GN, not grid search.
- Geometry updates must emit artifact/provenance data.
- Public imports should come through this package facade.
- `run_alignment_smoke` is a tiny v2 wiring check. It is not the final
  optimiser.
- `solve_pose_only_lm` defaults to `phi_residual_rad`, `dx_px`, and `dz_px`;
  `alpha_rad` and `beta_rad` are supported opt-in pose DOFs for focused stages.
- `solve_setup_only_lm` defaults to `theta_offset_rad`, `det_u_px`, active
  `det_v_px`, and detector roll. Axis rotations are supported opt-in setup
  parameters; theta scale is supported only as an explicit opt-in setup
  parameter until identifiable scale policy activates it automatically.
- `solve_joint_schur_lm` is the first reference Schur setup+pose slice for the
  supported setup and pose DOFs. It has accepted/rejected damping adaptation and
  ratio-based trust-radius adaptation, but is not yet the final trust-region
  engine.
- `solve_joint_schur_lm` can opt into per-view gain/offset variable projection
  so affine acquisition drift is modelled as nuisance rather than geometry.
- `write_joint_schur_normal_eq_summary` writes the current Phase 6
  `normal_eq_summary.json` artifact.
- `run_alternating_solver_smoke` is the first Phase 7 vertical slice. It runs a
  deterministic 32^3 stopped-volume alternating smoke path and writes the
  initial/final geometry, pose CSVs, FISTA trace, alignment summary,
  verification report, and artifact index.

## Tests

- Existing alignment tests cover retained staged behavior.
- `tests/test_alternating_solver_smoke.py` covers the Phase 7 deterministic
  artifact smoke run.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
