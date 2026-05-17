# tomojax.align

## Purpose

`tomojax.align` owns alignment orchestration: alternating reconstruction and
geometry updates, continuation policy, observability reporting, gauge
canonicalisation, and solver traces.

This README and `api.py` define the public boundary. The package root keeps the
minimal product surface (`AlignConfig`, `align`, `align_multires`). Broader
typed configuration helpers live in `tomojax.align.api`. Diagnostic-only helpers
may remain directly importable during productionization, but they are not part
of `tomojax.align.api.__all__`; new diagnostic entrypoints belong in
`tomojax.verify`, `tomojax.bench`, or `tomojax dev ...`, not in the product
alignment namespace.

## Public API

- `AlignConfig`
- `AlignmentLossConfig`
- `AlignmentProfile`
- `AlignmentProfileInput`
- `AlignmentProfilePolicy`
- `ContinuationLevel`
- `ContinuationSchedule`
- `ContinuationScheduleName`
- `DofBounds`
- `DofSpec`
- `AlignmentLossSchedule`
- `AlignmentLossSpec`
- `AlignmentState`
- `BaseGeometryArrays`
- `FixedVolumeProjectionObjective`
- `GeometryCalibrationState`
- `L2LossSpec`
- `L2OtsuLossSpec`
- `LossAdapter`
- `LossScheduleEntry`
- `ObjectiveProvenance`
- `ObjectiveResult`
- `PoseState`
- `SetupGeometryState`
- `FallbackPolicy`
- `GaugeFixMode`
- `GaugePolicy`
- `GeometryUpdateVolumeSource`
- `JointSchurLMConfig`
- `JointSchurLMResult`
- `PoseOnlyLMConfig`
- `PoseOnlyLMResult`
- `ResolvedAlignmentSchedule`
- `ResolvedAlignmentStage`
- `AlignmentSchedule`
- `AlignmentStage`
- `SetupOnlyLMConfig`
- `SetupOnlyLMResult`
- `adapt_joint_schur_damping`
- `adapt_joint_schur_trust_radius`
- `align`
- `align_multires`
- `alignment_profile_policy`
- `apply_alignment_state`
- `build_loss_adapter`
- `dof_spec`
- `geometry_with_axis_state`
- `level_detector_grid`
- `loss_spec_name`
- `normalize_alignment_dofs`
- `normalize_alignment_profile`
- `normalize_bounds`
- `normalize_geometry_dofs`
- `parse_loss_schedule`
- `parse_loss_spec`
- `project_and_score_stack`
- `project_stack`
- `profile_policy_from_config`
- `reference_continuation_schedule`
- `resolve_alignment_schedule`
- `resolve_loss_for_level`
- `resolve_profiled_cli_defaults`
- `schedule_preset`
- `schur_step_from_jacobian`
- `se3_from_5d`
- `solve_joint_schur_lm`
- `solve_pose_only_lm`
- `solve_setup_only_lm`
- `summarize_geometry_calibration_stats`
- `validate_loss_schedule_levels`

Diagnostic compatibility imports remain available for existing tests and
tooling, but are deliberately not advertised by `__all__`:

- `JointSchurDiagnostics`
- `joint_schur_normal_eq_summary`
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
- Product code should import alignment helpers from `tomojax.align.api` or the
  package-root product facade, not nested compatibility aliases.
- `tomojax.align.checkpoint`, `tomojax.align.diagnostics`,
  `tomojax.align.motion_models`, `tomojax.align.params_export`, and
  `tomojax.align.losses` are intentionally not registered compatibility
  modules. Use their owning deep-module paths.
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
- `JointSchurDiagnostics` remains a transition-only diagnostic import on this
  facade. `joint_schur_normal_eq_summary` and
  `write_joint_schur_normal_eq_summary` are verify-owned report helpers that
  remain directly importable here only for compatibility.

## Tests

- Existing alignment tests cover retained staged behavior.
- `tests/test_alternating_solver_smoke.py` covers the deterministic diagnostic
  artifact run.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
