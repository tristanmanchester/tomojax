# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8/9 stopped-reconstruction volume-axis semantics
- Goal: fix the core volume-axis/gauge semantics that make stopped
  reconstruction absorb setup geometry before adding any more orchestration
  knobs.

### Scope

- In scope:
  - Add explicit core volume-axis constants and focused projector semantics
    tests.
  - Fix volume support, center penalty, and stopped-volume anchoring to use
    core axes: x/u axis 0, y/beam axis 1, z/v axis 2.
  - Remove periodic `jnp.roll` recentering from stopped geometry update
    volumes.
  - Run the minimal stopped det_u-only setup diagnostic before returning to
    broader setup gates.
  - Add candidate-refresh acceptance now that the corrected stopped det_u-only
    gate still fails while materially improving det_u.
  - Record the next `128^3`/256-view stopped setup scale gate as evidence
    before moving to broader Phase 8/9 functionality.
  - Record the hard stopped-preview x-gauge projection diagnostic and do not
    keep it unless the gate improves.
- Out of scope:
  - Adding more alignment orchestration knobs.
  - Chasing theta endpoint=180 sampling.
  - Reintroducing rotate-and-sum or projector selectors.
  - Full five-case benchmark reruns before the minimal det_u stopped setup
    gate gives evidence.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Add core axis constants and export them from `tomojax.geometry`.
- [x] Add focused projector/axis semantics tests.
- [x] Fix support masks, center penalty, and stopped-volume anchoring axes.
- [x] Run focused CPU tests, ruff, basedpyright, and `just imports`.
- [x] Run the minimal 64^3/64-view stopped det_u-only CUDA diagnostic.
- [x] Add candidate-refresh acceptance for stopped geometry updates.
- [x] Rerun focused CPU tests, ruff, basedpyright, and `just imports`.
- [x] Rerun the minimal 64^3/64-view stopped det_u-only CUDA diagnostic.
- [x] Update `docs/implementation_log.md`, add a concise benchmark note, and
      commit the slice.
- [x] Run 128^3/256-view stopped det_u-only CUDA setup scale gate.
- [x] Update `docs/implementation_log.md`, add a concise benchmark note, and
      commit the scale-gate evidence.
- [x] Test hard stopped-preview det_u-axis gauge projection as a diagnostic.
- [x] Rerun focused CPU tests, ruff, basedpyright, and `just imports`.
- [x] Rerun the minimal 64^3/64-view stopped det_u-only CUDA diagnostic.
- [x] Revert the no-benefit hard-gauge code path, update logs, and commit the
      diagnostic evidence.

### Validation

Current slice:

- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_forward_reference.py::test_core_volume_axis_constants_match_projector_convention
  tests/test_forward_reference.py::test_core_volume_axis_translations_match_detector_axes
  tests/test_reference_fista.py::test_reference_fista_center_l2_penalty_enters_regulariser
  tests/test_reference_fista.py::test_reference_fista_center_l2_uses_core_x_y_axes
  tests/test_reference_fista.py::test_centered_volume_support_generates_cylinder_and_sphere
  tests/test_alternating_geometry_update_policy.py::test_coarse_setup_global_anchoring_recenters_stopped_volume
  tests/test_alternating_geometry_update_policy.py::test_anchoring_releases_outside_coarse_setup_global`
  passed: 7 tests in 6.24 seconds.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu uv run ruff check
  src/tomojax/geometry src/tomojax/recon/_fista_reference.py
  src/tomojax/recon/_support.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_forward_reference.py tests/test_reference_fista.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu
  PYTHONPATH=.venv/lib/python3.12/site-packages uv run basedpyright
  src/tomojax/recon/_fista_reference.py src/tomojax/recon/_support.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_forward_reference.py tests/test_reference_fista.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `env UV_CACHE_DIR=.uv-cache JAX_PLATFORM_NAME=cpu just imports` passed.
- CUDA minimal stopped det_u-only gate completed on `cuda:0`. Artifact:
  `.artifacts/phase8_axis_gauge/runs/64_stopped_detu_only_axis_fix_cuda/`.
  Status failed, but det_u improved from 7.25 px to 2.87216 px and Schur
  accepted the update.

Historical validation below remains as prior execution log context.

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_setup_only_geometry_update_solver_recovers_setup_without_pose
  tests/test_alternating_geometry_update_policy.py::test_setup_only_geometry_update_solver_requires_frozen_pose
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 3 tests in 38.72 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py src/tomojax/align/api.py
  src/tomojax/cli/align_auto.py tests/test_alternating_geometry_update_policy.py
  tests/test_align_auto_cli.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the setup-only solver option.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_stopped_preview_policy_constrains_first_preview_only
  tests/test_alternating_geometry_update_policy.py::test_stopped_preview_no_fista_policy_skips_first_preview_reconstruction_only
  tests/test_alternating_geometry_update_policy.py::test_stopped_preview_policy_reuses_first_preview_for_later_geometry_updates
  -q` passed: 3 tests in 0.72 seconds.
- `uv run ruff check src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the no-FISTA first-preview policy.
- CUDA 128^3/256-view supported-only stopped-reconstruction gate completed on
  `cuda:0` in 178.47 seconds with the no-FISTA first-preview policy. Artifact:
  `.artifacts/phase8_no_fista_first_preview/runs/128_supported_only_256views_no_fista_first_gpu/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_can_exclude_heldout_view
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_defaults_to_all_views
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 3 tests in 34.11 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_can_exclude_heldout_view
  tests/test_alternating_geometry_update_policy.py::test_preview_reconstruction_mask_source_defaults_to_all_views
  tests/test_alternating_geometry_update_policy.py::test_train_view_reconstruction_disables_coarse_early_exit
  -q` passed: 3 tests in 0.88 seconds.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py
  src/tomojax/align/_alternating_verification.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py
  src/tomojax/align/api.py src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py
  src/tomojax/align/_alternating_verification.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- CUDA 128^3/256-view supported-only train-view reconstruction gate completed
  on `cuda:0` in 218.47 seconds after disabling coarse early exit. Artifact:
  `.artifacts/phase8_train_view_reconstruction/runs/128_supported_only_256views_train_views_no_skip_gpu/`.
- `just imports` passed after the train-view reconstruction policy.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_fixed_truth_geometry_updates_use_level_residual_sigma
  tests/test_alternating_geometry_update_policy.py::test_stopped_geometry_updates_keep_estimated_residual_sigma_floor
  -q` passed: 2 tests in 0.69 seconds.
- `uv run ruff check src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the fixed-truth sigma policy.
- Fixed-truth `synth128_pose_random_extreme` CUDA oracle completed on `cuda:0`
  in 217.02 seconds. Artifact:
  `.artifacts/phase8_fixed_truth_sigma/runs/synth128_pose_random_extreme_fixed_truth_no_nuisance_fit_cuda/`.
- Direct true-volume all-5 pose trust probe completed on `cuda:0`. Disabling
  pose trust improved dx/dz but worsened alpha/beta and phi.
- Direct true-volume phi/dx/dz-only no-trust iteration probe completed on
  `cuda:0`. At 12 iterations it recovered dx/dz to sub-pixel but left phi near
  `0.105` rad and alpha/beta at the initial zero-pose error.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_pose_trust_radius_uses_level_default_when_unset
  tests/test_alternating_geometry_update_policy.py::test_pose_trust_radius_negative_sentinel_disables_clipping
  tests/test_alternating_geometry_update_policy.py::test_pose_trust_radius_can_override_level_radius
  tests/test_align_auto_cli.py::test_align_auto_accepts_geometry_update_volume_source
  -q` passed: 4 tests in 38.80 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the pose trust-radius option.
- Fixed-truth `synth128_pose_random_extreme` phi/dx/dz no-trust CUDA
  diagnostic completed on `cuda:0` in 213.10 seconds. Artifact:
  `.artifacts/phase8_pose_trust_option/runs/pose_random_fixed_truth_phi_dxdz_no_trust_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_alpha_beta_activation_policy_freezes_angular_pose_until_configured_level
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 2 tests in 34.18 seconds.
- `uv run ruff check src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/align/_alternating_types.py
  src/tomojax/cli/align_auto.py tests/test_alternating_geometry_update_policy.py
  tests/test_align_auto_cli.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/align/_alternating_types.py
  src/tomojax/cli/align_auto.py tests/test_alternating_geometry_update_policy.py
  tests/test_align_auto_cli.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after alpha/beta activation policy.
- Fixed-truth `synth128_pose_random_extreme` alpha/beta-final no-trust CUDA
  diagnostic completed on `cuda:0` in 215.97 seconds. Artifact:
  `.artifacts/phase8_alpha_beta_staging/runs/pose_random_fixed_truth_alpha_beta_final_no_trust_cuda/`.
- Direct phi-only polish from the staged result completed on `cuda:0`; 16
  iterations reduced phi RMSE from about `0.1258` to `0.0547` rad while
  preserving dx/dz recovery.
- Direct alpha/beta/phi polish reduced phi similarly but worsened alpha/beta,
  so a dedicated phi-only polish is the better next implementation target.
- `just imports` passed after the diagnostic log update.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_phi_polish_runs_phi_only_geometry_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_accepts_geometry_update_volume_source
  -q` passed: 3 tests in 44.80 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the phi-only polish stage.
- CUDA `synth128_pose_random_extreme` fixed-truth phi-polish gate completed on
  `cuda:0` in 327.57 seconds from the artifact. Artifact:
  `.artifacts/phase8_phi_polish_stage/runs/pose_random_fixed_truth_phi_polish16_cuda/`.
  The stage reduced theta-realized RMSE to `0.045132` rad and accepted the
  final Schur update, but the benchmark still fails detector-shift and
  alpha/beta tolerances.
- Direct true-volume final pose polish probes completed on `cuda:0`. Opening
  `det_u_px` plus all five pose DOFs removed the global det_u gauge floor in
  isolated solves, and fresh restarted solves repaired the single endpoint
  outlier from a written full-gate artifact.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_final_pose_polish_can_open_det_u_with_all_pose_dofs
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  -q` passed: 2 tests in 8.31 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  src/tomojax/align/_alternating_orchestration.py
  src/tomojax/align/_alternating_types.py src/tomojax/cli/align_auto.py
  tests/test_alternating_geometry_update_policy.py tests/test_align_auto_cli.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the final pose polish stage.
- CUDA `synth128_pose_random_extreme` fixed-truth phi+final-pose-polish gate
  completed on `cuda:0` in 764.26 seconds from the artifact. Artifact:
  `.artifacts/phase8_final_pose_polish/runs/pose_random_fixed_truth_phi16_final_pose48_restart_cuda/`.
  Alpha/beta and theta passed, Schur train loss fell to `0.001048`, but
  detector-shift RMSE still failed due a flagged endpoint outlier at view 255.
- CUDA `synth128_pose_random_extreme` fixed-truth 64-update final-pose gate
  without bad-view-aware verification completed on `cuda:0` in 909.84 seconds
  from the artifact and still failed strict full-view detector-shift recovery.
  Artifact:
  `.artifacts/phase8_final_pose_polish/runs/pose_random_fixed_truth_phi16_final_pose64_restart_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py::test_geometry_recovery_can_exclude_flagged_bad_view
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 2 tests in 50.18 seconds.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the weak-view verification change.
- CUDA `synth128_pose_random_extreme` fixed-truth weak-view recovery gate passed
  on `cuda:0` in 910.44 seconds from the artifact. Artifact:
  `.artifacts/phase8_weak_view_recovery/runs/pose_random_fixed_truth_phi16_final_pose64_bad_view_exclusion_cuda/`.
  View 255 was excluded by robust residual outlier detection; effective
  `det_u=0.000279 px`, `det_v=0.062866 px`, `theta=0.000909 rad`, and
  `alpha_beta=0.001509 rad` all passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py -q` passed: 8 tests in
  0.89 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 49.86 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_orchestration.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the early anchoring change.
- CUDA setup-global stopped-reconstruction rerun completed on the existing
  128^3/256-view sidecar in 128.65 seconds. Artifact:
  `.artifacts/phase8_early_anchor/128_setup_global_stopped_cuda/`.
- Detector-u-first staging probe completed on the same sidecar in 267.28
  seconds but was not kept as source code because it worsened theta and axis
  recovery. Artifact:
  `.artifacts/phase8_staged_setup/128_setup_global_stopped_cuda/`.
- Constrained-preview diagnostics completed on the same sidecar with
  cylindrical support (`preview_tv_scale=1` and `10`) and spherical support
  (`preview_tv_scale=1`). Artifacts under
  `.artifacts/phase8_constrained_preview/`.
- True-geometry reconstruction oracle diagnostic completed on `cuda:0` in
  159.64 seconds. Artifact:
  `.artifacts/phase8_true_geometry_recon_oracle/128_setup_global_true_recon_schur_cuda/`.
- True-geometry 32-iteration oracle passed all setup-global criteria in 277.27
  seconds. Artifact:
  `.artifacts/phase8_true_geometry_recon_oracle/128_setup_global_true_recon32_schur_cuda/`.
- Production-like stopped 8/32/32 continuation improved volume/residual but not
  geometry. Artifact:
  `.artifacts/phase8_more_iterations_after_anchor/128_setup_global_stopped_8_32_32_cuda/`.
- Constrained theta/det_u coarse-stage policy probe completed but worsened
  theta and was not promoted. Artifact:
  `.artifacts/phase8_staged_constrained_policy_probe/128_setup_global_theta_detu_then_full_cuda/`.
- Projection-centroid volume-gauge transfer probe completed but still failed
  setup-global recovery. Artifact:
  `.artifacts/phase8_volume_gauge_transfer_probe/128_setup_global_projection_com_transfer_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 20 tests in 268.87 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_reference_fista.py
  tests/test_vertical_smoke.py
  tests/test_joint_schur_lm.py::test_schur_step_matches_dense_normal_solve
  tests/test_joint_schur_lm.py::test_joint_schur_streamed_normals_put_pure_setup_error_in_setup_gauge
  -q` passed: 10 tests in 84.38 seconds.
- `uv run ruff check src/tomojax/recon/_backprojection_accumulation.py
  src/tomojax/recon/_reference.py src/tomojax/recon/_fista_reference.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/forward/_projector.py
  tests/test_joint_schur_lm.py` passed.
- `uv run basedpyright src/tomojax/recon/_backprojection_accumulation.py
  src/tomojax/recon/_reference.py src/tomojax/recon/_fista_reference.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/forward/_projector.py
  tests/test_joint_schur_lm.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the streaming changes.
- GPU diagnostic artifacts:
  `.artifacts/phase8_streamed_schur_probe/logs/128_setup_global_full5_fixed_truth_cuda_rerun2/`
  and
  `.artifacts/phase8_streamed_schur_probe/logs/64_setup_global_full5_fixed_truth_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed after the batched Schur scan: 20 tests in 100.05 seconds.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py
  src/tomojax/forward/_filters.py tests/test_joint_schur_lm.py` passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py
  src/tomojax/forward/_filters.py tests/test_joint_schur_lm.py` passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the batched Schur update.
- 64^3/64-view full-5DOF fixed-truth CUDA probe completed in about 79 seconds
  with peak sampled GPU memory 735 MiB.
- 128^3/256-view full-5DOF fixed-truth CUDA probe completed in about 253
  seconds with peak sampled GPU memory 1265 MiB.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_theta_scale_setup_update
  -q` passed: 5 tests in 9.67 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 49.91 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the Schur staging policy update.
- CUDA setup-global fixed-truth rerun with `theta_scale` and all five pose DOFs
  requested resolved to active setup-only Schur updates, passed all 4/4 manifest
  geometry criteria, and peaked at 1259 MiB sampled GPU memory. Artifact:
  `.artifacts/phase8_setup_staging_policy/128_setup_global_policy_filtered_setup_only_fixed_truth_cuda/`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Existing weak-DOF policy evidence should make policy criteria real when it is
  sufficient, without adding duplicate benchmark fields.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.
- [x] Parallel laminography acquisition metadata committed: `7aa086c`.
- [x] det_v observability gating evidence committed: `7c1e0fe`.
- [x] Synthetic unsupported-term classification committed: `28e336f`.
- [x] Benchmark criterion aliases committed: `fe83427`.
- [x] Laminography solver residuals committed: `7002d42`.
- [x] Recovered det_v policy criterion committed: `f6fe3c4`.
- [x] Backend policy criterion evaluation committed: `b040829`.
- [x] Calibrated-grid backend provenance committed: `a0b69db`.
- [x] Missing-policy criterion reasons committed: `9034b91`.
- [x] 128^3 supported-only GPU scale gate committed: `d2fbd5a`.
- [x] Active Schur DOFs in observability committed: `7ab5013`.
- [x] Smoke expectation cleanup committed: `44dda7e`.
- [x] Nuisance-corrected failure gate committed: `f374d58`.

### Risks

- Risk: full active setup/pose updates at `128^3` may expose memory or runtime
  regressions before producing numerical recovery evidence.
- Mitigation: record command, device, peak GPU memory, runtime, and failure
  artifact path per case rather than shrinking the benchmark.
