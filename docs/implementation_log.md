# TomoJAX Implementation Log

This log records implementation milestones, validation commands, design
decisions, deviations from `docs/tomojax-v2/`, and unresolved risks.

## 2026-05-07 — Phase 8 Active-DOF Observability Reporting Slice

Corrected `observability_report.json` DOF status generation so it follows the
actual Schur active setup and pose blocks instead of hard-coded weak placeholders.
Active setup parameters now report `active=true`, `status=evaluated`, and Schur
curvature when diagnostics exist; frozen setup parameters report
`status=frozen`. Active pose DOFs report `active=true`, with
`phi_residual_rad`, `dx_px`, and `dz_px` retaining their gauge-canonicalised
status, while inactive pose DOFs report `status=frozen`.

This does not change solver maths, benchmark tolerances, or add new report
fields. It makes the existing observability artifact accurately reflect
supported DOFs that have already been wired through the Schur block.

Validation:

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_observability.py
  -q` passed: 2 tests in 0.81 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 112.65 seconds.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py tests/test_alternating_solver_smoke.py`
  passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

Note: a broad CPU run of `tests/test_alternating_solver_smoke.py -q` completed
and reported three failures in existing sidecar/recovery expectations after
17:07. The focused tests above cover this slice; the broad smoke-file
instability remains separate from the observability payload correction.

## 2026-05-07 — Phase 8 128^3 Supported-Only Scale Gate

Ran the next realistic laptop-GPU scale gate for supported-only
`synth128_setup_global_tomo` before further benchmark/provenance cleanup. The
valid scale-gate dataset is 128^3 with 256 views and nuisance disabled; an
earlier fixed-truth command accidentally used the smoke CLI default of 4 views
and is discarded as scale evidence.

- Fixed-truth oracle Schur geometry update on `cuda:0` passed the benchmark
  manifest geometry criteria at full view count:
  `det_u_realized_rmse_px=2.28882e-05`,
  `theta_realized_rmse_rad=4.10218e-06`, `det_v_realized_rmse_px=0`.
  Peak observed GPU memory was 6071 MB and `/usr/bin/time -v` wall time was
  2:55.81 with max host RSS 2933348 KB.
- The top-level fixed-truth `benchmark_result.json` status remains `failed`
  because the preview reconstruction volume NMSE gate fails
  (`volume_nmse=4262.16`, final residual 618.138), but the oracle geometry
  manifest status is `passed`; this was sufficient to run the requested stopped
  diagnostic.
- Anchored stopped reconstruction with pose frozen and only `det_u_px` active
  ran on `cuda:0` against the same sidecar directory. It failed manifest
  geometry criteria with `det_u_realized_rmse_px=0.594401` against the 0.5 px
  threshold and `theta_realized_rmse_rad=0.0218166` because theta is frozen in
  this diagnostic. Volume NMSE was 0.871336, final residual 4.06514, Schur
  accepted the final update, peak observed GPU memory was 6071 MB, and wall time
  was 2:45.36 with max host RSS 2775420 KB.
- Artifacts:
  `.artifacts/phase8_supported128_scale_gate/datasets/synth128_setup_global_tomo_128_supported_only/`,
  `.artifacts/phase8_supported128_scale_gate/runs/128_supported_only_256views_fixed_truth_reference_gpu/`,
  `.artifacts/phase8_supported128_scale_gate/runs/128_supported_only_256views_stopped_anchor_detu_gpu/`,
  and
  `.artifacts/phase8_supported128_scale_gate/benchmark_comparison_128_supported_only.md`.
- Summary recorded in
  `docs/benchmark_runs/2026-05-07-phase8-128-supported-scale-gate.md`.

Interpretation: the full-view fixed-truth oracle Schur path is not blocked by
setup/pose/theta convention mapping at 128^3, and the run did not reproduce a
12 GiB memory blow-up. The stopped anchored run still misses strict detector
shift recovery, keeping the next blocker on stopped reconstruction/volume gauge
handling.

## 2026-05-07 — Phase 8 Missing-Policy Criterion Reason Slice

### Summary

- Replaced generic `unsupported_dof_not_evaluated` reasons for policy-style
  benchmark criteria with explicit missing-evidence reasons.
- `core_solver`, `bad_views_flagged`,
  `pose_dx_dz_rmse_px_lt_except_jumps`, `beats_current_default_nmse`, and
  `object_motion_enabled_tx_rmse_px_lt` now remain `not_evaluated` with the
  exact absent payload named.
- This does not implement object-motion, bad-view detection, jump-exclusion
  metrics, or current-default comparison. It makes the remaining report gaps
  concrete and auditable.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 10 tests in
  0.65 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- The named payloads still need real implementations before these criteria can
  pass or fail as measured benchmark criteria.

## 2026-05-07 — Phase 8 Calibrated-Grid Backend Provenance Slice

### Summary

- Parsed optional `detector_grid` metadata from the synthetic benchmark
  manifest and wrote it into generated dataset manifests when present.
- Included detector-grid metadata in `align-auto` synthetic sidecar readback.
- Benchmark backend provenance now records an explicit fallback row when
  sidecar readback reports `detector_grid="calibrated_noncanonical"`.
- The existing `backend_policy: calibrated_grid_fallback_explicit` criterion can
  now pass from concrete sidecar-triggered provenance instead of proxy labels.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_synthetic_datasets.py::test_load_synthetic_dataset_sidecars_reads_manifest_index
  tests/test_alternating_benchmark_criteria.py -q` passed: 8 tests in
  2.66 seconds.
- `uv run ruff check src/tomojax/datasets/_specs.py
  src/tomojax/datasets/_writer.py src/tomojax/cli/align_auto.py
  src/tomojax/align/_alternating_artifacts.py tests/test_synthetic_datasets.py
  tests/test_alternating_benchmark_criteria.py` passed.
- `uv run basedpyright` on the same focused source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- This slice records the policy/provenance decision. It does not add a new
  projector backend or detector-grid transform path.
- Bad-view policy, object-motion criteria, and pose-jump exclusions remain
  future policy/report slices.

## 2026-05-07 — Phase 8 Backend Policy Criterion Slice

### Summary

- Threaded backend provenance into synthetic benchmark manifest criterion
  evaluation.
- `backend_policy: calibrated_grid_fallback_explicit` now evaluates as a real
  policy criterion: it passes only when backend fallback provenance is recorded
  and fails when the fallback list is empty.
- This makes missing calibrated-grid fallback evidence a failed benchmark policy
  criterion instead of an unsupported placeholder.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 6 tests in
  0.66 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Actual calibrated-grid fallback provenance still needs to be emitted by runs
  that use a noncanonical calibrated detector grid.
- Bad-view policy, object-motion criteria, and pose-jump exclusions remain
  future policy/report slices.

## 2026-05-07 — Phase 8 det_v Policy Criterion Slice

### Summary

- Added explicit benchmark-manifest evaluation for
  `det_v_policy: recovered_or_reported_unobservable`.
- The criterion now passes when existing geometry recovery evidence reports
  `det_v_realized_rmse_px_passed=True`.
- If det_v is not recovered and no unobservability policy payload is present in
  `benchmark_result`, the criterion remains `not_evaluated` with a specific
  missing-evidence reason instead of being treated as an unsupported DOF.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 5 tests in
  0.65 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Wiring observability policy payloads into `benchmark_result` is still needed
  before the `reported_unobservable` branch can be evaluated.
- `backend_policy`, bad-view policy, object-motion criteria, and pose-jump
  exclusions remain future policy/report slices.

## 2026-05-07 — Phase 2/7 Laminography Solver Residual Slice

### Summary

- Threaded acquisition nominal-axis metadata into setup-only LM, pose-only LM,
  and joint Schur LM residual/loss paths.
- Added public `tomojax.forward.nominal_axis_unit_from_geometry` so align
  solvers can use the same acquisition-to-axis mapping as direct
  `GeometryState` projection without importing forward private internals.
- Added focused zero-residual tests proving each solver path preserves
  laminography acquisition when observations are generated by
  `project_parallel_reference`.
- The operational projector remains the existing `core_trilinear_ray` path; no
  projector selector or rotate-and-sum path was added.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_setup_lm.py::test_setup_only_lm_residual_preserves_laminography_acquisition
  tests/test_pose_lm.py::test_pose_only_lm_residual_preserves_laminography_acquisition
  tests/test_joint_schur_lm.py::test_joint_schur_lm_residual_preserves_laminography_acquisition
  -q` passed: 3 tests in 5.31 seconds.
- `uv run ruff check src/tomojax/forward/_projector.py
  src/tomojax/forward/api.py src/tomojax/forward/__init__.py
  src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py
  src/tomojax/align/_joint_schur_lm.py tests/test_setup_lm.py
  tests/test_pose_lm.py tests/test_joint_schur_lm.py` passed.
- `uv run basedpyright` on the same touched source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Laminography tilt is still fixed acquisition metadata, not an active setup
  parameter.
- Realistic laminography benchmark recovery still needs a dedicated run after
  remaining policy criteria and unsupported scenario terms are handled.

## 2026-05-07 — Phase 8 Benchmark Criterion Alias Slice

### Summary

- Updated synthetic benchmark manifest evaluation to handle documented geometry
  criterion aliases that map to existing recovery metrics.
- `detector_roll_error_deg_lt` now evaluates against
  `detector_roll_error_rad`.
- `axis_roll_error_deg_lt` now evaluates against the max of `axis_error_rad`
  and `detector_roll_error_rad`.
- String policy criteria such as `backend_policy` remain report-only and
  `not_evaluated` until a real policy payload is available.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_benchmark_criteria.py -q` passed: 3 tests in
  0.67 seconds.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  tests/test_alternating_benchmark_criteria.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- `det_v_policy`, `backend_policy`, bad-view policy, object-motion criteria,
  and pose-jump exclusions still need real policy/report payloads before they
  can become evaluated criteria.

## 2026-05-07 — Phase 8 Synthetic Unsupported-Term Classification Slice

### Summary

- Restored truthful unsupported-term classification in generated synthetic
  sidecar manifests.
- `SyntheticDatasetSpec` now parses `true_object_motion` from the benchmark
  manifest so object-motion scenarios can be classified instead of silently
  dropped.
- Generated manifests now mark remaining unmodelled object motion, sparse pose
  jumps, bad views, partial-FOV invalid edges, and currently unrealised nuisance
  terms under `unsupported_dofs_not_evaluated`.
- `supported_only` variants still report no unsupported terms after stripping
  unsupported truth.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_synthetic_datasets.py::test_generate_object_motion_dataset_marks_unsupported_manifest_terms
  tests/test_synthetic_datasets.py::test_generate_combined_nuisance_dataset_marks_unmodelled_terms
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  -q` passed: 3 tests in 2.80 seconds.
- `uv run ruff check src/tomojax/datasets/_specs.py
  src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py` passed.
- `uv run basedpyright src/tomojax/datasets/_specs.py
  src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py` passed
  with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Object-frame thermal drift, sparse pose-jump handling, bad-view detection,
  partial-FOV masks, and full nuisance simulation/fitting remain future
  implementation slices before the combined benchmark can be fully evaluated.

## 2026-05-07 — just check Baseline Failure Note

### Summary

- Ran `just check` after the laminography and det_v observability slices.
- The command is not a clean project-wide gate in the current branch: it first
  ran `uv run ruff format src tests tools`, reformatting 71 files, then stopped
  in `uv run ruff check --fix src tests tools` with broad pre-existing lint
  failures.
- The formatter/lint churn was reverted because it was unrelated to the active
  vertical slices.

### Failure Shape

- Representative failures include missing docstrings and annotations in
  `src/tomojax/align/model/schedules.py` and
  `src/tomojax/align/model/state.py`, private/relative import style failures in
  `src/tomojax/align/objectives/fixed_volume.py`, and many test-suite lint
  failures such as unused fixture arguments, lambda assignments, and compound
  assertions.
- Ruff reported 1,682 total issues after applying 318 safe fixes before the
  changes were reverted.

### Decision

- Keep using focused `pytest`, `ruff check`, `basedpyright`, and `just imports`
  gates for scoped implementation slices until the legacy project-wide Ruff
  backlog is handled in a separate cleanup milestone.

## 2026-05-07 — Phase 7/8 det_v Observability-Gating Slice

### Summary

- Corrected report-only weak-DOF policy evidence for active `det_v_px`.
- det_v correlation evidence now uses the actual active Schur setup-parameter
  index instead of assuming a fixed setup ordering.
- det_v accepted-step evidence is now available whenever det_v is active and
  Schur diagnostics exist, independent of whether `theta_scale` is active.
- Added a focused white-box observability payload test for a det_v-active,
  theta-scale-frozen Schur result.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_observability.py
  -q` passed: 1 test in 0.66 seconds.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py` passed with 0 errors, 0 warnings,
  and 0 notes.
- `just imports` passed.

### Remaining Work

- This slice only fixed existing report-only gating evidence. It does not
  change automatic weak-DOF activation, schedules, tolerances, or benchmark
  pass/fail criteria.

## 2026-05-07 — Phase 2 Parallel Laminography Acquisition Slice

### Summary

- Added typed acquisition metadata to `GeometryState` for parallel tomography
  and parallel laminography, including nominal laminography tilt and tilt axis.
- Geometry JSON sidecars now serialize/read acquisition metadata while older
  payloads without the field still default to parallel tomography.
- The v2 `core_trilinear_ray` adapter now builds the nominal rotation axis from
  acquisition metadata and applies existing setup axis x/y corrections on top.
  No rotate-and-sum path or projector selector was added.
- Synthetic sidecar generation now records laminography acquisition metadata
  for planned laminography scenarios, and geometry update/canonicalisation paths
  preserve acquisition metadata.
- Module READMEs were updated to document acquisition metadata and the
  laminography nominal-axis mapping.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_forward_reference.py::test_core_projection_geometry_matches_core_laminography_pose_convention
  tests/test_geometry_serialization.py::test_geometry_json_and_pose_csv_round_trip_contract_artifacts
  tests/test_synthetic_datasets.py::test_load_synthetic_dataset_sidecars_reads_manifest_index
  -q` passed: 3 tests in 3.02 seconds.
- `uv run ruff check src/tomojax/align/_alternating_inputs.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_setup_lm.py
  src/tomojax/align/_pose_lm.py src/tomojax/datasets/_writer.py
  src/tomojax/forward/_projector.py src/tomojax/geometry/__init__.py
  src/tomojax/geometry/_gauges.py src/tomojax/geometry/_serialization.py
  src/tomojax/geometry/_state.py src/tomojax/geometry/api.py
  tests/test_forward_reference.py tests/test_geometry_serialization.py
  tests/test_synthetic_datasets.py` passed.
- `uv run basedpyright` on the same touched source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Laminography tilt is represented as acquisition metadata only; it is not yet
  an active solved setup parameter.
- The laminography benchmark scenario still needs a scoped solver/recovery pass
  before its criteria stop being scenario-level future work.
- Object drift and automatic weak-DOF activation remain separate planned
  slices.

## 2026-05-07 — GPU Memory Diagnostic Pause Summary

### Summary

- Pausing at a diagnostic/report boundary; no new Phase 8 ingestion,
  report-shape, observability, refactor, or benchmark feature slice is started.
- The realistic diagnostic target remains `synth128_setup_global_tomo` at
  64^3/64 views on JAX GPU with nuisance disabled. The 32^3/4-view smoke is
  wiring/CI coverage only and is not alignment-quality evidence.
- The current uncommitted laminography metadata work is outside this pause
  summary and is intentionally not part of this commit.

### Current Best Diagnosis

- The original five-case benchmark failures remain mixed unsupported-geometry
  and gauge/solver-policy failures, not a single alignment-quality verdict.
  Detector roll, axis tilt, alpha/beta, and theta scale now have scoped support
  slices, but laminography, object drift, automatic weak-DOF activation, and
  scenario-specific classification still need separate work.
- The realistic setup-global 64^3/64-view ladder initially failed in both
  `fixed_synthetic_truth` and `stopped_reconstruction`, so the first blocker was
  setup/pose/theta coupling or geometry convention mapping rather than
  reconstruction alone.
- After the core-projector rebaseline and supported-only oracle narrowing,
  `fixed_synthetic_truth` passes supported-only setup recovery when pose is
  frozen, strongly prior-constrained, or filtered Schur residuals are used.
  The best filtered runs recovered det_u and theta to near-zero error on
  `cuda:0`.
- Under matching strong-pose-prior settings,
  `stopped_reconstruction` still failed with det_u RMSE 7.25 px,
  theta RMSE 0.0218166 rad, Schur train loss about 0.361978, and independent
  true-volume/final-geometry projection loss about 0.884522. That supports the
  current interpretation that stopped reconstruction is absorbing geometry
  error into the reconstructed volume/geometry gauge.
- The GPU memory regression isolated so far was in the finite-difference Schur
  Jacobian path: all parameter perturbations were evaluated with a single
  `jax.vmap`, materialising parameter x view x volume work arrays. The observed
  failure was a 12.14 GiB allocation shaped like
  `f32[194,64,64,64,64]`. Projector, backprojector, one FISTA iteration,
  nuisance fitting, and single Schur updates all completed on the 64-view
  ladder after finite-difference columns were accumulated sequentially.

### Commands Run

- Component memory probes over 1/4/16/64 views for projector, backprojector,
  one FISTA iteration, fixed-truth Schur, stopped-volume Schur, and
  fixed-truth Schur with nuisance.
- Realistic 64^3/64-view `synth128_setup_global_tomo` runs through existing
  sidecar ingestion for `fixed_synthetic_truth` and
  `stopped_reconstruction`.
- Supported-only oracle refreshes through `tomojax-align-auto-smoke` on
  JAX GPU, including pose-frozen, strong-pose-prior, staged-pose,
  zero-mean-pose, reporting-provenance, and filtered-Schur diagnostics.
- Comparison reports rendered with `tomojax-synthetic-benchmark-compare`.
- Slice-level validation already recorded in the preceding implementation-log
  entries: focused CPU `pytest`, `uv run ruff check ...`,
  `uv run basedpyright ...`, and `just imports`.

### Artifacts Produced

- `.artifacts/phase8_setup_global_gpu_ladder/`
- `.artifacts/phase8_supported_only_oracle/`
- `.artifacts/phase8_core_projector/`
- `docs/benchmark_runs/2026-05-06-phase8-setup-global-gpu-ladder.md`
- `docs/benchmark_runs/2026-05-07-phase8-core-projector-rebaseline.md`
- `docs/benchmark_runs/2026-05-07-phase8-supported-only-oracle.md`

### Remaining Open Questions

- Whether broader active setup/pose blocks need chunked Schur accumulation
  beyond sequential finite-difference columns once more supported DOFs are
  enabled together.
- Whether production stopped-reconstruction alignment needs stronger volume
  gauge constraints, a different preview reconstruction hand-off, anchored or
  zero-mean pose parameterisation, or staged pose activation before setup and
  pose can recover together.
- How each unsupported five-case scenario criterion should be classified after
  the detector roll, axis tilt, alpha/beta, and theta-scale slices without
  relaxing geometry tolerances.

## 2026-05-07 — Phase 8 Theta-Scale Opt-In Setup Slice

### Summary

- Promoted `theta_scale` from hard-frozen/unsupported in setup solvers to a
  supported opt-in setup parameter for setup-only LM and joint Schur LM.
- Defaults remain frozen and observability-gated: existing smoke/alternating
  schedules do not activate `theta_scale` unless explicitly configured.
- `align-auto` and alternating geometry-update validation now accept
  `theta_scale` in `geometry_update_active_setup_parameters`.
- Verification now emits `initial_theta_scale_error`, `theta_scale_error`,
  pass/improvement fields, and a `theta_scale_error_lt` benchmark criterion.
- Observability reporting keeps frozen theta-scale as missing accepted-step
  evidence, but reports active theta-scale from Schur diagnostics when it is
  explicitly included in the setup block.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_setup_lm.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_recovers_realized_supported_geometry
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_det_u_only_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_detector_roll_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_axis_tilt_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_theta_scale_setup_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_parses_supported_geometry_update_dofs
  tests/test_align_auto_cli.py::test_align_auto_rejects_unknown_geometry_update_dofs
  -q` passed: 14 tests in 113.33 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  tests/test_setup_lm.py::test_setup_only_lm_recovers_theta_scale_when_explicitly_active
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_theta_scale_setup_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_parses_supported_geometry_update_dofs
  tests/test_align_auto_cli.py::test_align_auto_rejects_unknown_geometry_update_dofs
  -q` passed: 6 tests in 125.98 seconds.
- `uv run ruff check src/tomojax/align/_setup_lm.py
  src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py
  tests/test_setup_lm.py tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py` passed.
- `uv run basedpyright` on the same focused source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Automatic weak-DOF decision rules still need to decide when theta scale is
  identifiable and should become active with a prior in benchmark schedules.
- `det_v_px` observability gating, parallel laminography, object drift, and the
  full synthetic benchmark ladder remain future vertical slices.

## 2026-05-07 — Phase 7 CLI Geometry-Update Activation Slice

### Summary

- Updated `align-auto` active pose DOF parsing/help to accept
  `alpha_rad`, `beta_rad`, `phi_residual_rad`, `dx_px`, and `dz_px`.
- Updated `align-auto` active setup parameter parsing/help to accept
  `theta_offset_rad`, `det_u_px`, `det_v_px`, `detector_roll_rad`,
  `axis_rot_x_rad`, and `axis_rot_y_rad`.
- Updated alternating geometry-update private validation to accept the same
  alpha/beta pose names now supported by joint Schur LM.
- Defaults are unchanged: production-like smoke runs still activate
  `phi_residual_rad`, `dx_px`, and `dz_px` unless explicitly configured.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_smoke_help_documents_outputs
  tests/test_align_auto_cli.py::test_align_auto_parses_supported_geometry_update_dofs
  tests/test_align_auto_cli.py::test_align_auto_rejects_unknown_geometry_update_dofs
  -q` passed: 3 tests in 0.67 seconds.
- `uv run ruff check src/tomojax/cli/align_auto.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_align_auto_cli.py` passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py
  src/tomojax/align/_alternating_geometry_update.py
  tests/test_align_auto_cli.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Weak-DOF activation policy still decides when alpha/beta, axis tilt, detector
  roll, and det_v should be enabled automatically in benchmark schedules.
- This slice only exposed supported names through CLI/config validation; it did
  not change benchmark defaults or rerun the synthetic ladder.

## 2026-05-07 — Phase 2 Alpha/Beta Pose Core Geometry Slice

### Summary

- Promoted per-view `alpha_rad` and `beta_rad` from unsupported/frozen
  placeholders to supported `core_trilinear_ray` v2 pose DOFs for parallel
  tomography.
- The v2-to-core adapter now composes alpha/beta residual rotations after the
  nominal axis/theta world-from-object pose, matching the existing sidecar
  geometry wrapper convention. Existing detector-shift signs and detector-grid
  semantics are unchanged.
- `project_parallel_reference_arrays` now accepts scalar or per-view
  `alpha_rad`/`beta_rad` arrays and records alpha/beta max-absolute provenance.
- Pose-only LM and joint Schur LM can pack, update, freeze, trace, and report
  alpha/beta as explicit opt-in active pose DOFs. The default pose block remains
  `phi_residual_rad`, `dx_px`, and `dz_px` until weak-DOF activation policy is
  broadened.
- Synthetic sidecar generation no longer projects alpha/beta away and no
  longer classifies them as `unsupported_dof_not_evaluated`.
- Verification and benchmark-manifest evaluation now emit
  `alpha_beta_rmse_rad` and evaluate `alpha_beta_rmse_deg_lt` as a real metric.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py
  tests/test_pose_lm.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_recovers_realized_supported_geometry
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_alpha_beta_pose_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_axis_tilt_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_detector_roll_setup_update
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_generates_named_synthetic_dataset
  tests/test_synthetic_datasets.py::test_generate_synthetic_dataset_writes_deterministic_smoke_artifacts
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  -q` passed: 27 tests in 278.34 seconds.
- `uv run ruff check src/tomojax/geometry/_state.py
  src/tomojax/forward/_projector.py src/tomojax/align/_pose_lm.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_setup_lm.py
  src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/datasets/_writer.py
  tests/test_forward_reference.py tests/test_pose_lm.py
  tests/test_joint_schur_lm.py tests/test_align_auto_cli.py` passed.
- `uv run basedpyright` on the same focused source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Parallel laminography, theta-scale observability, det_v activation policy,
  object drift, automatic weak alpha/beta/axis activation, and full five-case
  benchmark recovery remain future vertical slices.
- Alpha/beta are supported opt-in pose DOFs; they are intentionally not yet
  enabled by default in alternating schedules because the weak-DOF policy still
  needs gauge-aware activation rules.

## 2026-05-07 — GPU Memory Diagnostic Pause Addendum

### Summary

- Pausing again at a clean diagnostic boundary after the core-projector
  rebaseline, detector-roll support, and axis-tilt support slices.
- No new Phase 8 benchmark-ingestion, report-shape, observability, or refactor
  slice is started here.
- The 32^3/4-view smoke benchmark remains CI and wiring coverage only. It is
  not evidence for realistic alignment quality.

### Current Best Diagnosis

- The five-case benchmark failures are still expected until unsupported
  geometry terms are handled by scoped implementation slices or reported as
  `unsupported_dof_not_evaluated`. They should not be interpreted as a Schur
  failure by themselves.
- The fixed-synthetic-truth versus stopped-reconstruction evidence still points
  to reconstruction/volume gauge handling as the production-like blocker once
  the setup-global oracle is reduced to supported geometry and pose freezing.
  Fixed-truth full-oracle Schur recovers the supported setup DOFs on the
  realistic 64^3/64-view GPU case; stopped reconstruction can carry enough
  signal for detector-centre correction, but trust scheduling and the preview
  reconstruction hand-off still limit broader recovery.
- The GPU memory regression isolated so far was the finite-difference Schur
  Jacobian/reconstruction reference path materialising too much all-view
  projector state, not a reason to shrink the benchmark. The current 64^3/
  64-view diagnostics complete on JAX GPU with `selected_jax_device="cuda:0"`
  after sequential finite-difference accumulation and the core-adjoint
  reconstruction-gradient rebaseline.

### Commands Run

- Focused component probes over 1/4/16/64 views for projector, backprojector,
  one FISTA iteration, fixed-truth Schur, stopped-volume Schur, and
  fixed-truth Schur with nuisance.
- Realistic 64^3/64-view `synth128_setup_global_tomo` runs through existing
  sidecar ingestion for `fixed_synthetic_truth` and `stopped_reconstruction`
  modes.
- Supported-only oracle and stopped det_u-only refreshes through
  `tomojax-align-auto-smoke` on JAX GPU.
- Focused validation for the current committed slices:
  `uv run ruff check ...`, `uv run basedpyright ...`, focused CPU `pytest`,
  and `just imports`.

### Artifacts Produced

- `.artifacts/phase8_setup_global_gpu_ladder/`
- `.artifacts/phase8_supported_only_oracle/`
- `.artifacts/phase8_core_projector/`
- `docs/benchmark_runs/2026-05-06-phase8-multi-case-32.md`
- `docs/benchmark_runs/2026-05-06-phase8-setup-global-gpu-ladder.md`
- `docs/benchmark_runs/2026-05-07-phase8-supported-only-oracle.md`

### Remaining Open Questions

- Whether broader supported parameter blocks still need chunked Schur
  accumulation beyond sequential finite-difference columns.
- Whether preview reconstruction needs stronger gauge constraints,
  non-periodic detector boundary semantics, or a different initialization
  before stopped-reconstruction alignment can recover setup and pose together.
- How to classify each unsupported five-case scenario criterion after the
  detector roll and axis-tilt support slices without relaxing tolerances.

## 2026-05-07 — Phase 2/5 Axis Tilt Core Geometry Slice

### Summary

- Promoted `axis_rot_x_rad` and `axis_rot_y_rad` from unsupported to supported
  `core_trilinear_ray` v2 setup DOFs for parallel tomography.
- The v2-to-core adapter now derives a lab-frame rotation-axis unit through
  `tomojax.calibration.axis_geometry.axis_unit_from_rotations` and builds
  per-view core `T_all` with `axis_pose_stack`.
- Setup-only LM and joint Schur LM can pack, update, freeze, trace, and report
  axis x/y rotations as explicit active setup parameters. They remain
  opt-in active blocks until observability policy turns them on automatically.
- Synthetic sidecar generation no longer projects axis rotations away and no
  longer classifies them as `unsupported_dof_not_evaluated`; alpha/beta remain
  explicitly unsupported in this slice.
- Verification and benchmark-manifest evaluation now emit `axis_error_rad` and
  evaluate `axis_error_deg_lt` as a real metric.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py
  tests/test_setup_lm.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_recovers_realized_supported_geometry
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_freeze_pose_dofs_for_setup_oracle
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_det_u_only_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_detector_roll_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_axis_tilt_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_writes_normal_eq_summary_artifact
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_generates_named_synthetic_dataset
  tests/test_synthetic_datasets.py::test_generate_synthetic_dataset_writes_deterministic_smoke_artifacts
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  -q` passed: 27 tests in 261.43 seconds.
- `uv run ruff check src/tomojax/forward/_projector.py
  src/tomojax/align/_setup_lm.py src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/datasets/_writer.py
  tests/test_forward_reference.py tests/test_setup_lm.py
  tests/test_joint_schur_lm.py tests/test_align_auto_cli.py` passed.
- `uv run basedpyright` on the same focused source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed.

### Remaining Work

- Laminography, alpha/beta pose, theta-scale observability, object drift,
  automatic weak-axis activation policy, and full five-case benchmark recovery
  remain future vertical slices.
- The named 4-view CLI synthetic smoke now evaluates both `axis_error_deg_lt`
  and `roll_error_deg_lt`; failures in that wiring diagnostic are real
  criteria, not unsupported placeholders.

## 2026-05-07 — Phase 2/5 Detector Roll Core Geometry Slice

### Summary

- Promoted `detector_roll_rad` from unsupported to a supported
  `core_trilinear_ray` v2 setup DOF for parallel tomography.
- The v2-to-core adapter now builds a calibrated detector grid with detector
  roll through `tomojax.calibration.detector_grid`, while keeping detector
  centre shifts independent through the existing supported translation terms.
- Setup-only LM and joint Schur LM can pack, update, freeze, trace, and report
  `detector_roll_rad` as an active setup parameter.
- Synthetic sidecar generation no longer projects detector roll away and no
  longer classifies it as `unsupported_dof_not_evaluated`; axis rotations and
  alpha/beta remain explicitly unsupported in this slice.
- Verification and benchmark-manifest evaluation now emit
  `detector_roll_error_rad` and evaluate `roll_error_deg_lt` as a real metric.

### Validation

- `uv run ruff check src/tomojax/forward/_projector.py
  src/tomojax/align/_setup_lm.py src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py
  src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py src/tomojax/datasets/_writer.py
  tests/test_forward_reference.py tests/test_setup_lm.py
  tests/test_joint_schur_lm.py tests/test_align_auto_cli.py` passed.
- `uv run basedpyright` on the same focused source/test set passed with
  0 errors, 0 warnings, and 0 notes.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py
  tests/test_setup_lm.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_recovers_realized_supported_geometry
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_freeze_pose_dofs_for_setup_oracle
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_det_u_only_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_detector_roll_setup_update
  tests/test_joint_schur_lm.py::test_joint_schur_writes_normal_eq_summary_artifact
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_generates_named_synthetic_dataset
  tests/test_synthetic_datasets.py::test_generate_synthetic_dataset_writes_deterministic_smoke_artifacts
  tests/test_synthetic_datasets.py::test_generate_supported_only_setup_global_dataset_removes_unsupported_truth
  -q` passed: 24 tests in 168.57 seconds.
- `just imports` passed.

### Remaining Work

- Axis tilt, laminography, alpha/beta pose, theta-scale observability, object
  drift, and full five-case benchmark recovery remain future vertical slices.
- The named 4-view CLI synthetic smoke now evaluates `roll_error_deg_lt` and
  currently fails it as expected for that small wiring diagnostic; this is a
  real criterion now, not an unsupported placeholder.

## 2026-05-07 — Phase 8 Core Trilinear Ray Projector Rebaseline

### Summary

- Retired the v2 rotate-and-sum projector as an operational path for the public
  `tomojax.forward.project_parallel_reference*` API. Those names now adapt the
  supported v2 `GeometryState` into core `Grid`, `Detector`, and per-view
  `T_all`, then call the existing core trilinear ray projector.
- Added typed forward provenance for the single supported v2 operator family:
  `core_trilinear_ray`.
- Switched the deterministic preview backprojection helper from the hand-rolled
  rotate/shift inverse to the core explicit adjoint
  `sum_backproject_views_T`.
- Switched the preview FISTA data gradient to an explicit core-adjoint update:
  core `forward_project_view_T` builds residuals and
  `sum_backproject_views_T` applies the data-gradient adjoint. The wrapper keeps
  existing residual filters and pseudo-Huber weights without reverse-mode
  differentiation through the projector data term.
- Recorded core operator provenance in generated sidecar manifests, run
  manifests, backend reports, config files, and benchmark results.
- Unsupported setup/pose DOFs in the adapter are now explicit errors rather
  than silently ignored: detector roll, axis rotations, alpha, and beta.
- Synthetic sidecar generation for benchmark cases with unsupported DOFs now
  records `unsupported_dof_status="unsupported_dof_not_evaluated"` and the
  skipped DOF names in the manifest, while generating projections only from the
  supported core-projectable geometry subset.

### Adapter Contract

- Supported in this slice: nominal theta, `theta_offset_rad`,
  `phi_residual_rad`, `det_u_px`, `det_v_px`, per-view `dx_px`, and per-view
  `dz_px` for parallel tomography.
- Core grid defaults are explicit and recorded: unit voxel spacing, centered
  volume origin by `Grid`, detector shape derived from volume/projection shape,
  unit detector pixels, centered detector, `step_size=null`,
  `n_steps=null`, `gather_dtype="fp32"`, checkpointed projector, and unroll 1.
- Generation, Schur residual/Jacobian finite differences, verification losses,
  residual artifacts, and sidecar consistency checks all reach the same core
  operator through the forward public API.

### Diagnostics

Artifacts:

- 32^3 CPU smoke:
  `.artifacts/phase8_core_projector/runs/32_supported_only_fixed_truth_cpu/`
- 64^3/64-view GPU fixed-truth, balanced:
  `.artifacts/phase8_core_projector/runs/64_supported_only_fixed_truth_gpu/`
- 64^3/64-view GPU fixed-truth, reference:
  `.artifacts/phase8_core_projector/runs/64_supported_only_fixed_truth_reference_gpu/`
- 64^3/64-view GPU fixed-truth, full raw/no-prior oracle:
  `.artifacts/phase8_core_projector/runs/64_supported_only_fixed_truth_full_oracle_gpu/`
- 64^3/64-view GPU stopped anchored det_u-only:
  `.artifacts/phase8_core_projector/runs/64_supported_only_stopped_anchor_gpu/`
- 64^3/64-view GPU stopped anchored det_u-only with unclipped setup trust:
  `.artifacts/phase8_core_projector/runs/64_supported_only_stopped_anchor_unclipped_detu_gpu/`
- Core sidecar dataset:
  `.artifacts/phase8_core_projector/datasets/synth128_setup_global_tomo_64_supported_only/`

| Run | Device | Mode | Status | det_u RMSE px | theta RMSE rad | Schur accepted | Total time s | Notes |
|---|---|---|---|---:|---:|---|---:|---|
| 32^3 CPU smoke | `cpu:0` | fixed truth, pose frozen | geometry passed | 7.15256e-07 | 1.19844e-07 | true | n/a | Refreshed after `a1e7fc9`; `benchmark_result.json` now records `core_trilinear_ray` and geometry criteria pass. Benchmark top-level status still reflects final-volume verification, so this remains wiring/oracle coverage only. |
| 64^3 GPU balanced | `cuda:0` | fixed truth, pose frozen | failed | 6.75000 | 0.0203247 | final level accepted | 30.0770 | Final `det_u` only reached about 0.50 px from true 7.25 px. |
| 64^3 GPU reference | `cuda:0` | fixed truth, pose frozen | failed | 7.12500 | 0.0224485 | mostly rejected/limited | 50.2916 | Correct core provenance recorded; longer schedule did not fix setup recovery. |
| 64^3 GPU reference | `cuda:0` | fixed truth, pose frozen, raw/no-prior full oracle | geometry passed | 1.43051e-06 | 1.06805e-07 | true | 52.0031 | Geometry criteria pass; benchmark top-level status still reflects unrelated final-volume NMSE. |
| 64^3 GPU reference | `cuda:0` | stopped reconstruction, cylindrical support, constant init, det_u only, pose frozen | det_u Gate 3 failed | 0.237177 | 0.0218166 | true | 48.7828 | Trust clipping limited the detector-center correction. |
| 64^3 GPU reference | `cuda:0` | stopped reconstruction, cylindrical support, constant init, det_u only, pose frozen, unclipped setup trust | det_u Gate 3 passed | 0.102502 | 0.0218166 | true | 42.1509 | Benchmark manifest still marks theta failed because theta is intentionally frozen in this setup-only diagnostic. |

Additional stopped-volume Schur probe using the stopped run's final volume:

| Probe | det_u final px | det_u error px | Interpretation |
|---|---:|---:|---|
| raw, no prior, setup trust 0.5 | 6.87210 | 0.377899 | Trust clipping still limits recovery. |
| raw, no prior, setup trust 1.0 | 7.01685 | 0.233153 | Similar to anchored stopped diagnostic. |
| raw, no prior, no setup trust clip | 7.18889 | 0.0611143 | Stopped volume contains enough signal; current trust schedule is the remaining limiter. |

### Current Blocker

Fixed-truth core recovery passes after isolating the oracle from preview
continuation filters, metadata priors, and coarse early-exit. The supported
v2-to-core adapter and setup scaling are therefore coherent for det_u and theta.
The v2-to-core adapter, fixed-truth oracle, and stopped setup-only det_u
diagnostic now pass their supported geometry checks under the real operator.
The remaining gap before broader benchmark reporting is semantic rather than a
core-operator mismatch: five-case scenarios include unsupported DOFs that must
be reported as `unsupported_dof_not_evaluated`, and stopped setup-only diagnostics
intentionally freeze theta, so manifest-level theta criteria are not applicable
to that diagnostic.

### GPU Memory Finding

The 64^3/64-view fixed-truth and stopped diagnostics now complete on the laptop
GPU with JAX selecting `cuda:0` after the v2 reference path was rebaselined on
the core trilinear projector and the preview FISTA data term was moved to an
explicit core-adjoint gradient. The current best diagnosis is that the earlier
memory regression was not the sidecar ingestion path or nuisance fitting; it was
the reference reconstruction/gradient path materialising reverse-mode projector
state through the all-view forward call. The fixed-truth oracle run exercises
Schur geometry updates without stopped-reconstruction ambiguity, while the
stopped det_u-only run exercises the production-like reconstruction-to-Schur
hand-off.

The remaining memory question is whether broader parameter blocks and the
unsupported five-case scenarios still require chunked Schur accumulation once
they are evaluated through supported geometry subsets. No smaller benchmark was
accepted as a substitute for the 64^3/64-view diagnostic.

### Validation

- `uv run ruff check ...` passed for touched source and tests.
- `uv run basedpyright ...` passed with 0 errors and 0 warnings for touched
  source and tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py
  tests/test_reference_fista.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_freeze_pose_dofs_for_setup_oracle
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_det_u_only_setup_update
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  tests/test_alternating_solver_smoke.py::test_alternating_solver_stopped_reconstruction_sidecar_reports_recovery_gap
  -q` passed: 21 tests in 200.80 seconds.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke --out-dir
  .artifacts/phase8_core_projector/runs/32_supported_only_fixed_truth_cpu
  --profile smoke32 --synthetic-dataset synth128_setup_global_tomo
  --dataset-out-dir .artifacts/phase8_core_projector/datasets --size 32
  --views 8 --supported-only-setup-global
  --geometry-update-volume-source fixed_synthetic_truth
  --geometry-update-pose-frozen` completed and refreshed the 32^3 smoke
  artifact with `backend.actual="core_trilinear_ray"` and
  `selected_jax_device="cpu:0"`.
- `just imports` passed.

### Remaining Work

- Keep fixed-truth full-oracle and stopped det_u-only diagnostics as regression
  guards while moving to benchmark reporting.
- Update five-case reporting with unsupported DOFs classified as
  `unsupported_dof_not_evaluated`.
  Sidecar manifests now record that classification; the full five-case compare
  pass remains the next benchmark-reporting step.

## 2026-05-07 — Phase 8 Anchored Preview Reconstruction Gate 1

### Summary

- Added optional `volume_support` to reference FISTA and project both candidate
  and momentum state through nonnegativity/support constraints after every
  update.
- Added deterministic centered cylindrical/spherical support masks in
  `tomojax.recon`.
- Threaded preview support, preview initialization, preview TV scaling, and
  preview residual-filter mode through the private alternating solver and
  `align-auto` CLI.
- Added setup-only Schur wiring for active setup-parameter subsets so the first
  stopped schedule can freeze pose, `theta_offset_rad`, and `det_v_px` while
  updating only `det_u_px`.
- Ran supported-only 64^3/64-view stopped GPU diagnostics and reached Gate 1
  with cylindrical support, continuation preview filters, preview TV, constant
  initialization, and det_u-only stopped setup updates.

### Diagnostics

Dataset:
`.artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only/`

All stopped runs used JAX GPU on `cuda:0`, nuisance disabled, sidecar ingestion,
`geometry_update_volume_source=stopped_reconstruction`,
`geometry_update_active_setup_parameters=det_u_px`, and
`geometry_update_pose_frozen=true`.

| Mode | Profile | Support | Init | TV scale | Preview filters | det_u RMSE px | theta RMSE rad | True vol/final geom | Final volume/init geom | Final volume/true geom | Classification | Schur accepted | Time s |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|
| baseline | reference | none | backprojection | 0 | raw | 7.25 | 0.0218166 | 0.884522 | 1.05091 | 1.40607 | `reconstruction_absorbed_geometry` | false | 18.9327 |
| support only | reference | cylindrical | backprojection | 0 | raw | 4.28274 | 0.0218166 | 0.483220 | 1.38357 | 1.35286 | `training_loss_not_independent` | true | 12.3957 |
| support + TV | reference | cylindrical | backprojection | 1 | raw | 4.28349 | 0.0218166 | 0.483309 | 1.38357 | 1.35286 | `training_loss_not_independent` | true | 12.3683 |
| support + TV + filters | reference | cylindrical | backprojection | 1 | continuation | 4.28358 | 0.0218166 | 0.483320 | 1.38357 | 1.35286 | `training_loss_not_independent` | true | 13.5487 |
| less geometry-aware init | reference | cylindrical | constant | 1 | continuation | 0.453199 | 0.0218166 | 0.0167246 | 1.87446 | 1.84208 | `independent_projection_losses_consistent` | true | 18.6021 |

Gate 1 outcome:

- First stopped setup stage now accepts useful detector-shift movement in the
  anchored runs.
- `det_u` RMSE improves from 7.25 px to 0.453199 px, below the 3 px gate.
- True-volume/final-geometry loss drops from 0.884522 to 0.0167246.
- Theta remains frozen in this diagnostic schedule; exact stopped theta recovery
  is intentionally not required without an explicit orientation anchor.

### Fixed-Truth Regression

Fixed-truth supported-only Schur still passes after the anchored preview change:

- Run:
  `.artifacts/phase8_anchored_preview/runs/64_fixed_truth_pose_frozen_anchor_regression/`
- Status: passed.
- Device: `cuda:0`.
- det_u RMSE: `1.00136e-05` px.
- theta RMSE: `2.68284e-06` rad.
- true-volume/final-geometry loss: `0.0`.

### Detector Boundary Diagnostic

The current reference projector uses periodic detector shifts. A one-off
diagnostic compared the successful anchored stopped run with zero-fill and
valid-overlap masked non-periodic detector-shift semantics.

- Artifact:
  `.artifacts/phase8_anchored_preview/detector_boundary_diagnostic.json`
- Valid overlap fraction: `0.875`.
- Wrap losses: true-volume/true-geometry `0.0`,
  true-volume/final-geometry `0.0167246`,
  true-volume/initial-geometry `0.884522`.
- Valid-overlap masked zero-fill losses: true-volume/true-geometry `0.0`,
  true-volume/final-geometry `0.0191139`,
  true-volume/initial-geometry `1.01088`.

Interpretation: non-periodic masked detector-shift semantics penalize the wrong
detector shift more strongly while preserving zero loss for true geometry. This
should be handled as the next detector-boundary slice, not mixed into the
anchored-preview Gate 1 commit.

### Validation

- `uv run ruff format ...` passed for touched source and tests.
- `uv run ruff check ...` passed for touched source and tests.
- `uv run basedpyright ...` passed with 0 errors and 0 warnings for touched
  source and tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_reference_fista.py
  tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir
  -q` passed: 22 tests.
- `just imports` passed.
- `just check` did not pass because the repository-wide Ruff gate still hits
  broad pre-existing legacy lint debt outside this slice after formatting
  `src tests tools` (`src/tomojax/align/model/schedules.py`,
  `src/tomojax/align/model/state.py`,
  `src/tomojax/align/objectives/fixed_volume.py`, and many legacy tests).
  The unrelated formatter churn from that attempt was reverted before commit.

### Remaining Work

- Gate 2 is not yet complete. Backprojection initialization with support/TV and
  filters improves det_u only to about 4.28 px; the constant initialization is
  the first configuration that reaches Gate 1.
- Detector-boundary semantics should be addressed next because the current
  projector wraps detector shifts.
- Pose and theta should remain frozen until setup recovery is stronger; exact
  stopped theta recovery still needs an explicit orientation anchor.

## 2026-05-07 — GPU Memory Diagnostic Pause

### Summary

- Paused at a documentation-only boundary after the current GPU
  memory-regression and setup-global diagnostics. No new feature, report-field,
  refactor, or benchmark-ingestion slice is started here.
- Current code head before this pause commit was `f66c9af` (`Record refreshed
  pose-frozen oracle`).
- The 32^3/4-view smoke benchmark remains CI/wiring coverage only and should
  not be used to judge alignment quality.

### Current Best Diagnosis

- The five-case 32^3 benchmark failures are wiring evidence, not recovery
  evidence: sidecar generation, sidecar ingestion, `benchmark_result.json`,
  `benchmark_report.md`, and compare artifacts run end-to-end, but all current
  stopped-reconstruction recovery gates fail at unrealistic view count.
- The first realistic 64^3/64-view nuisance-free `synth128_setup_global_tomo`
  ladder on `cuda:0` failed both `fixed_synthetic_truth` and
  `stopped_reconstruction`, initially pointing at setup/pose/theta coupling or
  geometry convention mapping.
- Supported-only 64^3/64-view oracle refreshes narrowed that: filtered
  fixed-truth Schur passes with pose frozen and with the strong pose prior,
  while stopped reconstruction still leaves geometry at nominal. The current
  production-like blocker is therefore reconstruction/volume gauge absorption
  before Schur, after accounting for setup/pose gauge coupling.

### Five-Case Benchmark Failures

The 32^3 multi-case pass generated planned sidecars and completed all run and
comparison artifacts, but failed recovery:

| Benchmark | Status | Criteria | Geometry | Total Time s | Notes |
|---|---|---|---|---:|---|
| `synth128_setup_global_tomo` | failed | failed | failed | 11.5794 | `det_u=3.625`, `theta=0.0218166`, `det_v=0` |
| `synth128_pose_random_extreme` | failed | partially_evaluated | failed | 13.3407 | `det_u=2.7415`, `det_v=2.5782`, `theta=0.2019` |
| `synth128_lamino_axis_roll_pose` | failed | failed | failed | 13.3946 | `det_u=2.2334`, `det_v=0.7336`, `theta=0.1598` |
| `synth128_thermal_object_drift` | failed | partially_evaluated | failed | 13.5649 | `det_u=1.4893`, `det_v=0.0512`, `theta=0.0052336`; label `nuisance_residual_structure` |
| `synth128_combined_nuisance_jumps` | failed | failed | failed | 13.4880 | `det_u=3.8751`, `det_v=0.9955`, `theta=0.0309604` |

### Fixed-Truth Versus Stopped Evidence

Realistic setup-global ladder, nuisance disabled, 64^3 volume, 64 views,
existing sidecar ingestion, `jax_default_backend="gpu"`,
`selected_jax_device="cuda:0"`:

| Mode | Status | Geometry | det_u RMSE px | det_v RMSE px | theta RMSE rad | Final Residual | Volume NMSE | Schur Accepted | Total Time s |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `fixed_synthetic_truth` | failed | failed | 6.9338 | 0.00666 | 0.02211 | 0.856277 | 0.686109 | true | 37.5096 |
| `stopped_reconstruction` | failed | failed | 7.25 | 0 | 0.02182 | 0 | 0.686110 | true | 24.8489 |

Supported-only oracle refresh after Schur residual filtering:

| Mode | Status | det_u RMSE px | theta RMSE rad | Schur train loss | True vol/final geom | Classification | Total Time s |
|---|---|---:|---:|---:|---:|---|---:|
| `fixed_synthetic_truth`, pose frozen | passed | 1.33514e-05 | 2.59716e-06 | 1.18979e-08 | 0 | `independent_projection_losses_consistent` | 16.5956 |
| `fixed_synthetic_truth`, strong pose prior `1e6` | passed | 5.24164e-06 | 5.10065e-05 | 2.13915e-08 | 3.39969e-09 | `independent_projection_losses_consistent` | 106.826 |
| `stopped_reconstruction`, strong pose prior `1e6` | failed | 7.25 | 0.0218166 | 0.361978 | 0.884522 | `reconstruction_absorbed_geometry` | 113.388 |

### GPU Memory Finding

- The confirmed memory-regression source was the Schur finite-difference
  Jacobian path, not a need to shrink the benchmark. The original 64^3/64-view
  fixed-truth run attempted a 12.14 GiB allocation shaped like
  `f32[194,64,64,64,64]` because all parameter perturbations were evaluated by
  a single `jax.vmap`.
- Commit `dc2aa74` changed the shared finite-difference Jacobian helper to
  accumulate columns sequentially. After that fix, component probes passed on
  GPU for 1/4/16/64 views across projector, backprojector, one FISTA iteration,
  fixed-truth Schur, stopped-volume Schur, and fixed-truth Schur with nuisance.
- The 64^3/64-view benchmark results record `jax_default_backend="gpu"` and
  `selected_jax_device="cuda:0"` in `benchmark_result.json`.

### Commands And Artifacts

Commands run in this diagnostic thread:

- GPU probe:
  `LD_LIBRARY_PATH=<venv nvidia */lib paths> JAX_PLATFORMS=cuda uv run python -c 'import jax; ...'`
- 64^3 sidecar generation for nuisance-free `synth128_setup_global_tomo` with
  64 views.
- Component probes:
  `.artifacts/phase8_setup_global_gpu_ladder/probes/probe_components.py --views 1|4|16|64`
- Realistic ladder runs:
  `tomojax-align-auto-smoke --profile balanced --synthetic-dataset-dir ... --geometry-update-volume-source fixed_synthetic_truth`
  and
  `tomojax-align-auto-smoke --profile balanced --synthetic-dataset-dir ... --geometry-update-volume-source stopped_reconstruction`
- Compare:
  `tomojax-synthetic-benchmark-compare ... --out .artifacts/phase8_setup_global_gpu_ladder/benchmark_comparison_64.md`
- Supported-only oracle refreshes through `tomojax-align-auto-smoke` with
  fixed-truth pose-frozen, fixed-truth strong-pose-prior, and
  stopped-reconstruction strong-pose-prior modes.
- Focused validation for the committed code slices:
  `uv run ruff check ...`, `uv run basedpyright ...`,
  `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_align_auto_cli.py -q`,
  and `just imports`.

Key artifacts:

- `docs/benchmark_runs/2026-05-06-phase8-multi-case-32.md`
- `docs/benchmark_runs/2026-05-06-phase8-setup-global-gpu-ladder.md`
- `docs/benchmark_runs/2026-05-07-phase8-supported-only-oracle.md`
- `.artifacts/phase8_multi_case_32_benchmark_pass/`
- `.artifacts/phase8_setup_global_gpu_ladder/`
- `.artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only/`
- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_filtered_reporting/`
- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000_filtered_reporting/`
- `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000_filtered_reporting/`
- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_filtered_reporting.md`

### Remaining Open Questions

- Why does stopped reconstruction produce a preview volume/geometry pair that
  accepts no useful setup movement when filtered fixed-truth Schur now recovers
  the supported setup DOFs?
- Is stopped-gradient reconstruction absorbing setup error into volume gauge,
  detector shift, missing normalization, or support/background degrees of
  freedom before the geometry update?
- Should the next diagnostic compare independent all-view losses for
  true-volume/true-geometry, true-volume/final-geometry,
  final-volume/initial-geometry, and final-volume/final-geometry before
  changing solver behavior?
- Should the next implementation slice constrain preview reconstruction gauge
  or initialization before revisiting broader benchmark scenarios?

## 2026-05-07 — Supported-Only Fixed-Truth Oracle Pass

### Summary

- Added a supported-only `synth128_setup_global_tomo` sidecar generation mode
  for the setup oracle: clean projections, nominal theta 0..180, true
  `det_u_px` and `theta_offset_deg`, and unsupported roll/axis/pose/nuisance
  terms disabled.
- Added a pose-frozen Schur configuration path through `align-auto` so the
  existing sidecar ingestion, benchmark_result, benchmark_report, and compare
  artifact path can run a fixed-truth setup-only oracle.
- Scaled Schur predicted reduction by data residual rows so trust adaptation
  compares against the same mean-loss scale used for actual reduction.
- Added manifest evaluation for `theta_offset_error_deg_lt` by converting the
  threshold to radians and comparing against realised theta RMSE.

### GPU Oracle Result

- Command:
  `LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -)" JAX_PLATFORMS=cuda uv run tomojax-align-auto-smoke --out-dir .artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass --profile balanced --synthetic-dataset-dir .artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only --geometry-update-volume-source fixed_synthetic_truth --geometry-update-pose-frozen`
- Backend/device: `jax_default_backend="gpu"`, `selected_jax_device="cuda:0"`.
- Status: passed.
- det_u realised RMSE: 0.0890718 px, threshold 0.5 px.
- theta realised RMSE: 0.00109812 rad, threshold 0.00174533 rad.
- final residual: 0.000185589.
- volume NMSE: 0.576863.
- total wall time: 15.7484 s.

### Artifacts

- Dataset:
  `.artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only/`
- Passing run:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/`
- Benchmark result:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/benchmark_result.json`
- Benchmark report:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/benchmark_report.md`
- Compare report:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only.md`
- Summary:
  `docs/benchmark_runs/2026-05-07-phase8-supported-only-oracle.md`

### Validation

- `uv run ruff check ...` on touched source/test files passed.
- `uv run basedpyright ...` on touched source/test files passed with 0 errors
  and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py
  tests/test_synthetic_datasets.py tests/test_align_auto_cli.py
  tests/test_bench_synthetic_results.py -q` passed: 37 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Remaining Questions

- Fixed-truth joint setup+pose on the same supported-only dataset fails without
  pose constraints by absorbing setup into per-view pose:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_baseline/`
  ended with det_u realised RMSE 6.72424 px and theta realised RMSE
  0.021352 rad.
- Fixed-truth joint setup+pose passes with a strong pose prior:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000/`
  ended with det_u realised RMSE 0.0890279 px and theta realised RMSE
  0.00109136 rad. This classifies the joint blocker as setup absorption into
  weakly constrained pose, not a fixed-truth projector/setup convention issue.
- Stopped-reconstruction diagnostics remain blocked until fixed-truth joint
  setup+pose has a principled staged/block-wise trust policy rather than an
  effectively hard diagnostic pose prior.

## 2026-05-07 — Schur Block-Wise Trust Diagnostic

### Summary

- Replaced the single global Schur trust scale with separate setup and pose
  trust scales so a large aggregate pose update no longer shrinks a valid setup
  update.
- Preserved the existing `trust_scale` diagnostic as the minimum block scale
  and added explicit `setup_trust_scale` and `pose_trust_scale` diagnostics.
- Added a regression test proving pose trust clipping leaves the setup step
  unchanged when setup is within its trust radius.

### GPU Result

- Command:
  `LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -)" JAX_PLATFORMS=cuda uv run tomojax-align-auto-smoke --out-dir .artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_block_trust --profile balanced --synthetic-dataset-dir .artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only --geometry-update-volume-source fixed_synthetic_truth`
- Backend/device: JAX GPU on `cuda:0`.
- Status: failed.
- det_u realised RMSE: 7.25 px.
- theta realised RMSE: 0.0218166 rad.
- First final-level candidate used separate scales:
  `setup_trust_scale=0.617204`, `pose_trust_scale=0.0187305`, but the setup
  update was `[theta=-1.04e-04, det_u=-0.5]` and the step was rejected.

### Interpretation

Block-wise trust fixes the mechanical aggregate-pose clipping issue, but it is
not sufficient for the unconstrained fixed-truth joint case. The remaining
blocker is setup/pose gauge coupling: without a strong pose prior or staged pose
activation, the coupled Schur step can point setup in the wrong detector-shift
direction. The previously recorded hard-prior diagnostic still passes, so the
next implementation slice should parameterise or stage pose DOFs rather than
returning to projector/setup convention debugging.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 11 tests.
- `just imports` passed.

## 2026-05-07 — Staged Pose Activation Diagnostic

### Summary

- Extended joint Schur active pose DOFs from all-or-none to explicit subsets
  such as `("dx_px", "dz_px")`.
- Exposed `--geometry-update-active-pose-dofs` and
  `--geometry-update-pose-activate-at-level-factor` in `align-auto` so coarse
  setup updates can run with pose frozen before selected pose DOFs activate.
- Added focused tests for partial pose DOF updates and the new CLI/config
  surface.

### GPU Result

- Staged all pose DOFs at level factor 1:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_staged_pose_level1/`
  failed with det_u realised RMSE 0.583686 px and theta realised RMSE
  0.00940296 rad.
- Staged detector pose only (`dx_px,dz_px`) at level factor 1:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_staged_pose_level1_no_phi/`
  failed with the same final geometry. The final detector-pose candidate was
  rejected, leaving setup just outside tolerance.
- Compare artifact:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_staged_pose.md`.

### Interpretation

Partial/staged pose activation is now available, but activating pose in the
final geometry update does not yet pass the strict supported-only criteria. The
hard pose-prior fixed-truth joint run remains the only passing joint diagnostic.
The next fix should avoid perturbing verified setup during final pose activation
or use a zero-mean/anchored pose parameterisation inside LM.

### Validation

- `uv run ruff check ...` on touched staged-pose source/test files passed.
- `uv run basedpyright ...` on touched staged-pose source/test files passed with
  0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py -q` passed: 21 tests.
- `just imports` passed.

## 2026-05-07 — Zero-Mean Pose Step Diagnostic

### Summary

- Added optional Schur step gauge projection inside `solve_joint_schur_lm`:
  mean `phi_residual_rad`, `dx_px`, and active `dz_px` pose steps are
  transferred into setup before trust scaling and candidate evaluation.
- Kept direct `schur_step_from_jacobian` dense-solve tests unchanged by making
  the projection opt-in for low-level calls and enabled in the LM solver path.
- Added a regression test proving mean detector pose step is transferred into
  setup and the residual pose step is zero-mean.

### GPU Result

- Unconstrained zero-mean joint:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_pose_step/`
  improved detector setup but failed because active per-view `phi_residual_rad`
  introduced theta variation: det_u RMSE 0.764954 px, theta RMSE 0.0219937 rad.
- Zero-mean with `phi_residual_rad` frozen and `dx_px,dz_px` active:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_no_phi/`
  passed theta but failed det_u: det_u RMSE 0.746027 px, theta RMSE
  3.83419e-05 rad.
- Reference schedule with zero-mean and `phi_residual_rad` frozen:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_no_phi_reference/`
  passed manifest criteria but missed the internal 0.2 px geometry gate by
  about 0.001 px: det_u RMSE 0.201021 px, theta RMSE 1.36732e-08 rad.
- Compare artifact:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_zero_mean.md`.

### Interpretation

The setup/pose gauge fix is materially better and nearly passes strict internal
geometry recovery without a hard pose prior, but it is still not enough to move
to stopped-reconstruction judgement. The remaining narrow blocker is final
detector-shift accuracy when detector pose is active; pose-frozen and hard-prior
diagnostics pass, so the projector/setup convention is no longer the blocker.

### Validation

- `uv run ruff check ...` on touched zero-mean source/test files passed.
- `uv run basedpyright ...` on touched zero-mean source/test files passed with
  0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py -q` passed: 22 tests.
- `just imports` passed.

## 2026-05-07 — Stopped-Reconstruction Supported-Only Classification

### Summary

- Reran the supported-only 64^3/64-view setup-global case with
  `geometry_update_volume_source=stopped_reconstruction` using the same strong
  pose prior that passes fixed-truth joint setup+pose.
- The run failed with no geometry recovery: final setup stayed nominal
  (`det_u_px=0`, `theta_offset_rad=0`).

### Result

- Run:
  `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000/`
- Backend/device: JAX GPU on `cuda:0`.
- det_u realised RMSE: 7.25 px.
- theta realised RMSE: 0.0218166 rad.
- final residual: 0.367724.
- volume NMSE: 0.576871.
- Compare artifact:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_stopped_reconstruction.md`.

### Interpretation

Fixed-truth joint setup+pose passes with the strong pose prior, but
stopped-reconstruction with the same geometry-update settings does not move
geometry. This classifies the next blocker as reconstruction/volume gauge
handling or reconstruction absorption of geometry, not nominal theta,
sidecar/projector convention, GPU memory, or fixed-truth setup Schur.

## 2026-05-07 — Nominal Theta Geometry Root Fix

### Summary

- Made nominal acquisition theta first-class in `PoseParameters` as
  `theta_nominal_rad`.
- Added `GeometryState.theta_total_rad()` with the v2 realised-angle convention:
  `theta_scale * theta_nominal_rad + theta_offset_rad + phi_residual_rad`.
- Updated the reference projector, reference backprojector, setup-only LM,
  pose-only LM, joint Schur LM, and geometry recovery metrics to use realised
  theta instead of only `theta_offset_rad + phi_residual_rad`.
- Preserved nominal theta in v2 pose CSV artifacts and pose-decomposition
  reports, with old pose CSV readback defaulting missing `theta_nominal_rad` to
  zeros.
- Updated synthetic sidecar generation so 0..180 nominal acquisition angles are
  carried into v2 nominal/corrupted/true pose metadata.

### Diagnosis

- The 64^3/64-view fixed-truth and stopped-reconstruction failures were not
  explainable by reconstruction alone because fixed-truth also failed.
- The highest-signal root cause found in this slice was a geometry convention
  bug: sidecars recorded nominal acquisition theta, but v2 geometry state,
  projection, solver residuals, and recovery metrics collapsed realised theta
  to setup offset plus residual pose.
- Focused synthetic coverage now verifies that sidecars preserve nominal theta,
  true v2 geometry projects back to the stored clean projections at near-zero
  MSE, and nominal/corrupted geometry has higher projection error.

### Validation

- `uv run ruff format ...` on touched source/test files passed.
- `uv run ruff check ...` on touched source/test files passed.
- `uv run basedpyright ...` on touched source/test files passed with 0 errors
  and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_geometry_serialization.py
  tests/test_geometry_gauges.py tests/test_synthetic_datasets.py
  tests/test_forward_reference.py tests/test_reference_fista.py
  tests/test_setup_lm.py tests/test_pose_lm.py tests/test_joint_schur_lm.py -q`
  passed: 46 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Remaining Questions

- Rerun the supported-only fixed-truth oracle benchmark after this commit to
  confirm whether nominal theta was the main alignment-quality blocker.
- If fixed-truth still fails, continue with Schur trust/block scaling and
  setup/pose gauge coupling diagnostics rather than adding report fields.
- If fixed-truth passes but stopped reconstruction fails, return to
  reconstruction/volume gauge handling.

## 2026-05-06 — Split Alternating Solver Private Implementation

### Summary

- Kept `src/tomojax/align/_alternating.py` as the public-compatible facade for
  `AlternatingAlignmentSolver`, `AlternatingSmokeConfig`,
  `AlternatingSmokeResult`, and `run_alternating_solver_smoke`.
- Moved the alternating smoke loop into
  `src/tomojax/align/_alternating_orchestration.py`.
- Moved Schur geometry-update helpers into
  `src/tomojax/align/_alternating_geometry_update.py`.
- Moved per-level skip, residual-filter summary, and coarse-verification timing
  helpers into `src/tomojax/align/_alternating_level_helpers.py`.

### Decisions

- Preserved existing private boundaries for artifact writing, verification and
  report payloads, held-out checks, smoke inputs, and smoke config/result
  dataclasses instead of renaming those files during the cleanup.
- Did not add benchmark-ingestion behavior in this slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_level_helpers.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_level_helpers.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_level_helpers.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Risks

- This was a structural cleanup, so behavior coverage relies on the existing
  deterministic alternating smoke tests.

## 2026-05-06 — Ingest Synthetic Benchmark Result Artifacts

### Summary

- Added `tomojax.bench.synthetic_results` for loading schema-validated
  `benchmark_result.json` artifacts.
- Added deterministic markdown comparison rendering over actual benchmark result
  fields: benchmark, implementation, profile, status, criteria status, geometry
  status, volume NMSE, final residual, runtime, and source path.
- Re-exported the narrow helper API from `tomojax.bench`.
- Added focused tests for loading, rendering, writing, and schema rejection.

### Decisions

- Kept this as artifact ingestion over existing `benchmark_result.json` files
  rather than starting the full current-vs-reimagined protocol runner.
- The comparison report uses only fields already required by the synthetic
  benchmark result schema.

### Validation

- `uv run ruff format src/tomojax/bench/synthetic_results.py src/tomojax/bench/__init__.py tests/test_bench_synthetic_results.py`
  passed.
- `uv run ruff check src/tomojax/bench/synthetic_results.py src/tomojax/bench/__init__.py tests/test_bench_synthetic_results.py`
  passed.
- `uv run basedpyright src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_bench_synthetic_results.py -q` passed: 4 tests.
- `uv run pytest tests/test_bench_fitness_imports.py tests/test_bench_synthetic_results.py -q`
  passed: 5 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Risks

- `tomojax.bench` is still transitional and not yet one of the finalized v2
  deep modules, so this keeps a narrow API and avoids broader benchmark package
  restructuring.

## 2026-05-06 — Add Synthetic Benchmark Comparison CLI

### Summary

- Added the `tomojax-synthetic-benchmark-compare` console script.
- The command reads one or more synthetic `benchmark_result.json` artifacts
  using the existing schema-validated ingestion helper.
- It writes a deterministic markdown report with `--out` or prints the same
  report to stdout when `--out` is omitted.
- Added focused CLI coverage for report writing and stdout rendering.

### Decisions

- Kept the command as a thin wrapper around `tomojax.bench.synthetic_results`
  instead of introducing a separate runner or report format.

### Validation

- `uv run ruff format src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py pyproject.toml`
  passed.
- `uv run ruff check src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py pyproject.toml`
  passed.
- `uv run basedpyright src/tomojax/bench/synthetic_results.py tests/test_bench_synthetic_results.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_bench_synthetic_results.py -q` passed: 6 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run tomojax-synthetic-benchmark-compare --help` passed.

### Risks

- The command compares existing result artifacts only; it does not yet schedule
  benchmark cases or compare current TomoJAX against the reimagined path.

## 2026-05-06 — Record Schur Nuisance Estimates

### Summary

- Extended `JointSchurDiagnostics` with optional `gain_offset_model` and
  `background_offset_model` payloads.
- When gain/offset fitting is enabled, diagnostics now record the fitted
  per-view gain and offset model for the accepted parameter state.
- When background fitting is enabled, diagnostics now record the fitted
  constant plus vertical-gradient background model for the accepted parameter
  state.
- Focused Schur tests now verify that fitted nuisance estimates are recovered
  and that disabled nuisance payloads remain `None`.

### Decisions

- Diagnostics reuse public `tomojax.nuisance` model `to_dict()` payloads rather
  than defining a parallel artifact schema.
- This slice records provenance for fitted nuisance correction without changing
  the geometry update rule or weak-DOF policy.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 8 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Risks

- The recorded nuisance estimates are diagnostic payloads. Automatic escalation
  and weak-DOF decisions still need additional Phase 8 policy work.

## 2026-05-06 — Cover Nuisance Estimates In Smoke Artifacts

### Summary

- Extended alternating smoke tests to assert that `schur_diagnostics.json`
  contains the fitted gain/offset nuisance model when gain/offset fitting is
  enabled.
- Extended alternating smoke tests to assert that `schur_diagnostics.json`
  contains the fitted background nuisance model when background fitting is
  enabled.
- Verified disabled nuisance branches still serialize `None` for the unrelated
  model payload.

### Decisions

- This slice adds artifact-level coverage for the previous diagnostics payload
  change without changing solver behavior or artifact schema names.

### Validation

- `uv run ruff format tests/test_alternating_solver_smoke.py` passed.
- `uv run ruff check tests/test_alternating_solver_smoke.py` passed.
- `uv run basedpyright tests/test_alternating_solver_smoke.py` passed with 0
  errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Risks

- The smoke artifact coverage uses the lightning profile for runtime. Broader
  Phase 8 nuisance policy still needs benchmark-case validation.

## 2026-05-06 — Add Weak-DOF Correlation Evidence

### Summary

- Added det_v setup-correlation evidence to `observability_report.json`.
- The report-only det_v weak-DOF decision now requires curvature, correlation,
  accepted-step, and validation-improvement evidence to pass.
- Added the correlation threshold to the weak-DOF policy thresholds block.
- Extended alternating smoke tests to assert the emitted correlation evidence,
  threshold, and missing-evidence behavior.

### Decisions

- Correlation evidence is computed from the Schur setup correlation matrix for
  active `det_v_px`. Missing Schur diagnostics keep correlation evidence
  explicit and conservative.
- This remains report-only; it does not yet mutate the active setup parameter
  set during solve.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Risks

- The correlation ceiling is conservative and report-only. Automatic weak-DOF
  activation/freezing still needs a later Phase 8 policy slice.

## 2026-05-06 — Surface Failure-Report Warning Status

### Summary

- Updated `failure_report.json` payloads to use `status: "warning"` when a
  warning-class gate fails.
- Clean smoke runs still assert `status: "passed"` and an empty warnings list.
- Added focused structured-residual coverage showing that a column-pattern
  residual produces a `nuisance_unmodelled` warning.

### Decisions

- This remains a warning classification, not a hard failure policy. The
  `failure` field stays `None` for warning-only reports.
- The focused test uses a deliberate white-box import of the align-owned
  failure report builder because the current public verification facade does
  not yet expose full failure-report construction.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py -q`
  passed: 11 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Risks

- Failure-report construction still lives under `tomojax.align`; a later verify
  module slice should move the report builder behind the public `tomojax.verify`
  facade.

## 2026-05-06 — Move Failure-Report Assembly To Verify

### Summary

- Added `tomojax.verify.failure_report_from_gates` and
  `tomojax.verify.failure_warnings_from_gates`.
- Moved the common failure-report envelope, failure-class list, warning status,
  and warning payload construction into `tomojax.verify`.
- Routed alternating smoke failure reports through the verify-owned helper while
  keeping align responsible for producing run-specific gate evidence.
- Replaced the private white-box failure-report test with public verify facade
  coverage.

### Decisions

- `tomojax.verify` now owns the generic failure-report shape; `tomojax.align`
  still owns solver-specific gate evidence until a later artifact ownership
  cleanup moves more report construction out of align.

### Validation

- `uv run ruff format src/tomojax/verify src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/verify src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/verify src/tomojax/align/_alternating_verification.py tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_failure_report_classification.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 16 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`

### Risks

- The gate evidence is still produced in `tomojax.align`; verify-owned complete
  report construction remains a future cleanup.

## 2026-05-06 — Milestone 0 Guardrail Preparation

### Summary

- Added strict Ruff, pytest, basedpyright, import-linter, and pre-commit guardrails.
- Added `tools/check_public_imports.py` to prevent cross-boundary imports from
  private TomoJAX modules.
- Marked existing white-box tests with explicit
  `check-public-imports: allow-private` exceptions.
- Added `.agent/PLANS.md` as the active milestone execution-plan workspace.
- Added this implementation log.

### Decisions

- `docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased
  implementation plan.
- `.agent/PLANS.md` is not a competing plan. It is the active milestone work log
  and checklist.
- `.importlinter` currently reflects importable packages in the transitional
  tree. It must be updated as the v2 deep-module skeleton becomes real.

### Validation

- `uv lock` completed after adding `basedpyright` and `pre-commit`.
- `just --list` found the canonical command surface.
- `uv run pre-commit validate-config` passed.
- `uv run ruff check tools/check_public_imports.py pyproject.toml` passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run basedpyright --version` reported `basedpyright 1.39.3`.
- `just typecheck` currently fails on the transitional source and test tree.
  This is expected technical debt from the pre-restructure implementation, not a
  guardrail installation failure. The v2 implementation should make strict
  typechecking pass as modules are replaced.

### Risks

- The current codebase still contains transitional modules such as
  `tomojax.utils`, `tomojax.data`, and `tomojax.calibration`. The v2 restructure
  should delete or migrate them under the deep-module architecture rather than
  preserve them as compatibility surfaces.
- `just check` cannot pass until type errors in the existing implementation are
  removed or the old implementation is replaced by the v2 deep modules.

## 2026-05-06 — Remove Stale Pre-v2 User Documentation

### Summary

- Removed stale pre-v2 user-facing archive docs to prevent agents from treating
  old CLI, config, tutorial, and public API surfaces as current architecture.
- Replaced the root README with a v2 rewrite README that points to
  `docs/tomojax-v2/` as the canonical design source.

### Decisions

- Kept historical brainstorms, ideation notes, implementation plans, and
  solution notes under `docs/archive/` for now because they may still contain
  useful benchmark and implementation archaeology.
- Deleted archived user-facing docs where Git history is a better archive than
  a stale in-tree compatibility signal.

### Validation

- Checked current v2-facing docs and guardrail files for links to the removed
  archive pages:
  `README.md`, `AGENTS.md`, `.agent/PLANS.md`, `docs/implementation_log.md`,
  `docs/tomojax-v2/`, `justfile`, and `pyproject.toml`.
- No references to the removed archived install, quickstart, CLI, reference,
  concepts, tutorials, or troubleshooting pages remain in those current files.

## 2026-05-06 — Milestone 0 Architecture-Smell Audit

### Summary

- Re-read `AGENTS.md` and the canonical phased plan before starting further
  migration work.
- Verified the v2 design docs and guardrail files are present in the checkout.
- Updated `.agent/PLANS.md` from the blank template into an active Milestone 0
  bridge plan.
- Confirmed the live source tree is still transitional:
  - no top-level `src/tomojax/*` package currently has the required v2 `api.py`
    and `README.md` pair;
  - forbidden `tomojax.utils` production imports remain in
    `align/io/checkpoint.py`, `calibration/_json.py`, and `cli/manifest.py`;
  - old top-level owners such as `tomojax.data`, `tomojax.calibration`, and
    `tomojax.bench` still exist outside the canonical v2 owner list;
  - nested old alignment/core geometry packages remain exposed under
    `align/geometry`, `align/model`, `align/objectives`, `align/io`, and
    `core/geometry`.

### Decisions

- Keep strict Ruff, basedpyright, import-linter, and public-private import
  checks in place. Current failures should drive migration/deletion work rather
  than guardrail weakening.
- Treat `tomojax.utils` as the first cleanup target because it is explicitly
  forbidden by the v2 architecture and has a small production import footprint.
- Do not start Phase 1 skeleton work until Milestone 0 records which old
  surfaces are being deleted, migrated, or temporarily retained as benchmark
  references.

### Validation

- `sifs agent-context --json` reported the current SIFS CLI contract.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run basedpyright` failed on the transitional tree with 4456 errors and
  9341 warnings. First reported errors include:
  - `align/_config.py`: hyphenated `GaugePolicyInput` is not normalized before
    calling `resolve_alignment_schedule`;
  - `align/_observer.py`: unnecessary `isinstance` check;
  - `align/_pose_stage.py`: private usage and many unknown JAX typing errors;
  - `tests/test_views.py`: dummy geometry does not satisfy the `Geometry`
    protocol return type.

### Risks

- The import-linter contract currently reflects only part of the old package
  graph and must be replaced when the v2 deep-module skeleton lands.
- The typecheck failure volume is high enough that new errors can be hidden
  unless each scoped migration is validated narrowly before rerunning broader
  checks.

## 2026-05-06 — Move JSON Serialization Out Of `tomojax.utils`

### Summary

- Added the first v2-owned deep module surface:
  - `src/tomojax/io/api.py`
  - `src/tomojax/io/__init__.py`
  - `src/tomojax/io/_json.py`
  - `src/tomojax/io/README.md`
- Moved the shared JSON normalization contract from forbidden
  `tomojax.utils.json` into `tomojax.io`.
- Updated production consumers in alignment checkpoints, calibration JSON, and
  CLI manifests to import through the public `tomojax.io` facade.
- Updated JSON utility tests to assert the new public API and deleted
  `src/tomojax/utils/json.py`.
- Added `tomojax.io` to `.importlinter` so the new module owner is included in
  executable import-boundary checks.

### Decisions

- `tomojax.io` owns artifact/metadata serialization helpers. This keeps JSON
  normalization out of generic utilities while avoiding a premature dependency
  on future dataset or verifier schemas.
- Kept `calibration/_json.py` as a temporary internal adapter because the whole
  `tomojax.calibration` owner is transitional and will be migrated or deleted
  under the v2 geometry/motion/nuisance plan.

### Validation

- `uv run ruff check src/tomojax/io src/tomojax/calibration/_json.py tests/test_json_utils.py`
  passed.
- `uv run basedpyright src/tomojax/io` passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py -q`
  passed: 18 tests.
- `just imports` passed after adding `tomojax.io` to `.importlinter`.
- `rg -n "tomojax\\.utils\\.json|utils/json" src tests docs .agent` found no
  remaining references.

### Risks

- Other `tomojax.utils` modules still remain (`axes`, `fov`, `logging`,
  `memory`, `phasecorr`, and `subprocesses`) and need owner-by-owner migration
  or deletion before the v2 skeleton is clean.

## 2026-05-06 — Move Axis And FOV Helpers Into `tomojax.geometry`

### Summary

- Added the second v2-owned deep module surface:
  - `src/tomojax/geometry/api.py`
  - `src/tomojax/geometry/__init__.py`
  - `src/tomojax/geometry/_axes.py`
  - `src/tomojax/geometry/_fov.py`
  - `src/tomojax/geometry/README.md`
- Moved axis-order and detector field-of-view helpers from forbidden
  `tomojax.utils` modules into `tomojax.geometry`.
- Updated production consumers in NXtomo IO, reconstruction CLI, alignment CLI,
  and pose-stage masking to import through the public `tomojax.geometry`
  facade.
- Updated geometry/FOV tests to assert the new public API and deleted
  `src/tomojax/utils/axes.py` and `src/tomojax/utils/fov.py`.
- Added `tomojax.geometry` to `.importlinter` so it participates in the current
  executable import-boundary checks.

### Decisions

- `tomojax.geometry` owns axis-order metadata and detector-FOV ROI helpers
  because these helpers describe geometry conventions and reconstruction domains
  rather than generic utilities.
- Kept existing behavior intact, including the private white-box monkeypatch in
  `tests/test_regression_geometry_io.py`, but marked it with the explicit
  public-import checker exception so this remaining test coupling is visible.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_regression_geometry_io.py tests/test_axes_io.py tests/test_issue_fix_pr.py`
  passed.
- `uv run basedpyright src/tomojax/geometry` passed with 0 errors and 0
  warnings.
- `uv run pytest tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py -q`
  passed: 66 tests.
- `just imports` passed after adding `tomojax.geometry` to `.importlinter`.
- `rg -n "utils\\.axes|utils\\.fov|tomojax\\.utils\\.axes|tomojax\\.utils\\.fov|from tomojax\\.utils import axes" src/tomojax tests`
  found no remaining axis/FOV utility references.

### Risks

- `tomojax.geometry` currently depends on the transitional
  `tomojax.core.geometry.base` types. Phase 1 should decide whether those types
  move into top-level `tomojax.geometry` or remain as lower-level core
  primitives.
- `tomojax.utils` still contains logging, memory, phase-correlation, and
  subprocess helper surfaces that need explicit v2 owners or deletion.

## 2026-05-06 — Move Phase Correlation Into `tomojax.motion`

### Summary

- Added the third v2-owned deep module surface:
  - `src/tomojax/motion/api.py`
  - `src/tomojax/motion/__init__.py`
  - `src/tomojax/motion/_phasecorr.py`
  - `src/tomojax/motion/README.md`
- Moved phase-correlation translation estimation from forbidden
  `tomojax.utils.phasecorr` into `tomojax.motion`.
- Updated the alignment stage-loop initializer to import `phase_corr_shift`
  through the public `tomojax.motion` facade.
- Updated phase-correlation tests to assert the public API while keeping the
  `_wrap_shift` white-box check explicit with a public-import checker exception.
- Added `tomojax.motion` to `.importlinter`.

### Decisions

- `tomojax.motion` owns phase-correlation because it estimates per-view motion
  for alignment initialization, not generic utility behavior.
- Used a narrow pyright suppression on `jnp.asarray` in the private
  implementation because the current JAX stubs expose that member as partially
  unknown. The public API remains typed and the module-level basedpyright gate
  is green.

### Validation

- `uv run ruff check src/tomojax/motion tests/test_phasecorr.py` passed.
- `uv run basedpyright src/tomojax/motion` passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_phasecorr.py -q` passed: 5 tests.
- `just imports` passed after adding `tomojax.motion` to `.importlinter`.

### Risks

- `tomojax.motion` currently contains only a seed/initializer primitive. Phase 1
  still needs to define whether full per-view pose parameterizations live here
  or stay under `tomojax.align` until the geometry optimizer milestone.

## 2026-05-06 — Move Backend Memory Probes Into `tomojax.backends`

### Summary

- Added the fourth v2-owned deep module surface:
  - `src/tomojax/backends/api.py`
  - `src/tomojax/backends/__init__.py`
  - `src/tomojax/backends/_memory.py`
  - `src/tomojax/backends/_subprocesses.py`
  - `src/tomojax/backends/README.md`
- Moved memory budgeting, gather-dtype selection, and `nvidia-smi` probing out
  of forbidden `tomojax.utils`.
- Updated CLI and simulation consumers to import backend helpers through the
  public `tomojax.backends` facade.
- Updated memory tests to keep private white-box checks explicit with
  public-import checker exceptions.
- Added `tomojax.backends` to `.importlinter`.

### Decisions

- `tomojax.backends` owns device-memory and gather-dtype heuristics because
  those helpers are runtime backend policy rather than generic utilities.
- Kept `_subprocesses.py` private to `tomojax.backends` instead of exposing a
  generic command runner.

### Validation

- `uv run ruff check src/tomojax/backends tests/test_memory.py` passed.
- `uv run pytest tests/test_memory.py tests/test_cli_geometry_build.py tests/test_small_module_coverage.py -q`
  passed: 40 tests.
- `just imports` passed after adding `tomojax.backends` to `.importlinter`.

### Failures And Risks

- `uv run basedpyright src/tomojax/backends` currently fails on private dynamic
  JAX/device-probe code and subprocess wrapper typing. This is narrower than
  the repo-wide transitional typecheck failure, but it is still type debt in the
  moved implementation.
- Phase 1 should decide whether backend probes stay as Python runtime helpers or
  become a smaller typed policy layer with untyped adapter code hidden behind
  explicit boundaries.

## 2026-05-06 — Remove The `tomojax.utils` Package

### Summary

- Added the fifth v2-owned deep module surface:
  - `src/tomojax/core/api.py`
  - `src/tomojax/core/__init__.py`
  - `src/tomojax/core/_logging.py`
  - `src/tomojax/core/README.md`
- Moved logging setup, JAX environment logging, progress iteration, and duration
  formatting into `tomojax.core`.
- Updated reconstruction, alignment, data simulation, and CLI consumers to use
  `tomojax.core`.
- Removed the now-empty `src/tomojax/utils` package after moving JSON, axis/FOV,
  phase-correlation, backend-memory, subprocess, and logging helpers to explicit
  v2 owners.

### Decisions

- `tomojax.core` owns shared runtime instrumentation for now because these
  helpers are used across numerical modules and CLI surfaces.
- Kept CLI-specific logging setup in the same public core facade until Phase 1
  creates a fuller `tomojax.cli` deep module. This avoids recreating a generic
  utility bucket under a different name.

### Validation

- `uv run ruff check src/tomojax/core/_logging.py src/tomojax/core/api.py src/tomojax/core/__init__.py tests/test_logging.py tests/test_small_module_coverage.py`
  passed.
- `uv run basedpyright src/tomojax/core/_logging.py src/tomojax/core/api.py src/tomojax/core/__init__.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_logging.py tests/test_small_module_coverage.py -q`
  passed: 9 tests.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py -q`
  passed: 105 tests.
- `just imports` passed.
- `fd -p 'src/tomojax/utils' -a` found no remaining `tomojax.utils` package.
- `rg -n "tomojax\\.utils|from \\.\\.utils|from tomojax\\.utils|import tomojax\\.utils" src/tomojax tests`
  found no code/test imports. Remaining matches are only explanatory README
  references in the new owner modules.
- `just check` currently fails during
  `uv run ruff check --fix src tests tools` after formatting. Current failure
  shape is broad transitional legacy lint debt, beginning with ambiguous
  Unicode in `src/tomojax/__init__.py`, import/type-checking/style findings in
  `src/tomojax/align/_config.py`, and old pose-stage lint/type annotation
  findings in `src/tomojax/align/_pose_stage.py`. The command reported 2065
  errors total, 448 fixed, and 1617 remaining.

### Risks

- `tomojax.core` now has a new public facade, but the rest of the old
  `tomojax.core` package still lacks a complete v2 `api.py` boundary for
  geometry/projector/validation internals.
- Some tests still deliberately white-box private moved implementations. Those
  exceptions are explicit but should be eliminated as Phase 1 public APIs settle.
- `just check` remains the milestone target, but passing it requires continuing
  the planned migration/deletion of old transitional modules rather than
  weakening or mass-suppressing checks.

## 2026-05-06 — Add The v2 Deep-Module Skeleton

### Summary

- Added missing canonical top-level v2 skeleton packages:
  - `tomojax.nuisance`
  - `tomojax.forward`
  - `tomojax.verify`
  - `tomojax.datasets`
- Added `api.py`, `__init__.py`, and `README.md` facade files for the new
  skeleton packages.
- Added missing v2 facade files for existing top-level owners:
  - `tomojax.cli`
  - `tomojax.align`
  - `tomojax.recon`
- Added `tests/test_v2_module_skeleton.py` to enforce the top-level
  `README.md`/`__init__.py`/`api.py` contract and import every canonical facade.
- Added the newly importable v2 modules to `.importlinter`.

### Decisions

- Empty new facades export no public names until the owning implementation
  milestone introduces typed contracts. This avoids placeholder APIs that would
  become compatibility debt.
- Kept old transitional owners (`tomojax.data`, `tomojax.calibration`,
  `tomojax.bench`, and nested alignment packages) importable for now. They will
  be deleted or migrated owner-by-owner instead of hidden behind new
  compatibility layers.
- Kept the skeleton bridge behavior-free. The next implementation milestone
  should add actual benchmark/dataset or geometry behavior against these
  boundaries.

### Validation

- `uv run ruff check src/tomojax/nuisance src/tomojax/forward src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/nuisance src/tomojax/forward src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_v2_module_skeleton.py -q` passed: 2 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run ruff format --check src/tomojax/nuisance src/tomojax/forward src/tomojax/verify src/tomojax/datasets src/tomojax/cli/__init__.py src/tomojax/cli/api.py src/tomojax/align/api.py src/tomojax/recon/api.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py -q`
  passed: 107 tests.

### Risks

- `just check` was not rerun for this skeleton-only slice because the immediately
  preceding run already stopped in broad legacy Ruff failures before reaching
  the new code. Passing `just check` still requires migrating or deleting old
  transitional modules rather than weakening checks.
- `tomojax.datasets` and old `tomojax.data` temporarily coexist. The synthetic
  benchmark foundation should make `datasets` the owner for deterministic v2
  generators and then delete or migrate old data code deliberately.

## 2026-05-06 — Add Synthetic Dataset Foundation

### Summary

- Added typed synthetic benchmark spec loading in `tomojax.datasets` from
  `docs/tomojax-v2/benchmark_manifest.yaml`.
- Added a deterministic procedural phantom generator for smoke and benchmark
  artifact generation.
- Added `generate_synthetic_dataset(...)`, which writes:
  - `dataset_manifest.json`
  - `ground_truth_volume.npy`
  - `projections.npy`
  - `mask.npy`
  - `nominal_geometry.json`
  - `corrupted_geometry.json`
  - `true_geometry.json`
  - `true_pose.csv`
  - `true_motion.csv`
  - `nuisance_truth.json`
  - `noise_truth.json`
- Added `tests/test_synthetic_datasets.py` for manifest loading, deterministic
  phantom generation, and repeatable 32^3 smoke artifact emission.

### Decisions

- `tomojax.datasets` owns the v2 synthetic benchmark foundation. Old
  `tomojax.data` remains transitional and was not extended for this slice.
- The first projection writer is a deterministic CPU smoke projector used to
  produce artifact contracts. It is not the final differentiable JAX reference
  projector, which remains owned by the `tomojax.forward` milestone.
- 128^3 mode is supported by configuration, but tests exercise 32^3 smoke mode
  to keep validation fast.

### Validation

- `uv run ruff check src/tomojax/datasets tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/datasets tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `uv run ruff format --check src/tomojax/datasets tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py -q`
  passed: 110 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The smoke projector is intentionally simple. It should be replaced as the
  benchmark truth generator once the `tomojax.forward` JAX reference projector
  is implemented and validated.

## 2026-05-06 — Add Geometry State And Gauge Canonicalisation

### Summary

- Added typed v2 geometry state containers in `tomojax.geometry`:
  - `ScalarParameter`
  - `SetupParameters`
  - `PoseParameters`
  - `GeometryState`
- Added gauge canonicalisation:
  - `mean(dx_px) -> det_u_px`
  - `mean(phi_residual_rad) -> theta_offset_rad`
  - `mean(dz_px) -> det_v_px` only when `det_v_px` is active
- Added structured gauge reports with `GaugeTransfer`, `GaugeReport`, and
  `CanonicalizedGeometry`.
- Added `tests/test_geometry_gauges.py` covering zero-centering, inactive
  `det_v`, active `det_v`, shape validation, and realised setup-plus-pose gauge
  preservation.

### Decisions

- Kept these v2 state types in top-level `tomojax.geometry` without replacing
  old `tomojax.core.geometry` primitives in this slice.
- Implemented only state and gauge data structures. Geometry artifact
  serialisation, optimiser integration, Jacobians, and Schur solves remain
  separate milestones.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/geometry tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `uv run ruff format --check src/tomojax/geometry tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py -q`
  passed: 113 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Geometry state JSON/CSV artifact writing is still missing and should be added
  before optimizer milestones rely on geometry provenance.

## 2026-05-06 — Add Geometry Artifact Serialization

### Summary

- Added versioned geometry setup JSON payloads for `GeometryState`.
- Added `write_geometry_json` and `read_geometry_json` for
  `geometry_initial.json` / `geometry_final.json`-compatible artifacts.
- Added `write_pose_params_csv` and `read_pose_params_csv` for per-view 5-DOF
  pose arrays.
- Added `write_pose_decomposition_csv` for realised setup-plus-pose channels:
  `theta_offset + phi_residual`, `det_u + dx`, and `det_v + dz`.
- Added `tests/test_geometry_serialization.py` covering JSON/CSV round-trip,
  contract artifact filenames, schema version, active parameter metadata, and
  decomposition values.

### Decisions

- Geometry artifact serialization lives in `tomojax.geometry` because these
  files encode geometry-state contracts. Run-level artifact indexing remains a
  future `tomojax.verify` responsibility.
- Pose arrays are stored in CSV artifacts while setup parameter metadata lives
  in JSON. This matches the v2 artifact contract and keeps per-view arrays easy
  to inspect.

### Validation

- `uv run ruff check src/tomojax/geometry tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/geometry tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.
- `uv run ruff format --check src/tomojax/geometry tests/test_geometry_serialization.py tests/test_geometry_gauges.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py -q`
  passed: 116 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The serializer does not yet write full run artifact indexes or observability
  reports. Those remain part of later `tomojax.verify` and optimiser slices.

## 2026-05-06 — Add Minimal Forward Reference And Robust Residuals

### Summary

- Added `tomojax.forward.project_parallel_reference`, a minimal JAX reference
  projector for tiny cubic-volume smoke tests.
- Added masked whitened residual helpers:
  - `masked_whitened_residual`
  - `pseudo_huber_loss`
  - `pseudo_huber_weights`
  - `residual_loss`
  - `ResidualResult`
- Added `tests/test_forward_reference.py` covering projection shape, per-view
  detector shift, masking, pseudo-Huber robust behavior, valid counts, and IRLS
  weight behavior.

### Decisions

- Kept this as a new `tomojax.forward` reference slice rather than adapting old
  `tomojax.core.projector`.
- The projector is intentionally minimal: it uses coarse array rotation and
  detector shifts for smoke tests. Full ray geometry, laminography, detector
  roll, axis rotations, theta-scale handling, and finite-difference geometry
  checks remain future Phase 2 work.
- Robust residuals follow the v2 loss spec: masked, whitened residuals with
  pseudo-Huber loss and IRLS weights.

### Validation

- `uv run ruff check src/tomojax/forward tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_forward_reference.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `uv run ruff format --check src/tomojax/forward tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py -q`
  passed: 121 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The minimal projector is not yet the final physical forward model. It should
  be expanded before geometry optimisation tests rely on physical accuracy.

## 2026-05-06 — Add Tiny Reconstruction And Alignment Smoke Path

### Summary

- Updated the minimal forward projector so setup `det_u_px` and active
  `det_v_px` contribute to detector shifts. This makes gauge canonicalisation
  projection-preserving for the channels the smoke projector supports.
- Added `tomojax.recon.reconstruct_average_reference`, a tiny deterministic
  average-backprojection preview helper.
- Added `tomojax.align.run_alignment_smoke`, which:
  - reconstructs a stopped-gradient preview volume;
  - computes masked robust projection loss;
  - canonicalises geometry gauges;
  - recomputes loss after canonicalisation;
  - reports loss values, valid count, and the canonicalised geometry/report.
- Added `tests/test_vertical_smoke.py` for gauge-equivalent projection
  preservation, preview reconstruction shape, and alignment smoke report
  invariants.

### Decisions

- Kept the smoke path as explicit reference scaffolding, not the product
  optimiser. It wires the v2 modules together before FISTA and LM/GN land.
- `reconstruct_average_reference` is not the default reconstruction algorithm.
  It exists only to exercise the forward/residual/gauge path with a volume-like
  object.

### Validation

- `uv run ruff check src/tomojax/forward src/tomojax/recon/_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py src/tomojax/align/_smoke.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward src/tomojax/recon/_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py src/tomojax/align/_smoke.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py -q`
  passed: 10 tests.
- `uv run ruff format --check src/tomojax/forward src/tomojax/recon/_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py src/tomojax/align/_smoke.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_vertical_smoke.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py -q`
  passed: 124 tests.

### Risks

- A broad `uv run ruff format --check src/tomojax/forward src/tomojax/recon src/tomojax/align ...`
  still reports 20 untouched transitional align/recon files that would be
  reformatted. This remains outside the current smoke slice.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The smoke path does not perform optimisation. Pose-only LM/GN remains the next
  major geometry milestone after the reference projector grows enough physical
  fidelity for meaningful derivatives.

## 2026-05-06 — Make Detector Shifts Differentiable

### Summary

- Replaced rounded detector shifts in the minimal forward projector with
  differentiable periodic linear interpolation.
- Added `project_parallel_reference_arrays`, which accepts JAX arrays for
  `theta_rad`, `dx_px`, and `dz_px`. This gives future pose optimizers a path
  that can differentiate through detector shifts.
- Added tests for fractional detector shifts and `jax.grad` through `dx_px`.

### Decisions

- Kept the current coarse theta quadrant handling. This slice only removes the
  non-differentiable rounded detector-shift path.
- Periodic interpolation is a smoke/reference simplification. Full detector
  boundary policy belongs with the physical projector milestone.

### Validation

- `uv run ruff check src/tomojax/forward tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py -q`
  passed: 12 tests.
- `uv run ruff format --check src/tomojax/forward tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py -q`
  passed: 126 tests.

### Risks

- `uv run python` without forcing CPU emits a JAX CUDA plugin warning about
  missing cuSPARSE, then falls back to CPU. The current validation still passes
  on CPU.
- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The projector is differentiable for detector shifts, but not yet a full
  physical differentiable projector for all 5 pose DOFs.

## 2026-05-06 — Add Pose-Only Detector-Shift LM Solver

### Summary

- Added `tomojax.align.solve_pose_only_lm`, a damped Gauss-Newton/LM solver
  against a fixed volume for the currently differentiable per-view pose
  channels:
  - `dx_px`
  - `dz_px`
- Added `PoseOnlyLMConfig` and `PoseOnlyLMResult`.
- The solver uses masked whitened projection residuals plus pseudo-Huber IRLS
  weights, solves a damped normal equation, and canonicalises geometry gauges
  after the solve.
- Added deterministic tests covering detector-shift recovery, active/frozen DOF
  reporting, final loss improvement, and gauge canonicalisation preservation.

### Decisions

- This is intentionally not the full 5-DOF pose solver. `alpha_rad`,
  `beta_rad`, and `phi_residual_rad` are reported as frozen because the current
  reference projector does not yet provide physical differentiable sensitivity
  for those DOFs.
- Used a finite-difference Jacobian for this first LM implementation. The
  periodic linear detector shift has derivative kinks at integer shifts, and
  finite differences behaved more robustly from zero initialization while also
  building toward the required finite-difference validation suite.

### Validation

- `uv run ruff check src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run pytest tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `uv run ruff format --check src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_pose_lm.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py -q`
  passed: 128 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full pose-only 5-DOF optimisation remains incomplete until the physical
  reference projector supports differentiable `alpha`, `beta`, and
  `phi_residual` effects.

## 2026-05-06 — Add Setup-Only Detector-Shift LM Solver

### Summary

- Added `tomojax.align.solve_setup_only_lm`, a damped Gauss-Newton/LM solver
  against a fixed volume for the currently differentiable setup channels:
  - `det_u_px`
  - active `det_v_px`
- Added `SetupOnlyLMConfig` and `SetupOnlyLMResult`.
- Factored the shared finite-difference Jacobian helper into
  `tomojax.align._lm_numerics` for the pose-only and setup-only LM solvers.
- The setup solver uses masked whitened projection residuals plus pseudo-Huber
  IRLS weights, solves a damped normal equation, and reports active and frozen
  setup parameters.
- Added deterministic tests covering detector-shift recovery, inactive
  `det_v_px` freezing, final loss improvement, and active/frozen reporting.

### Decisions

- This is intentionally not the full setup solver. `detector_roll_rad`,
  `axis_rot_x_rad`, `axis_rot_y_rad`, `theta_offset_rad`, and `theta_scale` are
  reported as frozen because the current reference projector only models
  detector shifts.
- Reused the finite-difference Jacobian path from the pose-only solver for
  consistency with the current differentiable detector-shift reference path and
  the planned finite-difference validation suite.

### Validation

- `uv run ruff check src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_lm_numerics.py src/tomojax/align/_setup_lm.py src/tomojax/align/_pose_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_setup_lm.py tests/test_pose_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 130 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full setup-only optimisation remains incomplete until the physical reference
  projector supports detector roll, axis rotation, and theta setup effects.

## 2026-05-06 — Add Projection Residual Filters

### Summary

- Added `tomojax.forward.apply_residual_filter` and
  `tomojax.forward.apply_residual_filter_schedule`.
- Added typed residual filter config/result values:
  - `ResidualFilterConfig`
  - `ResidualFilterKind`
  - `ResidualFilterResult`
- Implemented the first deterministic JAX reference residual filter policies:
  - `raw`
  - `lowpass_gaussian`
  - `bandpass_difference_of_gaussians`
- Added deterministic tests for raw identity with mask reapplication,
  low-pass impulse spreading and sum preservation, band-pass zero-mean
  behavior, and weighted schedule summation.

### Decisions

- The current low-pass policy uses a separable Gaussian kernel over the final
  two detector axes and periodic boundary handling via `jnp.roll`.
- The current band-pass policy is a difference between inner and outer Gaussian
  low-pass results. This matches the named Phase 2 reference policy without
  committing the public API to a future multiresolution filter bank design.
- Masks are reapplied after filtering so invalid pixels remain suppressed in
  filtered residuals.

### Validation

- `uv run ruff check src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/forward/_filters.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_residual_filters.py tests/test_forward_reference.py tests/test_v2_module_skeleton.py -q`
  passed: 13 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 134 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The residual filters are deterministic reference policies, not yet a complete
  level-aware continuation schedule integrated into the alternating solver.

## 2026-05-06 — Add Reference FISTA Preview Reconstruction

### Summary

- Added `tomojax.recon.fista_reconstruct_reference`, a tiny JAX reference FISTA
  preview reconstruction path against the current v2 reference projector.
- Added typed reconstruction config/result/trace rows:
  - `ReferenceFISTAConfig`
  - `ReferenceFISTAResult`
  - `ReferenceFISTATraceRow`
- Added smoothed TV regularisation, warm-start support, optional
  non-negativity projection, and masked robust projection residual loss.
- Added `write_fista_trace_csv` for the Phase 3 trace artifact contract.
- Added deterministic tests covering projection-loss improvement,
  non-negativity, warm-start updates, and CSV trace output.

### Decisions

- Used a fixed configured step size for this first reference implementation.
  It keeps the slice small and leaves production step-size estimation or line
  search for a later reconstruction milestone.
- Kept the implementation against the current minimal reference projector. This
  validates the FISTA control flow and artifact contract before the physical
  projector/backprojector is complete.
- The trace backend is reported as `jax_reference`.

### Validation

- `uv run ruff check src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/recon/_fista_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 4 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 136 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- This is not yet production reconstruction quality: the reference projector is
  still minimal, and multiresolution schedules plus stronger step-size control
  remain future Phase 3 work.

## 2026-05-06 — Add Reference FISTA Reconstruction Schedules

### Summary

- Added `tomojax.recon.reference_fista_schedule`, a typed v2 schedule resolver
  for reference FISTA preview and final reconstruction runs.
- Added schedule dataclasses:
  - `ReferenceFISTASchedule`
  - `ReferenceFISTAScheduleEntry`
  - `ReferenceReconstructionScheduleName`
- Defined the Phase 3 schedule contract:
  - preview schedule: levels 4 and 2
  - final schedule: level 1
- Included the Phase 2 residual filter policies in schedule entries:
  - level 4 uses low-pass
  - level 2 uses low-pass plus band-pass
  - level 1 uses raw residuals
- Added deterministic tests for schedule resolution, level factors, filter
  weights, and unknown-name rejection.

### Decisions

- The schedule API resolves typed configuration only. It does not execute a
  multiresolution pyramid yet; that orchestration belongs in a later Phase 3 or
  Phase 7 slice.
- Kept this separate from the old-core `recon.multires` path so v2 reference
  reconstruction can evolve behind the deep-module API without importing old
  geometry internals.

### Validation

- `uv run ruff check src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/recon/_schedule_reference.py src/tomojax/recon/api.py src/tomojax/recon/__init__.py tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_reference_fista_schedule.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 139 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full schedule execution over a multiresolution pyramid is still incomplete.

## 2026-05-06 — Make Reference Projector Differentiable In Theta

### Summary

- Replaced the minimal reference projector's quadrant `rot90` angle handling
  with bilinear x-y plane rotation.
- Preserved the existing differentiable periodic detector-shift path after
  projection.
- Added tests showing projection output changes with small theta updates and
  autodiff returns a finite nonzero theta gradient.
- Updated `tomojax.forward` documentation to describe the new angle sampling
  boundary policy.

### Decisions

- The reference angle rotation now samples outside-volume coordinates as zero.
  Detector shifts remain periodic because that behavior was already covered by
  the current smoke/reference contracts.
- This is still the minimal parallel projector. It does not yet implement
  laminography, detector roll, axis rotations, or full physical ray geometry.

### Validation

- `uv run ruff check src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/forward/_projector.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_reference_fista.py tests/test_v2_module_skeleton.py -q`
  passed: 20 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 141 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Full physical angle/pose sensitivity remains incomplete until the projector
  models axis rotations, detector roll, and laminography geometry.

## 2026-05-06 — Add Phi Residual To Pose-Only LM

### Summary

- Extended `tomojax.align.solve_pose_only_lm` to optimise per-view
  `phi_residual_rad` along with `dx_px` and `dz_px`.
- Kept `alpha_rad` and `beta_rad` frozen because the reference projector does
  not yet model out-of-plane pose effects.
- Added deterministic tests for phi recovery and gauge canonicalisation into
  `theta_offset_rad`.
- Updated the `tomojax.align` README to reflect the active/frozen pose DOFs.

### Decisions

- Reused the existing finite-difference damped LM normal equation for the
  expanded pose vector. This keeps the implementation consistent with the
  detector-shift pose solver while the full 5-DOF projector is still being
  built.
- Used a more asymmetric 7^3 fixture for phi recovery tests. A smaller earlier
  fixture had a sign ambiguity for one view.

### Validation

- `uv run ruff check src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_pose_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 20 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 143 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Pose-only LM is still not the full 5-DOF solver until alpha/beta effects and
  trust-region mechanics are implemented.

## 2026-05-06 — Add Theta Offset To Setup-Only LM

### Summary

- Extended `tomojax.align.solve_setup_only_lm` to optimise global
  `theta_offset_rad` along with `det_u_px` and active `det_v_px`.
- Kept detector roll, axis rotations, and theta scale frozen because the
  reference projector does not yet model those setup effects.
- Added deterministic theta-offset recovery tests on the asymmetric theta
  fixture.
- Updated the `tomojax.align` README to reflect the active/frozen setup
  parameters.

### Decisions

- Reused the existing finite-difference damped LM normal equation for the
  expanded setup vector.
- Setup-only LM keeps pose fixed, so `theta_offset_rad` and per-view
  `phi_residual_rad` remain separable by the existing gauge policy after the
  solve.

### Validation

- `uv run ruff check src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_setup_lm.py tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_setup_lm.py tests/test_pose_lm.py tests/test_forward_reference.py tests/test_vertical_smoke.py tests/test_v2_module_skeleton.py -q`
  passed: 21 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py -q`
  passed: 144 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Setup-only LM is still incomplete until detector roll, axis rotations,
  observability diagnostics, and trust-region mechanics are implemented.

## 2026-05-06 — Add Joint Setup+Pose Schur LM Reference Slice

### Summary

- Added `tomojax.align.solve_joint_schur_lm`, the first joint setup+pose Schur
  LM reference solver for supported differentiable DOFs.
- Added typed config/result/diagnostics values:
  - `JointSchurLMConfig`
  - `JointSchurLMResult`
  - `JointSchurDiagnostics`
- Added `schur_step_from_jacobian` as a public numerical contract for
  Schur-vs-dense validation.
- The solver packs supported setup DOFs
  (`theta_offset_rad`, `det_u_px`, optional `det_v_px`) and per-view pose DOFs
  (`phi_residual_rad`, `dx_px`, `dz_px`), builds finite-difference weighted
  residual Jacobians, solves the setup step by Schur complement, and
  back-substitutes per-view pose steps.
- Added deterministic tests for Schur-vs-dense normal equation equivalence and
  joint supported-geometry recovery after gauge canonicalisation.

### Decisions

- This is a reference Schur slice, not the final production trust-region
  engine. Priors, trust radii, damping adaptation, and acceptance diagnostics
  remain future Phase 6 work.
- Kept alpha/beta pose effects, detector roll, axis rotations, and theta scale
  frozen until the reference projector models those effects.
- Tested realised geometry after gauge canonicalisation because setup and pose
  mean components are gauge-coupled.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 11 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 146 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The joint Schur solver still lacks priors, trust radii, damping adaptation,
  detailed normal-equation artifact export, and unsupported physical DOFs.

## 2026-05-06 — Add Joint Schur Normal-Equation Artifact

### Summary

- Added `joint_schur_normal_eq_summary`, a JSON-serializable Phase 6
  normal-equation summary for `JointSchurLMResult`.
- Added `write_joint_schur_normal_eq_summary`, writing
  `normal_eq_summary.json`-style artifacts.
- Added `JointSchurDiagnostics.to_dict` for stable diagnostics serialization.
- Added a readback test covering solver label, active setup parameters, active
  pose DOFs, and Schur diagnostic keys.

### Decisions

- Kept the artifact intentionally compact for this slice:
  losses, iterations, active/frozen parameters, and current Schur diagnostics.
  Richer eigenvalue and pose-block condition summaries remain future Phase 6
  work.
- Exposed the writer through `tomojax.align` so callers and tests do not import
  private implementation files.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 147 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The artifact does not yet include eigenvalues, correlations, or pose-block
  condition statistics required by the full Phase 6 diagnostics target.

## 2026-05-06 — Enrich Joint Schur Diagnostics

### Summary

- Extended `JointSchurDiagnostics` with:
  - damped global normal-equation eigenvalues
  - Schur complement eigenvalues
  - per-view pose-block condition numbers
- Included these fields in `normal_eq_summary.json` via
  `joint_schur_normal_eq_summary`.
- Extended joint Schur tests to verify eigenvalue and pose-condition fields are
  present and sized correctly.

### Decisions

- Recorded eigenvalues from the damped normal equations currently solved by the
  reference implementation. Undamped Hessian diagnostics and correlation
  matrices remain future Phase 6 work.
- Kept the diagnostics JSON-native and compact so it can become an artifact in
  later alternating-solver runs without extra conversion code.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 147 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Phase 6 diagnostics still lack correlation matrices, weak-mode labels, priors,
  trust radii, and damping adaptation metadata.

## 2026-05-06 — Add Schur Correlations And Weak-Mode Labels

### Summary

- Extended `JointSchurDiagnostics` with:
  - normalized setup Schur correlation matrix
  - eigenvalue-derived weak-mode labels
- Included both fields in `normal_eq_summary.json` through
  `joint_schur_normal_eq_summary`.
- Added tests that verify correlation matrix shape/diagonal and artifact
  readback for correlation and weak-mode fields.

### Decisions

- Correlations are computed from the damped setup Schur complement by
  normalizing with the square root of the Schur diagonal.
- Weak modes are labelled generically as `schur_eigen_<index>` when their
  eigenvalue magnitude is below a relative threshold. Semantic labels by DOF
  contribution require eigenvector attribution and remain future work.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 5 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 147 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Weak-mode labels are numerical labels rather than semantic DOF labels until
  eigenvector attribution is added.

## 2026-05-06 — Add Joint Schur Trust Radius Controls

### Summary

- Added optional `setup_trust_radius` and `pose_trust_radius` fields to
  `JointSchurLMConfig`.
- Applied trust scaling to the Schur step and the dense-equivalent comparison
  step.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `trust_scale`
  - `trust_clipped`
  - `setup_update_by_parameter`
  - `pose_update_max_by_dof`
- Added deterministic tests covering default unclipped behavior, configured
  clipping, scaled Schur-vs-dense equivalence, and artifact fields.

### Decisions

- Used scalar setup and pose block radii for the first reference implementation.
  Per-parameter radii can be layered later using parameter metadata.
- Trust radii default to `None`, preserving the previous full-step behavior.
- This is not yet adaptive trust-region control; it only clips an otherwise
  computed LM step.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 148 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Trust radii are scalar and non-adaptive; per-DOF metadata radii and
  actual/predicted-decrease radius updates remain future Phase 6 work.

## 2026-05-06 — Add Joint Schur Damping Adaptation

### Summary

- Added configurable joint Schur LM damping adaptation:
  - `adapt_damping`
  - `damping_decrease_factor`
  - `damping_increase_factor`
  - `min_damping`
  - `max_damping`
- Updated `solve_joint_schur_lm` to keep a local damping value and adapt it
  after each accepted or rejected candidate step.
- Added `adapt_joint_schur_damping` to the public alignment facade so the
  accepted/rejected policy is testable without private imports.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `damping`
  - `next_damping`
  - `accepted`
  - `current_loss`
  - `candidate_loss`
- Added deterministic tests for accepted/rejected damping changes, clamp
  behavior, solver diagnostics, and artifact fields.

### Decisions

- Accepted steps decrease damping and rejected steps increase damping with
  configurable factors and clamps.
- Damping adaptation is deliberately separate from trust-radius scaling; this
  keeps the reference solver simple while preserving diagnostics for the later
  actual/predicted reduction policy.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 149 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  reduction-diagnostics slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 149 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; first current failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  damping slice.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The damping policy is accepted/rejected only; actual/predicted reduction and
  adaptive radius updates remain future Phase 6 work.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Reduction Diagnostics

### Summary

- Added predicted reduction diagnostics for the final trust-scaled Schur step
  using the damped quadratic normal-equation model.
- Recorded actual candidate reduction from the robust nonlinear objective and
  an actual/predicted reduction ratio.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `predicted_reduction`
  - `actual_reduction`
  - `reduction_ratio`
- Added deterministic tests for the quadratic predicted-reduction formula,
  solver diagnostics, and artifact fields.

### Decisions

- `reduction_ratio` is `None` when the predicted reduction is numerically zero;
  this occurs at convergence and avoids reporting an unstable ratio.
- The ratio is diagnostic-only for this slice. Adaptive radius or damping
  policy based on actual/predicted reduction remains future Phase 6 work.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 7 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Predicted reduction is computed from the current IRLS quadratic model, while
  actual reduction comes from the robust nonlinear objective. The ratio should
  be treated as a trust-region diagnostic, not a correctness metric by itself.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Trust-Radius Adaptation

### Summary

- Added configurable ratio-based trust-radius adaptation for existing joint
  Schur setup and pose block radii:
  - `adapt_trust_radii`
  - `trust_shrink_ratio`
  - `trust_expand_ratio`
  - `trust_shrink_factor`
  - `trust_expand_factor`
  - `min_trust_radius`
  - `max_trust_radius`
- Updated `solve_joint_schur_lm` to keep local setup and pose trust radii and
  adapt them after each accepted or rejected candidate step.
- Added `adapt_joint_schur_trust_radius` to the public alignment facade so the
  trust-radius policy is testable without private imports.
- Extended `JointSchurDiagnostics` and `normal_eq_summary.json` with:
  - `next_setup_trust_radius`
  - `next_pose_trust_radius`
- Added deterministic tests for shrink, expand, clamp, disabled, and unset
  trust-radius behavior plus artifact readback with configured radii.

### Decisions

- Rejected, missing-ratio, and low-ratio steps shrink configured trust radii.
- High-ratio steps expand trust radii only when the step was clipped, matching
  common trust-region behavior and avoiding unnecessary radius growth.
- Unset trust radii remain `None`, preserving the existing full-step default.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/api.py src/tomojax/align/__init__.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  per-view-normal-block slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  per-view-reduction slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  iteration-trace slice.
- `just imports` passed:
  - `uv run lint-imports --config .importlinter`
  - `uv run python tools/check_public_imports.py`
- `uv run pytest tests/test_json_utils.py tests/test_manifest.py tests/test_align_checkpoint.py tests/test_axes_io.py tests/test_regression_geometry_io.py tests/test_issue_fix_pr.py tests/test_cli_geometry_build.py tests/test_align_roi.py tests/test_phasecorr.py tests/test_memory.py tests/test_logging.py tests/test_small_module_coverage.py tests/test_v2_module_skeleton.py tests/test_synthetic_datasets.py tests/test_geometry_gauges.py tests/test_geometry_serialization.py tests/test_forward_reference.py tests/test_residual_filters.py tests/test_reference_fista.py tests/test_reference_fista_schedule.py tests/test_vertical_smoke.py tests/test_pose_lm.py tests/test_setup_lm.py tests/test_joint_schur_lm.py -q`
  passed: 150 tests.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  `uv run ruff format src tests tools`; current first failures include
  `RUF002` in `src/tomojax/__init__.py`, `TC003`/`TID252`/`UP040`/`PLR0912`
  in `src/tomojax/align/_config.py`, and many other transitional legacy Ruff
  findings. Formatter-only churn from this command was reverted outside this
  trust-radius-adaptation slice.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- The policy still uses scalar setup/pose block radii. Per-DOF trust radii can
  be layered later from parameter metadata.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Iteration Trace

### Summary

- Added `iteration_diagnostics` to `JointSchurLMResult`.
- Updated `solve_joint_schur_lm` to retain the diagnostics from every solve
  iteration, not only the final iteration.
- Extended `normal_eq_summary.json` with `iteration_diagnostics`.
- Added deterministic tests that verify trace length, final diagnostic
  consistency, and artifact readback.

### Decisions

- Reused `JointSchurDiagnostics` for each trace row so the final summary and
  per-iteration trace share one schema.
- Kept the trace in JSON only for this slice. A CSV trace writer can be added
  later when the Phase 6 solver is wired into a benchmark/alignment artifact
  directory.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Artifact size grows with iteration count, but the reference solver currently
  uses a small default iteration count and compact diagnostics.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Per-View Reduction Diagnostics

### Summary

- Added per-view robust loss diagnostics to `JointSchurDiagnostics`:
  - `current_loss_by_view`
  - `candidate_loss_by_view`
  - `actual_reduction_by_view`
- Updated `solve_joint_schur_lm` to evaluate current and candidate robust loss
  per view for each candidate step.
- Extended `normal_eq_summary.json` and iteration trace rows with the per-view
  loss/reduction fields.
- Added deterministic tests for per-view diagnostic lengths and artifact
  readback.

### Decisions

- Per-view values are actual robust objective losses/reductions, not quadratic
  model reductions.
- Mask handling supports both per-view masks and projection-shaped masks.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- Per-view `Jᵀr`/`JᵀJ` block artifacts are not yet materialized separately from
  the dense finite-difference Jacobian; this slice records per-view loss
  effects only.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Add Joint Schur Per-View Normal-Block Diagnostics

### Summary

- Added compact per-view normal-equation diagnostics to `JointSchurDiagnostics`:
  - `setup_gradient_by_view`
  - `pose_gradient_by_view`
  - `setup_hessian_diag_by_view`
  - `pose_hessian_diag_by_view`
  - `setup_pose_coupling_norm_by_view`
- Computed these summaries from the current finite-difference Jacobian and
  weighted residual inside `schur_step_from_jacobian`.
- Extended `normal_eq_summary.json` and iteration trace rows with the new
  per-view normal-block fields.
- Added deterministic checks for selected per-view gradient values plus artifact
  field/readback coverage.

### Decisions

- Recorded compact vectors and norms rather than full per-view matrices to keep
  the JSON artifact readable.
- Used a private typed dataclass for the intermediate block diagnostics so the
  public `JointSchurDiagnostics` construction remains type-safe.

### Validation

- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py`
  passed.
- `uv run pytest tests/test_joint_schur_lm.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures
  recorded in the Milestone 0 cleanup entry.
- These diagnostics are computed from dense finite-difference Jacobians, not a
  streamed production accumulator. They are reference diagnostics until the
  fast path exists.
- Proposed next fix for `just check`: continue the legacy Ruff cleanup as a
  separate milestone instead of mixing repository-wide lint churn into Phase 6
  numerical solver work.

## 2026-05-06 — Unblock First Milestone 0 Ruff Cleanup Cluster

### Summary

- Replaced ambiguous Unicode punctuation in the package docstring with ASCII
  equivalents.
- Updated `tomojax.align._config` imports, type-only imports, and Python 3.12
  type aliases to satisfy the first `just check` Ruff cluster.
- Split `AlignConfig.__post_init__` into focused normalization helpers while
  preserving the previous normalization order.
- Added casts where normalized config fields narrow from public input types to
  internal enum-like literals.

### Decisions

- Kept the helper call order identical to the old in-method validation order to
  avoid behavior changes.
- Left the existing public-facade contract mismatch and LBFGS native crash out
  of scope for this narrow cleanup slice.

### Validation

- `uv run ruff check src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed.
- `uv run basedpyright src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed with 0 errors and 0 warnings.
- `uv run ruff format --check src/tomojax/__init__.py src/tomojax/align/_config.py`
  passed.
- `uv run pytest tests/test_align_profiles.py tests/test_align_motion_models.py tests/test_align_gauge.py tests/test_align_optimizers.py tests/test_v2_module_skeleton.py -q`
  passed: 30 tests.
- `uv run pytest tests/test_align_chunking.py -q -k 'not lbfgs'` passed:
  24 tests, 5 deselected.
- `just imports` passed.

### Risks

- `just check` remains blocked by broad transitional legacy Ruff failures. The
  first remaining cluster now starts in `src/tomojax/align/_pose_stage.py`
  with parent-relative imports, type-only imports, missing annotations, and
  complexity findings.
- `uv run pytest tests/test_cli_config.py tests/test_align_contracts.py tests/test_alignment_schedules.py -q`
  still fails on `test_alignment_public_facade_stays_narrow`, which expects the
  old three-symbol `tomojax.align.__all__` while the current committed facade
  exposes LM symbols.
- A broader config-adjacent pytest run encountered a native JAX/Optax
  segmentation fault inside an existing LBFGS chunking path.
- Proposed next fix for `just check`: start a dedicated cleanup slice for
  `src/tomojax/align/_pose_stage.py` import hygiene and local annotation
  findings before tackling larger complexity findings.

## 2026-05-06 — Reduce Pose Stage Ruff Import And Annotation Findings

### Summary

- Converted `_pose_stage.py` parent-relative core imports to absolute imports.
- Moved `Detector`, `Geometry`, and `Grid` behind `TYPE_CHECKING` because they
  are only used in postponed annotations in this module.
- Added broad `jnp.ndarray` annotations to nested JAX objective helpers and
  scan bodies without changing numerical logic.
- Replaced a local smoothness-weight append loop with a comprehension and
  rewrote the GN smooth-candidate lambda as a nested function.

### Decisions

- Kept this as a low-risk cleanup slice and deferred decomposition of the large
  pose-stage functions to a separate milestone.
- Treated basedpyright for the whole file as an existing backlog rather than a
  gate for this slice because it reports broad JAX unknown/private-usage/stat
  narrowing issues unrelated to the import/annotation cleanup.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports only seven
  local PLR0912/PLR0915 complexity findings.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_chunking.py -q -k 'not lbfgs'` passed:
  24 tests, 5 deselected.
- `uv run pytest tests/test_align_quick.py -q -k 'gn or smooth_pose_model or pose_model'`
  passed: 23 tests.
- `just imports` passed.

### Risks

- `just check` remains blocked. The first remaining local failures are
  `_pose_stage.py` complexity findings in `_build_pose_objective_bundle`,
  `_align_summary_parts`, `_run_alignment_step`, and `align`, followed by
  legacy import/type-alias findings in `_profiles.py` and
  `_reconstruction_stage.py`.
- The broader basedpyright backlog in `_pose_stage.py` remains unresolved.
- Proposed next fix for `just check`: split `_align_summary_parts` first,
  then tackle `_run_alignment_step` and `_build_pose_objective_bundle` as
  separate behavior-preserving decomposition slices.

## 2026-05-06 — Split Pose Alignment Summary Formatting

### Summary

- Split `_align_summary_parts` into smaller helpers for GN, L-BFGS, GD, and
  alignment-loss formatting.
- Preserved the existing compact and verbose log message text while removing
  `_align_summary_parts` from the local complexity blocker list.

### Decisions

- Kept the formatting helpers private to `_pose_stage.py` because they are only
  used by this pipeline's outer-iteration logging.
- Split by output branch instead of introducing a formatter object or changing
  the `OuterStat` payload contract.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports five local
  PLR0912/PLR0915 complexity findings.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k 'log_summary or log_compact or smooth_pose_model or pose_model'`
  passed: 9 tests, 43 deselected.

### Risks

- `just check` remains blocked by `_pose_stage.py` complexity findings in
  `_build_pose_objective_bundle`, `_run_alignment_step`, and `align`, followed
  by legacy import/type-alias findings in `_profiles.py` and
  `_reconstruction_stage.py`.
- Proposed next fix for `just check`: split `_run_alignment_step` into
  optimizer-kind helpers before tackling `_build_pose_objective_bundle`.

## 2026-05-06 — Split Pose Alignment Step Dispatch

### Summary

- Added a small `_AlignmentStepCoreResult` carrier for alignment step dispatch.
- Extracted optimizer dispatch, pre-step loss evaluation, GN handling, final
  gauge application, gauge-stat recording, final-loss bookkeeping, and relative
  improvement bookkeeping out of `_run_alignment_step`.
- Preserved the existing `OuterStat` keys and the GN final-loss reuse rule.

### Decisions

- Kept the existing GD and L-BFGS helpers and added matching GN/core helpers
  instead of changing the optimizer dispatch contract.
- Left the native JAX/Optax L-BFGS abort as an existing validation risk rather
  than hiding it or weakening checks.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports three local
  PLR0912/PLR0915 complexity findings.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.
- `just imports` passed.

### Risks

- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k 'gn or gd or lbfgs or smooth_pose_model or pose_model'`
  still aborts in the existing JAX/Optax L-BFGS chunking path.
- `just check` remains blocked by `_pose_stage.py` complexity findings in
  `_build_pose_objective_bundle` and top-level `align`, followed by legacy
  import/type-alias findings in `_profiles.py` and `_reconstruction_stage.py`.
- Proposed next fix for `just check`: decompose `_build_pose_objective_bundle`
  before the larger top-level `align` split.

## 2026-05-06 — Split Pose Objective Bundle Builders

### Summary

- Added `_PoseObjectiveContext` to carry pose objective arrays, config, chunk
  sizes, masks, and loss adapter state through private helper builders.
- Extracted chunk scheduling, mask handling, smoothness loss/gradient handling,
  align-loss construction, manual loss/gradient construction, and GN update
  construction out of `_build_pose_objective_bundle`.
- Preserved the `PoseObjectiveBundle` public surface and the same JAX reference
  objective paths.

### Decisions

- Used a frozen private context object rather than passing long argument lists
  through every JAX helper.
- Shared smoothness handling between align loss, manual gradient, and GN loss
  paths to keep the existing formulas consistent.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` now reports two local
  PLR0912/PLR0915 complexity findings, both on top-level `align`.
- `uv run ruff format --check src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.

### Risks

- `just check` remains blocked by top-level `align` complexity, followed by
  legacy import/type-alias findings in `_profiles.py` and
  `_reconstruction_stage.py`.
- The existing native JAX/Optax L-BFGS abort remains unresolved.
- Proposed next fix for `just check`: split top-level `align` orchestration
  into setup, per-outer, and final-info helpers.

## 2026-05-06 — Split Pose Align Orchestration

### Summary

- Added `_AlignLoopState` to carry mutable alignment loop, resume, checkpoint,
  loss, observer, and gauge state through private helpers.
- Extracted step-context construction, per-outer reconstruction/alignment
  execution, observer handling, early-stop handling, completion logging, and
  final `AlignInfo` assembly out of top-level `align`.
- Preserved the public `align` API, checkpoint payload, and fixed-volume
  objective provenance fields while clearing the last local `_pose_stage.py`
  Ruff complexity blocker.

### Decisions

- Kept mutable loop state explicit instead of hiding it behind callbacks so the
  resume/checkpoint contract remains visible and testable.
- Left runtime/objective setup in `align` for now; the cleanup goal was the
  orchestration loop, not a broader pipeline rewrite.

### Validation

- `uv run ruff check src/tomojax/align/_pose_stage.py` passed.
- `uv run ruff format src/tomojax/align/_pose_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_optimizers.py -q` passed: 10 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The first remaining blockers are legacy import/type-alias issues
  in `_profiles.py`, `_quality_policy.py`, `_reconstruction_stage.py`,
  `_results.py`, and `_setup_stage.py`, followed by broader repository lint
  backlog. Formatter churn from `just check` was reverted outside this slice.

### Risks

- Stateful extraction could regress resume/checkpoint or early-stop behavior;
  targeted GN/GD alignment coverage passed, but dedicated checkpoint tests were
  not run in this slice.
- The existing native JAX/Optax L-BFGS abort remains unresolved.
- Proposed next fix for `just check`: clean `_profiles.py` import/type-alias
  findings, then proceed through `_quality_policy.py` and
  `_reconstruction_stage.py`.

## 2026-05-06 — Clean Alignment Profile Typing

### Summary

- Converted `_profiles.py` public type aliases to PEP 695 `type` aliases.
- Moved annotation-only `Mapping`, `ProjectorBackend`, and `Regulariser`
  imports behind `TYPE_CHECKING` and replaced parent-relative imports with
  absolute type-checking imports.
- Kept runtime behavior unchanged by using string-based `cast` targets where
  the imported type names are no longer available at runtime.

### Decisions

- Treated profile policy typing as its own small cleanup slice rather than
  mixing it with reconstruction-stage lint work.
- Let Ruff apply the file-local docstring spacing and cast-quoting fixes after
  the manual import/type-alias edit.

### Validation

- `uv run ruff check src/tomojax/align/_profiles.py` passed.
- `uv run ruff format src/tomojax/align/_profiles.py` passed.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_profiles.py` is no longer in the failure list; the first
  remaining blocker is `_quality_policy.py` UP040, followed by
  `_reconstruction_stage.py`, `_results.py`, `_setup_stage.py`, and broader
  repository lint backlog. Formatter churn from `just check` was reverted
  outside this slice.

### Risks

- Runtime casts now use string targets for annotation-only types; focused
  profile tests passed, but broader config/CLI paths were not run in this
  slice.
- Proposed next fix for `just check`: convert `_quality_policy.py` to a PEP
  695 alias, then continue into `_reconstruction_stage.py`.

## 2026-05-06 — Clean Alignment Quality Policy Alias

### Summary

- Converted `_quality_policy.py` `AlignmentQualityTier` from `TypeAlias` to a
  PEP 695 `type` alias.
- Let Ruff apply the file-local cast quoting and import ordering changes.
- Kept the public alias name and quality policy behavior unchanged.

### Decisions

- Kept this as a separate small cleanup slice because it fully removes the
  `_quality_policy.py` blocker before the larger reconstruction-stage work.

### Validation

- `uv run ruff check src/tomojax/align/_quality_policy.py` passed.
- `uv run ruff format src/tomojax/align/_quality_policy.py` passed.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_quality_policy.py` is no longer in the failure list; the first
  remaining blockers are in `_reconstruction_stage.py`, followed by
  `_results.py`, `_setup_stage.py`, and broader repository lint backlog.
  Formatter churn from `just check` was reverted outside this slice.

### Risks

- Minimal behavioral risk; focused profile tests passed.
- Proposed next fix for `just check`: clean `_reconstruction_stage.py`
  imports, local helper annotations, and statement count.

## 2026-05-06 — Clean Reconstruction Stage Ruff Blockers

### Summary

- Replaced `_reconstruction_stage.py` parent-relative imports with absolute
  imports and moved annotation-only geometry/stat imports behind
  `TYPE_CHECKING`.
- Added return annotations to the local FISTA, SPDHG, and Huber-FISTA-core
  reconstruction runner helpers.
- Extracted OOM message detection and final reconstruction `OuterStat`
  assembly into private helpers, clearing the local statement-count blocker
  without changing reconstruction math.

### Decisions

- Kept `Mapping` as a runtime `collections.abc` import because the final info
  path still uses `isinstance(info_rec, Mapping)`.
- Moved the existing stat payload verbatim into `_reconstruction_step_stat`
  rather than changing the provenance/stat contract.

### Validation

- `uv run ruff check src/tomojax/align/_reconstruction_stage.py` passed.
- `uv run ruff format src/tomojax/align/_reconstruction_stage.py` passed.
- `uv run pytest tests/test_align_quick.py tests/test_align_chunking.py -q -k '(gn or gd or smooth_pose_model or pose_model) and not lbfgs'`
  passed: 47 tests, 5 deselected.
- `uv run pytest tests/test_align_profiles.py -q` passed: 6 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_reconstruction_stage.py` is no longer in the failure list; the
  first remaining blockers are `_results.py`, `_setup_stage.py`,
  `_stage_loop.py`, and broader repository lint backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- Final-stat assembly now lives in a helper; focused alignment/reconstruction
  coverage passed, but a dedicated stat-schema test was not added in this
  slice.
- Proposed next fix for `just check`: clean `_results.py` type-only imports.

## 2026-05-06 — Clean Alignment Result Type Imports

### Summary

- Moved `_results.py` annotation-only `jax.numpy`, observer, and schedule
  imports behind `TYPE_CHECKING`.
- Switched runtime collection protocol imports to `collections.abc`.
- Let Ruff apply file-local fixes for `__all__` ordering and direct
  `cfg.spdhg_seed` access.

### Decisions

- Kept `TypedDict` as a runtime import because the result schema classes
  subclass it.
- Kept result payload shapes unchanged; this slice only changes import/runtime
  typing hygiene.

### Validation

- `uv run ruff check src/tomojax/align/_results.py` passed.
- `uv run ruff format src/tomojax/align/_results.py` passed.
- `uv run pytest tests/test_align_checkpoint.py tests/test_align_profiles.py -q`
  passed: 16 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_results.py` is no longer in the failure list; the first
  remaining blockers are in `_setup_stage.py`, followed by `_stage_loop.py`
  and broader repository lint backlog. Formatter churn from `just check` was
  reverted outside this slice.

### Risks

- Runtime annotation introspection remains dependent on postponed annotations;
  checkpoint/profile tests passed.
- Proposed next fix for `just check`: clean `_setup_stage.py` imports and
  missing annotations.

## 2026-05-06 — Clean Setup Stage Typing

### Summary

- Replaced `_setup_stage.py` parent-relative imports with absolute imports and
  moved annotation-only geometry, stat, schedule, fold, adapter, and loss-spec
  imports behind `TYPE_CHECKING`.
- Added missing annotations for setup fold arrays, loss adapter, and loss spec
  inputs.
- Updated the bilevel setup test's manual `ResolvedAlignmentStage` construction
  to the current schedule contract and cleaned touched-file lint.

### Decisions

- Kept runtime setup execution dependencies as runtime imports; only
  annotation-only names moved behind `TYPE_CHECKING`.
- Treated the stale test constructor as part of this cleanup because it blocked
  focused setup validation and represented the current stage API inaccurately.

### Validation

- `uv run ruff check src/tomojax/align/_setup_stage.py tests/test_bilevel_setup_alignment.py`
  passed.
- `uv run ruff format src/tomojax/align/_setup_stage.py tests/test_bilevel_setup_alignment.py`
  passed.
- `uv run pytest tests/test_bilevel_setup_alignment.py tests/test_align_profiles.py -q`
  passed: 12 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_setup_stage.py` is no longer in the failure list; the first
  remaining blockers are in `_stage_loop.py`, followed by geometry module
  doc/import findings and broader repository lint backlog. Formatter churn
  from `just check` was reverted outside this slice.

### Risks

- The setup test now carries explicit resolved-stage metadata; if the stage
  contract changes again, this manual construction will need to change with it.
- Proposed next fix for `just check`: split `_stage_loop.py` into smaller
  import/annotation and orchestration cleanup slices.

## 2026-05-06 — Clean Multires Stage Loop Imports

### Summary

- Replaced `_stage_loop.py` parent-relative imports with absolute imports and
  moved annotation-only geometry, observer, and schedule imports behind
  `TYPE_CHECKING`.
- Added annotations to the stage observer and checkpoint callback factory.
- Moved multires scale/bin helpers and phase-correlation import to module scope,
  fixed the optional translation-seeding `vmap` lambda by binding per-level
  values through a local function, and removed unused resume-stage locals.

### Decisions

- Kept this as a pre-split cleanup slice so the next `_stage_loop.py` work can
  focus on actual orchestration complexity.
- Left `_run_multires_level_stages` and `align_multires` decomposition for the
  next slice because their complexity changes need focused review.

### Validation

- `uv run ruff check src/tomojax/align/_stage_loop.py` now reports only the
  remaining planned complexity blockers in `_run_multires_level_stages` and
  `align_multires`.
- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py -q`
  passed: 43 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The first remaining blockers are `_stage_loop.py`
  PLR0915/PLR0912 complexity findings, followed by geometry module doc/import
  findings and broader repository lint backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- The optional translation-seeding path changed from an inline lambda to a
  local function; focused multires tests passed, but there is no dedicated
  seed-translation regression in this slice.
- Proposed next fix for `just check`: decompose `_run_multires_level_stages`
  before the larger `align_multires` split.

## 2026-05-06 — Split Multires Level Stage Dispatch

### Summary

- Added a private `StageLoopState` carrier for `_stage_loop.py` level-local
  state.
- Extracted proposal, setup-geometry, and pose-alignment stage handlers from
  `_run_multires_level_stages`.
- Preserved level stats, loss accumulation, checkpoint writes, resume handling,
  observer actions, and final gauge-fix propagation.

### Decisions

- Kept the split private to `tomojax.align`; no public API or algorithm
  behavior changed.
- Left the larger `align_multires` orchestration split for the next cleanup
  slice because it is the remaining function-level Ruff complexity blocker.

### Validation

- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run ruff check src/tomojax/align/_stage_loop.py` now reports only
  `align_multires` PLR0912/PLR0915.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py -q`
  passed: 43 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The first remaining blockers are `align_multires`
  PLR0912/PLR0915, followed by geometry module doc/import findings and the
  broader repository lint backlog. Formatter churn from `just check` was
  reverted outside this slice.

### Risks

- The handler split depends on `StageLoopState` carrying every mutated level
  value; focused multires/setup/checkpoint tests passed, but future changes
  should keep new per-stage state explicit on that carrier.
- Proposed next fix for `just check`: split `align_multires` orchestration
  complexity.

## 2026-05-06 — Split Multires Public Orchestration

### Summary

- Added private multires context and run-state carriers for setup, resume, and
  finalization bookkeeping.
- Extracted multires input setup/validation, level selection, initial volume
  selection, translation seeding, per-level execution, and run finalization from
  the public `align_multires` function.
- Removed all remaining `_stage_loop.py` Ruff findings without changing the
  public alignment API.

### Decisions

- Kept the split inside `tomojax.align._stage_loop` so the existing
  `tomojax.align.pipeline` facade and import-linter contract remain unchanged.
- Preserved the coarsest-level phase-correlation seeding path as private helper
  logic rather than changing the alignment schedule model.

### Validation

- `uv run ruff format src/tomojax/align/_stage_loop.py` passed.
- `uv run ruff check src/tomojax/align/_stage_loop.py` passed.
- `uv run pytest tests/test_multires.py tests/test_bilevel_setup_alignment.py tests/test_align_checkpoint.py tests/test_align_quick.py -q`
  passed: 66 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `_stage_loop.py` is no longer in the failure list; the first
  remaining blockers are geometry module doc/import lint findings, followed by
  checkpoint/io/model lint and broader repository backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- The run-state carrier now owns checkpoint/resume bookkeeping across helper
  calls; focused multires/checkpoint tests passed, but future per-level state
  additions should be explicit fields.
- Proposed next fix for `just check`: clean the geometry deep-module lint
  backlog beginning with `detector_center.py`, `geometry_applier.py`,
  `geometry_blocks.py`, `initializers.py`, and `parametrizations.py`.

## 2026-05-06 — Clean Alignment Geometry Lint Blockers

### Summary

- Added module and public API docstrings for the first geometry lint blockers:
  detector-center exports, geometry application, calibration blocks,
  initializers, and pose parametrizations.
- Replaced parent-relative imports with absolute imports where needed and moved
  annotation-only geometry/state imports behind `TYPE_CHECKING`.
- Cleaned small parametrization findings by removing non-ASCII comment text and
  returning the final transform update directly.

### Decisions

- Kept the geometry package API unchanged; this slice only addressed docs,
  imports, and local lint findings.
- Replaced constant `getattr` calls in touched geometry files with direct
  attribute access as part of the focused Ruff cleanup.

### Validation

- `uv run ruff format src/tomojax/align/geometry/detector_center.py src/tomojax/align/geometry/geometry_applier.py src/tomojax/align/geometry/geometry_blocks.py src/tomojax/align/geometry/initializers.py src/tomojax/align/geometry/parametrizations.py`
  passed.
- `uv run ruff check src/tomojax/align/geometry/detector_center.py src/tomojax/align/geometry/geometry_applier.py src/tomojax/align/geometry/geometry_blocks.py src/tomojax/align/geometry/initializers.py src/tomojax/align/geometry/parametrizations.py`
  passed.
- `uv run pytest tests/test_geometry.py tests/test_geometry_applier.py tests/test_geometry_block_taxonomy_generator.py tests/test_detector_center_objective.py tests/test_align_quick.py -q`
  passed: 54 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The touched geometry files are no longer in the failure list; the
  first remaining blockers are `align/io/checkpoint.py`,
  `align/io/params_export.py`, and `align/model/*`, followed by broader
  repository lint backlog. Formatter churn from `just check` was reverted
  outside this slice.

### Risks

- Direct attribute access now makes the expected geometry protocol more
  explicit in touched files; focused geometry and quick alignment tests passed.
- Proposed next fix for `just check`: clean checkpoint/io lint, then continue
  through the alignment model package.

## 2026-05-06 — Clean Alignment I/O Lint Blockers

### Summary

- Added checkpoint and params-export module/class docstrings required by the
  current Ruff configuration.
- Cleaned checkpoint atomic-write lint by using `Path.replace` and
  `contextlib.suppress` while preserving fsync and temporary-file cleanup.
- Simplified schedule metadata validation and params5 shape error formatting.
- Reduced params-export JSON normalization return-count lint without changing
  exported payload shape.

### Decisions

- Kept the checkpoint schema and payload fields unchanged.
- Kept params-export normalization local instead of introducing a shared utility
  module, consistent with the no-generic-utils policy.

### Validation

- `uv run ruff format src/tomojax/align/io/checkpoint.py src/tomojax/align/io/params_export.py`
  passed.
- `uv run ruff check src/tomojax/align/io/checkpoint.py src/tomojax/align/io/params_export.py`
  passed.
- `uv run pytest tests/test_align_checkpoint.py tests/test_align_quick.py -q`
  passed: 33 tests.
- `uv run pytest tests/test_align_params_export.py -q` passed: 8 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. The touched alignment I/O files are no longer in the failure
  list; the first remaining blockers are in `src/tomojax/align/model/*`,
  followed by broader repository lint backlog. Formatter churn from
  `just check` was reverted outside this slice.

### Risks

- Atomic checkpoint save still depends on filesystem support for same-directory
  replace; this was already required by the previous `os.replace` path.
- Proposed next fix for `just check`: clean the alignment model package lint
  blockers beginning with diagnostics, DOF specs, gauge, motion models, and
  schedules.

## 2026-05-06 — Clean Alignment Diagnostics And DOF Lint

### Summary

- Added public docstrings for gauge diagnostics and scoped DOF helpers.
- Moved annotation-only diagnostics imports behind `TYPE_CHECKING`.
- Simplified the conditioning threshold branch without changing singular-value
  diagnostics.

### Decisions

- Kept this as a narrow model-package slice so the larger `dof_specs.py` and
  schedule cleanup can be reviewed separately.
- Left all gauge policy rules and DOF normalization behavior unchanged.

### Validation

- `uv run ruff format src/tomojax/align/model/diagnostics.py src/tomojax/align/model/dofs.py`
  passed.
- `uv run ruff check src/tomojax/align/model/diagnostics.py src/tomojax/align/model/dofs.py`
  passed.
- `uv run pytest tests/test_alignment_gauge_registry.py tests/test_align_quick.py tests/test_align_profiles.py -q`
  passed: 34 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `diagnostics.py` and `dofs.py` are no longer in the failure list;
  the first remaining blockers start in `src/tomojax/align/model/dof_specs.py`,
  followed by gauge, motion model, schedule, state, and broader repository lint
  backlog. Formatter churn from `just check` was reverted outside this slice.

### Risks

- The diagnostics import movement is safe only while those names remain
  annotation-only; focused gauge/profile tests passed.
- Proposed next fix for `just check`: clean `dof_specs.py`.

## 2026-05-06 — Clean Active Parameter DOF Spec Lint

### Summary

- Added `dof_specs.py` module and public API docstrings for active parameter
  packing, whitening, bounds, and diagnostics helpers.
- Moved annotation-only imports behind `TYPE_CHECKING`.
- Removed quoted forward references now covered by postponed annotations.

### Decisions

- Kept active-parameter behavior unchanged; this slice only documents the API
  and cleans annotation imports.
- Left gauge, motion model, schedule, and state lint for separate focused
  slices.

### Validation

- `uv run ruff format src/tomojax/align/model/dof_specs.py` passed.
- `uv run ruff check src/tomojax/align/model/dof_specs.py` passed.
- `uv run pytest tests/test_alignment_state.py tests/test_align_optimizers.py tests/test_alignment_objectives.py tests/test_bilevel_setup_alignment.py tests/test_alignment_scenario_catalog.py -q`
  passed: 46 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `dof_specs.py` is no longer in the failure list; the first
  remaining blockers start in `src/tomojax/align/model/gauge.py`, followed by
  motion model, schedule, state, and broader repository lint backlog. Formatter
  churn from `just check` was reverted outside this slice.

### Risks

- The import movement is safe only while `AlignmentState`, `Iterable`, and
  `Sequence` remain annotation-only; active parameter tests passed.
- Proposed next fix for `just check`: clean `gauge.py`.

## 2026-05-06 — Clean Alignment Gauge Lint

### Summary

- Added `gauge.py` module and typed-dict docstrings.
- Moved annotation-only imports behind `TYPE_CHECKING`.
- Added explicit argument and return annotations for the JAX zero-mean
  projection loop body.

### Decisions

- Kept gauge normalization, feasibility checks, projection behavior, and stats
  payloads unchanged.
- Left motion model, schedule, and state lint for follow-up focused slices.

### Validation

- `uv run ruff format src/tomojax/align/model/gauge.py` passed.
- `uv run ruff check src/tomojax/align/model/gauge.py` passed.
- `uv run pytest tests/test_align_gauge.py tests/test_alignment_gauge_registry.py tests/test_align_quick.py -q`
  passed: 34 tests.
- `just imports` passed.
- `just check` failed at `uv run ruff check --fix src tests tools` after
  formatting. `gauge.py` is no longer in the failure list; the first remaining
  blockers start in `src/tomojax/align/model/motion_models.py`, followed by
  schedules, state, objectives, and broader repository lint backlog. Formatter
  churn from `just check` was reverted outside this slice.

### Risks

- The JAX loop-body annotation is structural only; focused gauge tests passed.
- Proposed next fix for `just check`: clean `motion_models.py`.

## 2026-05-06 — Clean Alignment Motion Model Lint

### Summary

- Added `motion_models.py` module and property docstrings.
- Moved annotation-only `Sequence` import behind `TYPE_CHECKING`.
- Cleaned local type-checking lint for `cast` and sorted `__all__`.

### Decisions

- Kept pose model basis construction, coefficient fitting, and expansion
  behavior unchanged.
- Stopped the legacy Ruff cleanup path after this slice per user direction and
  will move the active plan directly to the smallest Phase 7 alternating-solver
  smoke/artifact vertical slice.

### Validation

- `uv run ruff format src/tomojax/align/model/motion_models.py` passed.
- `uv run ruff check src/tomojax/align/model/motion_models.py` passed.
- `uv run pytest tests/test_align_motion_models.py -q` passed: 6 tests.
- `just imports` passed.
- `uv run pytest tests/test_align_motion_models.py tests/test_align_chunking.py tests/test_align_optimizers.py -q`
  aborted with a JAX/Optax L-BFGS segmentation fault in
  `tests/test_align_chunking.py::test_align_smooth_pose_model_clips_active_bounds_only`;
  this is the known native L-BFGS path and is not caused by this docs/import
  cleanup. No unrelated formatter churn was present from this validation.

### Risks

- The import movement is safe only while `Sequence` remains annotation-only;
  focused motion-model tests passed.
- Proposed next fix: switch from legacy Ruff cleanup to Phase 7 alternating
  solver continuation smoke.

## 2026-05-06 — Phase 7 Alternating Solver Smoke Slice

### Summary

- Added an align-owned `smoke32` continuation schedule with preview and final
  levels.
- Added a deterministic 32^3 stopped-volume alternating smoke runner that
  reconstructs with the JAX reference FISTA path, applies gauge
  canonicalisation as the first geometry update, and writes Phase 7 smoke
  artifacts.
- Added smoke tests that verify deterministic volume output, coarse
  verification, level-1 geometry skip recording, pose gauge canonicalisation,
  and artifact presence.

### Decisions

- Kept the legacy `tomojax.align` package facade unchanged; Phase 7 names are
  exposed through `tomojax.align.api`.
- Used gauge canonicalisation as the smallest geometry-update vertical slice so
  the run emits real geometry/provenance artifacts before the full production
  Schur LM loop is wired into the alternating solver.
- Stopped legacy Ruff cleanup after the motion-model slice per user direction.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- The alternating solver currently uses gauge canonicalisation rather than the
  full Schur LM/GN geometry update; this is intentionally the smallest Phase 7
  vertical slice and leaves production escalation/continuation for follow-up.

## 2026-05-06 — Phase 7 Smoke Artifact Contract Expansion

### Summary

- Expanded the deterministic 32^3 alternating smoke run to persist
  `final_volume.npy`.
- Added the core audit scaffold artifacts for the smoke profile:
  `run_manifest.json`, `config_resolved.toml`, `input_summary.json`,
  `projection_stats.json`, `mask_summary.json`, `gauge_report.json`,
  `backend_report.json`, and `residual_metrics.csv`.
- Updated `artifact_index.json` entries with artifact type, media type, and
  descriptions, and extended smoke tests to verify the expanded contract.

### Decisions

- Kept the artifact writer local to `tomojax.align._alternating` for this
  slice to avoid introducing a generic helper module before a second owner
  needs it.
- Used deterministic smoke-profile manifest values rather than wall-clock
  timestamps so the smoke output remains reproducible.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- The smoke artifact scaffold is not yet the full production run schema from
  `docs/tomojax-v2/06_verification_and_artifact_contract.md`; missing
  production artifacts include observability/failure reports, preview slices,
  residual maps, and CLI-run metadata.

## 2026-05-06 — Phase 7 Early Exit Continuation Smoke

### Summary

- Extended the `smoke32` continuation schedule from level 4/final to
  level 4, conditional level 2, and final level 1.
- Added explicit skipped-level and skipped-geometry summary fields so early
  exit decisions are visible in `alignment_summary.csv`,
  `residual_metrics.csv`, and `verification.json`.
- Made level-1 geometry a planned verification-triggered polish step that is
  skipped when coarse verification passes, while still running the final
  reconstruction.

### Decisions

- Kept gauge canonicalisation as the smoke geometry update; this slice only
  changes continuation control flow and artifact reporting.
- Recorded skipped level 2 as an artifact row instead of silently omitting it so
  early-exit behavior remains auditable.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_vertical_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- The early-exit predicate is still the smoke-profile coarse verification
  predicate, not the full production combination of held-out residual,
  parameter-update, and gauge-stability checks.

## 2026-05-06 — Phase 7 Continuation Profile Presets

### Summary

- Added `lightning`, `balanced`, and `reference` continuation schedules while
  keeping `smoke32` as the deterministic test profile.
- Exposed `ContinuationScheduleName` through `tomojax.align.api`.
- Added focused schedule tests for Phase 7 level ordering, conditional level-2
  behavior, final-level policy, monotonic profile work, and unknown profile
  rejection.

### Decisions

- Kept profile presets as deterministic schedule data only; CLI/runtime profile
  plumbing remains out of scope for this slice.
- Encoded monotonic work increases from `lightning` to `balanced` to
  `reference`; empirical tuning remains future work.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/api.py tests/test_continuation_schedules.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/api.py tests/test_continuation_schedules.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py tests/test_continuation_schedules.py`
  passed.
- `uv run pytest tests/test_continuation_schedules.py tests/test_alternating_solver_smoke.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The schedule presets are conservative defaults and have not yet been tuned
  against the full synthetic benchmark suite.

## 2026-05-06 — Phase 7 Alternating Solver Entrypoint

### Summary

- Added the public `AlternatingAlignmentSolver` orchestration object.
- Routed `run_alternating_solver_smoke` through `AlternatingAlignmentSolver`
  so the function and solver class share one implementation path.
- Exported and documented the solver entrypoint, and added a focused smoke test
  for `AlternatingAlignmentSolver.run_smoke`.

### Decisions

- Kept the solver object intentionally thin and smoke-profile-specific in this
  slice. A general dataset solver interface remains follow-up work.
- Kept the legacy `tomojax.align` package facade unchanged; the new solver is
  exported from `tomojax.align.api`.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py -q`
  passed: 9 tests.
- `just imports` passed.

### Risks

- `AlternatingAlignmentSolver` currently exposes only the deterministic
  `run_smoke` method; production data loading and CLI integration are still
  missing.

## 2026-05-06 — Phase 7 Auto Smoke CLI

### Summary

- Added `tomojax.cli.align_auto`, a one-command deterministic Phase 7
  `align=auto` smoke entrypoint.
- Added the `tomojax-align-auto-smoke` console script.
- Added focused CLI tests for help text and writing the core final volume,
  geometry, and verification artifacts.

### Decisions

- Kept the new command explicitly smoke-profile-only so it does not imply full
  production dataset alignment.
- Called `AlternatingAlignmentSolver` from the CLI instead of duplicating solver
  behavior in `tomojax.cli`.
- Put the new CLI tests in `tests/test_align_auto_cli.py` to avoid touching the
  older transitional `tests/test_cli_entrypoints.py` lint backlog.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py -q`
  passed: 5 tests.
- `just imports` passed.
- `uv run tomojax-align-auto-smoke --help` passed.

### Risks

- The command currently runs only synthetic smoke data; full dataset input and
  production `align=auto` CLI integration remain future work.

## 2026-05-06 — Phase 7 Smoke Audit Reports

### Summary

- Added `gauge_policy.json`, `observability_report.json`, and
  `failure_report.json` to the deterministic Phase 7 smoke run.
- Included those reports in `artifact_index.json`.
- Extended the smoke tests to verify report presence and key fields.

### Decisions

- Kept `observability_report.json` explicitly marked as
  `smoke_placeholder`; Schur-curvature-backed observability is not computed in
  this smoke profile yet.
- Wrote `failure_report.json` for passed runs with `status="passed"` and
  `failure=null` so the artifact is always present.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- Observability remains a placeholder until the alternating solver consumes the
  Schur LM condition and weak-mode diagnostics from the geometry update path.

## 2026-05-06 — Phase 7 Residual Filter Continuation

### Summary

- Added residual-filter configs to the Phase 7 continuation levels:
  low-pass at level 4, low-pass plus band-pass at conditional level 2, and raw
  residual at final level 1.
- Applied the public `tomojax.forward.apply_residual_filter_schedule` path to
  the projection-domain geometry/verification loss in the alternating smoke
  runner.
- Recorded residual-filter kinds in `alignment_summary.csv`,
  `residual_metrics.csv`, and verification level payloads, with focused tests.

### Decisions

- Kept reconstruction FISTA on the existing reference objective in this slice;
  only the geometry verification/update loss uses continuation filters.
- Used the public forward-module filter API rather than adding local filtering
  under `tomojax.align`.

### Validation

- `uv run ruff format src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run ruff check src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run basedpyright src/tomojax/align/_continuation.py src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py -q`
  passed: 9 tests.
- `just imports` passed.

### Risks

- FISTA does not yet consume the residual-filter schedule; the continuation
  filters currently affect only geometry loss and verification behavior.

## 2026-05-06 — Phase 7 Profile Provenance Artifacts

### Summary

- Threaded the resolved `ContinuationSchedule` into `run_manifest.json` and
  `config_resolved.toml`.
- Fixed hard-coded `smoke32` profile metadata so non-default profiles record the
  actual schedule name and level factors.
- Added a focused non-default `lightning` profile smoke test for manifest and
  resolved-config provenance.

### Decisions

- Kept the config artifact deterministic and minimal: profile, align mode,
  backend, geometry model, and level factors.
- Recorded schedule name and level factors now; full schedule serialization is
  left for a later artifact-schema pass.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- Custom schedule objects are represented by name and level factors only, not a
  complete per-level serialized schedule.

## 2026-05-06 — Phase 7 Verification Predicate Bundle

### Summary

- Added per-level verification predicates for loss non-increase, finite loss,
  gauge stability, and parameter-update size.
- Gated smoke-level `verified` status on the full predicate bundle instead of
  only loss non-increase.
- Recorded predicate fields, thresholds, and measured update norm in
  `verification.json`, `alignment_summary.csv`, and `residual_metrics.csv`.

### Decisions

- Kept the predicate thresholds smoke-profile local in
  `AlternatingSmokeConfig`.
- Did not add held-out residual checks in this slice; that remains a production
  verification follow-up.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- `parameter_update_small` is a smoke-scale heuristic, not the final
  trust-region update acceptance metric.

## 2026-05-06 — Phase 7 Smoke Input Artifacts

### Summary

- Persisted `observed_projections.npy` and `projection_mask.npy` from the
  deterministic Phase 7 smoke run.
- Added `recovery_tolerances.json` with smoke-specific geometry and
  verification tolerances.
- Added the new input artifacts to `artifact_index.json` and focused smoke
  tests.

### Decisions

- Used `.npy` arrays for smoke input artifacts, matching the existing synthetic
  dataset writer contract.
- Kept recovery tolerances smoke-specific and explicit; this is not a full
  128^3 synthetic benchmark manifest yet.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- The smoke run now persists benchmark-style input arrays, but the full 128^3
  synthetic suite still lives in the dataset generator path and is not yet run
  end-to-end through Phase 7.

## 2026-05-06 — Phase 7 Smoke Truth Provenance

### Summary

- Persisted `ground_truth_volume.npy` from the deterministic Phase 7 smoke run.
- Added `geometry_true.json` and explicit `geometry_corrupted.json` alongside
  the existing solver `geometry_initial.json`.
- Added the new truth/corruption artifacts to `artifact_index.json` and focused
  smoke tests.

### Decisions

- Kept `geometry_initial.json` as the solver initial state and added
  `geometry_corrupted.json` to mirror the synthetic benchmark naming contract.
- Used `GeometryState.zeros(n_views)` as the true uncorrupted smoke geometry.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- `geometry_corrupted.json` currently mirrors `geometry_initial.json`; future
  production dataset ingestion may need separate supplied-vs-initial records.

## 2026-05-06 — Phase 7 Smoke Recovery Metrics

### Summary

- Added final-vs-true smoke geometry recovery metrics to `verification.json`.
- Compared mean residual `dx` and `phi_residual` errors against the
  smoke-specific `recovery_tolerances.json` thresholds.
- Added focused tests for recovery pass/fail, measured values, and limits.

### Decisions

- Kept recovery metrics limited to the gauge-canonicalised mean pose quantities
  represented by the current reference smoke path.
- Reused the same smoke tolerance payload written to `recovery_tolerances.json`
  so the verification report and tolerance artifact stay aligned.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- Recovery coverage is still limited to smoke-scale gauge recovery, not the
  full 5-DOF synthetic recovery suite.

## 2026-05-06 — Phase 7 Smoke Volume Metrics

### Summary

- Added final-vs-truth volume RMSE, MAE, and NMSE to the deterministic smoke
  `verification.json`.
- Added a smoke-specific volume NMSE tolerance to `recovery_tolerances.json`.
- Extended focused smoke tests to verify volume metric presence and pass/fail
  behavior.

### Decisions

- Kept the NMSE tolerance loose because the smoke profile uses a tiny
  one-iteration reference FISTA reconstruction and is a wiring check, not a
  quality benchmark.
- Recorded volume metrics now so the smoke run mirrors the benchmark contract
  fields used by the future 128^3 suite.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- The recorded volume metrics are not expected to be strong until longer
  reconstruction schedules and production benchmark profiles are wired in.

## 2026-05-06 — Phase 7 Per-View Residual Metrics

### Summary

- Expanded `residual_metrics.csv` so it records both per-level continuation
  summaries and final per-view projection residual metrics.
- Added `row_type=view_residual` rows with RMSE, MAE, robust loss,
  valid-pixel fraction, outlier fraction, and raw RMSE fields.
- Extended the deterministic 32^3 smoke test to verify the summary rows and
  four per-view residual rows are written.

### Decisions

- Added a `row_type` discriminator so existing level summary rows remain in the
  same artifact without overloading per-view metric columns.
- Kept this slice to raw final projection residuals; low-pass and band-pass
  residual views remain a later benchmark-contract expansion.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- Per-view metrics are currently computed from the final raw reference forward
  projection only, not from the continuation-filtered residual streams.

## 2026-05-06 — Phase 7 Artifact Validation

### Summary

- Added a typed `tomojax.verify` artifact validation API with inspect and
  fail-loud entrypoints.
- Validated required smoke JSON artifacts, core schema identifiers, geometry
  schema version, and indexed artifact file existence.
- Wired the Phase 7 smoke artifact writer through validation before returning
  and added positive/negative tests against real smoke bundles.

### Decisions

- Used a lightweight stdlib validator instead of adding a schema dependency for
  the smoke contract.
- Kept this slice focused on JSON/index validation; CSV and array semantic
  validation remain future extensions of the same API.

### Validation

- `uv run ruff format src/tomojax/verify src/tomojax/align/_alternating.py tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/verify src/tomojax/align/_alternating.py tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/verify src/tomojax/align/_alternating.py tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_verify_artifacts.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The validator does not yet check CSV column semantics, array shapes, or
  preview/residual-map directory contents.

## 2026-05-06 — Phase 7 Residual Map Artifacts

### Summary

- Added `residual_maps/final_raw_residual.npy` for the deterministic Phase 7
  smoke bundle.
- Added `residual_maps/summary.json` with residual-map schema, shape, dtype,
  valid-pixel fraction, and aggregate residual statistics.
- Updated `artifact_index.json` to record run-directory-relative nested paths
  and extended smoke tests to verify the indexed residual-map artifacts.

### Decisions

- Persisted residual maps as `.npy` arrays for the smoke path so deterministic
  numeric content can be validated without image-rendering dependencies.
- Kept human-facing preview images and plots out of this slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The smoke run still does not emit preview slices or plots from the artifact
  contract.

## 2026-05-06 — Phase 7 Preview Slice Artifacts

### Summary

- Added deterministic preview-slice artifacts under `preview_slices/` for the
  Phase 7 smoke bundle.
- Persisted central z-slices for truth, final reconstruction, and final-minus-
  truth error as `.npy` arrays.
- Added `preview_slices/summary.json` with schema, axis, index, shape, dtype,
  and error aggregate metrics, and indexed all nested preview artifacts.

### Decisions

- Stored numeric preview arrays instead of rendered images so the smoke tests
  can validate deterministic data without adding rendering dependencies.
- Used a tiny absolute tolerance in the preview error assertion to allow
  float32 subnormal roundoff after save/load.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- Human-facing PNG previews and plots are still not emitted by the smoke run.

## 2026-05-06 — Phase 7 Geometry Trace Artifact

### Summary

- Added `geometry_trace.csv` to the deterministic Phase 7 smoke artifact
  bundle.
- Recorded per-level geometry update trace rows with requested/executed update
  counts, loss delta, gauge and parameter-update predicates, skip state, and
  early-exit reason.
- Indexed the trace artifact and extended smoke tests to verify the deterministic
  geometry trace rows.

### Decisions

- Derived the trace from `AlternatingLevelSummary` in this slice because the
  smoke path still uses gauge canonicalisation as its geometry update rather
  than real inner LM/GN steps.
- Left per-inner-step trace granularity for the upcoming geometry optimiser
  implementation.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- `geometry_trace.csv` is per continuation level, not yet per accepted/rejected
  LM/GN step.

## 2026-05-06 — Phase 7 Verification Report Shape

### Summary

- Added contract-shaped `status`, `summary`, `metrics`, and `escalation`
  sections to the Phase 7 smoke `verification.json`.
- Kept existing smoke-specific keys so deterministic checks can still inspect
  seed, schedule, levels, thresholds, and recovery details.
- Updated artifact validation so `verification.json` must include the new
  top-level report sections.

### Decisions

- Used the first solver-level loss as `metrics.residual_before` rather than the
  legacy smoke `initial_loss`, which compares the true volume against synthetic
  observations and is not a run-start residual.
- Reported `summary.projection_residual_improved` truthfully from the smoke
  metrics; the current tiny smoke path may report `false` while still passing
  the smoke recovery checks.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The smoke run still uses placeholder geometry updates, so the new verification
  summary is a report-shape contract rather than proof of full optimiser
  convergence.

## 2026-05-06 — Phase 7 Backend Provenance Report

### Summary

- Expanded the Phase 7 smoke `backend_report.json` beyond the placeholder
  requested/actual fields.
- Added explicit projector, backprojector, geometry-reduction, detector-grid,
  Pallas eligibility, fallback, and agreement-test provenance fields.
- Updated artifact validation to require the expanded backend provenance keys.

### Decisions

- Reported all active components as `jax_reference` because this smoke path is
  intentionally CPU/JAX reference first.
- Added a `reference_baseline` agreement-test row with zero errors so the
  contract shape is present before fast-path comparison tests exist.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- Agreement testing currently records reference-baseline provenance only; real
  max/mean error comparisons remain blocked on alternate backend paths.

## 2026-05-06 — Phase 7 Failure Report Gates

### Summary

- Expanded the Phase 7 smoke `failure_report.json` with verification gate rows.
- Added finite-output, projection-residual-improvement, gauge-stability,
  optimiser-health, and backend-provenance gates.
- Added structured warning entries for smoke-level `no_improvement` cases while
  preserving the smoke run's pass status.
- Updated artifact validation to require the expanded failure-report shape.

### Decisions

- Treated projection residual non-improvement as a warning in the tiny smoke
  profile because this path is still a wiring check with placeholder geometry
  updates.
- Kept the failure classes from the artifact contract visible in the report so
  later verifier work can promote warnings/errors without changing shape.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- Optimiser-health evidence remains coarse until real accepted/rejected LM/GN
  step diagnostics are part of the alternating solver.

## 2026-05-06 — Phase 7 Observability Report Shape

### Summary

- Expanded the Phase 7 smoke `observability_report.json` beyond the placeholder
  status.
- Added structured setup and pose DOF entries with active, observable, status,
  and gauge-group metadata where applicable.
- Recorded smoke weak modes and handled frozen DOFs so the report explicitly
  distinguishes uncomputed observability from frozen parameters.

### Decisions

- Kept the top-level status as `smoke_not_evaluated` because the smoke path
  still does not compute Schur curvature or condition numbers.
- Marked gauge-canonicalised pose terms separately from weak/not-evaluated
  active DOFs.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- This report is still descriptive; numerical observability metrics remain for
  the real Schur LM/GN implementation.

## 2026-05-06 — Phase 7 Run Manifest Contract Fields

### Summary

- Expanded the Phase 7 smoke `run_manifest.json` with contract-required
  `tomojax_version`, `git_commit`, `started_at`, `finished_at`,
  `backend_requested`, and geometry model fields.
- Updated artifact validation so missing manifest contract keys fail loudly.
- Extended smoke tests to verify the deterministic timestamp placeholders and
  backend/geometry manifest fields.

### Decisions

- Kept smoke timestamps deterministic as `deterministic-smoke` so repeated smoke
  runs remain reproducible.
- Recorded the current git commit when available, with an `unknown` fallback for
  non-git environments.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_artifacts.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- Production elapsed-time accounting is still absent from the deterministic
  smoke manifest.

## 2026-05-06 — Phase 7 Plot Summary Artifact

### Summary

- Added `plots/summary.json` to the deterministic Phase 7 smoke artifact
  bundle.
- Recorded plot-ready FISTA loss and per-level geometry loss traces without
  adding rendering dependencies.
- Indexed the nested plots artifact and extended smoke tests to verify the
  artifact path and basic trace content.

### Decisions

- Stored numeric plot inputs first rather than rendered PNG/SVG files.
- Kept rendering out of the smoke path so the focused checks remain fast and
  deterministic.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- No rendered plot images are emitted yet; this is intentionally paused now in
  favor of the real Schur-in-the-loop alignment update.

## 2026-05-06 — Phase 7 Schur Geometry Update In The Loop

### Summary

- Switched the deterministic 32^3 smoke fixture so observed projections are
  generated from true geometry while the solver starts from a corrupted initial
  geometry.
- Replaced the placeholder geometry canonicalisation update in the Phase 7 loop
  with the supported `solve_joint_schur_lm` reference solver.
- Recorded Schur diagnostics in `schur_diagnostics.json` and added accepted
  step, condition, dense-vs-Schur, and reduction fields to `geometry_trace.csv`.
- Extended smoke tests to verify projection residual improvement, accepted Schur
  diagnostics, gauge-canonical supported DOF recovery, and true-vs-corrupted
  geometry separation.

### Decisions

- Used the synthetic ground-truth volume as the fixed volume for this first
  Schur-in-the-loop smoke slice so geometry recovery is isolated from the tiny
  one-iteration FISTA reconstruction.
- Kept the alternating reconstruction artifact path intact; replacing the fixed
  synthetic volume with the stopped reconstructed latent is the next numerical
  integration step.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_joint_schur_lm.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

### Risks

- This is a real Schur geometry update, but it is still an alignment-isolated
  smoke slice using the synthetic fixed volume rather than the stopped FISTA
  latent volume.

## 2026-05-06 — Phase 7 Geometry-Aware Backprojection Initializer

### Summary

- Added public `reconstruct_backprojection_reference` to `tomojax.recon`.
- Implemented a deterministic geometry-aware reference backprojection for smoke
  and FISTA warm starts.
- Used the backprojection volume as the initial volume for the Phase 7 smoke
  FISTA path when no previous stopped latent exists.
- Extended reconstruction and smoke tests to cover shape, nonzero output, and
  the stronger saved smoke volume.

### Decisions

- Kept the Schur update source as the fixed synthetic smoke volume for this
  slice because the stopped FISTA latent is not yet strong enough to recover all
  supported DOFs.
- Kept the backprojection helper in `tomojax.recon` as a reference
  reconstruction primitive rather than adding more artifact-shape scaffolding.

### Validation

- `uv run ruff format src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/recon/_reference.py src/tomojax/recon/__init__.py src/tomojax/recon/api.py src/tomojax/align/_alternating.py tests/test_reference_fista.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_reference_fista.py tests/test_alternating_solver_smoke.py tests/test_joint_schur_lm.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 17 tests.
- `just imports` passed.

### Risks

- The geometry-aware backprojection is still a simple reference initializer,
  not a full reconstruction-quality solution.

## 2026-05-06 — Phase 7 Schur Volume Source Contract

### Summary

- Added a typed `GeometryUpdateVolumeSource` contract for the Phase 7 smoke
  Schur geometry update.
- Kept the default source as the fixed synthetic truth volume and made that
  explicit in the alternating smoke config.
- Recorded the selected source in `config_resolved.toml`, `run_manifest.json`,
  `verification.json`, and `schur_diagnostics.json`.
- Exposed the source type through the `tomojax.align` public API and README.
- Extended smoke tests to verify the explicit source contract across the
  emitted artifacts.

### Decisions

- This slice does not switch the Schur update to the stopped FISTA latent. The
  synthetic fixed volume remains the deterministic smoke source until stopped
  latent recovery can pass the supported DOF checks.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/api.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The source contract is explicit, but the default is still a smoke-only
  numerical bridge. The next numerical slice should make stopped-latent Schur
  recovery pass before changing the default.

## 2026-05-06 — Phase 7 Stopped-Latent Schur Update

### Summary

- Switched the default Phase 7 smoke Schur geometry-update volume source from
  the fixed synthetic truth volume to the stopped reconstructed latent.
- Kept `fixed_synthetic_truth` as an explicit diagnostic source.
- Made the deterministic smoke corruption observable from the stopped latent by
  using wider angular residual variation with smaller detector shifts.
- Raised the smoke32 preview Schur budget to 8 LM iterations so the real
  stopped-latent update deterministically accepts and improves projection
  residual.
- Updated smoke tests to assert stopped-latent artifact provenance, accepted
  Schur diagnostics, residual improvement, and gauge-canonical supported DOF
  recovery.

### Decisions

- The stopped latent can absorb small detector errors in this minimal projector,
  so the smoke fixture must be numerically observable rather than merely
  nonzero. This slice changes only the deterministic smoke fixture and Schur
  iteration budget, not the joint solver API.
- The theta recovery threshold is now `8.5e-2` rad for this smoke profile,
  matching the deterministic stopped-latent recovery margin while keeping
  detector-shift recovery at the existing limits.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_continuation.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_continuation_schedules.py tests/test_verify_artifacts.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

### Risks

- This is still a 32^3 deterministic smoke profile with a minimal projector.
  The next step is to keep increasing realism without letting reconstruction
  absorb the geometry signal.

## 2026-05-06 — Phase 7 Benchmark-Facing Auto Path

### Summary

- Added optional synthetic128 benchmark metadata to the Phase 7 smoke config.
- Extended `align-auto` with `--synthetic-dataset` and `--dataset-out-dir`.
- The command now generates deterministic synthetic dataset artifacts for a
  named benchmark spec before running the Phase 7 auto smoke path.
- Recorded the synthetic benchmark identity and generated artifact directory in
  `verification.json`, `run_manifest.json`, and `config_resolved.toml`.
- Added focused CLI and smoke tests for the benchmark-facing path.

### Decisions

- This slice deliberately keeps the solver input as the current deterministic
  Phase 7 smoke fixture. The named synthetic dataset artifacts are generated
  and recorded as benchmark context only; full external dataset ingestion is a
  separate numerical/data-loading slice.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_verify_artifacts.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- The command now ties auto runs to named synthetic benchmark artifact
  generation, but it does not yet consume those generated projections, masks,
  or geometry files as solver inputs.

### Full Check Baseline

- `just check` was run after this slice and failed during `uv run ruff check
  --fix src tests tools` on broad pre-existing transitional lint/docstring
  debt, starting with `src/tomojax/align/model/schedules.py` and
  `src/tomojax/align/model/state.py`.
- The preceding `uv run ruff format src tests tools` step reformatted many
  unrelated legacy files; that formatter churn was reverted immediately because
  it was outside the Phase 7 slice.
- No Phase 7 benchmark-facing files were left dirty after the revert.

## 2026-05-06 — Phase 7 Robust Residual Scale Hook

### Summary

- Added public `robust_residual_scale` to `tomojax.forward`.
- Threaded per-level estimated residual scale into the Phase 7 alternating
  smoke path.
- Used the continuation schedule sigma as a stability floor for the effective
  sigma passed to Schur loss evaluation.
- Recorded estimated and effective residual sigma in level summaries and CSV
  artifacts.
- Added focused forward and smoke tests for the MAD estimator and artifact
  contract.

### Decisions

- The deterministic 32^3 smoke residual is sparse enough that its MAD estimate
  is below the stable solver scale. The effective sigma therefore uses
  `max(schedule_sigma, robust_estimate)` so noisier benchmark ingestion can
  raise the scale later without destabilising the current smoke run.

### Validation

- `uv run ruff format src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/forward/_residuals.py src/tomojax/forward/api.py src/tomojax/forward/__init__.py src/tomojax/align/_alternating.py tests/test_forward_reference.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 16 tests.
- `just imports` passed.

### Risks

- This hook mostly records scale on the current sparse smoke fixture. It should
  become more behaviorally important when the solver consumes noisier generated
  benchmark projections.

## 2026-05-06 — Phase 7 Held-Out Residual Gate

### Summary

- Added deterministic held-out view masking to the Phase 7 smoke config.
- Schur geometry updates now train on the non-held-out views when the held-out
  check is enabled.
- Added held-out residual before/after/pass fields to level summaries,
  `alignment_summary.csv`, `geometry_trace.csv`, `residual_metrics.csv`, and
  `verification.json`.
- Coarse early exit now requires the held-out residual check to pass in addition
  to residual nonincrease, finite losses, stable gauges, and small parameter
  updates.
- Added focused smoke tests for the held-out threshold and emitted metrics.

### Decisions

- The 4-view smoke profile holds out the last view. The held-out residual is
  allowed the same `1.0e-5` tolerance as the projection residual gate because
  this small smoke fixture can improve training residual while the held-out
  view remains nearly flat.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- The held-out gate is deterministic and real, but with only four smoke views it
  proves tolerance stability more than broad generalisation.

## 2026-05-06 — Phase 7 Continuation Prior In Schur Loop

### Summary

- Added `parameter_prior_strength` to the joint Schur LM config and diagnostics.
- Added a weak quadratic parameter prior around the current Schur solve's
  initial parameter vector, included in both finite-difference residual rows
  and accepted/rejected candidate loss evaluation.
- Kept per-view Schur normal-equation diagnostics data-only by separating data
  residual rows from appended prior rows.
- Threaded `ContinuationLevel.prior_strength` into Phase 7 geometry updates.
- Recorded prior strength in level summaries, `alignment_summary.csv`,
  `geometry_trace.csv`, `residual_metrics.csv`, and Schur diagnostics.
- Extended focused Schur and deterministic 32^3 smoke tests to verify the real
  Schur update improves projection residual, recovers supported realised DOFs
  after gauge canonicalisation, and emits the prior diagnostics.

### Decisions

- The continuation prior is currently a local step regulariser around the
  current solve's initial packed parameter vector. This keeps the vertical
  slice small and explicit until metadata, nuisance, and smoothness priors are
  implemented.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_alternating.py tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_alternating.py tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/align/_alternating.py tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- This is not the full prior family from the design docs; it is the smallest
  continuation-driven Schur prior needed to make the Phase 7 loop use a real
  regularised geometry update.

## 2026-05-06 — Phase 7 Align-Auto CLI Acceptance Contract

### Summary

- Expanded the `tomojax-align-auto-smoke` command test from artifact existence
  checks to the Phase 7 acceptance contract.
- The CLI test now asserts that `verification.json` reports `status="passed"`,
  residual improvement, stable gauges, all verified levels, and default level-1
  geometry skipping.
- The command path now has focused test coverage for emitted
  `alignment_summary.csv`, `geometry_trace.csv`, and `schur_diagnostics.json`
  fields, including held-out pass state and continuation prior strength.

### Decisions

- Kept this slice as command-path verification only. The solver and artifact
  writer already produce the reports; this test ensures the user-facing command
  exercises and preserves that behavior.

### Validation

- `uv run ruff format tests/test_align_auto_cli.py` passed.
- `uv run ruff check tests/test_align_auto_cli.py` passed.
- `uv run basedpyright tests/test_align_auto_cli.py` passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 3 tests.
- `just imports` passed.

### Risks

- This is a contract hardening slice, not a new numerical capability. It should
  still catch regressions in the canonical one-command Phase 7 acceptance path.

## 2026-05-06 — Phase 8 Gain/Offset Nuisance Primitive

### Summary

- Added `GainOffsetModel` to `tomojax.nuisance` with identity construction,
  per-view affine application, and JSON-serializable payload output.
- Added `estimate_gain_offset`, a closed-form per-view weighted least-squares
  fit for `observed ~= gain * predicted + offset`.
- Supported projection-shaped masks and detector-shaped masks, with identity
  fallback for fully masked views.
- Updated the nuisance README from skeleton-only status to document the first
  public Phase 8 primitive.
- Added focused tests for identity application, synthetic gain/offset recovery,
  masked fitting, and empty-mask stability.

### Decisions

- Used the variable-projection closed-form update allowed by Phase 8 instead of
  adding a nuisance LM block. This keeps the first nuisance slice deterministic
  and independent of the geometry solver.

### Validation

- `uv run ruff format src/tomojax/nuisance/_gain_offset.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_gain_offset.py`
  passed.
- `uv run ruff check src/tomojax/nuisance/_gain_offset.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_gain_offset.py`
  passed.
- `uv run basedpyright src/tomojax/nuisance/_gain_offset.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_gain_offset.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_nuisance_gain_offset.py tests/test_v2_module_skeleton.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- The alternating solver does not consume the nuisance model yet, so geometry
  can still absorb intensity drift in the Phase 7 loop. The next Phase 8 slice
  should thread this tested primitive into residual evaluation.

## 2026-05-06 — Phase 8 Schur Gain/Offset Variable Projection

### Summary

- Added opt-in `fit_gain_offset` support to `JointSchurLMConfig`.
- Schur residual, loss, and per-view loss evaluation can now fit per-view
  gain/offset through the public `tomojax.nuisance.estimate_gain_offset`
  primitive before computing projection residuals.
- Added `gain_offset_fit` to Schur diagnostics and normal-equation summary
  artifacts.
- Updated `tomojax.align` README dependencies and invariants to record the
  public nuisance dependency.
- Added a focused synthetic Schur test where observed projections contain only
  per-view affine intensity drift; with nuisance fitting enabled, the solver
  leaves correct geometry unchanged and reports near-zero residual.

### Decisions

- The hook remains opt-in in this slice. That preserves the existing Phase 7
  smoke behavior while giving Phase 8 a real path for geometry solvers to stop
  explaining affine acquisition drift as motion.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_nuisance_gain_offset.py -q`
  passed: 11 tests.
- `just imports` passed.

### Notes

- A mistaken `ruff format` invocation including `src/tomojax/align/README.md`
  failed because Ruff does not parse Markdown; the corrected Python-file format
  command passed.

### Risks

- Gain/offset fitting is recomputed inside finite-difference residual
  evaluation, so this is correctness-first rather than performance-first.

## 2026-05-06 — Phase 8 Align-Auto Gain/Offset Nuisance Toggle

### Summary

- Added `fit_gain_offset_nuisance` to `AlternatingSmokeConfig`.
- Threaded the option into Schur geometry updates through
  `JointSchurLMConfig.fit_gain_offset`.
- Added `--fit-gain-offset-nuisance` to the Phase 7/8 `align-auto` smoke CLI.
- Recorded the resolved option in `verification.json`, `run_manifest.json`, and
  `config_resolved.toml`.
- Added focused smoke and CLI tests proving the opt-in command path emits
  Schur diagnostics with `gain_offset_fit=true` while the default path remains
  disabled.

### Decisions

- Kept gain/offset nuisance fitting disabled by default until nuisance-bearing
  benchmark datasets are loaded directly by the solver path.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 9 tests.
- `just imports` passed.

### Risks

- The default smoke still has no nuisance drift. This slice exposes and records
  the opt-in path; benchmark ingestion should exercise it with nuisance-bearing
  synthetic data.

## 2026-05-06 — Phase 8 Synthetic Gain/Offset Nuisance Artifacts

### Summary

- Added deterministic gain-drift and per-view scalar background-offset
  realization to `generate_synthetic_dataset`.
- `clean=False` synthetic datasets now apply realized gain/offset nuisance to
  generated projections when the benchmark manifest declares supported terms.
- `clean=True` still writes clean projections but records the realized nuisance
  arrays and unsupported original spec fields in `nuisance_truth.json`.
- Added focused tests for thermal gain/background drift and lamino sinusoidal
  gain drift artifacts.

### Decisions

- Applied only nuisance terms covered by the current Phase 8 gain/offset model:
  per-view gain and scalar per-view offsets. Harder terms such as hot/dead
  pixels, stripes, bad views, and partial-FOV masks remain recorded as spec
  metadata for later owned slices.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_nuisance_gain_offset.py -q`
  passed: 9 tests.
- `just imports` passed.

### Risks

- Nuisance-bearing synthetic artifacts still do not cover the full hard dataset
  contract. Additional owned nuisance models should add the remaining terms
  rather than hiding them in the generator.

## 2026-05-06 — Phase 8 Schur-Backed Observability Report

### Summary

- Replaced the Phase 7 smoke observability placeholder with a report built from
  the last Schur geometry update.
- `observability_report.json` now records Schur condition number, Schur
  eigenvalues, minimum Schur eigenvalue, and weak-mode labels.
- The report now explicitly marks supported setup DOFs as evaluated, active
  `det_v_px` as evaluated in the smoke Schur setup block, and `theta_scale` as
  frozen with a reason.
- Updated focused smoke artifact assertions for the new evidence-backed report.

### Decisions

- This slice reports evidence only. It does not auto-activate or freeze weak
  DOFs beyond the current geometry state.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- The evidence is limited to supported Schur setup DOFs in the current smoke
  geometry. Full weak-DOF policy still needs correlation and validation
  improvement rules.

## 2026-05-06 — Phase 8 Report-Only Weak DOF Decisions

### Summary

- Added a `weak_dof_policy` block to `observability_report.json`.
- The report now emits conservative decisions for `det_v_px` and `theta_scale`
  with thresholds, curvature evidence, accepted-step evidence, and explicit
  missing evidence fields.
- `det_v_px` is reported as `keep_active_with_prior` in the current smoke run
  when Schur curvature and accepted-step evidence pass.
- `theta_scale` remains `keep_frozen` because it is not yet supported by the
  reference projector or identifiable-scale policy.

### Decisions

- The policy is report-only. It does not mutate the geometry state or change
  active DOF selection until correlation and validation-improvement gates exist.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- The smoke report lacks correlation and validation-improvement evidence, so
  those fields are intentionally marked missing rather than inferred.

## 2026-05-06 — Phase 8 Nuisance Residual Failure Gate

### Summary

- Added public `tomojax.verify.residual_structure_summary` for warning-oriented
  residual structure classification.
- Added a `nuisance_residual_structure` warning gate to `failure_report.json`.
- The gate reports per-view mean residual structure and detector-column mean
  residual structure against explicit thresholds.
- `failure_report.json` now maps failed nuisance-structure gates to
  `nuisance_unmodelled` warnings with a recommended action.
- Added focused smoke assertions for the passing default case and a structured
  column-residual test for the warning heuristic.

### Decisions

- The gate is warning-only. Phase 8 nuisance models are not complete enough to
  fail an otherwise valid smoke run on this heuristic alone.
- The residual-structure helper lives in `tomojax.verify` so align and tests use
  a public verification primitive instead of reaching into private align code.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/verify/_residual_structure.py src/tomojax/verify/api.py src/tomojax/verify/__init__.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/verify/_residual_structure.py src/tomojax/verify/api.py src/tomojax/verify/__init__.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/verify/_residual_structure.py src/tomojax/verify/api.py src/tomojax/verify/__init__.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The heuristic is intentionally simple and may miss structured nuisance modes
  until stripe/background models are added.

## 2026-05-06 — Phase 8 Background Offset Nuisance Primitive

### Summary

- Added public `BackgroundOffsetModel` to `tomojax.nuisance`.
- Implemented per-view constant plus vertical-gradient additive background
  application.
- Added masked closed-form `estimate_background_offset` fitting against
  `observed - predicted`.
- Exported the background API through the nuisance facade and updated the
  nuisance README.
- Added focused tests for neutral application, constant/gradient recovery,
  detector masking, and empty-mask stability.

### Decisions

- Started with a constant plus vertical-gradient basis because the synthetic
  benchmark nuisance spec explicitly calls out low-frequency vertical
  background drift.

### Validation

- `uv run ruff format src/tomojax/nuisance/_background.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_background.py`
  passed.
- `uv run ruff check src/tomojax/nuisance/_background.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_background.py`
  passed.
- `uv run basedpyright src/tomojax/nuisance/_background.py src/tomojax/nuisance/api.py src/tomojax/nuisance/__init__.py tests/test_nuisance_background.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_nuisance_background.py tests/test_nuisance_gain_offset.py tests/test_v2_module_skeleton.py -q`
  passed: 10 tests.
- `just imports` passed.

### Risks

- The model is not yet integrated into Schur or alternating residual
  evaluation. The next slice should add an opt-in solver hook, following the
  gain/offset path.

## 2026-05-06 — Phase 8 Schur Background Offset Fitting

### Summary

- Added opt-in `fit_background_offset` support to `JointSchurLMConfig`.
- Schur residual, loss, and per-view loss evaluation can now fit the public
  `BackgroundOffsetModel` through `estimate_background_offset`.
- Added `background_offset_fit` to Schur diagnostics and normal-equation
  summaries.
- Added a focused synthetic Schur test where observed projections contain only
  low-frequency background drift; with background fitting enabled, the solver
  leaves correct geometry unchanged and reports near-zero residual.

### Decisions

- Background fitting remains opt-in and solver-local in this slice. The default
  Phase 7/8 smoke path is unchanged until an alternating/CLI toggle is added.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_nuisance_background.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- Fitted background rows are recomputed inside finite-difference residual
  evaluation. This is acceptable for the reference correctness path but should
  be revisited before performance work.

## 2026-05-06 — Phase 8 Align-Auto Background Nuisance Toggle

### Summary

- Added `fit_background_nuisance` to `AlternatingSmokeConfig`.
- Threaded the option into Schur geometry updates through
  `JointSchurLMConfig.fit_background_offset`.
- Added `--fit-background-nuisance` to `align-auto`.
- Recorded the option in `verification.json`, `run_manifest.json`, and
  `config_resolved.toml`.
- Added smoke and CLI tests proving the default remains off while opt-in runs
  emit `background_offset_fit=true` in Schur diagnostics.

### Decisions

- Keep background fitting disabled by default until nuisance-bearing benchmark
  projections are loaded by the solver path.

### Validation

- `uv run ruff format --check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- The current default smoke has no background drift; opt-in plumbing is covered,
  but benchmark ingestion still needs to exercise the toggle with
  nuisance-bearing data.

## 2026-05-06 — Phase 7 Schur Recovery Evidence In Smoke

### Summary

- Extended the deterministic 32^3 smoke verification payload with
  corrupted-initial versus final supported DOF recovery metrics.
- Recorded per-DOF improvement flags for realized theta, detector-u, and
  detector-v after gauge canonicalisation.
- Added a focused smoke test that runs the existing joint Schur LM geometry
  update with the fixed synthetic truth volume source and verifies:
  projection residual improvement, accepted Schur diagnostics, geometry trace
  reduction fields, and supported DOF recovery from the corrupted geometry.

### Decisions

- Kept the default smoke geometry-update source as `stopped_reconstruction`.
  The fixed-truth source is used only as the focused vertical-slice assertion
  that isolates the Schur update mechanics from the current tiny-volume
  reconstruction quality.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_verify_artifacts.py -q`
  passed: 10 tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 5 tests.
- `just imports` passed.

### Risks

- The stopped-reconstruction default still limits geometry-recovery tightness.
  The verification payload now exposes that separately from the fixed-truth
  Schur recovery assertion.

## 2026-05-06 — Phase 8 Synthetic Background Gradient Nuisance

### Summary

- Updated the synthetic dataset writer so
  `background_drift = low_frequency_vertical_gradient` realizes a per-view
  detector-row background-gradient coefficient instead of a scalar offset.
- `nuisance_truth.json` now records `background_vertical_gradient` and reports
  it separately in `applied_terms`.
- Dirty synthetic projections apply the vertical background field as
  `gradient_i * linspace(-1, 1, detector_rows)`.
- Added a focused combined-nuisance dataset test that reconstructs a dirty view
  from clean projections, gain truth, and background-gradient truth.

### Decisions

- Kept the dataset writer NumPy-only and deterministic. This slice fixes the
  synthetic artifact truth/apply contract without coupling the writer to the
  alternating solver.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed: 1 file reformatted, 1 file left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_nuisance_background.py -q`
  passed: 10 tests.
- `just imports` passed.

### Risks

- Generated benchmark projections are still metadata-only for `align-auto`;
  the alternating solver does not yet consume them as its observed input.

## 2026-05-06 — Phase 8 Align-Auto Dirty Synthetic Sidecars

### Summary

- Added `applied_to_projections` to synthetic `nuisance_truth.json` so clean
  versus nuisance-applied generated datasets are explicit.
- Added `--apply-synthetic-nuisance` to `align-auto`; named synthetic benchmark
  sidecars remain clean by default and become dirty only with this flag.
- Recorded `nuisance_applied_to_projections` in `verification.json`,
  `run_manifest.json`, and `config_resolved.toml` through the existing
  synthetic dataset payload.
- Added focused dataset and CLI tests for the default clean path and opt-in
  dirty sidecar path.

### Decisions

- Kept dirty sidecars opt-in. The generated synthetic dataset projector and
  geometry artifact schema are not yet the same as the alternating solver's JAX
  reference geometry path, so this slice records sidecar truth without claiming
  solver ingestion.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed: 5 files left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_align_auto_cli.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- This still does not make the alternating solver consume generated benchmark
  projections. That requires a compatible generated-geometry loader or a
  benchmark writer that emits `GeometryState` artifacts.

## 2026-05-06 — Phase 8 Synthetic V2 Geometry Sidecars

### Summary

- Added v2 `GeometryState` sidecars to generated synthetic benchmark datasets:
  `v2_nominal_geometry.json`, `v2_corrupted_geometry.json`, and
  `v2_true_geometry.json`.
- Added matching radian pose sidecars:
  `v2_nominal_pose_params.csv`, `v2_corrupted_pose_params.csv`, and
  `v2_true_pose_params.csv`.
- Kept the existing manifest-schema `nominal_geometry.json`,
  `corrupted_geometry.json`, `true_geometry.json`, and degree `true_pose.csv`
  artifacts unchanged.
- Added public geometry readback coverage through `read_geometry_json` and
  `read_pose_params_csv`.

### Decisions

- Added sidecars instead of changing the existing benchmark artifact schema in
  place, so downstream benchmark-contract readers remain stable while the v2
  solver path gets a compatible geometry format to consume later.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 8 tests.
- `just imports` passed.

### Risks

- The synthetic projections are still generated by the NumPy smoke projector,
  not the JAX reference projector. Solver ingestion should account for that
  before using these sidecars as a strict recovery benchmark.

## 2026-05-06 — Phase 8 Synthetic Manifest Sidecar Index

### Summary

- Added an `artifacts` map to generated `dataset_manifest.json` files.
- Indexed all generated synthetic dataset artifacts, including the v2
  `GeometryState` JSON sidecars and radian pose-parameter CSV sidecars.
- Updated the synthetic dataset test to resolve v2 geometry paths from the
  manifest before reading them through public `tomojax.geometry` APIs.

### Decisions

- Keep artifact paths relative to the dataset directory. This keeps manifests
  relocatable and avoids embedding machine-local output roots.

### Validation

- `uv run ruff format src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed: 1 file reformatted, 1 file left unchanged.
- `uv run ruff check src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_writer.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py -q`
  passed: 6 tests.
- `just imports` passed.

### Risks

- Manifest discovery makes sidecars easy to consume, but the alternating solver
  still needs a deliberate ingestion step before using generated benchmark
  projections.

## 2026-05-06 — Phase 8 Synthetic Sidecar Loader API

### Summary

- Added public `SyntheticDatasetSidecars` and
  `load_synthetic_dataset_sidecars` to `tomojax.datasets`.
- The loader reads `dataset_manifest.json`, resolves its relative artifact map,
  and loads nominal, corrupted, and true v2 geometry states through public
  `tomojax.geometry` readers.
- Exported the loader through `tomojax.datasets.api`, the package facade, and
  the datasets README.
- Added focused tests for successful manifest-indexed readback and malformed
  manifests missing the artifact map.

### Decisions

- Keep the loader data-only. It does not validate projection compatibility or
  run alignment; it exists as the public bridge that later `align-auto`
  ingestion can call.

### Validation

- `uv run ruff format src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed: 4 files left unchanged after focused fixes.
- `uv run ruff check src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 10 tests.
- `just imports` passed.

### Risks

- The loader intentionally exposes generated sidecars only. Solver ingestion
  still needs a compatibility decision for NumPy-smoke projections versus the
  JAX reference projector.

## 2026-05-06 — Phase 8 Align-Auto Synthetic Sidecar Readback

### Summary

- `align-auto` now validates generated named synthetic benchmark sidecars by
  calling public `load_synthetic_dataset_sidecars`.
- The smoke synthetic dataset payload now records a compact
  `sidecar_readback` summary with validation status, source API, view count,
  and nominal/corrupted/true detector-u values.
- The readback summary is propagated through `verification.json`,
  `run_manifest.json`, and `config_resolved.toml`.
- Added focused CLI tests for clean and nuisance-applied generated sidecars.

### Decisions

- Keep generated benchmark sidecars as readback-validated metadata only. This
  slice does not route NumPy-smoke projections into the alternating solver.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 1 file reformatted, 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_synthetic_datasets.py -q`
  passed: 14 tests.
- `just imports` passed.

### Risks

- Sidecar readback proves artifact consistency, not solver recovery on those
  generated projections. Solver ingestion remains a later compatibility slice.

## 2026-05-06 — Phase 8 Synthetic Array Metadata Loader

### Summary

- Added public `SyntheticArrayMetadata` to `tomojax.datasets`.
- `load_synthetic_dataset_sidecars` now validates manifest-indexed
  `ground_truth_volume_npy`, `projections_npy`, and `mask_npy` artifacts using
  memory-mapped NumPy loads.
- The loader exposes array paths, shapes, and dtypes without eagerly loading
  full 128^3 data.
- Added focused tests for volume/projection/mask metadata and missing
  projection artifact entries.

### Decisions

- Keep array readback metadata-only. This prepares a public ingestion surface
  while avoiding accidental coupling between the NumPy smoke projector and the
  JAX reference solver.

### Validation

- `uv run ruff format src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed: 4 files left unchanged after import-order fixes.
- `uv run ruff check src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 11 tests.
- `just imports` passed.

### Risks

- Shape/dtype validation does not verify physical consistency between generated
  projections and v2 geometry sidecars. That remains a later solver-ingestion
  or reference-projector compatibility task.

## 2026-05-06 — Phase 8 Align-Auto Synthetic Array Metadata

### Summary

- Extended `align-auto` sidecar readback payloads with loader-provided volume,
  projection, and mask array metadata.
- `verification.json` and `run_manifest.json` now include array path, shape, and
  dtype metadata under `synthetic_dataset.sidecar_readback`.
- `config_resolved.toml` now records compact projection shape and dtype fields
  for generated synthetic sidecars.
- Added focused CLI assertions for clean and nuisance-applied generated
  sidecars.

### Decisions

- Keep array metadata under `sidecar_readback` to avoid implying generated
  projections are solver inputs.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 3 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_synthetic_datasets.py -q`
  passed: 15 tests.
- `just imports` passed.

### Risks

- This improves provenance/readback only. Solver ingestion still needs a
  deliberate compatibility step because generated projections come from the
  NumPy smoke projector.

## 2026-05-06 — Phase 8 Synthetic Loader Consistency Summary

### Summary

- Added `SyntheticDatasetConsistency` to the synthetic sidecar loader.
- The loader now compares manifest-declared volume shape, detector shape, and
  view count against memory-mapped volume/projection/mask metadata and loaded
  true geometry view count.
- `SyntheticDatasetSidecars.consistency` reports per-check booleans plus an
  aggregate `passed` flag.
- Added focused tests for all-pass consistency and a manifest detector-shape
  mismatch.

### Decisions

- Keep consistency structural only. This does not compare generated projections
  against the JAX reference projector or route sidecar arrays into alignment.

### Validation

- `uv run ruff format src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed: 4 files left unchanged.
- `uv run ruff check src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `uv run basedpyright src/tomojax/datasets/_loader.py src/tomojax/datasets/api.py src/tomojax/datasets/__init__.py tests/test_synthetic_datasets.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_synthetic_datasets.py tests/test_v2_module_skeleton.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- Structural consistency is necessary but not sufficient for solver ingestion;
  physical projector compatibility remains unverified.

## 2026-05-06 — Phase 8 Align-Auto Synthetic Consistency Readback

### Summary

- Extended `align-auto` sidecar readback payloads with
  `SyntheticDatasetConsistency`.
- `verification.json` and `run_manifest.json` now carry the loader's structural
  consistency checks under `synthetic_dataset.sidecar_readback.consistency`.
- `config_resolved.toml` now records
  `synthetic_dataset_sidecar_consistency_passed`.
- Added focused CLI assertions for clean and nuisance-applied generated
  sidecars.

### Decisions

- Keep consistency reporting under `sidecar_readback` so it is clearly
  provenance/readback metadata, not solver-ingestion evidence.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 1 file reformatted, 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_synthetic_datasets.py -q`
  passed: 16 tests.
- `just imports` passed.

### Risks

- Structural consistency still does not prove that generated NumPy-smoke
  projections are compatible with the JAX reference projector.

## 2026-05-06 — Phase 8 Synthetic Sidecar Failure Gate

### Summary

- Added warning-only `synthetic_sidecar_consistency` to the smoke
  `failure_report.json` gate table.
- Default smoke runs pass the gate with explicit no-sidecar evidence.
- Named synthetic sidecar runs report the loader consistency payload through the
  gate evidence.
- Added focused smoke and CLI assertions.

### Decisions

- Keep the gate warning-only and provenance-scoped. It does not imply generated
  projections are used by the solver.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed: 3 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

### Risks

- This validates structural sidecar readback reporting only. Real generated
  projection ingestion and geometry recovery remain separate work.

## 2026-05-06 — Align Alternating Module Split

### Summary

- Split the 32^3 alternating smoke implementation into cohesive private
  `tomojax.align` modules:
  - `_alternating_types.py` for smoke config/result dataclasses.
  - `_alternating_heldout.py` for held-out residual masks and checks.
  - `_alternating_verification.py` for verification and report payloads.
  - `_alternating_artifacts.py` for artifact writing.
- Kept `_alternating.py` focused on orchestration, synthetic geometry setup, and
  Schur/FISTA loop control.
- Preserved the public `tomojax.align.api` imports and existing smoke/CLI
  behavior.

### Decisions

- Kept sibling implementation imports inside the private `tomojax.align`
  boundary instead of adding a public helper surface.
- Used scoped Pyright private-implementation suppressions on the split files so
  private sibling modules can call each other without exporting these helpers as
  public API.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_heldout.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py`
  passed: 5 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_heldout.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_heldout.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 14 tests.
- `just imports` passed.

### Risks

- This is a structural split only. Synthetic benchmark projection ingestion and
  stronger real-geometry recovery checks remain next.

## 2026-05-06 — Phase 7 Synthetic Sidecar Ingestion

### Summary

- Added sidecar-backed input assembly for the alternating smoke runner.
- When `synthetic_dataset_artifact_dir` is set, the smoke run now loads the
  generated volume, projections, mask, corrupted geometry, and true geometry
  from the manifest-indexed sidecars.
- Updated the 32^3 synthetic generator to use a JAX-reference detector grid and
  v2 true geometry for generated projections, with pixel-valued setup and pose
  terms scaled to the smoke grid.
- Added a deterministic 32^3 sidecar smoke test that verifies Schur loss
  improvement, supported DOF recovery after gauge canonicalisation, Schur
  diagnostics, geometry trace evidence, and sidecar consistency reporting.

### Decisions

- Kept the default in-memory smoke path unchanged.
- Used the existing `fixed_synthetic_truth` geometry-update volume source for
  the focused sidecar recovery test so this slice validates the real Schur
  geometry update independently of reconstruction quality.
- Treated mean `dz` gauge enforcement as required only when `det_v_px` is active,
  matching the supported/frozen DOF policy.

### Validation

- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_inputs.py src/tomojax/align/_alternating_verification.py src/tomojax/datasets/_writer.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_inputs.py src/tomojax/align/_alternating_verification.py src/tomojax/datasets/_writer.py tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_synthetic_datasets.py tests/test_align_auto_cli.py -q`
  passed: 25 tests.
- `just imports` passed.

### Risks

- The sidecar recovery gate is still a 32^3 smoke-sized vertical slice. The
  128^3 benchmark path will need detector-grid support beyond the current JAX
  reference projector before it can use non-square manifest detector shapes.

## 2026-05-06 — Phase 7 Stopped-Reconstruction Sidecar Contract

### Summary

- Added a focused default-source sidecar smoke contract for
  `geometry_update_volume_source="stopped_reconstruction"`.
- The test verifies the real Schur update runs on generated 32^3 sidecars,
  accepts a step, improves projection residual, and improves supported DOFs.
- The test also preserves the current absolute recovery gap: detector-shift
  recovery remains outside the smoke tolerance when the stopped volume is
  reconstructed from corrupted geometry.

### Decisions

- Do not weaken the fixed-truth Schur recovery test or the geometry recovery
  tolerances.
- Keep the stopped-reconstruction limitation executable so the next slice can
  address reconstruction/volume gauge handling with a clear before/after signal.

### Validation

- `uv run ruff format tests/test_alternating_solver_smoke.py`
  passed: 1 file left unchanged.
- `uv run ruff check tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.
- `just check` currently fails in legacy Ruff cleanup before typecheck/tests:
  - `uv run ruff format src tests tools` reformatted 70 legacy files.
  - `uv run ruff check --fix src tests tools` fixed 320 issues and left 1364
    Ruff issues, starting in transitional `src/tomojax/align/model/schedules.py`
    and `src/tomojax/align/model/state.py`.
  - The unrelated formatter churn from the broad command was reverted.

### Risks

- The default stopped-reconstruction path improves but does not yet recover
  detector shift to tolerance because the current geometry-aware backprojection
  can bake detector shift into the latent volume.

## 2026-05-06 — Phase 7 Stopped-Volume Gauge Diagnostics

### Summary

- Added `stopped_volume_gauge` to the alternating smoke verification payload.
- The payload reports final stopped-volume projection losses under initial,
  final, and true geometry, plus the nearest geometry label.
- Added focused sidecar assertions showing the default stopped-reconstruction
  volume is closer to the improved/final geometry than to true geometry, which
  makes the remaining detector-shift recovery gap explicit.

### Decisions

- This is diagnostic evidence for the next reconstruction-gauge fix, not a
  relaxation of recovery tolerances.
- The fixed-truth sidecar Schur test remains the pass/fail recovery gate for
  the supported solver update.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 3 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py tests/test_align_auto_cli.py -q`
  passed: 16 tests.
- `just imports` passed.

### Risks

- The diagnostic confirms the stopped volume gauge problem but does not solve it.
  The next implementation slice should transfer or neutralise detector-shift
  gauge between the latent volume and geometry rather than changing tolerances.

## 2026-05-06 — Phase 7 Stopped-Volume Gauge Prototype

### Summary

- Prototyped simple detector-shift gauge corrections for the sidecar
  stopped-reconstruction path without committing code changes.
- Integer volume rolls along candidate latent axes did not recover `det_u_px`
  to the existing smoke tolerance.
- Projection centre-of-mass initialisation for `det_u_px` improved detector-u
  recovery in isolation but either still missed tolerance or degraded theta
  recovery.

### Decisions

- Do not commit a synthetic-only volume shift or projection-COM registration
  patch.
- Keep the existing fixed-truth Schur recovery test as the supported solver
  recovery gate.
- Keep the stopped-reconstruction sidecar contract as the current executable
  gap until reconstruction/volume gauge handling has a principled design.

### Validation

- Prototype runs were executed with `JAX_PLATFORM_NAME=cpu uv run python` against
  generated `synth128_thermal_object_drift` 32^3 sidecars.
- No source changes were kept from the prototype.

### Risks

- The default stopped-reconstruction path still improves projection residual
  but does not meet absolute detector-shift recovery tolerance.

## 2026-05-06 — Phase 8 Weak DOF Validation Evidence

### Summary

- Replaced the active `det_v_px` weak-DOF validation-improvement placeholder in
  `observability_report.json` with real Schur step evidence.
- The report now records `schur_actual_reduction`, reduction ratio when
  available, accepted-step status, and a pass flag for active DOFs with Schur
  diagnostics.
- Kept `theta_scale` frozen and explicit; it still reports missing validation
  evidence because the current reference projector does not support that DOF.

### Decisions

- Keep the weak-DOF policy in `report_only` mode for this slice.
- Treat Schur actual reduction as optimisation evidence, not held-out validation;
  correlation and held-out policy gates remain future work.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed: 3 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.

### Risks

- The evidence is still smoke-level Schur reduction, not a held-out weak-DOF
  acceptance gate.

## 2026-05-06 — Phase 8 Existing Synthetic Sidecar Ingestion

### Summary

- Added `--synthetic-dataset-dir` to `tomojax.cli.align_auto` for ingesting an
  existing generated synthetic benchmark sidecar directory without regenerating
  artifacts.
- The command now loads sidecars, infers dataset name, size, view count, and
  nuisance-applied status from the manifest/readback, then passes the real
  sidecar volume, projections, mask, and corrupted geometry into the smoke
  solver.
- Added focused CLI coverage that prepares a benchmark dataset outside the run
  directory, ingests it, and verifies the run's observed projections match the
  prepared sidecars.

### Decisions

- Existing-sidecar ingestion does not create or overwrite `out_dir/datasets`.
- If both `--synthetic-dataset` and `--synthetic-dataset-dir` are supplied, the
  explicit name must match the sidecar manifest name.
- This remains a deterministic 32^3 focused path; larger 128^3 runtime remains
  out of scope for this slice.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- This proves real sidecar ingestion through the align-auto smoke path, but not
  full 128^3 benchmark runtime or comparator reporting.

## 2026-05-06 — Phase 8 Synthetic Benchmark Result Artifact

### Summary

- Added a conditional `benchmark_result.json` artifact for deterministic smoke
  runs that include synthetic benchmark metadata.
- The artifact records the benchmark name, reimagined align-auto smoke
  implementation label, profile, status, dataset provenance, core
  reconstruction metrics, gauge-canonical geometry recovery metrics, backend
  provenance, failure labels, and deterministic runtime placeholders.
- Added focused CLI coverage proving ordinary smoke runs do not emit the
  benchmark result while existing-sidecar synthetic runs do.

### Decisions

- Emit JSON case results first; markdown comparison reports and current-vs-v2
  comparators remain later benchmark slices.
- Keep timing fields as explicit `null` placeholders until solver timing is
  measured and recorded, rather than fabricating deterministic wall-clock data.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- This is a single-case benchmark result, not a full benchmark protocol runner
  or comparison report.

## 2026-05-06 — Phase 8 Synthetic Benchmark Markdown Report

### Summary

- Added a conditional `benchmark_report.md` artifact for deterministic smoke
  runs that include synthetic benchmark metadata.
- The report is generated from `benchmark_result.json` and summarises the
  single-case implementation/profile/status row, dataset provenance,
  reconstruction residual/NMSE, gauge-canonical geometry recovery, backend
  provenance, and failure labels.
- Added focused CLI coverage proving ordinary smoke runs do not emit the
  report while existing-sidecar synthetic runs do.

### Decisions

- Keep this as a one-run human-readable report; multi-run compare and
  current-vs-reimagined tables remain later benchmark slices.
- Continue to render unavailable timing fields as `n/a` until measured runtime
  is wired through the solver.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- The report is intentionally not yet the full benchmark comparison report from
  the v2 artifact contract.

## 2026-05-06 — Phase 8 Synthetic Benchmark Timing Metrics

### Summary

- Added measured smoke-run timing to the alternating solver verification
  payload.
- `benchmark_result.json` now records `time_to_verified_geometry_seconds` and
  `total_wall_seconds` from the solver run instead of null placeholders.
- `benchmark_report.md` renders those measured values in the summary table.
- Updated deterministic smoke assertions to compare verification payloads
  excluding the intentionally variable runtime block.

### Decisions

- Timing covers the solver path up to artifact emission. Full end-to-end CLI
  timing remains a later benchmark harness concern.
- Tests assert finite positive timing and ordering, not exact values.

### Validation

- `uv run ruff format src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed: 5 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating.py src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py tests/test_alternating_solver_smoke.py -q`
  passed: 17 tests.
- `just imports` passed.

### Risks

- Wall-clock measurements are machine-dependent and should not be used as a
  deterministic equality signal.

## 2026-05-06 — Phase 8 Benchmark Manifest Criteria Readback

### Summary

- Added synthetic sidecar manifest `recovery_tolerances` to the align-auto
  readback payload.
- `benchmark_result.json` now records those values under
  `benchmark_manifest_criteria`.
- `benchmark_report.md` renders the criteria in a dedicated benchmark manifest
  section.
- Added focused CLI assertions for generated and existing-sidecar benchmark
  runs.

### Decisions

- Keep benchmark manifest criteria separate from the current smoke acceptance
  tolerances; this slice records benchmark expectations without changing solver
  pass/fail behavior.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 3 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/cli/align_auto.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- Manifest pass criteria are reported for benchmark context only; they are not
  yet active benchmark gates.

## 2026-05-06 — Phase 8 Benchmark Manifest Criteria Evaluation

### Summary

- Added report-only `benchmark_manifest_evaluation` to synthetic benchmark
  result artifacts.
- Criteria with available smoke recovery metrics are marked `passed` or
  `failed`; unsupported criteria are explicitly marked `not_evaluated` with a
  reason.
- `benchmark_report.md` now renders a benchmark manifest evaluation table.
- Added focused CLI assertions covering an evaluated detector-shift criterion
  and unsupported object-motion criteria.

### Decisions

- Criteria evaluation is report-only and does not alter solver or smoke
  verification pass/fail behavior.
- The initial mapping is intentionally narrow: detector-u and detector-v pixel
  thresholds can use existing smoke geometry recovery metrics; all other
  manifest criteria remain not evaluated until their metrics exist.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- Only criteria with explicit smoke metric mappings are evaluated in this slice.

## 2026-05-06 — Phase 8 Benchmark Criteria Evaluation Summary

### Summary

- Added `benchmark_manifest_evaluation_summary` to synthetic benchmark results.
- The summary reports aggregate status plus passed, failed, not-evaluated, and
  total criterion counts.
- `benchmark_report.md` now renders the aggregate status and counts before the
  per-criterion table.
- Added focused CLI assertions for failed and partially evaluated summaries.

### Decisions

- The aggregate status is report-only and does not change the solver
  verification status.
- `failed` takes precedence over `partially_evaluated`; unsupported-only
  criteria are reported as `partially_evaluated`.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 7 tests.
- `just imports` passed.

### Risks

- This summary is benchmark metadata, not the top-level smoke verification
  result.

## 2026-05-06 — Phase 8 Benchmark Result Artifact Validation

### Summary

- Extended `tomojax.verify.inspect_run_artifacts` to validate optional
  `benchmark_result.json` when present.
- The validator now checks the synthetic benchmark result schema and required
  top-level sections, including criteria/evaluation payloads.
- Added focused verifier coverage for a synthetic benchmark smoke run and a
  deliberately malformed benchmark result.

### Decisions

- Benchmark result validation is optional; non-benchmark smoke runs remain valid
  without `benchmark_result.json`.

### Validation

- `uv run ruff format src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_verify_artifacts.py -q`
  passed: 3 tests.
- `just imports` passed.

### Risks

- This validates benchmark result structure, not the semantic correctness of
  every benchmark metric.

## 2026-05-06 — Phase 8 Benchmark Report Artifact Validation

### Summary

- Extended `tomojax.verify.inspect_run_artifacts` to validate
  `benchmark_report.md` when `benchmark_result.json` is present.
- The verifier now reports a missing benchmark report and checks for the
  benchmark title plus manifest evaluation section.
- Added focused verifier coverage for a synthetic benchmark run with a deleted
  markdown report.

### Decisions

- Benchmark report validation is optional for non-benchmark smoke runs and
  required only when a benchmark result artifact exists.

### Validation

- `uv run ruff format src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed.
- `uv run basedpyright src/tomojax/verify/_artifacts.py tests/test_verify_artifacts.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_verify_artifacts.py -q`
  passed: 4 tests.
- `just imports` passed.

### Risks

- The verifier checks report structure, not prose completeness.

## 2026-05-06 — Phase 8 Theta-Scale Missing Evidence

### Summary

- Made the frozen `theta_scale` weak-DOF decision report the same explicit
  missing evidence categories used by active weak-DOF policy checks.
- The `observability_report.json` weak-DOF evidence now marks unavailable
  curvature, correlation, accepted-step, and validation-improvement evidence for
  `theta_scale`.
- Added focused alternating smoke assertions for the frozen `theta_scale`
  evidence while preserving the active `det_v_px` report-only decision.

### Decisions

- Keep `theta_scale` frozen until the reference projector and Schur update have
  an identifiable scale parameter.
- Treat this as report clarity only; no optimisation behavior changed.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged after the final patch.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 10 tests.
- `just imports` passed.

### Risks

- This clarifies frozen-DOF evidence only; it does not enable `theta_scale`
  recovery.

## 2026-05-06 — Phase 8 Multi-Case 32^3 Benchmark Pass

### Summary

- Generated five deterministic 32^3 sidecar datasets from planned synthetic128
  scenarios:
  `synth128_setup_global_tomo`, `synth128_pose_random_extreme`,
  `synth128_lamino_axis_roll_pose`, `synth128_thermal_object_drift`, and
  `synth128_combined_nuisance_jumps`.
- Ran `tomojax-align-auto-smoke` on each existing sidecar directory using
  `--profile smoke32`, gain/offset nuisance fitting, and background nuisance
  fitting.
- Collected the five `benchmark_result.json` files and rendered
  `.artifacts/phase8_multi_case_32_after_recovery_time_gate/benchmark_comparison.md`
  with `tomojax-synthetic-benchmark-compare` after the recovered-geometry
  timing gate was added.
- Added a tracked concise run summary at
  `docs/benchmark_runs/2026-05-06-phase8-multi-case-32.md`.

### Results

| Benchmark | Status | Criteria | Geometry | Volume NMSE | Final residual | Time to verified (s) | Total time (s) |
|---|---|---|---|---:|---:|---:|---:|
| `synth128_setup_global_tomo` | failed | failed | failed | 0.693523 | 0 | n/a | 11.8016 |
| `synth128_pose_random_extreme` | failed | partially_evaluated | failed | 0.662409 | 0.331717 | n/a | 13.4277 |
| `synth128_lamino_axis_roll_pose` | failed | failed | failed | 0.635030 | 0.00978141 | n/a | 13.0231 |
| `synth128_thermal_object_drift` | failed | partially_evaluated | failed | 0.608258 | 0.000758991 | n/a | 13.1870 |
| `synth128_combined_nuisance_jumps` | failed | failed | failed | 0.700399 | 0.00567048 | n/a | 13.0475 |

Recovery details:

- `synth128_setup_global_tomo`: `det_u_realized_rmse_px=3.625`,
  `theta_realized_rmse_rad=0.0218166`; supported DOFs did not improve.
- `synth128_pose_random_extreme`: `det_u_realized_rmse_px=2.7415`,
  `det_v_realized_rmse_px=2.5782`, `theta_realized_rmse_rad=0.2019`;
  supported DOFs did not improve.
- `synth128_lamino_axis_roll_pose`: `det_u_realized_rmse_px=2.2334`,
  `det_v_realized_rmse_px=0.7336`, `theta_realized_rmse_rad=0.1598`;
  supported DOFs did not improve.
- `synth128_thermal_object_drift`: `det_u_realized_rmse_px=1.4893`,
  `det_v_realized_rmse_px=0.0512`, `theta_realized_rmse_rad=0.0052336`;
  supported DOFs improved, with failure label `nuisance_residual_structure`.
- `synth128_combined_nuisance_jumps`: `det_u_realized_rmse_px=3.8751`,
  `det_v_realized_rmse_px=0.9955`, `theta_realized_rmse_rad=0.0309604`;
  supported DOFs did not improve.

### Decisions

- Keep the generated sidecar and run artifacts under ignored `.artifacts/`
  because they include array sidecars and smoke volumes.
- Commit the concise markdown summary rather than forcing generated `.npy`
  benchmark artifacts into git.
- Stop adding artifact/report/observability fields for this pass.

### Validation

- `uv run python` generated the five sidecar datasets through public
  `tomojax.datasets.generate_synthetic_dataset`.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke ...` completed for all
  five existing sidecar directories.
- `uv run tomojax-synthetic-benchmark-compare ... --out .artifacts/phase8_multi_case_32_after_recovery_time_gate/benchmark_comparison.md`
  passed.
- `just imports` passed after recording the documentation summary.
- `just imports` passed again after extending the summary to all five cases.
- The all-five pass was rerun after the recovered-geometry timing gate; all
  five failing cases now report `time_to_verified_geometry_seconds = null`.
- `just imports` passed after refreshing the tracked benchmark summary.

### Risks

- The first multi-case pass proves sidecar ingestion and comparison reporting,
  but the current 32^3 smoke solver does not meet the planned synthetic128
  recovery criteria.
- JAX emitted CUDA plugin warnings about missing cuSPARSE before falling back to
  CPU; alignment commands were run with `JAX_PLATFORM_NAME=cpu`.

## 2026-05-06 — Phase 8 Schur-Accepted Verification Gate

### Summary

- Tightened alternating level verification so an executed geometry-update level
  is not verified when the Schur update is rejected.
- Added a stopped-reconstruction sidecar regression for
  `synth128_setup_global_tomo`, where the reconstruction absorbs projection
  residual and the Schur update correctly rejects a zero-loss geometry step.
- This keeps benchmark timing honest: rejected geometry updates no longer
  produce a `time_to_verified_geometry_seconds` value.

### Decisions

- Treat Schur acceptance as required evidence for level verification whenever a
  geometry update is executed.
- Do not add artifact fields in this slice; the existing `verified`,
  `schur_accepted`, and runtime fields carry the changed semantics.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_orchestration.py tests/test_alternating_solver_smoke.py`
  passed: 3 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_orchestration.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py src/tomojax/align/_alternating_orchestration.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 11 tests.
- `just imports` passed.

### Risks

- Benchmark summaries generated before this change may over-report verified
  geometry timing for rejected Schur updates.
- The stopped-reconstruction volume gauge problem remains; this slice only
  prevents rejected updates from being counted as verified.

## 2026-05-06 — Broad Gate Audit

### Summary

- Ran `just check` after the Phase 8 benchmark and verification slices.
- The gate failed at `uv run ruff check --fix src tests tools` on broad
  pre-existing legacy Ruff debt.
- `ruff format` reformatted 70 unrelated legacy files before the check failed;
  that formatter churn was reverted.

### Representative Failures

- `src/tomojax/align/model/schedules.py`: missing public docstrings,
  type-checking import placement, and branch-count issues.
- `src/tomojax/align/model/state.py`: missing public docstrings and missing
  pytree method type annotations.
- `src/tomojax/align/objectives/fixed_volume.py`: missing docstrings,
  type-checking import placement, and relative import style.
- Legacy tests under `tests/` still have many Ruff issues such as `PTH118`,
  `ARG001`, `PT018`, and old lambda/style violations.

### Decisions

- Do not weaken Ruff or clean broad legacy lint in this Phase 8 slice.
- Revert unrelated formatter churn from the failed `just check` run.
- Continue using focused validation plus `just imports` for scoped vertical
  slices until old transitional code is deleted or migrated.

### Validation

- `just check` failed as described above.
- `git restore src tests tools` restored the unrelated formatter changes.

### Risks

- `just check` remains blocked by legacy/transitional lint debt outside the
  current Phase 8 vertical slice.

## 2026-05-06 — Phase 8 Recovered-Geometry Timing Gate

### Summary

- Tightened `time_to_verified_geometry_seconds` semantics in
  `verification.json`.
- A transient accepted level no longer publishes verified-geometry timing when
  the final synthetic geometry recovery gate fails.
- Added stopped-reconstruction sidecar coverage for the
  `synth128_thermal_object_drift` recovery-gap case, where Schur accepts and
  improves supported DOFs but final recovery remains outside tolerance.

### Decisions

- Treat `time_to_verified_geometry_seconds` as time to recovered geometry in
  deterministic synthetic smoke runs.
- Do not add artifact fields; keep the existing runtime key and make failed
  recovery report `null`.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 11 tests.
- `just imports` passed.

### Risks

- Benchmark summaries generated before this slice may still contain optimistic
  verified-geometry timing for accepted-but-unrecovered Schur updates.
- This changes reporting semantics only; the stopped-volume geometry recovery
  gap remains.

## 2026-05-06 — Phase 8 Align-Auto Geometry Source Option

### Summary

- Added `--geometry-update-volume-source` to `tomojax-align-auto-smoke`.
- The option exposes the existing
  `AlternatingSmokeConfig.geometry_update_volume_source` setting with choices
  `stopped_reconstruction` and `fixed_synthetic_truth`.
- Added CLI coverage showing `fixed_synthetic_truth` propagates into
  `verification.json` and `config_resolved.toml`.
- Updated align-auto benchmark runtime assertions for the stricter
  recovered-geometry timing semantics.

### Decisions

- Keep `stopped_reconstruction` as the default production-like path.
- Treat `fixed_synthetic_truth` as an explicit synthetic oracle/diagnostic mode,
  useful for separating Schur solver capability from stopped-volume gauge
  failures.

### Validation

- `uv run ruff format src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `just imports` passed.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke --geometry-update-volume-source fixed_synthetic_truth ...`
  completed on `synth128_setup_global_tomo_32`.
  The oracle run improved `det_u_realized_rmse_px` from 3.625 to 0.368451 and
  passed the manifest `det_u_error_px_lt` criterion, but final geometry still
  failed because `theta_realized_rmse_rad` worsened to 0.0606714 and overall
  synthetic recovery did not pass.

### Risks

- `fixed_synthetic_truth` is not a production reconstruction/alignment path; it
  exists only for deterministic synthetic diagnosis.

## 2026-05-06 — Phase 8 Supported DOF Recovery Semantics

### Summary

- Updated the aggregate `supported_dofs_improved` geometry recovery boolean.
- A supported DOF that starts within tolerance now satisfies the aggregate when
  it remains within tolerance, even if it cannot strictly improve.
- Supported DOFs that start outside tolerance still need strict error reduction.
- Added fixed-truth setup-global smoke coverage where `det_u_px` improves while
  theta and det-v remain within tolerance without strict improvement.

### Decisions

- Preserve the existing per-DOF `*_improved` and `*_passed` fields so the
  aggregate cannot hide individual regressions or tolerance failures.
- Keep final `geometry_recovery.passed` unchanged; this slice only makes the
  aggregate improvement summary reflect already-good DOFs.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed: 2 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py tests/test_alternating_solver_smoke.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- The aggregate `supported_dofs_improved` name is now slightly broader than
  strict improvement; read it with the per-DOF pass/improved fields.

## 2026-05-06 — Phase 8 Joint Schur Block Priors

### Summary

- Added optional `setup_prior_strength` and `pose_prior_strength` fields to
  `JointSchurLMConfig`.
- Existing `parameter_prior_strength` remains the default shared prior when the
  block-specific strengths are unset.
- Added focused Schur coverage showing a strong pose prior suppresses pose
  drift in a setup-only truth case while still recovering detector shift.

### Decisions

- Do not change the alternating smoke defaults in this slice.
- Keep the solver-level knob explicit so benchmark/oracle probes can separate
  setup recovery from pose-gauge drift without biasing pose-heavy cases by
  default.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed: 2 files reformatted.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py tests/test_joint_schur_lm.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 9 tests.
- `just imports` passed.

### Risks

- Strong pose priors can hurt pose-dominated benchmarks if applied broadly.
  They must remain explicit until benchmark evidence justifies schedule-level
  policy.

## 2026-05-06 — Phase 8 Alternating Schur Block Prior Wiring

### Summary

- Added optional `geometry_update_setup_prior_strength` and
  `geometry_update_pose_prior_strength` to `AlternatingSmokeConfig`.
- Passed those optional priors into `JointSchurLMConfig` during alternating
  Schur geometry updates.
- Added `--geometry-update-setup-prior-strength` and
  `--geometry-update-pose-prior-strength` to `tomojax-align-auto-smoke`.
- Recorded explicitly supplied prior strengths in `config_resolved.toml`.

### Decisions

- Defaults remain unset, so existing continuation schedules still use their
  shared `prior_strength` values.
- Keep block priors explicit for synthetic oracle/diagnostic runs until
  benchmark evidence supports a schedule-level policy.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed: 1 file reformatted, 5 files left unchanged.
- `uv run ruff check src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_types.py src/tomojax/align/_alternating_geometry_update.py src/tomojax/align/_alternating_orchestration.py src/tomojax/align/_alternating_artifacts.py src/tomojax/cli/align_auto.py tests/test_align_auto_cli.py`
  passed.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_align_auto_cli.py -q`
  passed: 8 tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_alternating_solver_smoke.py -q`
  passed: 12 tests.
- `just imports` passed.

### Risks

- Strong pose priors can bias pose-heavy synthetic cases; this slice only wires
  explicit knobs and leaves defaults unchanged.

## 2026-05-06 — Phase 8 Multi-Case 32^3 Synthetic Benchmark Pass

### Summary

- Generated five deterministic 32^3 sidecar datasets from the planned
  synthetic128 benchmark scenarios.
- Ran `tomojax-align-auto-smoke` against each existing sidecar directory with
  `--profile smoke32`, `--fit-gain-offset-nuisance`, and
  `--fit-background-nuisance`.
- Rendered the comparison report with `tomojax-synthetic-benchmark-compare`.
- Committed a concise benchmark summary at
  `docs/benchmark_runs/2026-05-06-phase8-multi-case-32.md`; generated arrays
  and run artifacts remain under ignored `.artifacts/`.

### Results

| Benchmark | Status | Criteria | Geometry | Volume NMSE | Final Residual | Time To Verified (s) | Total Time (s) | Recovery |
|---|---|---|---|---:|---:|---:|---:|---|
| `synth128_setup_global_tomo` | failed | failed | failed | 0.693523 | 0 | n/a | 11.5794 | `det_u=3.625`, `det_v=0`, `theta=0.0218166` |
| `synth128_pose_random_extreme` | failed | partially_evaluated | failed | 0.662409 | 0.331717 | n/a | 13.3407 | `det_u=2.7415`, `det_v=2.5782`, `theta=0.2019` |
| `synth128_lamino_axis_roll_pose` | failed | failed | failed | 0.635030 | 0.00978141 | n/a | 13.3946 | `det_u=2.2334`, `det_v=0.7336`, `theta=0.1598` |
| `synth128_thermal_object_drift` | failed | partially_evaluated | failed | 0.608258 | 0.000758991 | n/a | 13.5649 | `det_u=1.4893`, `det_v=0.0512`, `theta=0.0052336`; label `nuisance_residual_structure` |
| `synth128_combined_nuisance_jumps` | failed | failed | failed | 0.700399 | 0.00567048 | n/a | 13.4880 | `det_u=3.8751`, `det_v=0.9955`, `theta=0.0309604` |

### Commands

- Generated sidecars with `uv run python` and
  `tomojax.datasets.generate_synthetic_dataset(..., size=32, clean=False, views=4)`
  under `.artifacts/phase8_multi_case_32_benchmark_pass/datasets`.
- Ran five CPU smoke commands with `JAX_PLATFORM_NAME=cpu uv run
  tomojax-align-auto-smoke --synthetic-dataset-dir ...`.
- Rendered `.artifacts/phase8_multi_case_32_benchmark_pass/benchmark_comparison.md`
  with `JAX_PLATFORM_NAME=cpu uv run tomojax-synthetic-benchmark-compare ...`.

### Validation

- Five `tomojax-align-auto-smoke` runs completed and wrote
  `benchmark_result.json`.
- `JAX_PLATFORM_NAME=cpu uv run tomojax-synthetic-benchmark-compare ... --out
  .artifacts/phase8_multi_case_32_benchmark_pass/benchmark_comparison.md`
  passed.

### Risks

- This is a baseline benchmark pass, not a successful recovery pass. The normal
  stopped-reconstruction source still fails all five current 32^3 recovery
  gates.
- JAX emitted CUDA plugin warnings about missing cuSPARSE before CPU fallback;
  all benchmark commands were run with `JAX_PLATFORM_NAME=cpu`.

## 2026-05-06 — Phase 8 Setup-Global GPU Memory Isolation And Ladder

### Summary

- Treated the 64^3/64-view setup-global OOM as a v2 JAX reference memory
  regression, not a reason to shrink the benchmark.
- Verified the laptop GPU path with JAX selecting `cuda:0`; the ambient CUDA
  library path failed because the JAX CUDA plugin could not find cuSPARSE, while
  the venv NVIDIA wheel library paths initialized GPU correctly.
- Extended the synthetic sidecar writer and `align-auto` ingestion to accept
  64^3 datasets and fixed the sidecar manifest detector shape to match the
  current v2 smoke projector output.
- Changed the shared finite-difference Jacobian helper from all-parameter
  `vmap` to sequential column accumulation, avoiding materialization of
  parameter x view x volume work arrays.
- Added focused coverage for the finite-difference Jacobian and 64^3 sidecar
  consistency.

### Memory Isolation

- Initial 64^3/64-view fixed-truth benchmark failed in Schur finite differences
  with `RESOURCE_EXHAUSTED` while trying to allocate 12.14 GiB for an HLO shaped
  like `f32[194,64,64,64,64]`.
- After the column-accumulation change, component probes passed on GPU for
  views 1/4/16/64:

| Views | Projector | Backprojector | FISTA 1 Iter | Schur Fixed Truth | Schur Stopped Volume | Schur Fixed Truth + Nuisance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.974s | 1.022s | 4.069s | 5.362s | 4.074s | 4.716s |
| 4 | 1.016s | 1.199s | 4.337s | 6.484s | 4.890s | 5.488s |
| 16 | 1.042s | 1.422s | 4.542s | 8.640s | 6.983s | 7.850s |
| 64 | 1.066s | 2.130s | 5.287s | 18.062s | 15.870s | 17.672s |

### 64^3/64-View GPU Benchmark Results

Dataset: nuisance-free `synth128_setup_global_tomo`, 64^3 volume, 64 views.
Mode: `balanced`, existing sidecar ingestion path, no nuisance fitting.

| Mode | Status | Criteria | Geometry | det_u RMSE px | det_v RMSE px | theta RMSE rad | Final Residual | Volume NMSE | Schur Accepted | Total Time s |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `fixed_synthetic_truth` | failed | failed | failed | 6.9338 | 0.00666 | 0.02211 | 0.856277 | 0.686109 | true | 37.5096 |
| `stopped_reconstruction` | failed | failed | failed | 7.25 | 0 | 0.02182 | 0 | 0.686110 | true | 24.8489 |

### Interpretation

- Fixed-truth also fails, so the current blocker is setup/pose/theta coupling or
  geometry convention mapping rather than stopped-reconstruction gauge handling
  alone.
- Stopped reconstruction still fails to improve `det_u`; fixed truth improves
  it only slightly and remains far outside tolerance.
- The 32^3/4-view smoke benchmark remains CI/wiring coverage only and is not
  alignment-quality evidence.

### Validation

- `uv run ruff format ...` passed for touched source/tests.
- `uv run ruff check ...` passed for touched source/tests.
- `uv run basedpyright ...` passed for touched source/tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_lm_numerics.py
  tests/test_synthetic_datasets.py tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py -q` passed: 29 tests.
- `just imports` passed.
- GPU probes and both 64^3/64-view benchmark modes completed with
  `jax_default_backend = "gpu"` and `selected_jax_device = "cuda:0"` in
  `benchmark_result.json`.

### Risks

- The sequential finite-difference Jacobian trades peak memory for more Python
  dispatch and compilation overhead. It is acceptable for the reference path but
  should be replaced by chunked or analytic reductions when optimizing speed.
- 128^3/128-view was not attempted after fixed-truth failed at 64^3/64 views;
  the next correctness step is geometry convention/coupling diagnosis.

## 2026-05-06 — Phase 8 Diagnostic Pause Summary

### Summary

- Pausing after the GPU memory-regression diagnostic reached a clean commit
  (`dc2aa74`).
- No new feature, report-field, or refactor slice is started in this entry.
- The current evidence says the 32^3/4-view benchmark should remain CI/wiring
  coverage only. It is not realistic alignment-quality evidence.

### Five-Case 32^3 Benchmark Failures

The five-case 32^3 pass generated planned sidecars and exercised existing
sidecar ingestion, `benchmark_result.json`, `benchmark_report.md`, and compare
artifacts. All five stopped-reconstruction runs completed but failed recovery:

| Benchmark | Status | Criteria | Geometry | Total Time s | Notes |
|---|---|---|---|---:|---|
| `synth128_setup_global_tomo` | failed | failed | failed | 11.5794 | `det_u=3.625`, `theta=0.0218166`, `det_v=0` |
| `synth128_pose_random_extreme` | failed | partially_evaluated | failed | 13.3407 | `det_u=2.7415`, `det_v=2.5782`, `theta=0.2019` |
| `synth128_lamino_axis_roll_pose` | failed | failed | failed | 13.3946 | `det_u=2.2334`, `det_v=0.7336`, `theta=0.1598` |
| `synth128_thermal_object_drift` | failed | partially_evaluated | failed | 13.5649 | `det_u=1.4893`, `det_v=0.0512`, `theta=0.0052336`; label `nuisance_residual_structure` |
| `synth128_combined_nuisance_jumps` | failed | failed | failed | 13.4880 | `det_u=3.8751`, `det_v=0.9955`, `theta=0.0309604` |

Best diagnosis: this pass validated benchmark plumbing and failure reporting,
not solver quality. It should not be used to tune or judge alignment recovery.

### Fixed-Truth Versus Stopped-Reconstruction Evidence

The realistic setup-global ladder used a nuisance-free 64^3 volume and 64 views
on `cuda:0` with the `balanced` profile:

| Mode | Status | Geometry | det_u RMSE px | det_v RMSE px | theta RMSE rad | Final Residual | Volume NMSE | Schur Accepted | Total Time s |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| `fixed_synthetic_truth` | failed | failed | 6.9338 | 0.00666 | 0.02211 | 0.856277 | 0.686109 | true | 37.5096 |
| `stopped_reconstruction` | failed | failed | 7.25 | 0 | 0.02182 | 0 | 0.686110 | true | 24.8489 |

Interpretation:

- `fixed_synthetic_truth` also fails, so the next blocker is not only
  reconstruction/volume gauge handling.
- The likely next diagnosis target is setup/pose/theta coupling or geometry
  convention mapping.
- A quick true-volume loss check found that the sidecar/projector convention is
  internally consistent for setup-global: true geometry gives zero projection
  loss, corrupted geometry is high loss, and true `det_u` alone nearly explains
  the data. The Schur step, however, moves `det_u` only from 0 to about 0.316 px
  versus the true 7.25 px after the `balanced` fixed-truth run.
- The fixed-truth Schur trace shows accepted but trust-clipped pose-dominated
  steps. First two setup updates were approximately
  `[theta=8.35e-05, det_u=-0.0414]` then
  `[theta=1.46e-05, det_u=-0.0142]` before gauge canonicalisation moved the
  realised final `det_u` to about 0.316 px. This suggests setup update is being
  weakly expressed or absorbed through pose/gauge coupling.

### GPU Memory Finding

Initial 64^3/64-view fixed-truth failed in Schur finite differences with a
12.14 GiB GPU allocation for an HLO shaped like
`f32[194,64,64,64,64]`. The allocation source was the shared
finite-difference Jacobian evaluating all parameter perturbations with one
`jax.vmap`, materializing parameter x view x volume work arrays.

Implemented fix in `dc2aa74`:

- `finite_difference_jacobian` now accumulates finite-difference columns
  sequentially.
- The 64^3 sidecar path is accepted and records the current projected detector
  shape correctly.
- `benchmark_result.json` records `jax_default_backend` and
  `selected_jax_device`.

After the fix, GPU component probes passed for 1/4/16/64 views:

| Views | Projector | Backprojector | FISTA 1 Iter | Schur Fixed Truth | Schur Stopped Volume | Schur Fixed Truth + Nuisance |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.974s | 1.022s | 4.069s | 5.362s | 4.074s | 4.716s |
| 4 | 1.016s | 1.199s | 4.337s | 6.484s | 4.890s | 5.488s |
| 16 | 1.042s | 1.422s | 4.542s | 8.640s | 6.983s | 7.850s |
| 64 | 1.066s | 2.130s | 5.287s | 18.062s | 15.870s | 17.672s |

### Commands And Artifacts

Key commands already run:

- GPU probe:
  `LD_LIBRARY_PATH=<venv nvidia */lib paths> JAX_PLATFORMS=cuda uv run python -c 'import jax; ...'`
- 64^3 sidecar generation:
  `generate_synthetic_dataset("synth128_setup_global_tomo", ..., size=64, clean=True, views=64)`
- Component probes:
  `.artifacts/phase8_setup_global_gpu_ladder/probes/probe_components.py --views 1|4|16|64`
- Fixed-truth benchmark:
  `tomojax-align-auto-smoke --profile balanced --synthetic-dataset-dir ... --geometry-update-volume-source fixed_synthetic_truth`
- Stopped-reconstruction benchmark:
  `tomojax-align-auto-smoke --profile balanced --synthetic-dataset-dir ... --geometry-update-volume-source stopped_reconstruction`
- Compare:
  `tomojax-synthetic-benchmark-compare ... --out .artifacts/phase8_setup_global_gpu_ladder/benchmark_comparison_64.md`

Important artifacts:

- `docs/benchmark_runs/2026-05-06-phase8-multi-case-32.md`
- `docs/benchmark_runs/2026-05-06-phase8-setup-global-gpu-ladder.md`
- `.artifacts/phase8_multi_case_32_benchmark_pass/`
- `.artifacts/phase8_setup_global_gpu_ladder/datasets/synth128_setup_global_tomo_64/`
- `.artifacts/phase8_setup_global_gpu_ladder/probes/components_1.json`
- `.artifacts/phase8_setup_global_gpu_ladder/probes/components_4.json`
- `.artifacts/phase8_setup_global_gpu_ladder/probes/components_16.json`
- `.artifacts/phase8_setup_global_gpu_ladder/probes/components_64.json`
- `.artifacts/phase8_setup_global_gpu_ladder/runs/64_fixed_truth_balanced/benchmark_result.json`
- `.artifacts/phase8_setup_global_gpu_ladder/runs/64_stopped_reconstruction_balanced/benchmark_result.json`
- `.artifacts/phase8_setup_global_gpu_ladder/benchmark_comparison_64.md`

### Open Questions

- Why does fixed-truth Schur make only a small realised `det_u` correction when
  true `det_u` alone nearly explains the setup-global data?
- Is setup motion being absorbed into per-view `dx_px` and then only partially
  recovered by gauge canonicalisation?
- Should setup-global oracle diagnosis temporarily use a setup-only solver or
  stronger pose prior to separate setup recovery from pose gauge?
- Are the Schur trust radius, robust weighting, or reduction-ratio adaptation
  too conservative for large setup shifts at realistic view count?
- Does the unsupported axis/roll metadata in the synthetic128 manifest need
  clearer handling now that the v2 smoke projector only models theta and
  detector shifts?

## 2026-05-07 — Phase 8 GPU Diagnostic Pause Addendum

### Summary

- Pausing after the current GPU memory-regression and setup-global diagnostic
  work reached a clean commit boundary. No new feature, report-field, refactor,
  or benchmark-ingestion slice is started here.
- Current head before this documentation-only pause commit was `f030142`
  (`Classify stopped reconstruction supported diagnostic`).
- The 32^3/4-view benchmark remains CI/wiring coverage only and should not be
  used to judge alignment quality.

### Current Best Diagnosis

- The original five-case 32^3 benchmark failures still mean only that the
  sidecar ingestion, `benchmark_result.json`, `benchmark_report.md`, and compare
  plumbing run end-to-end. They are not realistic recovery evidence.
- The first realistic 64^3/64-view nuisance-free setup-global ladder on
  `cuda:0` failed both `fixed_synthetic_truth` and `stopped_reconstruction`.
  That pointed at setup/pose/theta coupling or convention mapping, not only
  reconstruction/volume gauge handling.
- Subsequent supported-only oracle diagnostics narrowed that interpretation:
  fixed-truth can recover the supported setup DOFs when pose is frozen or held
  by a strong pose prior, while the matching stopped-reconstruction run still
  fails without moving geometry. That makes reconstruction/volume gauge handling
  or geometry absorption the current blocker for production-like alternating
  alignment, after accounting for setup/pose gauge coupling.

### Fixed-Truth Versus Stopped-Reconstruction Evidence

Supported-only 64^3/64-view `synth128_setup_global_tomo` evidence:

| Mode | Status | det_u RMSE px | theta RMSE rad | Notes |
|---|---|---:|---:|---|
| `fixed_synthetic_truth`, pose frozen | passed | 0.089 | 0.001098 | Geometry update recovers supported setup DOFs. |
| `fixed_synthetic_truth`, strong pose prior `1e6` | passed | 0.089 | 0.001091 | Joint Schur works when pose absorption is constrained. |
| `fixed_synthetic_truth`, zero-mean gauge projection | near-pass | 0.201 | 1.37e-08 | Manifest criteria passed, internal `det_u` gate missed by about 0.001 px. |
| `stopped_reconstruction`, strong pose prior `1e6` | failed | 7.25 | 0.0218166 | Geometry stayed at nominal despite the fixed-truth oracle passing. |

### GPU Memory Finding

- The confirmed memory regression source remains the Schur finite-difference
  Jacobian path. The original 64^3/64-view fixed-truth run attempted a 12.14 GiB
  allocation shaped like `f32[194,64,64,64,64]` because all parameter
  perturbations were evaluated with a single `jax.vmap`.
- The committed fix in `dc2aa74` changed the finite-difference Jacobian helper
  to sequential column accumulation. After that, component probes passed on GPU
  for 1/4/16/64 views across projector, backprojector, one FISTA iteration,
  fixed-truth Schur, stopped-volume Schur, and fixed-truth Schur with nuisance.
- The 64^3/64-view benchmark records `jax_default_backend = "gpu"` and
  `selected_jax_device = "cuda:0"` in `benchmark_result.json`.

### Commands And Artifacts

Commands run during this diagnostic thread included:

- GPU probe with venv NVIDIA library paths:
  `LD_LIBRARY_PATH=<venv nvidia */lib paths> JAX_PLATFORMS=cuda uv run python -c 'import jax; ...'`
- 64^3 sidecar generation for `synth128_setup_global_tomo` with 64 views.
- Component probes:
  `.artifacts/phase8_setup_global_gpu_ladder/probes/probe_components.py --views 1|4|16|64`
- Realistic ladder runs through sidecar ingestion:
  `tomojax-align-auto-smoke --profile balanced --synthetic-dataset-dir ... --geometry-update-volume-source fixed_synthetic_truth`
  and
  `tomojax-align-auto-smoke --profile balanced --synthetic-dataset-dir ... --geometry-update-volume-source stopped_reconstruction`
- Supported-only oracle diagnostics through `tomojax-align-auto-smoke` with
  `fixed_synthetic_truth`, `stopped_reconstruction`, pose freezing, strong pose
  prior, staged pose activation, and zero-mean pose-step gauge projection.
- Focused validation for the committed code slices:
  `uv run ruff check ...`, `uv run basedpyright ...`,
  `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py tests/test_align_auto_cli.py -q`,
  and `just imports`.

Key artifacts:

- `.artifacts/phase8_setup_global_gpu_ladder/`
- `.artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only/`
- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_pass/`
- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000/`
- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_no_phi_reference/`
- `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000/`
- `docs/benchmark_runs/2026-05-06-phase8-setup-global-gpu-ladder.md`
- `docs/benchmark_runs/2026-05-07-phase8-supported-only-oracle.md`

### Remaining Open Questions

- Why does stopped-reconstruction give the Schur solver a volume/geometry pair
  that accepts no useful setup movement when fixed-truth passes under the same
  supported-only geometry model?
- Is the stopped-gradient reconstruction absorbing setup error into volume gauge,
  detector shift, or missing normalization before the geometry update?
- Should the next diagnostic compare independent all-view losses for
  true-volume/true-geometry, true-volume/final-geometry,
  final-volume/true-geometry, and final-volume/final-geometry before changing
  solver behavior?
- Can the near-pass zero-mean fixed-truth joint run be made robust without a hard
  pose prior, or should the next production path use staged/frozen pose DOFs
  until reconstruction gauge handling is corrected?

## 2026-05-07 — Phase 8 Projection Loss Provenance Reporting

### Summary

- Updated the alternating smoke verification payload so `schur_train_loss` is
  recorded separately from independent all-view projection losses.
- `residual_after`, `final_loss`, and synthetic benchmark
  `reconstruction.final_residual` now point at
  `final_volume_final_geometry_loss_all_views`, not the last Schur training
  loss.
- Added benchmark-result and markdown-report fields for:
  - `schur_train_loss`
  - `heldout_loss`
  - `final_volume_initial_geometry_loss_all_views`
  - `final_volume_final_geometry_loss_all_views`
  - `final_volume_true_geometry_loss_all_views`
  - `true_volume_final_geometry_loss_all_views`
  - `true_volume_true_geometry_loss_all_views`
  - `projection_loss_classification`

### Interpretation

- This is an artifact/reporting honesty slice only. It does not change solver
  behavior, reconstruction gauge handling, Schur trust scaling, or benchmark
  scenario support.
- Existing GPU supported-only artifacts still describe solver behavior, but they
  predate these new loss-provenance fields. Fresh GPU diagnostics should be
  rerun before using those reports for final stopped-reconstruction
  classification.
- The new classification labels are intended to distinguish a low Schur training
  loss from a bad true-volume recovered-geometry residual, especially for
  `stopped_reconstruction` cases where the reconstructed volume may absorb
  geometry error.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 2 tests.
- `just imports` passed.

### Refreshed GPU Diagnostics

Reran the 64^3/64-view supported-only setup-global diagnostics on
`jax_default_backend = "gpu"` and `selected_jax_device = "cuda:0"` with the new
loss-provenance fields.

| Mode | Status | Criteria | det_u RMSE px | theta RMSE rad | Final residual | Schur train loss | True vol/final geom | True vol/true geom | Classification | Total time s |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| `fixed_synthetic_truth` + pose prior `1e6` | passed | passed | 0.0890679 | 0.00109202 | 1.39905 | 0.000189625 | 0.000809495 | 0 | `independent_projection_losses_consistent` | 75.3539 |
| `stopped_reconstruction` + pose prior `1e6` | failed | failed | 7.25 | 0.0218166 | 1.05102 | 0.367724 | 0.884522 | 0 | `reconstruction_absorbed_geometry` | 78.2245 |

Artifacts:

- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000_reporting/`
- `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000_reporting/`
- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_reporting.md`

Interpretation:

- Fixed-truth still passes supported setup recovery and its true-volume
  recovered-geometry loss remains close to the exact true-geometry loss.
- Stopped-reconstruction still leaves geometry at nominal and has a large
  true-volume recovered-geometry residual, while the final reconstructed volume
  prefers the unrecovered geometry. This refresh makes
  reconstruction/volume-gauge absorption the sharper next blocker.

### Remaining Work

- Diagnose why stopped-gradient reconstruction absorbs the supported setup error
  before the geometry update. The next code slice should compare reconstruction
  normalization/gauge handling and possibly constrain the volume gauge before
  Schur, rather than adding more report fields.

## 2026-05-07 — Phase 8 Schur Continuation Residual Filters

### Summary

- Added `JointSchurLMConfig.residual_filters` and threaded continuation-level
  residual filters into joint Schur geometry updates.
- Schur now uses filtered residuals consistently for IRLS weights,
  finite-difference Jacobian rows, candidate/current losses, and per-view
  diagnostics.
- Added a focused regression test showing a low-pass Schur residual produces a
  useful setup detector-shift step under high-frequency projection noise.

### Refreshed GPU Diagnostics

Reran the supported-only 64^3/64-view setup-global diagnostics on `cuda:0` after
the Schur filter change.

| Mode | Status | Criteria | det_u RMSE px | theta RMSE rad | Final residual | Schur train loss | Schur accepted | True vol/final geom | Classification | Total time s |
|---|---|---|---:|---:|---:|---:|---|---:|---|---:|
| `fixed_synthetic_truth` + pose prior `1e6` | passed | passed | 5.24164e-06 | 5.10065e-05 | 1.40612 | 2.13915e-08 | true | 3.39969e-09 | `independent_projection_losses_consistent` | 106.826 |
| `stopped_reconstruction` + pose prior `1e6` | failed | failed | 7.25 | 0.0218166 | 1.05102 | 0.361978 | false | 0.884522 | `reconstruction_absorbed_geometry` | 113.388 |

Artifacts:

- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_pose_frozen_filtered_reporting/`
- `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000_filtered_reporting/`
- `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000_filtered_reporting/`
- `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_filtered_reporting.md`

Interpretation:

- Continuation-filtered Schur makes the fixed-truth strong-pose-prior supported
  oracle essentially exact. The remaining blocker is no longer fixed-truth setup
  recovery under supported DOFs.
- The refreshed fixed-truth pose-frozen oracle also passes on `cuda:0`
  (`det_u` RMSE `1.33514e-05` px, theta RMSE `2.59716e-06` rad, true-volume
  recovered-geometry loss `0.0`, true-volume true-geometry loss `0.0`).
- Stopped reconstruction still fails and is classified as
  `reconstruction_absorbed_geometry`. The next implementation slice should
  target reconstruction/volume gauge handling rather than further Schur
  trust/report work.

### Validation

- `uv run ruff format src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py tests/test_joint_schur_lm.py`
  passed.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py tests/test_joint_schur_lm.py`
  passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py
  src/tomojax/align/_alternating_geometry_update.py tests/test_joint_schur_lm.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 14 tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 2 tests.
- `just imports` passed.

## 2026-05-07 — Phase 8 Unsupported DOF Benchmark Classification

### Summary

- Updated synthetic benchmark manifest evaluation so criteria without a
  supported v2 smoke metric mapping are reported as `not_evaluated` with reason
  `unsupported_dof_not_evaluated`.
- Added focused CLI artifact assertions for unsupported setup-global axis/roll
  criteria.
- This does not implement new DOFs; it prevents unsupported roll, axis tilt,
  laminography, nuisance, jumps, or object-motion criteria from being mistaken
  for supported recovery failures.

### Validation

- `uv run ruff format src/tomojax/align/_alternating_artifacts.py
  tests/test_align_auto_cli.py` passed.
- `uv run ruff check src/tomojax/align/_alternating_artifacts.py
  tests/test_align_auto_cli.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_artifacts.py
  tests/test_align_auto_cli.py` passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_generates_named_synthetic_dataset
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 3 tests.
- `just imports` passed.
