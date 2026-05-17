# tomojax.bench

## Purpose

`tomojax.bench` is a developer and verification package, not a production public
API. It owns benchmark fixtures, diagnostic runners, report builders, and
evidence adapters that exercise production facades without making benchmark
machinery part of normal reconstruction or alignment workflows.

This module is developer/evidence-facing. Its commands are exposed through
`tomojax dev ...` rather than as installed top-level console scripts.

## Developer Facade

`tomojax.bench.api` and the package root re-export reusable developer benchmark
helpers. User workflows should reach benchmark commands only through
`tomojax dev ...`.

The exported developer surface is intentionally broad because it supports
regression evidence, article artifacts, and real-laminography developer
workflows. It is grouped as:

- stage and real-laminography contract constants:
  `FULL_REQUIRED_STAGES`, `PARTIAL_REQUIRED_STAGES`,
  `REAL_LAMINO_COR_ONLY_STAGE`, `REAL_LAMINO_PROFILE_CHOICES`,
  `REAL_LAMINO_PUBLICATION_IMAGES`, `REAL_LAMINO_REPORT_STAGED_PATH`,
  `REAL_LAMINO_STAGED_PATH`, `REFERENCE_REGRESSION_CONTRACT`,
  `REFERENCE_REGRESSION_STAGE_MAP`, `STAGED_LAMINO_CONTRACT`
- synthetic and smoke scenario types:
  `AcquisitionSpec`, `AlignmentScenario`, `AlignmentSmokeReport`,
  `AlternatingAlignmentSolver`, `AlternatingLevelSummary`,
  `AlternatingSmokeConfig`, `AlternatingSmokeResult`, `PhantomSpec`,
  `ScenarioExpectation`, `ScenarioSuite`
- article and visualization payload types:
  `AlignmentVisualizationPayload`, `ArticleRunProfile`, `ArticleScenario`,
  `ArticleScenarioComputationResult`, `ArticleScenarioRunArtifacts`,
  `ArticleScenarioRunResult`, `NaiveVisualizationPayload`, `VisualProfile`,
  `VisualScenario`
- benchmark result and configuration types:
  `ForwardProjectorBenchmarkConfig`, `ForwardProjectorSuiteCase`,
  `ForwardSinogramBenchmarkConfig`, `ForwardSinogramSuiteCase`,
  `RealLaminoGpuMonitor`, `RealLaminoRunContext`, `SpdhgDatasetSimulationPlan`,
  `SpdhgGeometryBundle`, `SpdhgReconstructionResults`,
  `SpdhgSimulationGeometryBundle`, `SyntheticBenchmarkResult`,
  `SyntheticComparisonArtifact`
- scenario/catalog builders:
  `article_phantom_metadata`, `article_scenario_catalog_for_kind`,
  `make_article_phantom`, `phantom_spec`, `scenario_by_slug`,
  `scenario_catalog`, `scenario_suite`, `validate_scenario_catalog`
- article run and manifest helpers:
  `array_finite_summary`, `article_alignment_metadata`,
  `article_scenario_catalog_payload`, `article_scenario_finite_report`,
  `article_scenario_supplied_payload`, `article_scenario_truth_payload`,
  `build_article_full_run_result`, `build_article_naive_run_result`,
  `build_article_nonfinite_run_result`, `build_article_run_manifest`,
  `build_article_scenario_run_result`, `execute_article_scenario_computation`,
  `scalar_finite_summary`, `write_article_master_panel`,
  `write_article_summary_csv`
- forward-projector benchmark helpers:
  `benchmark_backend`, `benchmark_sinogram_mode`,
  `make_forward_projector_fixture`, `make_forward_sinogram_fixture`,
  `preset_config`, `run_forward_projector_benchmark`,
  `run_forward_projector_suite`, `run_forward_sinogram_benchmark`,
  `run_forward_sinogram_suite`, `sinogram_suite_cases`, `suite_cases`,
  `write_benchmark_json`
- real-laminography planning, runtime, and report helpers:
  `append_real_lamino_csv`, `apply_real_lamino_profile_args`,
  `apply_real_lamino_profile_contract_args`,
  `apply_real_lamino_projection_background`, `binned_pixel_scale`,
  `build_real_lamino_report`, `map_real_lamino_global_z_to_binned`,
  `mark_real_lamino_stage_failed`, `normalize_real_lamino_runtime_args`,
  `optimize_reference_setup_geometry_bilevel_for_level`, `parse_shape3`,
  `pose_dx_dz_bounds`, `pose_phi_bounds`, `pose_polish_bounds`,
  `prepare_real_lamino_binned_fixture`, `real_lamino_artifact_validation_failures`,
  `real_lamino_checkpoint_validation_failures`, `real_lamino_commit_info`,
  `real_lamino_finite_fraction`, `real_lamino_global_z_to_local_index`,
  `real_lamino_global_z_to_phys`, `real_lamino_grid_origin_z`,
  `real_lamino_json_safe`, `real_lamino_local_z_to_global_index`,
  `real_lamino_loss_summary`, `real_lamino_method_constraints`,
  `real_lamino_phys_z_to_local_index`, `real_lamino_pose_params_summary`,
  `real_lamino_projection_stats`, `real_lamino_reference_regression_contract_payload`,
  `real_lamino_safe_params_summary`, `real_lamino_stat_validation_failures`,
  `real_lamino_success_payload`, `real_lamino_xy_at_global_z`,
  `reference_regression_level_outer_counts`, `relative_l2`,
  `resolve_fixture_bin_factor`, `save_real_lamino_z_stack`,
  `select_real_lamino_final_candidates`, `select_real_lamino_views`,
  `setup_det_u_bounds`,
  `update_real_lamino_status`, `validate_bin_factor`,
  `validate_real_lamino_loaded_input`, `validate_real_lamino_stage_output`,
  `view_indices_for_smoke_shape`, `write_real_lamino_geometry_trace`,
  `write_real_lamino_json`, `write_real_lamino_params_csv`,
  `write_real_lamino_planned_stage_manifests`, `write_real_lamino_residual_trace`,
  `write_real_lamino_stage_products`, `write_real_lamino_skipped_stage_manifests`
- image, visualization, and report utilities:
  `alignment_visualization_payload`, `build_spdhg_experiment_report`,
  `center_crop`, `current_baseline_payload`, `grid_aligned_xy`,
  `largest_centered_square_inside_rotated_frame`, `load_volume_array`,
  `naive_visualization_payload`, `real_lamino_orthos_image`,
  `render_tem_grid_pose_artifacts`,
  `resize_nearest_2d`, `save_slice_png`, `save_uint8_png`, `save_volume`,
  `scale_uint8`, `spdhg_experiment_markdown`, `window_normalize`,
  `write_current_baseline_artifacts`, `write_spdhg_benchmark_report`
- metric and comparison helpers:
  `compute_spdhg_benchmark_metrics`, `is_expected_spdhg_fallback_failure`,
  `load_current_baseline_artifact`, `load_synthetic_benchmark_result`,
  `load_synthetic_benchmark_results`, `psnr3d`, `ssim_center_slices`,
  `synthetic_benchmark_comparison_markdown`, `timed_repeats`,
  `total_variation`, `write_synthetic_benchmark_comparison_markdown`

## Owned Concepts

- Benchmark fixtures and benchmark result comparison helpers.
- Diagnostic performance probes.
- Synthetic alignment diagnostic runners.
- Synthetic result report helpers used by implementation and regression work.
- Real-laminography developer workflow contracts, report semantics, and bounded
  planning helpers used by scripts under `scripts/real_laminography`.
- Reference-regression adapters that intentionally bridge developer diagnostics
  to private product internals without exposing those internals as production
  alignment API.

## Allowed Dependencies

Benchmark code may depend on production public facades when it is exercising or
measuring them. The expected dependencies are:

- `tomojax.align`
- `tomojax.backends`
- `tomojax.core`
- `tomojax.data`
- `tomojax.datasets`
- `tomojax.forward`
- `tomojax.geometry`
- `tomojax.io`
- `tomojax.motion`
- `tomojax.nuisance`
- `tomojax.recon`
- `tomojax.verify`

Developer-only adapters may also import private production internals when the
adapter exists specifically to produce evidence for a not-yet-public path. Those
imports must remain inside `tomojax.bench` and must not become examples for
production modules.

## Forbidden Dependencies

Production modules should not depend on `tomojax.bench`. Benchmark code may
depend on public production facades to exercise them, but benchmark-only helpers
must not leak back into `tomojax.io`, `tomojax.recon`, `tomojax.align`,
`tomojax.forward`, or `tomojax.geometry`.

`tomojax.bench` must not be imported by production modules. The import-linter
`production-no-bench` contract enforces that direction for production packages.

## Numerical Invariants

- Benchmark fixtures must keep geometry conventions aligned with the production
  facades they exercise.
- Reported timings and accuracy metrics must label public API, supported
  accelerator API, and private helper paths separately.
- Reference-regression adapters must preserve enough configuration, seed,
  backend, and artifact metadata for a later rerun to explain the result.
- Synthetic alignment diagnostics must be deterministic from their declared
  seeds and scenario specs.
- Real-laminography report builders must distinguish completed, skipped, and
  failed stages instead of silently treating missing evidence as success.

## Artifact/Provenance Responsibilities

- benchmark JSON and Markdown summaries
- synthetic comparison artifacts
- article scenario manifests, finite reports, summary CSVs, and master panels
- real-laminography status files, staged manifests, geometry traces, residual
  traces, and report validation failures
- SPDHG experiment reports and saved comparison volumes/slices

## Testing Strategy

- Public facade/import tests keep `tomojax.bench` importable without making it a
  production dependency.
- Focused benchmark tests cover benchmark suite definitions, current-baseline
  payloads, forward projector measurements, sampled benchmark helpers, synthetic
  result comparison, visualization helpers, and expected fallback handling.
- Alignment diagnostic tests cover smoke scenarios, objective helpers,
  convergence criteria, and memory behavior.
- Real-laminography/report tests should assert artifact schemas and failure
  classification when those helpers change.
- Import-boundary tests and import-linter contracts must continue to block
  production packages from depending on this module.
