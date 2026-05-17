from __future__ import annotations

import tomojax.bench as bench_api


def test_bench_facade_exports_developer_benchmark_helpers() -> None:
    assert bench_api.AcquisitionSpec.__name__ == "AcquisitionSpec"
    assert bench_api.AlignmentScenario.__name__ == "AlignmentScenario"
    assert bench_api.AlignmentSmokeReport.__name__ == "AlignmentSmokeReport"
    assert bench_api.AlignmentVisualizationPayload.__name__ == "AlignmentVisualizationPayload"
    assert bench_api.AlternatingAlignmentSolver.__name__ == "AlternatingAlignmentSolver"
    assert bench_api.AlternatingLevelSummary.__name__ == "AlternatingLevelSummary"
    assert bench_api.AlternatingSmokeConfig.__name__ == "AlternatingSmokeConfig"
    assert bench_api.AlternatingSmokeResult.__name__ == "AlternatingSmokeResult"
    assert bench_api.ArticleRunProfile.__name__ == "ArticleRunProfile"
    assert bench_api.ArticleScenario.__name__ == "ArticleScenario"
    assert bench_api.ArticleScenarioComputationResult.__name__ == "ArticleScenarioComputationResult"
    assert bench_api.ArticleScenarioRunArtifacts.__name__ == "ArticleScenarioRunArtifacts"
    assert bench_api.ArticleScenarioRunResult.__name__ == "ArticleScenarioRunResult"
    assert bench_api.ForwardProjectorBenchmarkConfig.__name__ == "ForwardProjectorBenchmarkConfig"
    assert bench_api.ForwardProjectorSuiteCase.__name__ == "ForwardProjectorSuiteCase"
    assert bench_api.ForwardSinogramBenchmarkConfig.__name__ == "ForwardSinogramBenchmarkConfig"
    assert bench_api.ForwardSinogramSuiteCase.__name__ == "ForwardSinogramSuiteCase"
    assert bench_api.NaiveVisualizationPayload.__name__ == "NaiveVisualizationPayload"
    assert bench_api.PhantomSpec.__name__ == "PhantomSpec"
    assert bench_api.ScenarioExpectation.__name__ == "ScenarioExpectation"
    assert bench_api.ScenarioSuite.__name__ == "ScenarioSuite"
    assert bench_api.SyntheticBenchmarkResult.__name__ == "SyntheticBenchmarkResult"
    assert bench_api.SyntheticComparisonArtifact.__name__ == "SyntheticComparisonArtifact"
    assert bench_api.SpdhgGeometryBundle.__name__ == "SpdhgGeometryBundle"
    assert bench_api.VisualProfile.__name__ == "VisualProfile"
    assert bench_api.VisualScenario.__name__ == "VisualScenario"
    assert bench_api.REAL_LAMINO_PROFILE_CHOICES
    assert bench_api.REAL_LAMINO_REPORT_STAGED_PATH
    assert bench_api.REAL_LAMINO_STAGED_PATH
    assert bench_api.RealLaminoGpuMonitor.__name__ == "RealLaminoGpuMonitor"
    assert bench_api.RealLaminoRunContext.__name__ == "RealLaminoRunContext"
    assert callable(bench_api.build_real_lamino_report)
    assert callable(bench_api.build_article_run_manifest)
    assert callable(bench_api.build_article_scenario_run_result)
    assert callable(bench_api.execute_article_scenario_computation)
    assert callable(bench_api.article_alignment_metadata)
    assert callable(bench_api.article_scenario_catalog_payload)
    assert callable(bench_api.article_scenario_finite_report)
    assert callable(bench_api.article_scenario_supplied_payload)
    assert callable(bench_api.article_scenario_truth_payload)
    assert callable(bench_api.alignment_visualization_payload)
    assert callable(bench_api.naive_visualization_payload)
    assert callable(bench_api.write_article_master_panel)
    assert callable(bench_api.write_article_summary_csv)
    assert callable(bench_api.build_spdhg_experiment_report)
    assert callable(bench_api.current_baseline_payload)
    assert callable(bench_api.write_current_baseline_artifacts)
    assert callable(bench_api.run_alignment_smoke)
    assert callable(bench_api.run_alternating_solver_smoke)
    assert callable(bench_api.run_baseline_stage)
    assert callable(bench_api.run_best_final_reconstruction_stage)
    assert callable(bench_api.run_cor_only_fista_stage)
    assert callable(bench_api.scenario_catalog)
    assert callable(bench_api.make_article_phantom)
    assert callable(bench_api.optimize_reference_setup_geometry_bilevel_for_level)
    assert callable(bench_api.run_real_lamino_pose_stage)
    assert callable(bench_api.run_real_lamino_setup_stage)
    assert callable(bench_api.real_lamino_success_payload)
    assert callable(bench_api.real_lamino_loss_summary)
    assert callable(bench_api.real_lamino_pose_params_summary)
    assert callable(bench_api.real_lamino_safe_params_summary)
    assert callable(bench_api.mark_real_lamino_stage_failed)
    assert callable(bench_api.write_real_lamino_planned_stage_manifests)
    assert callable(bench_api.write_real_lamino_params_csv)
    assert callable(bench_api.write_real_lamino_skipped_stage_manifests)
    assert callable(bench_api.write_real_lamino_json)
    assert callable(bench_api.render_tem_grid_pose_artifacts)
    assert callable(bench_api.select_real_lamino_views)
    assert callable(bench_api.select_real_lamino_final_candidates)
    assert callable(bench_api.timed_repeats)
    assert callable(bench_api.validate_real_lamino_loaded_input)
    assert callable(bench_api.apply_real_lamino_projection_background)
    assert callable(bench_api.load_volume_array)
    assert callable(bench_api.grid_aligned_xy)
    assert callable(bench_api.largest_centered_square_inside_rotated_frame)
    assert callable(bench_api.resize_nearest_2d)
    assert callable(bench_api.save_uint8_png)
    assert callable(bench_api.resolve_fixture_bin_factor)
    assert callable(bench_api.prepare_real_lamino_binned_fixture)
    assert callable(bench_api.reference_regression_level_outer_counts)
    assert callable(bench_api.pose_polish_bounds)
    assert callable(bench_api.load_synthetic_benchmark_result)
    assert callable(bench_api.load_synthetic_benchmark_results)
    assert callable(bench_api.load_current_baseline_artifact)
    assert callable(bench_api.phantom_spec)
    assert callable(bench_api.scenario_by_slug)
    assert callable(bench_api.scenario_suite)
    assert callable(bench_api.synthetic_benchmark_comparison_markdown)
    assert callable(bench_api.validate_scenario_catalog)
    assert callable(bench_api.write_synthetic_benchmark_comparison_markdown)
    assert callable(bench_api.spdhg_experiment_markdown)
    assert callable(bench_api.compute_spdhg_benchmark_metrics)
    assert callable(bench_api.write_spdhg_benchmark_report)
