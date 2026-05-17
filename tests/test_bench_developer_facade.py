from __future__ import annotations

import tomojax.bench as bench_api
import tomojax.bench.api as bench_full_api


_REPRESENTATIVE_DEVELOPER_EVIDENCE_EXPORTS = (
    "AcquisitionSpec",
    "AlignmentScenario",
    "AlignmentSmokeReport",
    "ArticleScenarioRunResult",
    "ForwardProjectorBenchmarkConfig",
    "RealLaminoRunContext",
    "SyntheticBenchmarkResult",
    "build_article_run_manifest",
    "build_real_lamino_report",
    "current_baseline_payload",
    "run_alignment_smoke",
    "run_forward_projector_suite",
    "run_real_lamino_setup_stage",
    "scenario_catalog",
    "validate_real_lamino_stage_output",
    "write_current_baseline_artifacts",
)


def test_bench_facade_mirrors_developer_evidence_api_module() -> None:
    assert set(bench_api.__all__) == set(bench_full_api.__all__)
    for name in _REPRESENTATIVE_DEVELOPER_EVIDENCE_EXPORTS:
        assert name in bench_api.__all__
        assert getattr(bench_api, name) is getattr(bench_full_api, name)


def test_bench_facade_exports_representative_developer_evidence_helpers() -> None:
    assert bench_api.AcquisitionSpec.__name__ == "AcquisitionSpec"
    assert bench_api.AlignmentScenario.__name__ == "AlignmentScenario"
    assert bench_api.AlignmentSmokeReport.__name__ == "AlignmentSmokeReport"
    assert bench_api.AlternatingSmokeConfig.__name__ == "AlternatingSmokeConfig"
    assert bench_api.AlternatingSmokeResult.__name__ == "AlternatingSmokeResult"
    assert bench_api.ArticleRunProfile.__name__ == "ArticleRunProfile"
    assert bench_api.ArticleScenario.__name__ == "ArticleScenario"
    assert bench_api.ArticleScenarioRunResult.__name__ == "ArticleScenarioRunResult"
    assert bench_api.ForwardProjectorBenchmarkConfig.__name__ == "ForwardProjectorBenchmarkConfig"
    assert bench_api.PhantomSpec.__name__ == "PhantomSpec"
    assert bench_api.ScenarioSuite.__name__ == "ScenarioSuite"
    assert bench_api.SyntheticBenchmarkResult.__name__ == "SyntheticBenchmarkResult"
    assert bench_api.SyntheticComparisonArtifact.__name__ == "SyntheticComparisonArtifact"
    assert bench_api.SpdhgGeometryBundle.__name__ == "SpdhgGeometryBundle"
    assert bench_api.REAL_LAMINO_PROFILE_CHOICES
    assert bench_api.REAL_LAMINO_REPORT_STAGED_PATH
    assert bench_api.REAL_LAMINO_STAGED_PATH
    assert bench_api.RealLaminoRunContext.__name__ == "RealLaminoRunContext"
    assert callable(bench_api.build_real_lamino_report)
    assert callable(bench_api.build_article_run_manifest)
    assert callable(bench_api.build_article_scenario_run_result)
    assert callable(bench_api.article_alignment_metadata)
    assert callable(bench_api.article_scenario_catalog_payload)
    assert callable(bench_api.article_scenario_finite_report)
    assert callable(bench_api.build_spdhg_experiment_report)
    assert callable(bench_api.current_baseline_payload)
    assert callable(bench_api.write_current_baseline_artifacts)
    assert callable(bench_api.run_alignment_smoke)
    assert callable(bench_api.run_forward_projector_suite)
    assert callable(bench_api.scenario_catalog)
    assert callable(bench_api.make_article_phantom)
    assert callable(bench_api.run_real_lamino_setup_stage)
    assert callable(bench_api.real_lamino_success_payload)
    assert callable(bench_api.real_lamino_loss_summary)
    assert callable(bench_api.mark_real_lamino_stage_failed)
    assert callable(bench_api.write_real_lamino_planned_stage_manifests)
    assert callable(bench_api.write_real_lamino_json)
    assert callable(bench_api.select_real_lamino_views)
    assert callable(bench_api.timed_repeats)
    assert callable(bench_api.validate_real_lamino_loaded_input)
    assert callable(bench_api.validate_real_lamino_stage_output)
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
