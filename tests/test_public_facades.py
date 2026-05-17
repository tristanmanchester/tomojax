from __future__ import annotations

import tomojax.align as align_api
import tomojax.align.api as align_full_api
import tomojax.backends as backends_api
import tomojax.bench as bench_api
import tomojax.calibration as calibration_api
import tomojax.cli as cli_api
import tomojax.io as io_api
import tomojax.recon as recon_api


def test_alignment_facade_exports_documented_api() -> None:
    assert align_api.AlignConfig.__name__ == "AlignConfig"
    assert callable(align_api.align)
    assert callable(align_api.align_multires)
    assert align_full_api.DofSpec.__name__ == "DofSpec"
    assert align_full_api.AlignmentSchedule.__name__ == "AlignmentSchedule"
    assert align_full_api.AlignmentState.__name__ == "AlignmentState"
    assert align_full_api.BaseGeometryArrays.__name__ == "BaseGeometryArrays"
    assert align_full_api.GeometryCalibrationState.__name__ == "GeometryCalibrationState"
    assert align_full_api.PoseState.__name__ == "PoseState"
    assert align_full_api.SetupGeometryState.__name__ == "SetupGeometryState"
    assert callable(align_full_api.apply_alignment_state)
    assert callable(align_full_api.dof_spec)
    assert callable(align_full_api.geometry_with_axis_state)
    assert callable(align_full_api.level_detector_grid)
    assert callable(align_full_api.normalize_geometry_dofs)
    assert callable(align_full_api.schedule_preset)
    assert callable(align_full_api.summarize_geometry_calibration_stats)
    assert not hasattr(align_full_api, "run_alignment_smoke")
    assert not hasattr(align_full_api, "run_alternating_solver_smoke")
    assert not hasattr(align_full_api, "AlternatingSmokeConfig")


def test_reconstruction_facade_exports_documented_api() -> None:
    assert recon_api.FBPConfig.__name__ == "FBPConfig"
    assert recon_api.FistaConfig.__name__ == "FistaConfig"
    assert recon_api.SPDHGConfig.__name__ == "SPDHGConfig"
    assert callable(recon_api.fbp)
    assert callable(recon_api.fista_tv)
    assert callable(recon_api.spdhg_tv)


def test_io_facade_exports_dataset_boundary() -> None:
    assert io_api.PreprocessConfig.__name__ == "PreprocessConfig"
    assert io_api.PreprocessResult.__name__ == "PreprocessResult"
    assert io_api.ProjectionDataset.__name__ == "ProjectionDataset"
    assert io_api.ValidationReport.__name__ == "ValidationReport"
    assert callable(io_api.build_geometry_from_dataset_metadata)
    assert callable(io_api.load_dataset)
    assert callable(io_api.load_projection_payload)
    assert callable(io_api.load_tiff_stack)
    assert callable(io_api.preprocess_nxtomo)
    assert callable(io_api.save_dataset)
    assert callable(io_api.save_projection_payload)
    assert callable(io_api.validate_dataset)
    assert callable(io_api.normalize_json)


def test_calibration_facade_exports_only_schema_value_types() -> None:
    assert set(calibration_api.__all__) == {
        "CalibrationState",
        "CalibrationVariable",
        "DetectorPixelScale",
        "DetectorPixelValue",
    }


def test_cli_facade_exports_command_catalog() -> None:
    assert cli_api.CliCommand.__name__ == "CliCommand"
    assert cli_api.product_command_names() == (
        "inspect",
        "validate",
        "preprocess",
        "ingest",
        "convert",
        "recon",
        "align",
        "simulate",
        "dev",
    )
    assert "align-auto" not in cli_api.product_command_names()
    assert "align-auto" in cli_api.developer_command_names()
    assert "test-gpu" in cli_api.developer_command_names()


def test_backend_facade_exports_runtime_helpers() -> None:
    assert backends_api.ViewsPerBatchEstimate.__name__ == "ViewsPerBatchEstimate"
    assert callable(backends_api.default_gather_dtype)
    assert callable(backends_api.device_free_memory_bytes)
    assert callable(backends_api.estimate_views_per_batch)
    assert callable(backends_api.estimate_views_per_batch_info)
    assert callable(backends_api.run_command)
    assert callable(backends_api.check_output_command)


def test_bench_facade_exports_developer_benchmark_helpers() -> None:
    assert bench_api.AlignmentScenario.__name__ == "AlignmentScenario"
    assert bench_api.AlternatingSmokeConfig.__name__ == "AlternatingSmokeConfig"
    assert bench_api.ArticleScenario.__name__ == "ArticleScenario"
    assert bench_api.SyntheticBenchmarkResult.__name__ == "SyntheticBenchmarkResult"
    assert bench_api.REAL_LAMINO_PROFILE_CHOICES
    assert bench_api.REAL_LAMINO_STAGED_PATH
    assert callable(bench_api.run_alignment_smoke)
    assert callable(bench_api.run_alternating_solver_smoke)
    assert callable(bench_api.scenario_catalog)
    assert callable(bench_api.make_article_phantom)
    assert callable(bench_api.real_lamino_success_payload)
    assert callable(bench_api.resolve_fixture_bin_factor)
    assert callable(bench_api.pose_polish_bounds)
    assert callable(bench_api.load_synthetic_benchmark_result)
