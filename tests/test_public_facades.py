from __future__ import annotations

import tomojax.align as align_api
import tomojax.align.api as align_full_api
import tomojax.backends as backends_api
import tomojax.bench as bench_api
import tomojax.calibration as calibration_api
import tomojax.cli as cli_api
import tomojax.datasets as datasets_api
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
    assert align_full_api.FixedVolumeProjectionObjective.__name__ == "FixedVolumeProjectionObjective"
    assert align_full_api.GeometryCalibrationState.__name__ == "GeometryCalibrationState"
    assert align_full_api.ObjectiveProvenance.__name__ == "ObjectiveProvenance"
    assert align_full_api.ObjectiveResult.__name__ == "ObjectiveResult"
    assert align_full_api.PoseState.__name__ == "PoseState"
    assert align_full_api.SetupGeometryState.__name__ == "SetupGeometryState"
    assert callable(align_full_api.apply_alignment_state)
    assert callable(align_full_api.dof_spec)
    assert callable(align_full_api.geometry_with_axis_state)
    assert callable(align_full_api.level_detector_grid)
    assert callable(align_full_api.normalize_geometry_dofs)
    assert callable(align_full_api.project_and_score_stack)
    assert callable(align_full_api.project_stack)
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
    assert io_api.LoadedNXTomo.__name__ == "LoadedNXTomo"
    assert io_api.NXTomoMetadata.__name__ == "NXTomoMetadata"
    assert io_api.PreprocessConfig.__name__ == "PreprocessConfig"
    assert io_api.PreprocessResult.__name__ == "PreprocessResult"
    assert io_api.ProjectionDataset.__name__ == "ProjectionDataset"
    assert io_api.ValidationReport.__name__ == "ValidationReport"
    assert callable(io_api.absorption_to_transmission)
    assert callable(io_api.build_geometry_from_dataset_metadata)
    assert callable(io_api.constant_dark_field)
    assert callable(io_api.flat_dark_correct_frames_to_absorption)
    assert callable(io_api.flat_dark_to_absorption)
    assert callable(io_api.flat_dark_to_transmission)
    assert callable(io_api.load_dataset)
    assert callable(io_api.load_nxtomo)
    assert callable(io_api.load_projection_payload)
    assert callable(io_api.load_tiff_stack)
    assert callable(io_api.pad_to_multiples)
    assert callable(io_api.preprocess_nxtomo)
    assert callable(io_api.save_dataset)
    assert callable(io_api.save_nxtomo)
    assert callable(io_api.save_projection_payload)
    assert callable(io_api.spatial_bin)
    assert callable(io_api.summarize_angles)
    assert callable(io_api.validate_dataset)
    assert callable(io_api.validate_nxtomo)
    assert callable(io_api.transmission_to_absorption)
    assert callable(io_api.volume_chunks)
    assert callable(io_api.normalize_json)


def test_datasets_facade_exports_simulation_boundary() -> None:
    assert datasets_api.SimConfig.__name__ == "SimConfig"
    assert datasets_api.SimulatedData.__name__ == "SimulatedData"
    assert callable(datasets_api.make_benchmark_phantom)
    assert callable(datasets_api.make_phantom)
    assert callable(datasets_api.random_cubes_spheres)
    assert callable(datasets_api.simulate)
    assert callable(datasets_api.simulate_to_file)


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
    assert bench_api.SpdhgGeometryBundle.__name__ == "SpdhgGeometryBundle"
    assert bench_api.REAL_LAMINO_PROFILE_CHOICES
    assert bench_api.REAL_LAMINO_REPORT_STAGED_PATH
    assert bench_api.REAL_LAMINO_STAGED_PATH
    assert bench_api.RealLaminoGpuMonitor.__name__ == "RealLaminoGpuMonitor"
    assert callable(bench_api.build_real_lamino_report)
    assert callable(bench_api.build_article_run_manifest)
    assert callable(bench_api.article_scenario_catalog_payload)
    assert callable(bench_api.build_spdhg_experiment_report)
    assert callable(bench_api.run_alignment_smoke)
    assert callable(bench_api.run_alternating_solver_smoke)
    assert callable(bench_api.scenario_catalog)
    assert callable(bench_api.make_article_phantom)
    assert callable(bench_api.optimize_reference_setup_geometry_bilevel_for_level)
    assert callable(bench_api.real_lamino_success_payload)
    assert callable(bench_api.write_real_lamino_json)
    assert callable(bench_api.render_tem_grid_pose_artifacts)
    assert callable(bench_api.select_real_lamino_views)
    assert callable(bench_api.timed_repeats)
    assert callable(bench_api.load_volume_array)
    assert callable(bench_api.grid_aligned_xy)
    assert callable(bench_api.largest_centered_square_inside_rotated_frame)
    assert callable(bench_api.resolve_fixture_bin_factor)
    assert callable(bench_api.pose_polish_bounds)
    assert callable(bench_api.load_synthetic_benchmark_result)
    assert callable(bench_api.spdhg_experiment_markdown)
    assert callable(bench_api.compute_spdhg_benchmark_metrics)
    assert callable(bench_api.write_spdhg_benchmark_report)
