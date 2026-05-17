from __future__ import annotations

import tomojax.align as align_api
import tomojax.align.api as align_full_api
import tomojax.backends as backends_api
import tomojax.calibration as calibration_api
import tomojax.cli as cli_api
import tomojax.core as core_api
import tomojax.core.api as core_full_api
import tomojax.datasets as datasets_api
import tomojax.datasets.api as datasets_full_api
import tomojax.forward as forward_api
import tomojax.forward.api as forward_full_api
import tomojax.geometry as geometry_api
import tomojax.geometry.api as geometry_full_api
import tomojax.io as io_api
import tomojax.io.api as io_full_api
import tomojax.motion as motion_api
import tomojax.motion.api as motion_full_api
import tomojax.nuisance as nuisance_api
import tomojax.nuisance.api as nuisance_full_api
import tomojax.recon as recon_api
import tomojax.recon.api as recon_full_api
import tomojax.verify as verify_api
import tomojax.verify.api as verify_full_api


def _assert_facade_reexports_api(package: object, api: object) -> None:
    package_exports = set(package.__all__)
    api_exports = set(api.__all__)

    assert package_exports == api_exports
    for name in api_exports:
        assert getattr(package, name) is getattr(api, name)


def test_alignment_facade_exports_documented_api() -> None:
    assert align_api.AlignConfig.__name__ == "AlignConfig"
    assert callable(align_api.align)
    assert callable(align_api.align_multires)
    assert align_full_api.DofSpec.__name__ == "DofSpec"
    assert align_full_api.AlignmentSchedule.__name__ == "AlignmentSchedule"
    assert align_full_api.AlignmentState.__name__ == "AlignmentState"
    assert align_full_api.BaseGeometryArrays.__name__ == "BaseGeometryArrays"
    assert (
        align_full_api.FixedVolumeProjectionObjective.__name__ == "FixedVolumeProjectionObjective"
    )
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
    _assert_facade_reexports_api(recon_api, recon_full_api)

    assert recon_api.FBPConfig.__name__ == "FBPConfig"
    assert recon_api.FistaConfig.__name__ == "FistaConfig"
    assert recon_api.SPDHGConfig.__name__ == "SPDHGConfig"
    assert callable(recon_api.fbp)
    assert callable(recon_api.fista_tv)
    assert callable(recon_api.spdhg_tv)


def test_geometry_facade_exports_concrete_geometry_api() -> None:
    _assert_facade_reexports_api(geometry_api, geometry_full_api)

    assert geometry_api.Detector.__name__ == "Detector"
    assert geometry_api.Geometry.__name__ == "Geometry"
    assert geometry_api.Grid.__name__ == "Grid"
    assert geometry_api.LaminographyGeometry.__name__ == "LaminographyGeometry"
    assert geometry_api.ParallelGeometry.__name__ == "ParallelGeometry"
    assert geometry_api.RotationAxisGeometry.__name__ == "RotationAxisGeometry"
    assert callable(geometry_api.normalize_axis_unit)
    assert callable(geometry_api.stack_view_poses)


def test_io_facade_exports_dataset_boundary() -> None:
    _assert_facade_reexports_api(io_api, io_full_api)

    assert io_api.InspectionReport.__name__ == "InspectionReport"
    assert io_api.JsonValue is not None
    assert io_api.LoadedNXTomo.__name__ == "LoadedNXTomo"
    assert io_api.NXTomoMetadata.__name__ == "NXTomoMetadata"
    assert io_api.PreprocessConfig.__name__ == "PreprocessConfig"
    assert io_api.PreprocessResult.__name__ == "PreprocessResult"
    assert io_api.ProjectionDataset.__name__ == "ProjectionDataset"
    assert io_api.ValidationReport.__name__ == "ValidationReport"
    assert callable(io_api.absorption_to_transmission)
    assert callable(io_api.build_geometry_from_dataset_metadata)
    assert callable(io_api.constant_dark_field)
    assert callable(io_api.convert_dataset)
    assert callable(io_api.drop_none)
    assert callable(io_api.flat_dark_correct_frames_to_absorption)
    assert callable(io_api.flat_dark_to_absorption)
    assert callable(io_api.flat_dark_to_transmission)
    assert callable(io_api.format_inspection_report)
    assert callable(io_api.inspect_dataset)
    assert callable(io_api.load_dataset)
    assert callable(io_api.load_nxtomo)
    assert callable(io_api.load_projection_payload)
    assert callable(io_api.load_tiff_stack)
    assert callable(io_api.pad_to_multiples)
    assert callable(io_api.preprocess_nxtomo)
    assert callable(io_api.preprocess_tiff_stack)
    assert callable(io_api.read_json_object)
    assert callable(io_api.save_dataset)
    assert callable(io_api.save_nxtomo)
    assert callable(io_api.save_projection_payload)
    assert callable(io_api.save_projection_quicklook)
    assert callable(io_api.spatial_bin)
    assert callable(io_api.summarize_angles)
    assert callable(io_api.validate_dataset)
    assert callable(io_api.validate_nxtomo)
    assert callable(io_api.transmission_to_absorption)
    assert callable(io_api.volume_chunks)
    assert callable(io_api.write_json_object)
    assert callable(io_api.normalize_json)


def test_datasets_facade_exports_simulation_boundary() -> None:
    _assert_facade_reexports_api(datasets_api, datasets_full_api)

    assert datasets_api.SimConfig.__name__ == "SimConfig"
    assert datasets_api.SimulatedData.__name__ == "SimulatedData"
    assert datasets_api.SimulationArtefacts.__name__ == "SimulationArtefacts"
    assert datasets_api.SyntheticArrayMetadata.__name__ == "SyntheticArrayMetadata"
    assert datasets_api.SyntheticArtifactPaths.__name__ == "SyntheticArtifactPaths"
    assert datasets_api.SyntheticDatasetConsistency.__name__ == "SyntheticDatasetConsistency"
    assert datasets_api.SyntheticDatasetSidecars.__name__ == "SyntheticDatasetSidecars"
    assert datasets_api.SyntheticDatasetSpec.__name__ == "SyntheticDatasetSpec"
    assert callable(datasets_api.generate_synthetic_dataset)
    assert callable(datasets_api.load_synthetic128_specs)
    assert callable(datasets_api.load_synthetic_dataset_sidecars)
    assert callable(datasets_api.make_benchmark_phantom)
    assert callable(datasets_api.make_phantom)
    assert callable(datasets_api.random_cubes_spheres)
    assert callable(datasets_api.simulate)
    assert callable(datasets_api.simulate_to_file)
    assert callable(datasets_api.synthetic128_spec)
    assert callable(datasets_api.validate_simulation_artefacts)


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


def test_product_facades_reexport_their_api_modules() -> None:
    facade_pairs = (
        (core_api, core_full_api),
        (forward_api, forward_full_api),
        (geometry_api, geometry_full_api),
        (motion_api, motion_full_api),
        (nuisance_api, nuisance_full_api),
        (verify_api, verify_full_api),
    )

    for package, api in facade_pairs:
        _assert_facade_reexports_api(package, api)
