from __future__ import annotations

import tomojax.align as align_api
import tomojax.align.api as align_full_api
import tomojax.bench as bench_api
import tomojax.calibration as calibration_api
import tomojax.io as io_api
import tomojax.recon as recon_api


def test_alignment_facade_exports_documented_api() -> None:
    assert align_api.AlignConfig.__name__ == "AlignConfig"
    assert callable(align_api.align)
    assert callable(align_api.align_multires)
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


def test_bench_facade_exports_developer_benchmark_helpers() -> None:
    assert bench_api.AlignmentScenario.__name__ == "AlignmentScenario"
    assert bench_api.AlternatingSmokeConfig.__name__ == "AlternatingSmokeConfig"
    assert bench_api.SyntheticBenchmarkResult.__name__ == "SyntheticBenchmarkResult"
    assert callable(bench_api.run_alignment_smoke)
    assert callable(bench_api.run_alternating_solver_smoke)
    assert callable(bench_api.scenario_catalog)
    assert callable(bench_api.load_synthetic_benchmark_result)
