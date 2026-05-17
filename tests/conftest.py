from pathlib import Path
import sys

import pytest

# Ensure package imports work from repo checkout: add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SURFACE_TEST_FILES = frozenset(
    {
        "test_cli_entrypoints.py",
        "test_cli_public_surface.py",
        "test_convert.py",
        "test_golden_path_cli.py",
        "test_importability.py",
        "test_inspect_cli.py",
        "test_io_public_dataset.py",
        "test_public_facades.py",
        "test_validate_cli.py",
    }
)

NUMERICAL_TEST_PREFIXES = (
    "test_align_",
    "test_alignment_",
    "test_alternating_",
    "test_bench_",
    "test_bilevel_",
    "test_calibration_",
    "test_detector_center_",
    "test_fbp_",
    "test_forward_",
    "test_geometry_",
    "test_grad_",
    "test_integration",
    "test_joint_",
    "test_lm_",
    "test_loss_bench",
    "test_multires",
    "test_nuisance_",
    "test_object_motion_",
    "test_phantoms",
    "test_phasecorr",
    "test_pose_",
    "test_projector",
    "test_recon",
    "test_reference_fista",
    "test_residual_",
    "test_rich_phantom_",
    "test_setup_lm",
    "test_simulate",
    "test_simulation_",
    "test_spdhg",
    "test_support_",
    "test_synthetic_",
    "test_tv_",
    "test_vertical_",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep the fast feedback loop separate from numerical coverage.

    TomoJAX has many CPU tests that compile or run JAX tomography kernels but
    are not slow enough to deserve the `slow` marker individually. Auto-marking
    them keeps `just check` bounded while preserving an explicit numerical gate.
    """
    for item in items:
        filename = Path(str(item.fspath)).name
        if filename in SURFACE_TEST_FILES:
            item.add_marker(pytest.mark.surface)
        if filename.startswith(NUMERICAL_TEST_PREFIXES):
            item.add_marker(pytest.mark.numerical)
