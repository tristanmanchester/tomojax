from __future__ import annotations

from contextlib import nullcontext

import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.cli._runtime import transfer_guard_context

# check-public-imports: allow-private
from tomojax.core.backend_policy import backend_provenance, normalize_projector_backend

# check-public-imports: allow-private
from tomojax.core.validation import (
    validate_detector_grid,
    validate_projection_shape,
    validate_volume,
)
from tomojax.geometry import Detector, Grid, ParallelGeometry


def test_backend_policy_normalizes_aliases_and_reports_fallback_status() -> None:
    assert normalize_projector_backend(" default ") == "jax"
    assert normalize_projector_backend("CUDA") == "cuda"

    provenance = backend_provenance(
        requested_backend="cuda",
        actual_backend="jax",
        api_surface="unit-test",
        fallback_reason="unsupported detector tile",
        differentiability="performance_only",
    )

    assert provenance.status == "fallback"
    assert provenance.eligible_for_speed_claim is False
    assert provenance.to_dict() == {
        "requested_backend": "cuda",
        "actual_backend": "jax",
        "status": "fallback",
        "fallback_reason": "unsupported detector tile",
        "api_surface": "unit-test",
        "differentiability": "performance_only",
        "eligible_for_speed_claim": False,
    }

    with pytest.raises(ValueError, match="backend must be one of"):
        normalize_projector_backend("pallas")


def test_core_validation_errors_include_context_expected_shape_and_fix() -> None:
    grid = Grid(nx=2, ny=3, nz=4, vx=1.0, vy=1.0, vz=1.0)

    with pytest.raises(ValueError) as excinfo:
        validate_volume(jnp.zeros((2, 3, 5), dtype=jnp.float32), grid, context="ctx")

    message = str(excinfo.value)
    assert "ctx: volume has incompatible shape" in message
    assert "expected (nz, ny, nx)=(4, 3, 2) from grid" in message
    assert "Likely fix:" in message


def test_projection_shape_validation_uses_geometry_view_count() -> None:
    detector = Detector(nu=4, nv=3, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        detector=detector,
        angles=[0.0, 90.0],
    )

    with pytest.raises(ValueError, match=r"expected .*\(2, 3, 4\).*geometry/detector"):
        validate_projection_shape((1, 3, 4), detector, geometry=geometry, context="ctx")


def test_detector_grid_validation_requires_pair_of_flattened_vectors() -> None:
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)

    with pytest.raises(ValueError, match="expected a pair of detector-grid vectors"):
        validate_detector_grid(jnp.zeros((4,), dtype=jnp.float32), detector, context="ctx")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"det_grid\[1\].*expected .*\(4,\)"):
        validate_detector_grid(
            (
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((3,), dtype=jnp.float32),
            ),
            detector,
            context="ctx",
        )


def test_grid_and_detector_hold_plain_numbers_from_numpy_metadata() -> None:
    grid = Grid(nx=np.uint8(200), ny=np.int64(3), nz=2, vx=np.float32(0.5), vy=1, vz=1.0)
    detector = Detector(nu=np.uint16(640), nv=np.int32(1), du=np.float32(0.25), dv=1)

    assert grid == Grid(nx=200, ny=3, nz=2, vx=0.5, vy=1.0, vz=1.0)
    assert hash(grid) == hash(Grid(nx=200, ny=3, nz=2, vx=0.5, vy=1.0, vz=1.0))
    assert grid.nx * grid.nx == 40_000  # an 8-bit count would wrap
    assert type(detector.nu) is int and type(detector.du) is float
    with pytest.raises(TypeError):
        Grid(nx=64.5, ny=1, nz=1, vx=1.0, vy=1.0, vz=1.0)


def test_transfer_guard_context_returns_noop_when_disabled() -> None:
    assert isinstance(transfer_guard_context("off"), type(nullcontext()))
