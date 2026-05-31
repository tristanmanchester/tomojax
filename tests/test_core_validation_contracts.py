from __future__ import annotations

from contextlib import nullcontext

import jax.numpy as jnp
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
    assert normalize_projector_backend("PALLAS") == "pallas"

    provenance = backend_provenance(
        requested_backend="pallas",
        actual_backend="jax",
        api_surface="unit-test",
        fallback_reason="unsupported detector tile",
        differentiability="performance_only",
    )

    assert provenance.status == "fallback"
    assert provenance.eligible_for_speed_claim is False
    assert provenance.to_dict() == {
        "requested_backend": "pallas",
        "actual_backend": "jax",
        "status": "fallback",
        "fallback_reason": "unsupported detector tile",
        "api_surface": "unit-test",
        "differentiability": "performance_only",
        "eligible_for_speed_claim": False,
    }

    with pytest.raises(ValueError, match="projector_backend must be one of"):
        normalize_projector_backend("cuda")


def test_core_validation_errors_include_context_expected_shape_and_fix() -> None:
    grid = Grid(nx=2, ny=3, nz=4, vx=1.0, vy=1.0, vz=1.0)

    with pytest.raises(ValueError) as excinfo:
        validate_volume(jnp.zeros((2, 3, 5), dtype=jnp.float32), grid, context="ctx")

    message = str(excinfo.value)
    assert "ctx: volume has incompatible shape" in message
    assert "expected (nx, ny, nz)=(2, 3, 4) from grid" in message
    assert "Likely fix:" in message


def test_projection_shape_validation_uses_geometry_view_count() -> None:
    detector = Detector(nu=4, nv=3, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        detector=detector,
        thetas_deg=[0.0, 90.0],
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


def test_transfer_guard_context_returns_noop_when_disabled() -> None:
    assert isinstance(transfer_guard_context("off"), type(nullcontext()))
