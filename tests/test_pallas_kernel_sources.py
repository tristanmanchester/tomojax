from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.backends import resolve_pallas_module

# check-public-imports: allow-private
from tomojax.core import pallas_resolver, projector

# check-public-imports: allow-private
from tomojax.core.pallas import api as pallas_api
from tomojax.geometry import Detector, Grid


def test_resolve_pallas_module_imports_public_facade() -> None:
    capability = resolve_pallas_module()

    assert capability.unavailable_reason is None
    assert capability.module is not None
    assert callable(capability.module.forward_project_view_T_pallas)


def test_resolve_pallas_callable_rejects_non_callable_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pallas_resolver,
        "resolve_pallas_module",
        lambda: pallas_resolver.PallasModuleCapability(
            SimpleNamespace(not_a_function=object()),
            None,
        ),
    )

    fn, reason = pallas_resolver.resolve_pallas_callable("not_a_function")

    assert fn is None
    assert reason == "not_a_function_not_callable"


def test_public_pallas_entrypoint_uses_single_options_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_project(*_args: object, **kwargs: object) -> jnp.ndarray:
        captured.update(kwargs)
        return jnp.zeros((2, 2), dtype=jnp.float32)

    monkeypatch.setattr(pallas_api, "_forward_project_view_T_pallas", fake_project)

    result = pallas_api.forward_project_view_T_pallas(
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        options=pallas_api.PallasProjectorOptions(
            gather_dtype="bf16",
            interpret=True,
            state_mode="cached",
        ),
    )

    assert result.shape == (2, 2)
    assert captured["gather_dtype"] == "bf16"
    assert captured["interpret"] is True
    assert captured["state_mode"] == "cached"


def _tiny_projection_case() -> tuple[jnp.ndarray, Grid, Detector, jnp.ndarray]:
    return (
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
    )


def _tiny_pallas_options() -> pallas_api.PallasProjectorOptions:
    return pallas_api.PallasProjectorOptions(
        interpret=True,
        tile_shape=(1, 1),
        num_warps=1,
    )


def test_pallas_variant_metadata_normalizes_user_options() -> None:
    metadata = pallas_api.pallas_projector_variant_metadata(
        tile_shape=(4, 8),
        num_warps=2,
        kernel_variant="AUTO",
        layout_variant="DETECTOR_UV",
        state_mode="CACHED",
        gather_dtype="float16",
    )

    assert metadata == {
        "tile_shape": [4, 8],
        "num_warps": 2,
        "kernel_variant": "generic",
        "layout_variant": "detector_uv",
        "state_mode": "cached",
        "gather_dtype": "fp16",
    }


def test_pallas_variant_metadata_rejects_invalid_tile_and_warp_options() -> None:
    with pytest.raises(pallas_api.PallasProjectorUnsupported, match="tile_shape"):
        pallas_api.pallas_projector_variant_metadata(tile_shape=(0, 8))

    with pytest.raises(pallas_api.PallasProjectorUnsupported, match="num_warps"):
        pallas_api.pallas_projector_variant_metadata(num_warps=3)


def test_pallas_public_support_reason_reports_validation_failures() -> None:
    reason = pallas_api.pallas_projector_unsupported_reason(
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float16),
    )

    assert reason is not None
    assert "volume dtype must be float32" in reason


def test_single_view_pallas_interpret_matches_jax_reference() -> None:
    transform, grid, detector, volume = _tiny_projection_case()

    expected = projector.forward_project_view_T(
        transform,
        grid,
        detector,
        volume,
        projector_backend="jax",
    )
    actual = pallas_api.forward_project_view_T_pallas(
        transform,
        grid,
        detector,
        volume,
        options=_tiny_pallas_options(),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_pallas_residual_sse_interpret_matches_jax_reference_loss() -> None:
    transform, grid, detector, volume = _tiny_projection_case()
    pose_stack = transform[None, ...]
    target = jnp.zeros((1, detector.nv, detector.nu), dtype=jnp.float32)
    projection = projector.forward_project_view_T(
        transform,
        grid,
        detector,
        volume,
        projector_backend="jax",
    )

    actual = pallas_api.forward_project_residual_sse_T_pallas(
        pose_stack,
        grid,
        detector,
        volume,
        target,
        options=_tiny_pallas_options(),
    )
    expected = jnp.sum((projection[None, ...] - target) ** 2)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_single_view_pallas_unsupported_falls_back_to_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedPallas(ValueError):
        pass

    def resolve(name: str, *, missing_reason: str):
        if name == "PallasProjectorUnsupported":
            return UnsupportedPallas, None
        if name == "forward_project_view_T_pallas":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    UnsupportedPallas("detector tile unsupported")
                )
            ), None
        return None, missing_reason

    monkeypatch.setattr(projector, "resolve_pallas_callable", resolve)

    result = projector.forward_project_view_T(
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        projector_backend="pallas",
    )

    assert result.shape == (2, 2)


def test_single_view_pallas_unexpected_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(name: str, *, missing_reason: str):
        if name == "PallasProjectorUnsupported":
            return ValueError, None
        if name == "forward_project_view_T_pallas":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken kernel"))
            ), None
        return None, missing_reason

    monkeypatch.setattr(projector, "resolve_pallas_callable", resolve)

    with pytest.raises(RuntimeError, match="broken kernel"):
        projector.forward_project_view_T(
            jnp.eye(4, dtype=jnp.float32),
            Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
            Detector(nu=2, nv=2, du=1.0, dv=1.0),
            jnp.ones((2, 2, 2), dtype=jnp.float32),
            projector_backend="pallas",
        )


def test_projector_traversal_without_det_grid_does_not_use_host_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_host_cache(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("host detector grid cache should not be used during traversal")

    monkeypatch.setattr(projector, "_build_detector_grid_cached", fail_host_cache)

    result = projector.forward_project_view_T(
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
    )

    assert result.shape == (2, 2)
