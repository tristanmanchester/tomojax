from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.backends import resolve_cuda_module

# check-public-imports: allow-private
from tomojax.core import cuda_projector_resolver, projector

# check-public-imports: allow-private
from tomojax.core.cuda_projector import api as cuda_api
from tomojax.geometry import Detector, Grid, LaminographyGeometry


def test_resolve_cuda_module_imports_public_facade() -> None:
    capability = resolve_cuda_module()

    assert capability.unavailable_reason is None
    assert capability.module is not None
    assert callable(capability.module.forward_project_view_T_cuda)


def test_resolve_cuda_callable_rejects_non_callable_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cuda_projector_resolver,
        "resolve_cuda_module",
        lambda: cuda_projector_resolver.CudaModuleCapability(
            SimpleNamespace(not_a_function=object()),
            None,
        ),
    )

    fn, reason = cuda_projector_resolver.resolve_cuda_callable("not_a_function")

    assert fn is None
    assert reason == "not_a_function_not_callable"


def test_public_cuda_entrypoint_uses_single_options_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_project(*_args: object, **kwargs: object) -> jnp.ndarray:
        captured.update(kwargs)
        return jnp.zeros((2, 2), dtype=jnp.float32)

    monkeypatch.setattr(cuda_api, "_forward_project_view_T_cuda", fake_project)

    result = cuda_api.forward_project_view_T_cuda(
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        options=cuda_api.CudaProjectorOptions(
            gather_dtype="bf16",
            interpret=True,
            state_mode="cached",
        ),
    )

    assert result.shape == (2, 2)
    assert captured["gather_dtype"] == "bf16"
    assert captured["interpret"] is True
    assert captured["state_mode"] == "cached"


def test_public_cuda_stack_entrypoint_accepts_legacy_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_project(*_args: object, **kwargs: object) -> jnp.ndarray:
        captured.update(kwargs)
        return jnp.zeros((1, 2, 2), dtype=jnp.float32)

    monkeypatch.setattr(cuda_api, "_forward_project_views_T_cuda", fake_project)

    det_grid = (jnp.zeros(4), jnp.arange(4, dtype=jnp.float32))
    result = cuda_api.forward_project_views_T_cuda(
        jnp.eye(4, dtype=jnp.float32)[None, ...],
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        options=cuda_api.CudaProjectorOptions(
            gather_dtype="fp32", tile_shape=(4, 4), n_steps=11, interpret=True, det_grid=det_grid
        ),
        n_steps=None,
        interpret=False,
        det_grid=None,
        unroll=2,
        gather_dtype="bf16",
        tile_shape=(1, 2),
        num_warps=1,
        state_mode="cached",
    )

    assert result.shape == (1, 2, 2)
    assert captured["n_steps"] == 11
    assert captured["interpret"] is False
    assert captured["det_grid"] is det_grid
    assert captured["unroll"] == 2
    assert captured["gather_dtype"] == "bf16"
    assert captured["tile_shape"] == (1, 2)
    assert captured["num_warps"] == 1
    assert captured["state_mode"] == "cached"


def test_public_loss_grad_entrypoint_accepts_legacy_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_loss_grad(*_args: object, **kwargs: object) -> tuple[jnp.ndarray, jnp.ndarray]:
        captured.update(kwargs)
        return jnp.asarray(0.0, dtype=jnp.float32), jnp.ones((2, 2, 2), dtype=jnp.float32)

    monkeypatch.setattr(cuda_api, "_forward_project_loss_and_grad_T_cuda", fake_loss_grad)
    weights = jnp.ones((1, 1, 1), dtype=jnp.float32)

    loss, grad = cuda_api.forward_project_loss_and_grad_T_cuda(
        jnp.eye(4, dtype=jnp.float32)[None, ...],
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        weights=weights,
        compute_loss=False,
        unroll=3,
        gather_dtype="fp16",
        tile_shape=(2, 2),
        num_warps=2,
        kernel_variant="generic",
        layout_variant="detector_uv",
    )

    assert loss.shape == ()
    assert grad.shape == (2, 2, 2)
    assert captured["weights"] is weights
    assert captured["compute_loss"] is False
    assert captured["unroll"] == 3
    assert captured["gather_dtype"] == "fp16"
    assert captured["tile_shape"] == (2, 2)
    assert captured["num_warps"] == 2
    assert captured["kernel_variant"] == "generic"
    assert captured["layout_variant"] == "detector_uv"


def _tiny_projection_case() -> tuple[jnp.ndarray, Grid, Detector, jnp.ndarray]:
    return (
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
    )


def _tiny_cuda_options() -> cuda_api.CudaProjectorOptions:
    return cuda_api.CudaProjectorOptions(
        interpret=True,
        tile_shape=(1, 1),
        num_warps=1,
    )


def test_cuda_variant_metadata_normalizes_user_options() -> None:
    metadata = cuda_api.cuda_projector_variant_metadata(
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


def test_cuda_variant_metadata_rejects_invalid_tile_and_warp_options() -> None:
    with pytest.raises(cuda_api.CudaProjectorUnsupported, match="tile_shape"):
        cuda_api.cuda_projector_variant_metadata(tile_shape=(0, 8))

    with pytest.raises(cuda_api.CudaProjectorUnsupported, match="num_warps"):
        cuda_api.cuda_projector_variant_metadata(num_warps=3)


def test_cuda_public_support_reason_reports_validation_failures() -> None:
    reason = cuda_api.cuda_projector_unsupported_reason(
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float16),
    )

    assert reason is not None
    assert "volume dtype must be float32" in reason


def test_single_view_cuda_interpret_matches_jax_reference() -> None:
    transform, grid, detector, volume = _tiny_projection_case()

    expected = projector.forward_project_view_T(
        transform,
        grid,
        detector,
        volume,
        backend="jax",
    )
    actual = cuda_api.forward_project_view_T_cuda(
        transform,
        grid,
        detector,
        volume,
        options=_tiny_cuda_options(),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_cuda_residual_sse_interpret_matches_jax_reference_loss() -> None:
    transform, grid, detector, volume = _tiny_projection_case()
    pose_stack = transform[None, ...]
    target = jnp.zeros((1, detector.nv, detector.nu), dtype=jnp.float32)
    projection = projector.forward_project_view_T(
        transform,
        grid,
        detector,
        volume,
        backend="jax",
    )

    actual = cuda_api.forward_project_residual_sse_T_cuda(
        pose_stack,
        grid,
        detector,
        volume,
        target,
        options=_tiny_cuda_options(),
    )
    expected = jnp.sum((projection[None, ...] - target) ** 2)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_single_view_cuda_unsupported_falls_back_to_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedCuda(ValueError):
        pass

    def resolve(name: str, *, missing_reason: str):
        if name == "CudaProjectorUnsupported":
            return UnsupportedCuda, None
        if name == "forward_project_view_T_cuda":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    UnsupportedCuda("detector tile unsupported")
                )
            ), None
        return None, missing_reason

    monkeypatch.setattr(projector, "resolve_cuda_callable", resolve)

    result = projector.forward_project_view_T(
        jnp.eye(4, dtype=jnp.float32),
        Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        Detector(nu=2, nv=2, du=1.0, dv=1.0),
        jnp.ones((2, 2, 2), dtype=jnp.float32),
        backend="cuda",
    )

    assert result.shape == (2, 2)


def test_single_view_cuda_unexpected_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(name: str, *, missing_reason: str):
        if name == "CudaProjectorUnsupported":
            return ValueError, None
        if name == "forward_project_view_T_cuda":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken kernel"))
            ), None
        return None, missing_reason

    monkeypatch.setattr(projector, "resolve_cuda_callable", resolve)

    with pytest.raises(RuntimeError, match="broken kernel"):
        projector.forward_project_view_T(
            jnp.eye(4, dtype=jnp.float32),
            Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
            Detector(nu=2, nv=2, du=1.0, dv=1.0),
            jnp.ones((2, 2, 2), dtype=jnp.float32),
            backend="cuda",
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


@pytest.mark.parametrize(("backend", "auto"), [("cpu", "fp32"), ("gpu", "fp16"), ("tpu", "bf16")])
def test_cuda_gather_aliases_match_projector_precision(monkeypatch, backend, auto) -> None:
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    for aliases, canonical, dtype in (
        (("fp32", "float32", "single"), "fp32", jnp.float32),
        (("fp16", "float16", "half"), "fp16", jnp.float16),
        (("bf16", "bfloat16"), "bf16", jnp.bfloat16),
    ):
        for alias in aliases:
            assert projector.resolve_gather_target(alias.upper()) == dtype
            metadata = cuda_api.cuda_projector_variant_metadata(gather_dtype=alias.upper())
            assert metadata["gather_dtype"] == canonical
    assert cuda_api.cuda_projector_variant_metadata(gather_dtype="auto")["gather_dtype"] == auto


@pytest.mark.parametrize("precision", [None, "unsupported"])
def test_cuda_gather_errors_keep_backend_context(precision) -> None:
    with pytest.raises(ValueError) as jax_error:
        projector.resolve_gather_target(precision)
    with pytest.raises(cuda_api.CudaProjectorUnsupported) as cuda_error:
        cuda_api.cuda_projector_variant_metadata(gather_dtype=precision)
    assert str(cuda_error.value) == f"cuda_projector_unsupported: {jax_error.value}"


@pytest.mark.parametrize("step_size", [None, 0.3, 1.7])
def test_stack_traversal_metadata_matches_longest_view(step_size) -> None:
    grid, detector = Grid(19, 13, 11, 0.8, 1.1, 0.7), Detector(25, 17, 0.9, 0.8)
    geometry = LaminographyGeometry(grid, detector, [0, 37, 91], tilt_deg=37)
    poses = np.asarray([geometry.pose_for_view(i) for i in range(3)], dtype=np.float32)
    single = [
        cuda_api.cuda_projector_traversal_metadata(t, grid, step_size=step_size) for t in poses
    ]
    stacked = cuda_api.cuda_projector_traversal_metadata(poses, grid, step_size=step_size)
    assert stacked["resolved_n_steps"] == single[0]["resolved_n_steps"]
    assert stacked["effective_cuda_n_steps"] == max(x["effective_cuda_n_steps"] for x in single)
    assert (
        cuda_api.cuda_projector_traversal_metadata(poses[:1], grid, step_size=step_size)
        == single[0]
    )


def test_cuda_binding_forwards_options_without_copying_detector_grid(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def bind(*args, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(cuda_api, "_bind_forward_project_views_T_cuda", bind)
    coordinates = (jnp.arange(4, dtype=jnp.float32), jnp.zeros(4, dtype=jnp.float32))
    options = cuda_api.CudaProjectorOptions(det_grid=coordinates, block_state=False, unroll=2)
    pose, grid, detector, _ = _tiny_projection_case()
    assert (
        cuda_api.bind_forward_project_views_T_cuda(pose[None], grid, detector, options=options)
        is sentinel
    )
    assert captured["det_grid"] is coordinates
    assert captured["block_state"] is False and captured["unroll"] == 2
    assert "state_mode" not in captured


@pytest.mark.parametrize("stack", [False, True])
def test_cached_cuda_volume_validation_preserves_context(stack) -> None:
    pose, grid, detector, volume = _tiny_projection_case()
    bound = cuda_api.bind_forward_project_views_T_cuda(
        pose[None], grid, detector, options=_tiny_cuda_options()
    )
    for call in (
        bound,
        lambda x: cuda_api.forward_project_views_T_cuda_with_state(bound.state, x, interpret=True),
    ):
        with pytest.raises(ValueError, match="forward_project_views_T_cuda_with_state"):
            call(volume[:1])
        with pytest.raises(cuda_api.CudaProjectorUnsupported, match="volume dtype must be float32"):
            call(volume.astype(jnp.float16))
    support = (
        cuda_api.cuda_projector_sinogram_unsupported_reason
        if stack
        else cuda_api.cuda_projector_unsupported_reason
    )
    reason = support(
        pose[None] if stack else pose,
        grid,
        detector,
        volume,
        options=cuda_api.CudaProjectorOptions(state_mode="cached", layout_variant="detector_uv"),
    )
    assert "cached traversal state currently supports layout_variant='detector_vu' only" in reason
