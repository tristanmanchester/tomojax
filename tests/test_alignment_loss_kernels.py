from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.alignment._objectives import fixed_volume

# check-public-imports: allow-private
from tomojax.alignment._objectives.loss_adapters import build_loss_adapter

# check-public-imports: allow-private
from tomojax.alignment._objectives.loss_kernels import (
    _compute_otsu_threshold,
)
from tomojax.alignment.api import L2LossSpec
from tomojax.geometry import Detector, Grid


def _score_tiny_stack(**options):
    targets = jnp.zeros((1, 2, 2), dtype=jnp.float32)
    return fixed_volume.project_and_score_stack(
        pose_stack=jnp.eye(4, dtype=jnp.float32)[None, ...],
        grid=Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        detector=Detector(nu=2, nv=2, du=1.0, dv=1.0),
        volume=jnp.ones((2, 2, 2), dtype=jnp.float32),
        det_grid=(jnp.zeros((2, 2), dtype=jnp.float32), jnp.zeros((2, 2), dtype=jnp.float32)),
        targets=targets,
        loss_adapter=build_loss_adapter(L2LossSpec(), targets),
        backend="cuda",
        require_differentiable_projector=False,
        **options,
    )


def test_cuda_project_and_score_records_unsupported_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedCuda(ValueError):
        pass

    def resolve(name: str, *, missing_reason: str):
        if name == "cuda_projector_sinogram_unsupported_reason":
            return (lambda *_args, **_kwargs: None), None
        if name == "CudaProjectorUnsupported":
            return UnsupportedCuda, None
        if name == "forward_project_residual_sse_T_cuda":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    UnsupportedCuda("tile shape not supported")
                )
            ), None
        return None, missing_reason

    monkeypatch.setattr(fixed_volume, "resolve_cuda_callable", resolve)
    monkeypatch.setattr(
        fixed_volume,
        "project_views",
        lambda *_args, **_kwargs: jnp.ones((1, 2, 2), dtype=jnp.float32),
    )

    # The JAX fallback projects ones against zero targets: half the squared sum.
    assert float(_score_tiny_stack()) == pytest.approx(2.0)


def test_cuda_project_and_score_propagates_unexpected_fast_path_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(name: str, *, missing_reason: str):
        if name == "cuda_projector_sinogram_unsupported_reason":
            return (lambda *_args, **_kwargs: None), None
        if name == "CudaProjectorUnsupported":
            return ValueError, None
        if name == "forward_project_residual_sse_T_cuda":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken kernel"))
            ), None
        return None, missing_reason

    monkeypatch.setattr(fixed_volume, "resolve_cuda_callable", resolve)

    with pytest.raises(RuntimeError, match="broken kernel"):
        _score_tiny_stack()


def test_otsu_resolves_adjacent_float32_intensities():
    values = np.array([1.0, np.nextafter(np.float32(1), np.float32(2))])
    threshold = _compute_otsu_threshold(values.astype(np.float32))
    assert values[0] < threshold < values[1]


@pytest.mark.gpu
def test_cone_alignment_objective_uses_cuda_pose_derivatives():
    from tomojax.core.cone import cone_model, use_cuda_cone
    from tomojax.geometry import ConeBeam, ConeGeometry, stack_view_poses

    if not use_cuda_cone():
        pytest.skip("requires CUDA cone kernels")
    grid, detector = Grid(7, 6, 5, 0.7, 1.1, 0.9), Detector(9, 7, 1.3, 0.8)
    geometry = ConeGeometry(grid, detector, [13.7, 78.1, 147.3], ConeBeam(80, 160), tilt_deg=53)
    poses = stack_view_poses(geometry, 3)
    volume = jnp.asarray(np.random.default_rng(19).uniform(size=(7, 6, 5)), jnp.float32)
    options = dict(
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=(jnp.zeros((7, 9)), jnp.zeros((7, 9))),
        frames=jnp.asarray(cone_model(geometry).frames(3)),
        require_differentiable_projector=True,
    )
    provenance = fixed_volume.alignment_projector_backend_provenance(
        pose_stack=poses, backend="cuda", api_surface="test", **options
    )
    assert provenance.actual_backend == "cuda"
    assert provenance.differentiability == "gradient_safe"

    targets = jnp.full((3, 7, 9), 0.5, jnp.float32)
    adapter = build_loss_adapter(L2LossSpec(), targets)

    def value_grad(backend):
        def loss(t):
            return fixed_volume.project_and_score_stack(
                pose_stack=t, backend=backend, targets=targets, loss_adapter=adapter, **options
            )

        return jax.jit(jax.value_and_grad(loss))(poses)

    for actual, expected in zip(value_grad("cuda"), value_grad("jax"), strict=True):
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=2e-4 * float(jnp.max(jnp.abs(expected)))
        )
