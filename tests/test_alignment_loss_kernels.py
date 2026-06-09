from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# check-public-imports: allow-private
from tomojax.align._objectives import fixed_volume

# check-public-imports: allow-private
from tomojax.align._objectives.loss_kernels import _loss_barron, _loss_swd

# check-public-imports: allow-private
from tomojax.align._objectives.loss_state import LossState
from tomojax.geometry import Detector, Grid


def test_barron_loss_uses_general_alpha_coefficient() -> None:
    pred = jnp.full((2, 2), 2.0, dtype=jnp.float32)
    target = jnp.ones((2, 2), dtype=jnp.float32)
    alpha = 1.5
    beta = abs(alpha - 2.0)
    expected_per_element = (beta / alpha) * ((1.0 + 1.0 / beta) ** (alpha / 2.0) - 1.0)

    actual = _loss_barron(pred, target, LossState(kind="barron", params={"alpha": alpha, "c": 1.0}))

    assert float(actual) == pytest.approx(expected_per_element * 4.0)


def test_swd_loss_subsampling_uses_supplied_rng_key() -> None:
    pred = jnp.arange(25, dtype=jnp.float32).reshape(5, 5)
    base = jnp.arange(25, dtype=jnp.float32)
    target = (base**1.3 + (base % 3) * 7.0).reshape(5, 5)
    params = {"n_samples": 5.0}

    first = _loss_swd(
        pred,
        target,
        LossState(kind="swd", params=params, rng_key=jax.random.key(0)),
    )
    same_key = _loss_swd(
        pred,
        target,
        LossState(kind="swd", params=params, rng_key=jax.random.key(0)),
    )
    other_key = _loss_swd(
        pred,
        target,
        LossState(kind="swd", params=params, rng_key=jax.random.key(1)),
    )

    assert float(first) == pytest.approx(float(same_key))
    assert float(first) != pytest.approx(float(other_key))


def test_pallas_project_stack_records_unsupported_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedPallas(ValueError):
        pass

    def resolve(name: str, *, missing_reason: str):
        if name == "pallas_projector_sinogram_unsupported_reason":
            return (lambda *_args, **_kwargs: None), None
        if name == "PallasProjectorUnsupported":
            return UnsupportedPallas, None
        if name == "forward_project_views_T_pallas":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    UnsupportedPallas("tile shape not supported")
                )
            ), None
        return None, missing_reason

    monkeypatch.setattr(fixed_volume, "resolve_pallas_callable", resolve)
    monkeypatch.setattr(
        fixed_volume,
        "forward_project_view_T",
        lambda *_args, **_kwargs: jnp.ones((2, 2), dtype=jnp.float32),
    )

    result = fixed_volume.project_stack(
        pose_stack=jnp.eye(4, dtype=jnp.float32)[None, ...],
        grid=Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
        detector=Detector(nu=2, nv=2, du=1.0, dv=1.0),
        volume=jnp.ones((2, 2, 2), dtype=jnp.float32),
        det_grid=(jnp.zeros((2, 2), dtype=jnp.float32), jnp.zeros((2, 2), dtype=jnp.float32)),
        projector_backend="pallas",
        require_differentiable_projector=False,
    )

    assert result.shape == (1, 2, 2)
    assert float(jnp.sum(result)) == pytest.approx(4.0)


def test_pallas_project_stack_propagates_unexpected_fast_path_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def resolve(name: str, *, missing_reason: str):
        if name == "pallas_projector_sinogram_unsupported_reason":
            return (lambda *_args, **_kwargs: None), None
        if name == "PallasProjectorUnsupported":
            return ValueError, None
        if name == "forward_project_views_T_pallas":
            return (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("broken kernel"))
            ), None
        return None, missing_reason

    monkeypatch.setattr(fixed_volume, "resolve_pallas_callable", resolve)

    with pytest.raises(RuntimeError, match="broken kernel"):
        fixed_volume.project_stack(
            pose_stack=jnp.eye(4, dtype=jnp.float32)[None, ...],
            grid=Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
            detector=Detector(nu=2, nv=2, du=1.0, dv=1.0),
            volume=jnp.ones((2, 2, 2), dtype=jnp.float32),
            det_grid=(
                jnp.zeros((2, 2), dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.float32),
            ),
            projector_backend="pallas",
            require_differentiable_projector=False,
        )
