from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.io import ProjectionDataset, load_dataset, save_dataset
from tomojax.recon import FBPConfig, FistaConfig, fbp, fista_tv
from tomojax.recon.api import clear_filter_caches, default_fbp_scale

# check-public-imports: allow-private
from tomojax.recon.fbp import _run_fbp_generic_with_oom_fallback
from tomojax.recon.filters import get_filter_np


def _tiny_geometry() -> tuple[Grid, Detector, ParallelGeometry]:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.linspace(0.0, 180.0, 4, endpoint=False, dtype=np.float32),
    )
    return grid, detector, geometry


def test_default_fbp_scale_uses_positive_view_count() -> None:
    assert default_fbp_scale(4) == pytest.approx(np.pi / 4.0)
    with pytest.raises(ValueError, match="n_views must be positive"):
        default_fbp_scale(0)


def test_clear_filter_caches_rebuilds_read_only_filters() -> None:
    clear_filter_caches()
    first = get_filter_np("ramp", 4, 1.0)
    second = get_filter_np("ramp", 4, 1.0)
    assert first is second

    clear_filter_caches()
    rebuilt = get_filter_np("ramp", 4, 1.0)

    assert rebuilt is not first
    assert not rebuilt.flags.writeable


@pytest.mark.numerical
def test_fbp_ramp_filter_smoke_is_finite() -> None:
    grid, detector, geometry = _tiny_geometry()
    projections = jnp.ones((4, 4, 4), dtype=jnp.float32)

    volume = fbp(
        geometry,
        grid,
        detector,
        projections,
        config=FBPConfig(filter_name="ramp", views_per_batch=1),
    )

    assert volume.shape == (4, 4, 4)
    assert bool(jnp.all(jnp.isfinite(volume)))


def test_fbp_generic_fallback_helper_synchronizes_before_oom_backoff() -> None:
    calls: list[str] = []

    class AsyncFailure:
        def block_until_ready(self) -> None:
            raise RuntimeError("RESOURCE_EXHAUSTED: simulated async allocation failure")

    def fake_fast_path(*args: object, **kwargs: object) -> AsyncFailure:
        del args, kwargs
        calls.append("fast")
        return AsyncFailure()

    def fake_backoff(*args: object, **kwargs: object) -> jnp.ndarray:
        del args, kwargs
        calls.append("backoff")
        return jnp.ones((2, 2, 2), dtype=jnp.float32)

    volume = _run_fbp_generic_with_oom_fallback(
        fast_path=fake_fast_path,
        backoff_path=fake_backoff,
        view_progress=iter(range(4)),
        n_views=4,
    )

    assert calls == ["fast", "backoff"]
    np.testing.assert_allclose(np.asarray(volume), 1.0)


@pytest.mark.parametrize(("fail_call", "expected_batches"), [(1, [4, 2, 2, 2]), (2, [4, 4, 2])])
def test_fbp_chunk_backoff_preserves_accumulator_and_progress_on_async_oom(
    monkeypatch, fail_call, expected_batches
) -> None:
    import importlib

    module = importlib.import_module("tomojax.recon.fbp")
    calls = []
    progress = iter(range(5))

    class AsyncFailure:
        def __radd__(self, other):
            return self

        def block_until_ready(self):
            raise RuntimeError("RESOURCE_EXHAUSTED: asynchronous chunk allocation")

    def backproject(poses, filtered, **kwargs):
        calls.append(int(poses.shape[0]))
        if len(calls) == fail_call:
            return AsyncFailure()
        return jnp.ones((2, 2, 1), dtype=jnp.float32) * jnp.sum(filtered)

    monkeypatch.setattr(module, "_bp_batch_sum_jit", backproject)
    result = module._run_fbp_with_backoff(
        jnp.broadcast_to(jnp.eye(4), (5, 4, 4)),
        jnp.arange(1, 6, dtype=jnp.float32).reshape(5, 1, 1),
        batch_size=4,
        grid=Grid(2, 2, 1, 1.0, 1.0, 1.0),
        detector=Detector(1, 1, 1.0, 1.0),
        filter_name="ramp",
        projector_unroll=1,
        checkpoint_projector=True,
        gather_dtype="fp32",
        det_grid=None,
        view_progress=progress,
    )
    # A one-pixel ramp row is multiplied by h[0] = 1/4. All five views
    # contribute once, including when an OOM occurs after a successful chunk.
    np.testing.assert_allclose(result, (1 + 2 + 3 + 4 + 5) / 4)
    assert calls == expected_batches
    assert next(progress, None) is None


@pytest.mark.numerical
def test_fista_reconstruction_smoke_is_finite() -> None:
    grid, detector, geometry = _tiny_geometry()
    projections = jnp.ones((4, 4, 4), dtype=jnp.float32)

    volume, info = fista_tv(
        geometry,
        grid,
        detector,
        projections,
        config=FistaConfig(iters=1, lambda_tv=0.0, views_per_batch=1, power_iters=1),
    )

    assert volume.shape == (4, 4, 4)
    assert bool(jnp.all(jnp.isfinite(volume)))
    assert "loss" in info


def test_reconstruction_volume_roundtrips_through_dataset_contract(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    volume = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    dataset = ProjectionDataset(
        projections=np.ones((2, 2, 4), dtype=np.float32),
        angles_deg=np.asarray([0.0, 90.0], dtype=np.float32),
        volume=volume,
        detector=Detector(nu=4, nv=2, du=1.0, dv=1.0),
        grid=Grid(nx=4, ny=4, nz=2, vx=1.0, vy=1.0, vz=1.0),
        sample_name="roundtrip",
    )

    save_dataset(path, dataset)
    loaded = load_dataset(path)

    assert loaded.volume is not None
    np.testing.assert_allclose(loaded.volume, volume)
    assert loaded.grid == dataset.grid
