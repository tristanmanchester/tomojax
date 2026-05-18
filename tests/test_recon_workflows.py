from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.io import ProjectionDataset, load_dataset, save_dataset
from tomojax.recon import FBPConfig, FistaConfig, default_fbp_scale, fbp, fista_tv


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


def test_fbp_generic_fast_path_synchronizes_before_oom_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid, detector, geometry = _tiny_geometry()
    projections = jnp.ones((4, 4, 4), dtype=jnp.float32)
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
        return jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    monkeypatch.setitem(fbp.__globals__, "_can_use_direct_parallel_fbp", lambda *_: False)
    monkeypatch.setitem(fbp.__globals__, "_run_fbp_fast_path", fake_fast_path)
    monkeypatch.setitem(fbp.__globals__, "_run_fbp_with_backoff", fake_backoff)

    volume = fbp(
        geometry,
        grid,
        detector,
        projections,
        config=FBPConfig(filter_name="ramp", views_per_batch=4),
    )

    assert calls == ["fast", "backoff"]
    np.testing.assert_allclose(np.asarray(volume), np.pi / 4.0)


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
