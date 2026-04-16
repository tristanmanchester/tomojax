import sys
import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fista_tv import FistaConfig, fista_tv, grad_data_term
from tomojax.recon.multires import fista_multires, scale_detector, upsample_volume


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def make_case(nx=16, ny=16, nz=16, n_views=16):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    vol = vol.at[nx // 4 : 3 * nx // 4, ny // 4 : 3 * ny // 4, nz // 4 : 3 * nz // 4].set(1.0)
    projs = []
    for i in range(n_views):
        projs.append(forward_project_view(geom, grid, det, vol, view_index=i))
    projs = jnp.stack(projs, axis=0)
    return grid, det, geom, vol, projs


def test_multires_beats_single_level():
    grid, det, geom, vol, projs = make_case(12, 12, 12, 12)
    # Single-level
    x_single, info_single = fista_tv(
        geom,
        grid,
        det,
        projs,
        config=FistaConfig(iters=6, lambda_tv=0.001),
    )
    g_single, loss_single = grad_data_term(geom, grid, det, projs, x_single)

    # Two-level (2 -> 1) with same total iters
    x_multi, info_multi = fista_multires(
        geom, grid, det, projs, factors=(2, 1), iters_per_level=(3, 3), lambda_tv=0.001
    )
    g_multi, loss_multi = grad_data_term(geom, grid, det, projs, x_multi)

    assert loss_multi <= loss_single + 1e-3
    assert x_single.shape == vol.shape
    assert x_multi.shape == vol.shape
    assert len(info_single["loss"]) == 6
    assert len(info_multi["loss"]) == 6
    assert info_multi["factors"] == [2, 1]
    assert info_single["effective_iters"] <= 6
    assert float(jnp.linalg.norm(g_multi)) <= float(jnp.linalg.norm(g_single)) + 1e-3


def test_multires_rejects_mismatched_level_lengths():
    grid, det, geom, vol, projs = make_case(16, 16, 16, 8)
    with pytest.raises(ValueError, match="same length"):
        fista_multires(
            geom,
            grid,
            det,
            projs,
            factors=(4, 2, 1),
            iters_per_level=(5, 6),
            lambda_tv=0.001,
        )


def test_upsample_volume_resizes_when_target_shape_differs_even_if_factor_is_one():
    vol = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)

    up = upsample_volume(vol, factor=1, target_shape=(4, 4, 4))

    assert up.shape == (4, 4, 4)
    assert up.dtype == vol.dtype


def test_upsample_volume_is_noop_when_target_shape_matches():
    vol = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)

    up = upsample_volume(vol, factor=3, target_shape=vol.shape)

    assert up is vol
    np.testing.assert_array_equal(np.asarray(up), np.asarray(vol))


@pytest.mark.parametrize(
    ("nu", "factor"),
    [(50, 2), (99, 2), (100, 2), (50, 3), (99, 3), (99, 4), (100, 4)],
)
def test_scale_detector_matches_centered_projection_samples(nu, factor):
    det = Detector(nu=nu, nv=nu, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    scaled = scale_detector(det, factor)

    def sampled_positions(n: int, spacing: float, center: float):
        coords = (np.arange(n, dtype=np.float32) - (n / 2.0 - 0.5)) * spacing + center
        pad = (factor - (n % factor)) % factor
        left = pad // 2
        coords = np.pad(coords, (left, pad - left), mode="edge")
        sampled = coords[factor // 2 :: factor][: int(np.ceil(n / factor))]
        raw_indices = np.arange(int(np.ceil(n / factor))) * factor + (factor // 2) - left
        unclipped = raw_indices < n
        return sampled, unclipped

    sampled_u, unclipped_u = sampled_positions(det.nu, det.du, det.det_center[0])
    coarse_u = (
        np.arange(scaled.nu, dtype=np.float32) - (scaled.nu / 2.0 - 0.5)
    ) * scaled.du + scaled.det_center[0]
    np.testing.assert_allclose(coarse_u[unclipped_u], sampled_u[unclipped_u], atol=1e-6)

    sampled_v, unclipped_v = sampled_positions(det.nv, det.dv, det.det_center[1])
    coarse_v = (
        np.arange(scaled.nv, dtype=np.float32) - (scaled.nv / 2.0 - 0.5)
    ) * scaled.dv + scaled.det_center[1]
    np.testing.assert_allclose(coarse_v[unclipped_v], sampled_v[unclipped_v], atol=1e-6)
