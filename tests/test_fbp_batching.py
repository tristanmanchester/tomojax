import numpy as np
import jax
import jax.numpy as jnp
import pytest
from importlib import import_module

from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import forward_project_view, get_detector_grid_device
from tomojax.data.simulate import SimConfig, make_phantom
from tomojax.recon.fbp import FBPConfig, _default_fbp_scale, _fft_filter_rows, fbp
from tomojax.recon.filters import get_filter_np


def make_case(
    nx=12,
    ny=12,
    nz=12,
    n_views=12,
    *,
    vx=1.0,
    vy=1.0,
    vz=1.0,
    du=1.0,
    dv=1.0,
    det_center=(0.0, 0.0),
    nu=None,
    nv=None,
    asymmetric=False,
):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=vx, vy=vy, vz=vz)
    det = Detector(
        nu=nx if nu is None else nu,
        nv=nz if nv is None else nv,
        du=du,
        dv=dv,
        det_center=det_center,
    )
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)
    if asymmetric:
        vol = (
            vol.at[1 : max(2, nx // 2), 2 : max(3, ny - 2), 0 : max(1, nz // 3)].set(0.8)
            .at[max(0, nx - 4) : nx - 1, 1 : max(2, ny // 2), max(1, nz // 2) : nz - 1]
            .set(1.7)
            .at[nx // 2, max(0, ny - 3), nz // 2]
            .set(2.3)
        )
    else:
        vol = vol.at[
            nx // 4 : 3 * nx // 4, ny // 4 : 3 * ny // 4, nz // 4 : 3 * nz // 4
        ].set(1.0)
    projs = jnp.stack(
        [
            forward_project_view(geom, grid, det, vol, view_index=i)
            for i in range(n_views)
        ],
        axis=0,
    )
    return grid, det, geom, vol, projs


def _relative_l2(candidate, reference):
    candidate_np = np.asarray(candidate)
    reference_np = np.asarray(reference)
    denom = np.linalg.norm(reference_np.ravel()) or 1.0
    return np.linalg.norm((candidate_np - reference_np).ravel()) / denom


def assert_direct_parallel_fbp_matches_generic(
    grid,
    det,
    geom,
    projs,
    *,
    filter_name="ramp",
    max_relative_l2=0.04,
    max_abs=0.06,
):
    direct = fbp(geom, grid, det, projs, filter_name=filter_name, views_per_batch=0)
    generic = fbp(
        geom,
        grid,
        det,
        projs,
        filter_name=filter_name,
        views_per_batch=0,
        det_grid=get_detector_grid_device(det),
    )

    direct_np = np.asarray(direct)
    generic_np = np.asarray(generic)
    diff = direct_np - generic_np
    relative_l2 = np.linalg.norm(diff.ravel()) / np.linalg.norm(generic_np.ravel())
    assert relative_l2 <= max_relative_l2
    assert np.max(np.abs(diff)) <= max_abs


@pytest.mark.parametrize("nu", [7, 8, 15, 16])
@pytest.mark.parametrize("du", [0.75, 1.0, 1.25])
@pytest.mark.parametrize("filter_name", ["ramp", "shepp", "hann"])
def test_rfft_filter_rows_matches_full_fft_reference(nu, du, filter_name):
    rng = np.random.default_rng(0)
    rows = jnp.asarray(rng.normal(size=(5, nu)).astype(np.float32))
    transfer = jnp.asarray(get_filter_np(filter_name, nu, du), dtype=jnp.float32)

    expected = jnp.fft.ifft(jnp.fft.fft(rows, axis=-1) * transfer, axis=-1).real
    actual = _fft_filter_rows(rows, du=du, filter_name=filter_name)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-5, rtol=1e-5)


def test_rfft_filter_rows_rejects_unknown_filter():
    rows = jnp.ones((2, 8), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Unknown filter"):
        _fft_filter_rows(rows, du=1.0, filter_name="unknown")


def test_direct_parallel_fbp_matches_generic_path_on_small_fixture():
    grid, det, geom, vol, projs = make_case(10, 10, 10, 10)

    assert_direct_parallel_fbp_matches_generic(grid, det, geom, projs)


@pytest.mark.parametrize("filter_name", ["ramp", "shepp", "hann"])
def test_direct_parallel_fbp_matches_generic_for_filters(filter_name):
    grid, det, geom, vol, projs = make_case(10, 10, 10, 10)

    assert_direct_parallel_fbp_matches_generic(grid, det, geom, projs, filter_name=filter_name)


def test_direct_parallel_fbp_matches_generic_with_nonzero_detector_center():
    grid, det, geom, vol, projs = make_case(10, 10, 10, 10, det_center=(0.25, -0.5))

    assert_direct_parallel_fbp_matches_generic(grid, det, geom, projs)


def test_direct_parallel_fbp_matches_generic_with_nonunit_spacing():
    grid, det, geom, vol, projs = make_case(
        10,
        10,
        10,
        10,
        vx=1.0,
        vy=1.25,
        vz=0.75,
        du=1.0,
        dv=0.75,
    )

    assert_direct_parallel_fbp_matches_generic(
        grid,
        det,
        geom,
        projs,
        max_relative_l2=0.07,
        max_abs=0.12,
    )


def test_direct_parallel_fbp_matches_generic_with_nonsquare_detector():
    grid, det, geom, vol, projs = make_case(10, 8, 6, 10, nu=12, nv=7)

    assert_direct_parallel_fbp_matches_generic(grid, det, geom, projs)


def test_direct_parallel_fbp_matches_generic_on_noncubic_grid():
    grid, det, geom, vol, projs = make_case(10, 8, 6, 10)

    assert_direct_parallel_fbp_matches_generic(grid, det, geom, projs)


def test_direct_parallel_fbp_matches_generic_on_asymmetric_phantom():
    grid, det, geom, vol, projs = make_case(
        10,
        8,
        6,
        10,
        det_center=(0.5, -0.25),
        asymmetric=True,
    )

    assert_direct_parallel_fbp_matches_generic(
        grid,
        det,
        geom,
        projs,
        max_relative_l2=0.06,
        max_abs=0.08,
    )


def test_direct_parallel_fbp_backprojects_asymmetric_sinogram_like_generic(monkeypatch):
    fbp_mod = import_module("tomojax.recon.fbp")
    grid = Grid(nx=5, ny=5, nz=3, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=7, nv=3, du=1.0, dv=1.0, det_center=(0.25, -0.5))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 90.0])
    projs = jnp.arange(2 * det.nv * det.nu, dtype=jnp.float32).reshape(
        (2, det.nv, det.nu)
    )

    monkeypatch.setattr(fbp_mod, "_fft_filter_rows_jit", lambda rows, du, filter_name: rows)

    direct = fbp_mod.fbp(geom, grid, det, projs, scale=1.0)
    generic = fbp_mod.fbp(
        geom,
        grid,
        det,
        projs,
        scale=1.0,
        det_grid=get_detector_grid_device(det),
    )

    assert _relative_l2(direct, generic) <= 0.08
    assert np.max(np.abs(np.asarray(direct) - np.asarray(generic))) <= 3.1


def test_non_parallel_geometry_uses_generic_fbp_path(monkeypatch):
    fbp_mod = import_module("tomojax.recon.fbp")
    grid = Grid(nx=6, ny=6, nz=6, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=6, nv=6, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = LaminographyGeometry(grid=grid, detector=det, thetas_deg=[0.0, 30.0], tilt_deg=15.0)
    projs = jnp.ones((2, det.nv, det.nu), dtype=jnp.float32)
    calls = []

    def fail_direct(*args, **kwargs):
        raise AssertionError("direct parallel FBP should not run for non-ParallelGeometry")

    def fake_fast_path(*args, **kwargs):
        calls.append("generic")
        return jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    monkeypatch.setattr(fbp_mod, "_run_parallel_fbp_direct_jit", fail_direct)
    monkeypatch.setattr(fbp_mod, "_run_fbp_fast_path", fake_fast_path)

    rec = fbp_mod.fbp(geom, grid, det, projs, filter_name="ramp", views_per_batch=2, scale=1.0)

    assert calls == ["generic"]
    np.testing.assert_allclose(np.asarray(rec), np.ones((grid.nx, grid.ny, grid.nz)))


def test_parallel_geometry_with_explicit_detector_grid_uses_generic_fbp_path(monkeypatch):
    fbp_mod = import_module("tomojax.recon.fbp")
    grid = Grid(nx=6, ny=6, nz=6, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=6, nv=6, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 30.0])
    projs = jnp.ones((2, det.nv, det.nu), dtype=jnp.float32)
    calls = []

    def fail_direct(*args, **kwargs):
        raise AssertionError("direct parallel FBP should not run with explicit det_grid")

    def fake_fast_path(*args, **kwargs):
        calls.append("generic")
        return jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    monkeypatch.setattr(fbp_mod, "_run_parallel_fbp_direct_jit", fail_direct)
    monkeypatch.setattr(fbp_mod, "_run_fbp_fast_path", fake_fast_path)

    rec = fbp_mod.fbp(
        geom,
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=2,
        det_grid=get_detector_grid_device(det),
        scale=1.0,
    )

    assert calls == ["generic"]
    np.testing.assert_allclose(np.asarray(rec), np.ones((grid.nx, grid.ny, grid.nz)))


def test_direct_parallel_fbp_oom_falls_back_to_generic(monkeypatch):
    fbp_mod = import_module("tomojax.recon.fbp")
    grid, det, geom, vol, projs = make_case(6, 6, 6, 4)
    calls = []

    def fake_direct(*args, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: simulated direct OOM")

    def fake_fast_path(*args, **kwargs):
        calls.append("generic")
        return jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    monkeypatch.setattr(fbp_mod, "_run_parallel_fbp_direct_jit", fake_direct)
    monkeypatch.setattr(fbp_mod, "_run_fbp_fast_path", fake_fast_path)

    rec = fbp_mod.fbp(geom, grid, det, projs, scale=1.0)

    assert calls == ["generic"]
    np.testing.assert_allclose(np.asarray(rec), np.ones((grid.nx, grid.ny, grid.nz)))


def test_direct_parallel_fbp_non_oom_error_propagates(monkeypatch):
    fbp_mod = import_module("tomojax.recon.fbp")
    grid, det, geom, vol, projs = make_case(6, 6, 6, 4)

    def fake_direct(*args, **kwargs):
        raise RuntimeError("simulated semantic failure")

    monkeypatch.setattr(fbp_mod, "_run_parallel_fbp_direct_jit", fake_direct)

    with pytest.raises(RuntimeError, match="semantic failure"):
        fbp_mod.fbp(geom, grid, det, projs, scale=1.0)


def test_fbp_direct_path_is_gradable_wrt_projections():
    grid, det, geom, vol, projs = make_case(6, 6, 6, 6)

    def loss(p):
        return jnp.sum(fbp(geom, grid, det, p, filter_name="ramp", scale=1.0))

    grad = jax.grad(loss)(projs)

    assert grad.shape == projs.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_fbp_generic_path_is_gradable_wrt_projections():
    grid, det, geom, vol, projs = make_case(6, 6, 6, 6)
    det_grid = get_detector_grid_device(det)

    def loss(p):
        return jnp.sum(
            fbp(
                geom,
                grid,
                det,
                p,
                filter_name="ramp",
                scale=1.0,
                det_grid=det_grid,
            )
        )

    grad = jax.grad(loss)(projs)

    assert grad.shape == projs.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_fbp_direct_path_jit_smoke():
    grid, det, geom, vol, projs = make_case(6, 6, 6, 6)

    @jax.jit
    def loss(p):
        return jnp.sum(fbp(geom, grid, det, p, filter_name="ramp", scale=1.0))

    value = loss(projs)

    assert bool(jnp.isfinite(value))


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires real Pallas lowering")
def test_pallas_parallel_fbp_helper_matches_generic_on_guard_geometry():
    fbp_mod = import_module("tomojax.recon.fbp")
    size = 64
    views = 90
    cfg = SimConfig(
        nx=size,
        ny=size,
        nz=size,
        nu=size,
        nv=size,
        n_views=views,
        rotation_deg=180.0,
        geometry="parallel",
        phantom="random_shapes",
        n_cubes=8,
        n_spheres=7,
        min_size=4,
        max_size=32,
        seed=42,
    )
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    det = Detector(size, size, 1.0, 1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(
        grid=grid,
        detector=det,
        thetas_deg=np.linspace(0.0, 180.0, views, endpoint=False).astype(np.float32),
    )
    volume = jnp.asarray(make_phantom(cfg), dtype=jnp.float32)
    projs = jnp.stack(
        [
            forward_project_view(geom, grid, det, volume, view_index=i)
            for i in range(views)
        ],
        axis=0,
    )
    poses = stack_view_poses(geom, views)

    pallas = fbp_mod._run_parallel_fbp_direct_pallas(
        poses,
        projs,
        grid=grid,
        detector=det,
        filter_name="ramp",
    ) * jnp.float32(_default_fbp_scale(views))
    generic = fbp(
        geom,
        grid,
        det,
        projs,
        filter_name="ramp",
        det_grid=get_detector_grid_device(det),
    )

    assert _relative_l2(pallas, generic) <= 0.02
    assert np.max(np.abs(np.asarray(pallas) - np.asarray(generic))) <= 0.03


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires real Pallas lowering")
def test_pallas_parallel_fbp_helper_changes_with_projection_input():
    fbp_mod = import_module("tomojax.recon.fbp")
    grid, det, geom, vol, projs = make_case(32, 32, 32, 32, asymmetric=True)
    poses = stack_view_poses(geom, int(projs.shape[0]))

    base = fbp_mod._run_parallel_fbp_direct_pallas(
        poses,
        projs,
        grid=grid,
        detector=det,
        filter_name="ramp",
    )
    changed = fbp_mod._run_parallel_fbp_direct_pallas(
        poses,
        projs.at[:, :, projs.shape[-1] // 2].add(jnp.float32(0.01)),
        grid=grid,
        detector=det,
        filter_name="ramp",
    )

    assert _relative_l2(changed, base) > 1e-5


def test_fbp_batch_equivalence():
    grid, det, geom, vol, projs = make_case(12, 12, 12, 12)
    rec_all = fbp(geom, grid, det, projs, filter_name="ramp", views_per_batch=0)
    rec_b3 = fbp(
        geom,
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=3,
    )
    assert np.allclose(np.asarray(rec_all), np.asarray(rec_b3), atol=1e-3)


def test_fbp_batch_equivalence_with_padded_tail():
    grid, det, geom, vol, projs = make_case(12, 12, 12, 10)
    rec_all = fbp(geom, grid, det, projs, filter_name="ramp", views_per_batch=0)
    rec_b4 = fbp(
        geom,
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=4,
    )
    assert np.allclose(np.asarray(rec_all), np.asarray(rec_b4), atol=1e-3)


def test_fbp_config_matches_legacy_keywords():
    grid, det, geom, vol, projs = make_case(8, 8, 8, 8)
    rec_config = fbp(
        geom,
        grid,
        det,
        projs,
        config=FBPConfig(filter_name="ramp", views_per_batch=2),
    )
    rec_legacy = fbp(
        geom,
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=2,
    )

    assert np.allclose(np.asarray(rec_config), np.asarray(rec_legacy), atol=1e-3)


def test_fbp_legacy_keywords_override_config():
    grid, det, geom, vol, projs = make_case(8, 8, 8, 8)
    rec_config = fbp(
        geom,
        grid,
        det,
        projs,
        config=FBPConfig(filter_name="hann", views_per_batch=1),
        filter_name="ramp",
        views_per_batch=2,
    )
    rec_legacy = fbp(
        geom,
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=2,
    )

    assert np.allclose(np.asarray(rec_config), np.asarray(rec_legacy), atol=1e-3)


def test_fbp_oom_retries_until_all_views_are_processed(monkeypatch):
    fbp_mod = import_module("tomojax.recon.fbp")

    grid = Grid(nx=2, ny=2, nz=1, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=2, nv=1, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    class DummyGeom:
        def pose_for_view(self, i: int):
            return np.eye(4, dtype=np.float32)

    projs = jnp.ones((5, 1, 2), dtype=jnp.float32)

    fast_path_calls = []
    row_trace_shapes = []
    batch_trace_shapes = []

    def fake_fast_path(*args, **kwargs):
        fast_path_calls.append("called")
        raise RuntimeError("RESOURCE_EXHAUSTED: simulated")

    def fake_filter_rows(rows, du, filter_name):
        row_trace_shapes.append(int(rows.shape[0]))
        return rows

    def fake_bp_batch_sum(T_chunk, filt_chunk, *, grid, detector, **kwargs):
        batch_trace_shapes.append(int(T_chunk.shape[0]))
        return jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32) * jnp.sum(
            jnp.mean(filt_chunk, axis=(1, 2)),
            dtype=jnp.float32,
        )

    monkeypatch.setattr(fbp_mod, "_run_fbp_fast_path", fake_fast_path)
    monkeypatch.setattr(fbp_mod, "_fft_filter_rows_jit", fake_filter_rows)
    monkeypatch.setattr(fbp_mod, "_bp_batch_sum_jit", fake_bp_batch_sum)

    rec = fbp_mod.fbp(
        DummyGeom(),
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=4,
        scale=1.0 / float(projs.shape[0]),
    )

    assert np.allclose(np.asarray(rec), np.ones((2, 2, 1), dtype=np.float32))
    assert fast_path_calls == ["called"]
    assert row_trace_shapes == [4, 4]
    assert batch_trace_shapes == [4, 4]
