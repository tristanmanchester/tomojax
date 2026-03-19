import numpy as np
import jax.numpy as jnp
from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view
from tomojax.recon.fbp import fbp


def make_case(nx=12, ny=12, nz=12, n_views=12):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = np.linspace(0, 180, n_views, endpoint=False)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32).at[
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


def test_fbp_oom_retries_until_all_views_are_processed(monkeypatch):
    import tomojax.recon.fbp as fbp_mod

    grid = Grid(nx=2, ny=2, nz=1, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=2, nv=1, du=1.0, dv=1.0, det_center=(0.0, 0.0))

    class DummyGeom:
        def pose_for_view(self, i: int):
            return np.eye(4, dtype=np.float32)

    projs = jnp.ones((5, 1, 2), dtype=jnp.float32)

    monkeypatch.setattr(
        fbp_mod,
        "_fft_filter_rows",
        lambda rows, du, filter_name: rows,
    )
    monkeypatch.setattr(
        fbp_mod,
        "_bp_one_jit",
        lambda T, grid, detector, F, **kwargs: jnp.ones(
            (grid.nx, grid.ny, grid.nz), dtype=jnp.float32
        ),
    )

    calls_by_chunk = {}

    def fake_scan(body, init, xs, length=None, unroll=1):
        T_chunk, filt_chunk = xs
        cur = int(T_chunk.shape[0])
        calls_by_chunk[cur] = calls_by_chunk.get(cur, 0) + 1
        if cur == 4 and calls_by_chunk[cur] == 1:
            raise RuntimeError("RESOURCE_EXHAUSTED: simulated")
        acc = init
        for i in range(cur):
            acc, _ = body(acc, (T_chunk[i], filt_chunk[i]))
        return acc, None

    monkeypatch.setattr(fbp_mod.jax.lax, "scan", fake_scan)

    rec = fbp_mod.fbp(
        DummyGeom(),
        grid,
        det,
        projs,
        filter_name="ramp",
        views_per_batch=4,
    )

    assert np.allclose(np.asarray(rec), np.ones((2, 2, 1), dtype=np.float32))
    assert calls_by_chunk[4] == 1
    assert calls_by_chunk[2] >= 2
