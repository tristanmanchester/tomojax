"""SPDHG's volume state is rewritten in place, so a full-size scan's solve fits beside it."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.recon import SPDHGConfig
from tomojax.recon.spdhg_tv import (  # check-public-imports: allow-private
    _prepare_spdhg_runtime,
    _run_spdhg_scan,
    _tv_dual_step,
    _update_spdhg_tv_duals,
)


@pytest.mark.parametrize("regularizer", ["tv", "huber_tv"])
def test_slab_dual_update_gives_the_whole_volume_bits(regularizer):
    # 512 x 512 planes make four-plane slabs: two slabs and a three-plane tail.
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(11, 512, 512)), jnp.float32)
    p = [jnp.asarray(rng.normal(size=x.shape) * 0.02, jnp.float32) for _ in range(3)]
    scales, sigma, weight, delta = (0.7, 1.3, 0.9), 0.37, 0.05, 0.11
    runtime = SimpleNamespace(
        step_sizes=SimpleNamespace(sigma_tv=sigma),
        metric=SimpleNamespace(gradient_scale_xyz=scales, weight=weight),
        regularizer=regularizer,
        huber_delta=delta,
    )

    @jax.jit
    def slabs(x, p):
        return _update_spdhg_tv_duals(SimpleNamespace(p1=p[0], p2=p[1], p3=p[2]), x, runtime)

    @jax.jit
    def whole(x, p):
        # Neumann forward differences along x, y and z, over the whole volume at once.
        def forward(axis):
            edge = jnp.zeros_like(jax.lax.slice_in_dim(x, 0, 1, axis=axis))
            return jnp.concatenate([jnp.diff(x, axis=axis), edge], axis=axis)

        gradient = tuple(c * forward(axis) for c, axis in zip(scales, (2, 1, 0), strict=True))
        return _tv_dual_step(tuple(p), gradient, runtime)

    # Both compiled, as in the solve, so both contract the same multiply-adds.
    for a, e in zip(slabs(x, p), whole(x, p), strict=True):
        np.testing.assert_array_equal(a, e)


@pytest.mark.gpu
def test_the_solve_keeps_at_most_three_volumes_beside_its_donated_state():
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    # Large planes, as on a full-size scan: a slab is two of them. Main kept four
    # whole-volume copies here (the TV duals and the data backprojection), donating none.
    grid, detector = Grid(1024, 1024, 8, 1.0, 1.0, 1.0), Detector(1024, 8, 1.0, 1.0)
    geometry = LaminographyGeometry(grid, detector, np.linspace(0, 360, 8, endpoint=False), 30)
    data = np.random.default_rng(4).random((8, 8, 1024), dtype=np.float32)
    config = SPDHGConfig(iterations=2, tv_weight=0.01, views_per_batch=4, stream=False)
    runtime = _prepare_spdhg_runtime(
        geometry, grid, detector, data, weights=None, init_x=None, config=config, det_grid=None
    )
    volumes = tuple(jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32) for _ in range(5))
    memory = _run_spdhg_scan.lower(grid, detector, runtime, volumes).compile().memory_analysis()
    volume = 4 * grid.nx * grid.ny * grid.nz
    assert memory.alias_size_in_bytes == 5 * volume
    assert memory.temp_size_in_bytes <= 3 * volume
