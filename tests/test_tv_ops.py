from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.recon._tv_ops import div3, grad3


def test_grad3_and_div3_are_negative_adjoints():
    u = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    px = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 1.0) / 10.0
    py = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 2.0) / 11.0
    pz = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 3.0) / 12.0

    dx, dy, dz = grad3(u)
    lhs = jnp.vdot(dx, px) + jnp.vdot(dy, py) + jnp.vdot(dz, pz)
    rhs = -jnp.vdot(u, div3(px, py, pz))

    assert lhs == pytest.approx(float(rhs), rel=1e-6, abs=1e-6)


def test_div3_handles_singleton_axes():
    px = jnp.ones((1, 2, 3), dtype=jnp.float32)
    py = jnp.ones((1, 1, 3), dtype=jnp.float32)
    pz = jnp.ones((1, 2, 1), dtype=jnp.float32)

    div = div3(px, py, pz)

    assert div.shape == (1, 2, 3)
    assert np.isfinite(np.asarray(div)).all()
